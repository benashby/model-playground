"""Full-duplex realtime client: one send stream, one receive stream, concurrently."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from . import protocol as proto
from .audio import from_bytes, paced_chunks, write_wav
from .policy import Decision, PendingCall, decide
from .tools import Registry


# Agent speech vs streamed silence. Measured on captured agent audio: silent
# chunks sit at RMS 0-a few, voiced chunks run 150-1400 (p50 153, p95 796). Any
# threshold in the tens separates them cleanly; 50 is comfortably inside the gap.
VOICED_RMS = 50.0


def _is_voiced(pcm: bytes) -> bool:
    """True if this wire chunk carries speech rather than streamed silence."""
    if not pcm:
        return False
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return bool(samples.size and np.sqrt((samples ** 2).mean()) > VOICED_RMS)


@dataclass
class AgentSpec:
    """Everything that defines the agent under test."""

    instructions: str
    registry: Registry
    name: str = "unnamed"

    def session_event(self) -> dict[str, Any]:
        # ASCII-only, per the model card's constraint on system prompts.
        ascii_instructions = self.instructions.encode("ascii", "backslashreplace").decode("ascii")
        return proto.session_update(ascii_instructions, self.registry.schemas())


@dataclass
class Transcript:
    """Timestamped event log. The actual artefact of a run."""

    t0: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        entry = {"t": round(time.monotonic() - self.t0, 4), "kind": kind, **fields}
        self.events.append(entry)
        return entry

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for e in self.events:
                fh.write(json.dumps(e) + "\n")


class DuplexSession:
    """Drives one conversation against the model.

    Two asyncio tasks run for the whole session: a sender walking the caller
    audio at wall-clock rate, and a receiver draining the socket. They are
    independent on purpose -- that independence IS full duplex. A request/response
    loop here would quietly serialize the thing under test.
    """

    def __init__(
        self,
        url: str,
        spec: AgentSpec,
        *,
        transcript: Transcript | None = None,
        verbose: bool = True,
        on_audio: Callable[[bytes], None] | None = None,
    ) -> None:
        self.url = url
        self.spec = spec
        self.log = transcript or Transcript()
        self.verbose = verbose
        # Called with raw PCM16 as each delta lands, for live playback. Must not
        # block -- it runs inside the receive loop.
        self.on_audio = on_audio

        self.ws: websockets.ClientConnection | None = None
        self.output_audio: list[np.ndarray] = []
        self.pending: dict[str, PendingCall] = {}
        self.held: dict[str, str] = {}
        # Set when the caller stops talking, cleared by the first agent audio
        # delta. Without it the moment the agent STARTS speaking is never
        # timestamped -- only the transcript .done, which lands a whole
        # utterance later and overstates turn-taking latency by seconds.
        self._awaiting_reply = False
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._done = asyncio.Event()
        # Incremental caller transcript for the current utterance. Feeds
        # policy.decide(), whose backchannel branch is dead code without it.
        self._partial: str = ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def run(self, caller_audio: np.ndarray, *, speed: float = 1.0, linger: float = 8.0) -> None:
        """Drive the session from a fixed array, paced at wall clock."""

        async def chunks() -> AsyncIterator[bytes]:
            async for _idx, chunk in paced_chunks(caller_audio, speed=speed):
                yield chunk

        await self.run_stream(chunks(), linger=linger)

    async def run_stream(self, source: AsyncIterator[bytes], *, linger: float = 8.0) -> None:
        """Drive the session from an arbitrary async source of PCM16 chunks.

        A live microphone is already real-time by construction, so it needs no
        pacing -- it simply yields as fast as the device produces. A file needs
        paced_chunks to impose the same property artificially. Both arrive here.
        """
        async with websockets.connect(self.url, max_size=None, ping_interval=20) as ws:
            self.ws = ws
            await self._send(self.spec.session_event())
            self.log.record("session.update.sent", tools=list(self.spec.registry.tools))

            receiver = asyncio.create_task(self._receive_loop(), name="receiver")
            sender = asyncio.create_task(self._send_loop(source), name="sender")

            try:
                await sender
                # Caller audio is exhausted, but the agent is very likely still
                # mid-utterance -- and any in-flight tool still has to land.
                # Cutting the socket here is the most common way to make a model
                # look like it failed to answer when it simply was not done.
                self.log.record("caller.audio.exhausted", linger_s=linger)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._done.wait(), timeout=linger)
            finally:
                for t in self._tool_tasks:
                    t.cancel()
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
                with contextlib.suppress(Exception):
                    await self._send(proto.session_close())

    async def _send(self, event: dict[str, Any]) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(event))

    # ------------------------------------------------------------------
    # outbound: the caller side of the conversation
    # ------------------------------------------------------------------

    async def _send_loop(self, source: AsyncIterator[bytes]) -> None:
        first = True
        async for chunk in source:
            await self._send(proto.audio_append(chunk))
            if first:
                self.log.record("caller.audio.first_chunk")
                first = False

    # ------------------------------------------------------------------
    # inbound: agent audio, transcripts, tool calls
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        assert self.ws is not None
        async for raw in self.ws:
            event = json.loads(raw)
            await self._dispatch(event)

    async def _dispatch(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")

        if etype == "response.output_audio.delta":
            pcm = proto.decode_audio_delta(event)
            if self._awaiting_reply and _is_voiced(pcm):
                # THE turn-taking measurement: caller stopped -> agent AUDIBLE.
                #
                # Not "first audio delta". A full-duplex model streams its
                # output channel continuously and emits digital silence when it
                # is not speaking -- measured at 42-65% near-silent chunks, with
                # a 5th-percentile RMS of exactly 0.0, across an agent track
                # whose duration matches the fixture's. So a delta always
                # arrives immediately after speech_stopped, and timing the first
                # one reports ~0.000 s for every turn: a plausible-looking
                # number that measures the transport, not the model.
                self.log.record("agent.audio.first_voiced")
                self._awaiting_reply = False
            self.output_audio.append(from_bytes(pcm))
            if self.on_audio is not None:
                self.on_audio(pcm)
            return

        if etype == "response.output_audio_transcript.delta":
            return  # too chatty to log per-token; the .done carries the text

        if etype == "conversation.item.input_audio_transcription.delta":
            # Accumulate so decide() sees what the caller is saying as they
            # interrupt. speech_started fires on the first token, so this is
            # usually a word or two -- a weak hint, which is how it is treated.
            self._partial += event.get("delta", "")
            return

        if etype == "input_audio_buffer.speech_stopped":
            self.log.record("caller.speech_stopped")
            self._awaiting_reply = True
            return

        if etype == "response.output_audio_transcript.done":
            text = event.get("transcript", "")
            self.log.record("agent.said", text=text)
            self._say(f"AGENT: {text}")
            return

        if etype == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript", "")
            self.log.record("caller.said", text=text)
            self._say(f"CALLER: {text}")
            self._partial = ""
            return

        if etype == "input_audio_buffer.speech_started":
            self.log.record("caller.speech_started", pending=list(self.pending))
            self._on_barge_in()
            return

        if etype == "response.function_call_arguments.done":
            self._on_tool_call(proto.ToolCall.parse(event))
            return

        if etype == "response.done":
            usage = event.get("response", {}).get("usage")
            self.log.record("response.done", usage=usage)
            return

        if etype == "session.end":
            self.log.record("session.end", stats=event.get("stats"))
            self._done.set()
            return

        if etype == "error" or event.get("code") in proto.ERROR_CODES:
            self.log.record("error", payload=event)
            self._say(f"!! ERROR {event.get('code')}: {event.get('message')}")
            return

        self.log.record("unhandled", type=etype)

    # ------------------------------------------------------------------
    # tool calling
    # ------------------------------------------------------------------

    def _on_tool_call(self, call: proto.ToolCall) -> None:
        tool = self.spec.registry.get(call.name)
        self.log.record("tool.call", name=call.name, call_id=call.call_id, args=call.arguments)
        self._say(f"TOOL -> {call.name}({json.dumps(call.arguments)})")

        if tool is None:
            # Answer anyway. Leaving a call_id unanswered can wedge the turn,
            # and "model invented a tool that was never registered" is a finding
            # worth capturing rather than a reason to crash.
            self.log.record("tool.unknown", name=call.name)
            self._reply_later(call.call_id, json.dumps({"error": f"unknown tool {call.name}"}))
            return

        self.pending[call.call_id] = PendingCall(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            started_at=time.monotonic(),
            elapsed=0.0,
            expected_latency=tool.latency,
        )

        # create_task, never await: awaiting here would block the receive loop,
        # which means the client would stop draining agent audio for the whole
        # tool duration. The model would still be talking; we simply would not
        # hear it. That single await would destroy the property under test.
        task = asyncio.create_task(self._run_tool(tool, call), name=f"tool:{call.name}")
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool(self, tool: Any, call: proto.ToolCall) -> None:
        started = time.monotonic()
        try:
            output = await tool.invoke(call.arguments)
        except Exception as exc:  # noqa: BLE001 - a tool raising is a valid scenario
            output = json.dumps({"error": type(exc).__name__, "detail": str(exc)[:200]})
            self.log.record("tool.raised", call_id=call.call_id, error=str(exc)[:200])

        duration = time.monotonic() - started
        entry = self.pending.pop(call.call_id, None)

        if entry is None:
            # Barge-in handling already resolved this call.
            decision = self.held.pop(call.call_id, None)
            self.log.record(
                "tool.completed_after_resolution",
                call_id=call.call_id,
                duration=round(duration, 4),
                disposition=decision or "cancelled",
            )
            if decision == Decision.COMPLETE_AND_HOLD.value:
                self.held[call.call_id] = output
            return

        self.log.record("tool.result", call_id=call.call_id, duration=round(duration, 4))
        await self._send(proto.function_call_output(call.call_id, output))

    def _reply_later(self, call_id: str, output: str) -> None:
        task = asyncio.create_task(self._send(proto.function_call_output(call_id, output)))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    # ------------------------------------------------------------------
    # barge-in
    # ------------------------------------------------------------------

    def _on_barge_in(self) -> None:
        if not self.pending:
            return

        now = time.monotonic()
        for call_id, entry in list(self.pending.items()):
            live = PendingCall(
                call_id=entry.call_id,
                name=entry.name,
                arguments=entry.arguments,
                started_at=entry.started_at,
                elapsed=now - entry.started_at,
                expected_latency=entry.expected_latency,
            )
            decision = decide(live, user_transcript_so_far=self._partial)
            self.log.record(
                "barge_in.decision",
                call_id=call_id,
                tool=entry.name,
                elapsed=round(live.elapsed, 4),
                decision=decision.value,
                heard=self._partial,
            )

            if decision is Decision.CANCEL:
                self.pending.pop(call_id, None)
            elif decision is Decision.COMPLETE_AND_HOLD:
                self.pending.pop(call_id, None)
                self.held[call_id] = decision.value
            # COMPLETE_AND_DELIVER: leave it pending; _run_tool sends normally.

    # ------------------------------------------------------------------

    def _say(self, line: str) -> None:
        if self.verbose:
            print(f"[{time.monotonic() - self.log.t0:7.2f}s] {line}", flush=True)

    def save_agent_audio(self, path: Path) -> float:
        """Write the captured agent side. Returns duration in seconds."""
        if not self.output_audio:
            self.log.record("agent.audio.empty")
            return 0.0
        merged = np.concatenate(self.output_audio)
        write_wav(path, merged)
        return len(merged) / proto.WIRE_RATE
