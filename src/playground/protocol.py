"""Event constructors and parsing for the NemotronLabs VoiceChat realtime API.

The wire protocol is the OpenAI Realtime API shape, served by the NIM container
at ws://<host>:9000/v1/realtime. Every message is a JSON object with a `type`
and an `event_id` (UUID). Audio rides as base64 PCM16.

Reference: NVIDIA-NeMo/Speech @ nemotron-labs-voicechat,
voicechat_realtime_instructions/api-reference.md
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any

# The server resamples internally (16 kHz in / 22.05 kHz out); the wire is 24 kHz
# both directions. Do not "helpfully" send 16 kHz -- the server assumes 24.
WIRE_RATE = 24_000
SAMPLE_WIDTH = 2  # PCM16, little-endian, mono

# ~80 ms per the deploy docs. 24000 * 0.08 * 2 bytes = 3840 bytes.
CHUNK_MS = 80
CHUNK_SAMPLES = WIRE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


def _event(type_: str, **fields: Any) -> dict[str, Any]:
    return {"type": type_, "event_id": str(uuid.uuid4()), **fields}


# --------------------------------------------------------------------------
# Client -> server
# --------------------------------------------------------------------------


def session_update(instructions: str | None, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Configure the session. Must be sent immediately after connect.

    `instructions` is applied only to the FIRST inference call -- it is not a
    persistent system prompt re-injected every turn. `tools` must be set before
    any turn that could trigger one, or the server answers `tools_not_set`.
    """
    return _event(
        "session.update",
        session={
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": WIRE_RATE}},
                "output": {"format": {"type": "audio/pcm", "rate": WIRE_RATE}},
            },
            "instructions": instructions,
            "tools": tools,
        },
    )


def audio_append(pcm: bytes) -> dict[str, Any]:
    """One chunk of caller audio. Expects raw PCM16 mono @ 24 kHz."""
    return _event("input_audio_buffer.append", audio=base64.b64encode(pcm).decode("ascii"))


def function_call_output(call_id: str, output: str) -> dict[str, Any]:
    """Hand a tool result back.

    NOTE: this event carries no `event_id` in the reference examples, but the
    server tolerates one. `output` must be a string -- serialize dicts yourself.
    The model card requires tool responses be ASCII-only.
    """
    return {
        "type": "conversation.item.create",
        "event_id": str(uuid.uuid4()),
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }


def session_close() -> dict[str, Any]:
    return _event("session.close")


# --------------------------------------------------------------------------
# Server -> client
# --------------------------------------------------------------------------

ERROR_CODES = {
    "inference_timeout",
    "inference_error",
    "session_timeout",
    "tools_not_set",
}


@dataclass(frozen=True)
class ToolCall:
    """Parsed from `response.function_call_arguments.done`."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, event: dict[str, Any]) -> "ToolCall":
        raw = event.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            # The model emitted malformed JSON. Surface it rather than crashing
            # the receive loop -- "does it emit valid JSON under duplex load" is
            # itself one of the things we are measuring.
            args = {"__unparsed__": raw}
        return cls(call_id=event["call_id"], name=event["name"], arguments=args)


def decode_audio_delta(event: dict[str, Any]) -> bytes:
    return base64.b64decode(event["delta"] if "delta" in event else event["audio"])
