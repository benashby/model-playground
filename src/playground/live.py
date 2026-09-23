"""Talk to the agent with your actual microphone.

Uses pw-record / pw-play subprocesses rather than a Python audio binding.
On NixOS that is the robust choice: pip-installed sounddevice/pyaudio wheels
expect a system libportaudio that a pure-pip venv has no reliable way to find,
whereas PipeWire's own CLI tools are already on PATH and speak raw PCM on
stdin/stdout.

    python -m playground.live scenarios/nvidia_demo.py

WEAR HEADPHONES. Without them the mic captures the agent's own voice and feeds
it straight back as caller audio. The model is full-duplex, so it will hear
itself mid-utterance, treat that as a barge-in, and talk over itself in a loop
that escalates until you kill it. This is not a subtle degradation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from .client import DuplexSession, Transcript
from .protocol import CHUNK_BYTES, WIRE_RATE

RECORD_CMD = [
    "pw-record",
    "--rate", str(WIRE_RATE),
    "--channels", "1",
    "--format", "s16",
    "--raw",
    # 20ms against the 80ms we send: the device hands us audio well before each
    # chunk boundary, so capture latency is never the thing adding to turn delay.
    "--latency", "20ms",
    "-",
]

PLAY_CMD = [
    "pw-play",
    "--rate", str(WIRE_RATE),
    "--channels", "1",
    "--format", "s16",
    "--raw",
    "--latency", "20ms",
    "-",
]


async def mic_chunks(proc: asyncio.subprocess.Process) -> AsyncIterator[bytes]:
    """Yield fixed-size PCM16 chunks from pw-record's stdout.

    No pacing here, deliberately: a capture device already runs at wall clock,
    so readexactly() blocks for exactly as long as the audio takes to exist.
    Adding paced_chunks on top would double-count the delay.
    """
    assert proc.stdout is not None
    while True:
        try:
            yield await proc.stdout.readexactly(CHUNK_BYTES)
        except asyncio.IncompleteReadError:
            return


async def amain(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from .__main__ import load_scenario

    spec = load_scenario(args.scenario)
    url = f"ws://{args.host}:{args.port}/v1/realtime"

    # stderr is INHERITED, not swallowed. These were briefly DEVNULL'd, which
    # hid `sndfile: failed to open audio file "-": Format not recognised` --
    # pw-play rejecting `--container raw` on stdin -- and turned a one-line
    # error into silent playback that looked like a model problem.
    rec = await asyncio.create_subprocess_exec(*RECORD_CMD, stdout=asyncio.subprocess.PIPE)
    play = await asyncio.create_subprocess_exec(*PLAY_CMD, stdin=asyncio.subprocess.PIPE)

    await asyncio.sleep(0.3)
    for name, proc in (("pw-record", rec), ("pw-play", play)):
        if proc.returncode is not None:
            raise SystemExit(f"{name} exited immediately with {proc.returncode} -- see error above")

    def speak(pcm: bytes) -> None:
        # Fire-and-forget into the player's pipe. Never await here -- this runs
        # inside the receive loop, and blocking it would stall tool dispatch and
        # transcript handling for as long as the sink took to accept the write.
        if play.stdin is not None and not play.stdin.is_closing():
            play.stdin.write(pcm)

    session = DuplexSession(url, spec, transcript=Transcript(), on_audio=speak)

    print(f"scenario : {spec.name} ({len(spec.registry.tools)} tools)")
    print(f"server   : {url}")
    print(f"mic      : pw-record @ {WIRE_RATE} Hz mono")
    print("\n  >>> HEADPHONES ON. Speak when ready. Ctrl-C to stop. <<<\n")

    # SIGINT alone is not enough. A session killed by SIGTERM -- `timeout`, a
    # supervisor, a closed terminal -- would otherwise skip the finally block
    # and lose the entire event log, which is the only artefact of the run.
    runner = asyncio.create_task(session.run_stream(mic_chunks(rec), linger=args.linger))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runner.cancel)

    try:
        await runner
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nstopping...")
    finally:
        # Save BEFORE touching the subprocesses. A group-wide SIGTERM (`timeout`,
        # a closed terminal) has already reaped pw-record, so terminate() raises
        # ProcessLookupError -- and when that ran first it escaped the finally
        # and destroyed the run's only artefact. Cleanup is best-effort; the
        # event log is not.
        dur = session.save_agent_audio(args.out_audio)
        session.log.save(args.out_log)
        print(f"\nagent audio : {args.out_audio} ({dur:.1f}s)")
        print(f"event log   : {args.out_log} ({len(session.log.events)} events)")

        for proc in (rec, play):
            with contextlib.suppress(ProcessLookupError, Exception):
                proc.terminate()
        if play.stdin is not None:
            with contextlib.suppress(Exception):
                play.stdin.close()
        for proc in (rec, play):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=3)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="playground.live")
    p.add_argument("scenario", type=Path, help="python file exporting SPEC")
    # No default host. This is a public repository and an external IP does not
    # belong in it; the address is supplied by the environment instead.
    # Set PLAYGROUND_HOST / PLAYGROUND_PORT in the environment to avoid passing
    # the flag every time.
    p.add_argument("--host", default=os.environ.get("PLAYGROUND_HOST"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PLAYGROUND_PORT", "9000")))
    p.add_argument("--linger", type=float, default=3.0)
    p.add_argument("--out-audio", type=Path, default=Path("logs/live-agent.wav"))
    p.add_argument("--out-log", type=Path, default=Path("logs/live.jsonl"))
    args = p.parse_args()
    if not args.host:
        p.error("no host: pass --host or set PLAYGROUND_HOST")
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
