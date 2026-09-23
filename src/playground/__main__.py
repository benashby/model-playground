"""CLI: stream one side of a recorded conversation at a VoiceChat server."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

from .audio import load_pcm16
from .client import AgentSpec, DuplexSession, Transcript


def load_scenario(path: Path) -> AgentSpec:
    """Import a scenario file and pull its module-level SPEC."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import scenario {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "SPEC"):
        raise SystemExit(f"{path} defines no module-level SPEC")
    return module.SPEC


def main() -> int:
    p = argparse.ArgumentParser(prog="playground")
    p.add_argument("scenario", type=Path, help="python file exporting SPEC")
    p.add_argument("--audio", type=Path, required=True, help="caller-side WAV/FLAC")
    p.add_argument(
        "--channel",
        type=int,
        default=0,
        help="channel to treat as the caller in a multi-channel recording (default 0)",
    )
    # No default host. This is a public repository and an external IP does not
    # belong in it; the address is supplied by the environment instead.
    # Set PLAYGROUND_HOST / PLAYGROUND_PORT in the environment to avoid passing
    # the flag every time.
    p.add_argument("--host", default=os.environ.get("PLAYGROUND_HOST"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PLAYGROUND_PORT", "9000")))
    p.add_argument("--speed", type=float, default=1.0, help="!= 1.0 invalidates latency numbers")
    p.add_argument("--linger", type=float, default=8.0, help="seconds to keep listening after input ends")
    p.add_argument("--out-audio", type=Path, default=Path("logs/agent.wav"))
    p.add_argument("--out-log", type=Path, default=Path("logs/session.jsonl"))
    args = p.parse_args()
    if not args.host:
        p.error("no host: pass --host or set PLAYGROUND_HOST")

    spec = load_scenario(args.scenario)
    caller = load_pcm16(args.audio, channel=args.channel)
    url = f"ws://{args.host}:{args.port}/v1/realtime"

    print(f"scenario : {spec.name} ({len(spec.registry.tools)} tools)")
    print(f"caller   : {args.audio.name} ch{args.channel}, {len(caller) / 24000:.1f}s")
    print(f"server   : {url}\n")

    session = DuplexSession(url, spec, transcript=Transcript())
    try:
        asyncio.run(session.run(caller, speed=args.speed, linger=args.linger))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        dur = session.save_agent_audio(args.out_audio)
        session.log.save(args.out_log)
        print(f"\nagent audio : {args.out_audio} ({dur:.1f}s)")
        print(f"event log   : {args.out_log} ({len(session.log.events)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
