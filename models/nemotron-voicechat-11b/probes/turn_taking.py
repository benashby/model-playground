#!/usr/bin/env python3
"""Turn-taking latency: caller stops speaking -> agent becomes audible.

NVIDIA's positioning claims ~450 ms. Until 2026-09-23 that claim was not
testable from anything this harness produced: the client consumed
``response.output_audio.delta`` without logging it, so the only agent-side
timestamp was ``response.output_audio_transcript.done`` -- which lands a whole
utterance later and overstates the number by seconds. See the note, section 11.3.

The obvious fix -- timestamp the first audio delta -- is also wrong, and wrong
in a way that looks fine: a full-duplex model streams its output channel
continuously and sends digital silence when it is not speaking, so a delta
always arrives immediately and every turn reports ~0.000 s. ``client.py`` now
records ``agent.audio.first_voiced``: the first chunk after
``caller.speech_stopped`` whose RMS clears the silence floor.

Two caveats it enforces rather than assumes:

  * The FIRST turn of a session is reported separately. Cold start carries a JIT
    compilation spike, and folding it into the distribution inflates every
    percentile while looking like ordinary variance.
  * Turns where a tool was outstanding are excluded from the headline figure.
    The model is waiting on the client, not thinking -- counting those measures
    the harness's injected latency, not the model's responsiveness.

    python models/nemotron-voicechat-11b/probes/turn_taking.py [results/*.jsonl]
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def gaps(path: Path):
    """Yield (index, seconds, tool_outstanding) per completed turn."""
    ev = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out, waiting, pending, idx = [], None, 0, 0
    for e in ev:
        k = e.get("kind")
        if k == "tool.call":
            pending += 1
        elif k == "tool.result":
            pending = max(0, pending - 1)
        elif k == "caller.speech_stopped":
            waiting = (e["t"], pending)
        elif k == "agent.audio.first_voiced" and waiting is not None:
            out.append((idx, round(e["t"] - waiting[0], 4), waiting[1] > 0))
            idx += 1
            waiting = None
    return out


def main(argv: list[str]) -> int:
    # Default to the dedicated turn-taking sessions. Sweep logs are NOT used:
    # the sweep predates the agent.audio.first_voiced fix, and its whole point
    # is holding tools open, which is the one condition this must exclude.
    files = [Path(p) for p in (argv or sorted(glob.glob(str(RESULTS / "turntaking-rep*.jsonl"))))]
    files = [f for f in files if f.exists()]
    if not files:
        print("no session logs found")
        return 1

    first_turn, clean, tool_bound = [], [], []
    for f in files:
        for idx, secs, had_tool in gaps(f):
            (first_turn if idx == 0 else tool_bound if had_tool else clean).append(secs)

    print(f"sessions: {len(files)}   turns: {len(first_turn)+len(clean)+len(tool_bound)}")
    for label, xs in (("first turn (cold/JIT)", first_turn),
                      ("tool outstanding", tool_bound),
                      ("CLEAN turns", clean)):
        if not xs:
            print(f"  {label:<24} n=0")
            continue
        xs = sorted(xs)
        p95 = xs[min(len(xs) - 1, int(.95 * len(xs)))]
        print(f"  {label:<24} n={len(xs):<4} min {xs[0]:.3f}  p50 {st.median(xs):.3f}  "
              f"p95 {p95:.3f}  max {xs[-1]:.3f}")
    if clean:
        print(f"\nvendor claim ~0.450 s; measured p50 {st.median(sorted(clean)):.3f} s "
              f"on clean turns (n={len(clean)})")
    return 0




# --- session runner -------------------------------------------------------
# `python turn_taking.py --run N` drives N sessions itself and writes
# results/turntaking-repK.jsonl, then analyses them. Kept here rather than in
# bargein_sweep.py because this measurement wants LOW tool latency: a pending
# tool makes the model wait on the client, which is not turn-taking.
def run_sessions(n: int, tag: str = "turntaking") -> None:
    import asyncio, importlib.util, json, os, sys as _sys
    root = Path(__file__).resolve().parents[3]
    _sys.path.insert(0, str(root / "src"))
    from playground.audio import load_pcm16
    from playground.client import DuplexSession, Transcript

    mod = importlib.util.spec_from_file_location("nvidia_demo", root / "scenarios" / "nvidia_demo.py")
    m = importlib.util.module_from_spec(mod); mod.loader.exec_module(m)
    for t in m.SPEC.registry.tools.values():
        t.latency = 0.05          # fast enough that no tool is ever outstanding

    host, port = os.environ["PLAYGROUND_HOST"], os.environ.get("PLAYGROUND_PORT", "9000")
    caller = load_pcm16(root / "audio" / "tool_call.wav", channel=0)
    for rep in range(1, n + 1):
        s = DuplexSession(f"ws://{host}:{port}/v1/realtime", m.SPEC, transcript=Transcript())
        asyncio.run(s.run(caller, speed=1.0, linger=8.0))
        (RESULTS / f"{tag}-rep{rep}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in s.log.events))
        print(f"  session {rep} done")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        tag = sys.argv[3] if len(sys.argv) > 3 else "turntaking"
        run_sessions(int(sys.argv[2]), tag)
        raise SystemExit(main(sorted(glob.glob(str(RESULTS / f"{tag}-rep*.jsonl")))))
    raise SystemExit(main(sys.argv[1:]))
