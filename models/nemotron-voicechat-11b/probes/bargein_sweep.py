#!/usr/bin/env python3
"""Does full-duplex tool calling hold up when a tool is slow and the caller interrupts?

THE CENTRAL QUESTION OF THIS REPOSITORY, and until now unanswered: across every
run recorded in September, ``speech_started`` always arrived with ``pending=[]``.
The barge-in policy is fully implemented and fully wired, and had never once
executed. A complete-looking code path that does nothing.

The reason was never the model -- it was the fixture. NVIDIA's ``tool_call.wav``
does contain real barge-in (the caller starts talking 0.31 s after the agent
begins its reply), but the injected tool latencies were 0.1 s and 1.5 s, so
every tool had long since completed by the time the caller spoke again.
Measured from ``logs/tool_call.jsonl``:

    tool.call @ 11.2449  ->  next caller.speech_started @ 20.2046   gap  8.96 s
    tool.call @ 26.9193  ->  next caller.speech_started @ 38.7683   gap 11.85 s

So the policy fires if, and only if, injected latency exceeds those gaps. That
makes tool latency the independent variable and needs no live microphone, no
spliced audio, and no performance -- just a sweep, reproducible to the sample.
A live mic would make the interrupt time a random variable; the fixture makes
it a controlled one.

The sweep deliberately brackets both gaps, and the upper end is not artificial:
``tools.py`` documents real tools spanning 50 ms (cache hit) to 10 s (cold
backend), so 8-14 s is the slow-but-real regime this question is about.

    PLAYGROUND_HOST=<host> python models/nemotron-voicechat-11b/probes/bargein_sweep.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from playground.audio import load_pcm16          # noqa: E402
from playground.client import DuplexSession, Transcript  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIXTURE = ROOT / "audio" / "tool_call.wav"

# Brackets the 8.96 s and 11.85 s gaps measured above. 1.5 s is the September
# baseline, included so a non-firing control is in the same dataset as the
# firing cases -- otherwise "it fired" has nothing to be contrasted with.
LATENCIES = [1.5, 8.0, 10.0, 12.0, 14.0]
REPEATS = int(os.environ.get("SWEEP_REPEATS", "3"))


def spec_with_latency(seconds: float):
    """NVIDIA's own demo tool set, with every latency overridden to `seconds`.

    Reusing the demo spec rather than inventing tools matters: tool_call.wav is
    NVIDIA's recording and exercises exactly these tools. A bespoke tool set
    would risk the model simply not calling anything, which would look
    identical to "the policy did not fire".
    """
    import importlib.util

    path = ROOT / "scenarios" / "nvidia_demo.py"
    spec_ = importlib.util.spec_from_file_location("nvidia_demo", path)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    agent = module.SPEC
    for tool in agent.registry.tools.values():
        tool.latency = seconds
    return agent


async def one_run(latency: float, rep: int) -> dict:
    host = os.environ.get("PLAYGROUND_HOST")
    port = os.environ.get("PLAYGROUND_PORT", "9000")
    if not host:
        sys.exit("PLAYGROUND_HOST is not set")

    agent = spec_with_latency(latency)
    caller = load_pcm16(FIXTURE, channel=0)
    session = DuplexSession(f"ws://{host}:{port}/v1/realtime", agent, transcript=Transcript())

    # linger must exceed the injected latency, or the run ends while a tool is
    # still in flight and the result is indistinguishable from a cancelled one.
    await session.run(caller, speed=1.0, linger=latency + 8.0)

    ev = session.log.events
    tag = f"lat{latency:g}-rep{rep}"
    (RESULTS / f"bargein-{tag}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in ev)
    )

    decisions = [e for e in ev if e.get("kind") == "barge_in.decision"]
    return {
        "latency": latency,
        "rep": rep,
        "tool_calls": sum(1 for e in ev if e.get("kind") == "tool.call"),
        "barge_ins": sum(1 for e in ev if e.get("kind") == "caller.speech_started"),
        "policy_fired": len(decisions),
        "decisions": [
            {"tool": d.get("tool"), "elapsed": d.get("elapsed"),
             "decision": d.get("decision"), "heard": d.get("heard")}
            for d in decisions
        ],
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    print(f"fixture={FIXTURE.name} repeats={REPEATS} latencies={LATENCIES}")
    print("(host redacted by design -- probe output is committed)\n")
    rows = []
    for latency in LATENCIES:
        for rep in range(1, REPEATS + 1):
            t0 = time.monotonic()
            row = asyncio.run(one_run(latency, rep))
            row["wall"] = round(time.monotonic() - t0, 1)
            rows.append(row)
            fired = row["policy_fired"]
            mark = "FIRED" if fired else "  -  "
            print(f"  lat={latency:>5.1f}s rep{rep}  tools={row['tool_calls']} "
                  f"bargeins={row['barge_ins']}  policy={mark} ({fired})  {row['wall']}s")
            for d in row["decisions"]:
                print(f"        -> {d['tool']} elapsed={d['elapsed']}s "
                      f"decision={d['decision']} heard={d['heard']!r}")
    (RESULTS / "bargein-sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RESULTS/'bargein-sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
