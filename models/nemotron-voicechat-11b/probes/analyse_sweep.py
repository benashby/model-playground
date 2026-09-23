#!/usr/bin/env python3
"""Summarise a barge-in sweep: firing threshold, dispositions, tool behaviour.

Reads the per-run logs bargein_sweep.py writes and reports the three things the
sweep exists to establish:

  1. At what injected tool latency does the barge-in policy actually fire?
  2. What disposition does it choose, and on what evidence?
  3. What does the MODEL do while a tool is outstanding -- which turned out to
     be the more interesting question.

    python models/nemotron-voicechat-11b/probes/analyse_sweep.py
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
# Tools the fixture's caller actually asks for. Anything else the model calls is
# its own invention -- the fixture is NVIDIA's own and its script is fixed.
REQUESTED = {"generate_random_number", "convert_currency"}


def load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    runs = defaultdict(list)
    for p in sorted(glob.glob(str(RESULTS / "bargein-lat*.jsonl"))):
        m = re.search(r"bargein-lat([\d.]+)-rep(\d+)", p)
        if not m or m.group(2) == "0":       # rep0 = ad-hoc confirmation run
            continue
        runs[float(m.group(1))].append(Path(p))

    print(f"{'latency':>8}  {'runs':>4}  {'fired':>10}  {'tools':>11}  {'unrequested':>11}  {'bargeins':>8}")
    print("-" * 66)
    detail = []
    for lat in sorted(runs):
        fired, tools, unreq, bargeins = [], [], [], []
        for f in runs[lat]:
            ev = load(f)
            calls = [e for e in ev if e.get("kind") == "tool.call"]
            fired.append(sum(1 for e in ev if e.get("kind") == "barge_in.decision"))
            tools.append(len(calls))
            unreq.append(sum(1 for c in calls if c.get("name") not in REQUESTED))
            bargeins.append(sum(1 for e in ev if e.get("kind") == "caller.speech_started"))
            for e in ev:
                if e.get("kind") == "barge_in.decision":
                    detail.append((lat, e.get("tool"), e.get("elapsed"),
                                   e.get("decision"), e.get("heard")))
        n = len(runs[lat])
        print(f"{lat:>7.1f}s  {n:>4}  {sum(1 for x in fired if x):>4}/{n:<5} "
              f"{st.mean(tools):>11.1f}  {st.mean(unreq):>11.1f}  {st.mean(bargeins):>8.1f}")

    print("\nevery policy decision recorded:")
    for lat, tool, elapsed, dec, heard in detail:
        print(f"  lat={lat:>5.1f}s  {tool:<24} elapsed={elapsed:>7}s  -> {dec:<20} heard={heard!r}")

    if detail:
        print("\ndispositions:", dict(Counter(d[3] for d in detail)))
        print("transcript available at decision time:",
              dict(Counter("empty" if not d[4] else "non-empty" for d in detail)))

    print("\nunrequested tool calls by latency (the fixture's caller never asks for news):")
    for lat in sorted(runs):
        names = Counter()
        for f in runs[lat]:
            for e in load(f):
                if e.get("kind") == "tool.call" and e.get("name") not in REQUESTED:
                    names[e["name"]] += 1
        print(f"  {lat:>5.1f}s  {dict(names) or '-'}")

    print("\nargument fidelity on convert_currency (caller says twenty-five):")
    for lat in sorted(runs):
        amounts = []
        for f in runs[lat]:
            for e in load(f):
                if e.get("kind") == "tool.call" and e.get("name") == "convert_currency":
                    amounts.append((e.get("args") or {}).get("amount"))
        print(f"  {lat:>5.1f}s  {amounts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
