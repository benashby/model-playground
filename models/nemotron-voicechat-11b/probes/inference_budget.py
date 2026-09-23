#!/usr/bin/env python3
"""Per-chunk compute headroom, read from the server's own logs.

The NIM container reports its own inference budget for every audio chunk:

    Inference #400: 67.2ms (budget: 160ms, 2.4x real-time, queue: 0)

This is the server grading itself, which makes it better evidence than anything
the client can time: it excludes network, client scheduling, and the harness
entirely. The 160 ms budget is the wall-clock the server has to process one
chunk and stay real-time; exceeding it means falling behind the caller.

Reads the log on stdin, so it needs no host access of its own and no hostname
appears here or in its output. The container logs are cumulative, so this reads
history rather than generating load -- pair it with a sweep to sample under
whatever conditions that sweep created.

    ssh <gpu-node> 'sudo docker logs voicechat 2>&1' \\
      | python models/nemotron-voicechat-11b/probes/inference_budget.py
"""

from __future__ import annotations

import re
import statistics as st
import sys

LINE = re.compile(r"Inference #(\d+): ([\d.]+)ms \(budget: (\d+)ms, [\d.]+x real-time, queue: (\d+)\)")


def main() -> int:
    out = sys.stdin.read()
    rows = [(int(a), float(b), int(c), int(d)) for a, b, c, d in LINE.findall(out)]
    if not rows:
        print("no inference lines on stdin -- has the container served a session?")
        return 1

    ms = sorted(r[1] for r in rows)
    budgets = {r[2] for r in rows}
    queues = [r[3] for r in rows]
    over = [x for x in ms if x > max(budgets)]

    print(f"samples      : {len(ms)}")
    print(f"budget       : {sorted(budgets)} ms")
    print(f"per-chunk ms : min {ms[0]:.1f}  p50 {st.median(ms):.1f}  "
          f"p95 {ms[int(.95 * len(ms)) - 1]:.1f}  max {ms[-1]:.1f}")
    print(f"mean         : {st.mean(ms):.1f} ms  (stdev {st.pstdev(ms):.1f})")
    print(f"headroom     : {max(budgets) / st.median(ms):.2f}x real-time at p50")
    print(f"over budget  : {len(over)} / {len(ms)} chunks")
    print(f"queue depth  : max {max(queues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
