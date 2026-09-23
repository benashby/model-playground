"""How much slower is the synthetic agent channel than the human caller channel?

Analysis only; runs no model. Parses section A of results/pk_exp.log (3 runs x
3 fixtures x 2 channels, warmup discarded) and, per fixture, compares the
median real-time factor of channel 1 (agent TTS) with channel 0 (human
caller):

    slowdown = 1 - RTF_ch1 / RTF_ch0

i.e. how much less audio per second of wall clock the agent channel gets
through. Prints the per-fixture values and their range.

Run from the repo root:

    python models/parakeet-redux/probes/channel_speed.py \
        > models/parakeet-redux/results/channel_speed.log
"""

import re
from pathlib import Path

LOG = Path("models/parakeet-redux/results/pk_exp.log")
row = re.compile(r"^\s+(\w+)\s+ch(\d)\s+([\d.]+)s\s+RTF\s+([\d.]+)/\s*([\d.]+)/\s*([\d.]+)")
med: dict[tuple[str, int], float] = {}
for line in LOG.read_text().splitlines():
    m = row.match(line)
    if m:
        med[(m.group(1), int(m.group(2)))] = float(m.group(5))

print(f"source: {LOG.name} section A, median RTF of 3 runs per channel")
vals = []
for fx in dict.fromkeys(k[0] for k in med):
    h, a = med[(fx, 0)], med[(fx, 1)]
    s = (1 - a / h) * 100
    vals.append(s)
    print(f"  {fx:14s} ch0 (human) {h:5.1f}x  ch1 (agent TTS) {a:5.1f}x  agent slower by {s:4.1f}%"
          f"  (human faster by {(h / a - 1) * 100:4.1f}%)")
print(f"  range over n={len(vals)} fixtures: agent slower by {min(vals):.1f}% to {max(vals):.1f}%")
