"""Does the `timestamps` mode change Parakeet Redux's text or its speed?

For each fixture (caller channel 0 of tool_call, interruptions, turn_taking,
native 24 kHz), each of the four modes ("none", "segment", "word",
"character") is timed 6 times. Modes are interleaved round-robin, with the
order rotated each round, so a slow drift in machine state cannot land on one
mode. Round 0 is a warmup and is discarded, leaving n=5 per mode per fixture.

Reports, per fixture and mode: min / median / max wall time, the median
relative to "none", and the number of segments. Then checks that the text is
byte-identical across all modes and all repeats.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/timestamps_modes.py \
        > models/parakeet-redux/results/timestamps_modes.log
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import moondream as md  # noqa: E402

from playground.audio import load_channel  # noqa: E402

FIXTURES = ["tool_call", "interruptions", "turn_taking"]
MODES = ["none", "segment", "word", "character"]
ROUNDS = 6  # round 0 discarded

with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    import numpy as np

    sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    all_rel: dict[str, list[float]] = {m: [] for m in MODES}
    for stem in FIXTURES:
        x, rate = load_channel(Path(f"audio/{stem}.wav"), 0)
        times: dict[str, list[float]] = {m: [] for m in MODES}
        texts: set[str] = set()
        nseg: dict[str, int] = {}
        for rnd in range(ROUNDS):
            order = MODES[rnd % 4:] + MODES[:rnd % 4]
            for mode in order:
                t0 = time.monotonic()
                r = sp.transcribe(audio=x, sample_rate=rate, timestamps=mode)
                el = time.monotonic() - t0
                if rnd > 0:
                    times[mode].append(el)
                texts.add(r["text"])
                nseg[mode] = len(r.get("segments") or [])
        dur = len(x) / rate
        base = statistics.median(times["none"])
        print(f"=== {stem} ch0, {dur:.1f} s audio, n={ROUNDS - 1} per mode (round 0 discarded)")
        for mode in MODES:
            t = times[mode]
            med = statistics.median(t)
            rel = (med / base - 1) * 100
            all_rel[mode].append(rel)
            print(f"  {mode:10s} min/med/max {min(t):.3f}/{med:.3f}/{max(t):.3f} s"
                  f"  median vs none {rel:+5.1f}%  segments={nseg[mode]}")
        print(f"  distinct texts across all modes and repeats: {len(texts)}")
    print("\n=== summary: median time relative to 'none', per fixture")
    for mode in MODES:
        print(f"  {mode:10s} " + "  ".join(f"{v:+5.1f}%" for v in all_rel[mode]))
