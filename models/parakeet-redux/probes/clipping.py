"""Does Photon reject loud audio? Peak level x sample rate, for Parakeet Redux.

Found while building the long-form test: a Map Task channel that touches full
scale made `transcribe` raise
`ValueError: PCM must contain finite samples in [-1, 1]`, although every input
sample was inside [-1, 1]. This probe maps where that happens.

  A. Synthetic: 3 s of silence with one 100-sample pulse of +peak and one of
     -peak, for peak in {0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0} and sample rate
     in {16000, 20000, 24000, 44100, 48000}, passed as a PCM array. Prints OK
     or the exception type for each cell.
  B. Real audio: channel 0 of HCRC Map Task q5nc2 (20 kHz, peaks at full
     scale), passed as an array at gains 1.0, 0.99, 0.95, 0.9, 0.8, 0.5, and
     once written to a 16-bit WAV and passed as a path.
  C. The peak level of channel 0 and 1 of every Map Task file and of the three
     fixtures, so it is visible which inputs are exposed.

No timing. Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/clipping.py \
        > models/parakeet-redux/results/clipping.log
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "src")
import moondream as md  # noqa: E402

from playground.audio import load_channel  # noqa: E402

PEAKS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
RATES = [16000, 20000, 24000, 44100, 48000]


def attempt(sp, **kw) -> str:
    try:
        r = sp.transcribe(**kw)
        return f"OK ({len(r['text'].split())} words)"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    print("### A. synthetic pulse, array input: rows = peak, columns = sample rate")
    print("  peak  " + "".join(f"{r:>9d}" for r in RATES))
    for peak in PEAKS:
        cells = []
        for rate in RATES:
            y = np.zeros(rate * 3, dtype=np.float32)
            y[rate // 2: rate // 2 + 100] = peak
            y[rate: rate + 100] = -peak
            try:
                sp.transcribe(audio=y, sample_rate=rate)
                cells.append("OK")
            except ValueError:
                cells.append("ValueErr")
            except Exception as e:  # noqa: BLE001
                cells.append(type(e).__name__[:8])
        print(f"  {peak:4.2f}  " + "".join(f"{c:>9s}" for c in cells))

    print("\n### B. real audio: Map Task q5nc2 channel 0")
    x, rate = load_channel(Path("audio/corpora/maptask/q5nc2.mix.wav"), 0)
    print(f"  {len(x) / rate:.1f} s at {rate} Hz, peak |x| = {np.abs(x).max():.5f}")
    for g in [1.0, 0.99, 0.95, 0.9, 0.8, 0.5]:
        print(f"  array, gain {g:4.2f}: {attempt(sp, audio=x * np.float32(g), sample_rate=rate)}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "q5nc2_ch0.wav"
        sf.write(str(wav), x, rate, subtype="PCM_16")
        print(f"  16-bit WAV path, gain 1.00: {attempt(sp, audio=str(wav))}")

print("\n### C. peak |sample| per input channel")
for p in sorted(Path("audio/corpora/maptask").glob("*.wav")) + [
    Path(f"audio/{s}.wav") for s in ("tool_call", "interruptions", "turn_taking")
]:
    d, rate = sf.read(str(p), dtype="float32", always_2d=True)
    peaks = " ".join(f"ch{c}={np.abs(d[:, c]).max():.4f}" for c in range(d.shape[1]))
    print(f"  {p.name:20s} {rate} Hz  {peaks}")
