"""Can Parakeet Redux take a 90-minute recording in one call, and what comes back?

Builds one long single-channel file by concatenating channel 0 of all eleven
HCRC Map Task dialogues in audio/corpora/maptask/ (20 kHz, in sorted filename
order), writes it as a 16-bit WAV to a temporary directory (outside the repo;
audio is never committed), and passes the PATH to
`transcribe(audio=path, timestamps="segment")`, the documented way to hand
Photon a long encoded file.

Four of the eleven channels reach full scale, and Photon rejects near-full-
scale audio at any rate other than 16 kHz (see clipping.py). The probe first
shows that failure on the unmodified file, then scales the whole concatenation
by GAIN = 0.5 (about -6 dB) and uses that scaled audio for every measurement
below, both the long file and the per-file runs, so the comparison is like for
like.

Reports: audio duration, wall time and RTF (n=1: a single 90-minute call; it
is a capability check, and its RTF is cross-checked against the per-file
runs below), segment count, segment length distribution and whether any
segment exceeds 30 s, whether segment times are monotonic and end near the
audio's end, word count, and peak resident memory of this process.

Then transcribes each of the eleven channels separately (PCM arrays at native
rate) and compares the concatenation of those transcripts with the single
long-form transcript using `textnorm.wer`, to show whether the long-form path
changes the words.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/longform.py \
        > models/parakeet-redux/results/longform.log
"""

from __future__ import annotations

import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import moondream as md  # noqa: E402

from playground.audio import load_channel  # noqa: E402
from textnorm import wer  # noqa: E402

files = sorted(Path("audio/corpora/maptask").glob("*.wav"))
parts, rates = [], set()
for f in files:
    x, rate = load_channel(f, 0)
    parts.append(x)
    rates.add(rate)
assert len(rates) == 1, rates
rate = rates.pop()
GAIN = np.float32(0.5)
long_raw = np.concatenate(parts)
long = long_raw * GAIN
parts = [x * GAIN for x in parts]
dur = len(long) / rate
print(f"input: channel 0 of {len(files)} Map Task files concatenated, {dur:.1f} s = {dur / 60:.1f} min at {rate} Hz")

with tempfile.TemporaryDirectory() as tmp, md.photon("moondream/parakeet-redux", device="cpu") as sp:
    raw = Path(tmp) / "long_raw.wav"
    sf.write(str(raw), long_raw, rate, subtype="PCM_16")
    print(f"peak |sample| before gain: {np.abs(long_raw).max():.4f}; after gain {GAIN}: {np.abs(long).max():.4f}")
    try:
        sp.transcribe(audio=str(raw), timestamps="segment")
        print("unscaled long file: accepted")
    except Exception as e:  # noqa: BLE001
        print(f"unscaled long file: {type(e).__name__}: {e}")
    raw.unlink()
    wav = Path(tmp) / "long.wav"
    sf.write(str(wav), long, rate, subtype="PCM_16")
    print(f"temporary WAV: {wav.stat().st_size / 1e6:.1f} MB")
    sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

    t0 = time.monotonic()
    r = sp.transcribe(audio=str(wav), timestamps="segment")
    el = time.monotonic() - t0
    segs = r["segments"]
    lens = [s["end"] - s["start"] for s in segs]
    starts = [s["start"] for s in segs]
    print("\n### single call on the whole file (n=1)")
    print(f"  wall {el:.1f} s  RTF {dur / el:.1f}x")
    print(f"  result duration_seconds={r.get('duration_seconds')} source_duration_seconds={r.get('source_duration_seconds')}")
    print(f"  segments: {len(segs)}; length min/med/max {min(lens):.2f}/{statistics.median(lens):.2f}/{max(lens):.2f} s;"
          f" over 30 s: {sum(v > 30.0 for v in lens)}")
    print(f"  starts monotonic: {all(a <= b for a, b in zip(starts, starts[1:]))};"
          f" first start {segs[0]['start']:.2f} s; last end {segs[-1]['end']:.2f} s of {dur:.1f} s")
    gaps = [b["start"] - a["end"] for a, b in zip(segs, segs[1:])]
    print(f"  largest gap between consecutive segments: {max(gaps):.1f} s")
    words_long = len(r["text"].split())
    print(f"  words: {words_long}")
    print(f"  peak RSS of this process so far: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

    print("\n### per-file transcription of the same eleven channels, concatenated")
    t0 = time.monotonic()
    per = [sp.transcribe(audio=x, sample_rate=rate, timestamps="none")["text"] for x in parts]
    el2 = time.monotonic() - t0
    joined = " ".join(per)
    print(f"  wall {el2:.1f} s  RTF {dur / el2:.1f}x  words {len(joined.split())}")
    w = wer(joined, r["text"])
    print(f"  long-form vs per-file: {w['errors']} word diffs / {w['ref_words']} words = {w['wer'] * 100:.2f}%"
          f" (S={w['S']} D={w['D']} I={w['I']})")
    print(f"  peak RSS of this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
