"""What does resampling before Photon do to Parakeet Redux's transcript?

Three input paths for the same channel of the same file:

  (a) native   : `playground.audio.load_channel` -> Photon at the file's own
                 rate. Photon resamples to the model's 16 kHz internally. This
                 is what `playground.asr` does, and the reference here.
  (b) harness  : `playground.audio.load_pcm16` -> Photon at 24 kHz. This is the
                 path the VoiceChat harness applies to every file: resample to
                 WIRE_RATE (24 kHz) with `audio._resample`. For a file
                 already at 24 kHz it is a no-op.
  (c) naive16k : `audio._resample(x, rate, 16000)` -> Photon at 16 kHz,
                 straight to the model's rate, the "obvious" shortcut. This
                 is what the old pk_exp.py section E measured, on one clip.

It measures whatever `audio._resample` currently is. results/resample.log was
produced with the original unfiltered linear interpolator;
results/resample-after-fix.log with the band-limited polyphase replacement.

Inputs: caller channel 0 of the three 24 kHz fixtures, and both channels of
two HCRC Map Task dialogues (20 kHz, so path (b) really does change the rate).

For each input, every pair is compared with `textnorm.wer` (a word-level edit
distance over normalised text): (b) vs (a), (c) vs (a), (c) vs (b). The number
of differing words, the reference word count, and the differing words
themselves are printed. The transcripts are deterministic (checked: two runs of
(a) are compared too), so one transcription per path is enough; n below is the
number of inputs.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/resample.py \
        > models/parakeet-redux/results/resample.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import moondream as md  # noqa: E402

from playground.audio import _resample, load_channel, load_pcm16  # noqa: E402
from playground.protocol import WIRE_RATE  # noqa: E402
from textnorm import errors_only, wer  # noqa: E402

INPUTS = [
    ("audio/tool_call.wav", 0),
    ("audio/interruptions.wav", 0),
    ("audio/turn_taking.wav", 0),
    ("audio/corpora/maptask/q4ec1.mix.wav", 0),
    ("audio/corpora/maptask/q4ec1.mix.wav", 1),
    ("audio/corpora/maptask/q1ec1.mix.wav", 0),
    ("audio/corpora/maptask/q1ec1.mix.wav", 1),
]


def show(label, r, limit=12):
    errs = errors_only(r["ops"])
    more = f" (+{len(errs) - limit} more)" if len(errs) > limit else ""
    print(f"    {label:9s} {r['errors']:3d} word diffs / {r['ref_words']:4d} ref words"
          f" = {r['wer'] * 100:5.2f}%  (S={r['S']} D={r['D']} I={r['I']})")
    for e in errs[:limit]:
        print(f"        {e}")
    if more:
        print(f"       {more}")


totals: dict[tuple[str, str], list[int]] = {}
with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    print(f"WIRE_RATE = {WIRE_RATE}")
    for path, ch in INPUTS:
        p = Path(path)
        xa, rate = load_channel(p, ch)
        xb = load_pcm16(p, ch)
        xc = _resample(xa, rate, 16000)
        ta = sp.transcribe(audio=xa, sample_rate=rate, timestamps="none")["text"]
        ta2 = sp.transcribe(audio=xa, sample_rate=rate, timestamps="none")["text"]
        tb = sp.transcribe(audio=xb, sample_rate=WIRE_RATE, timestamps="none")["text"]
        tc = sp.transcribe(audio=xc, sample_rate=16000, timestamps="none")["text"]
        b_noop = rate == WIRE_RATE and np.array_equal(xa, xb)
        print(f"=== {p.name} ch{ch}: {len(xa) / rate:.1f} s at {rate} Hz;"
              f" (b) input identical to (a): {b_noop}; (a) repeat identical: {ta == ta2}")
        print(f"    byte-identical text: b==a {tb == ta}, c==a {tc == ta}, c==b {tc == tb}")
        for key, lab, ref, hyp in (("b_vs_a", "(b)vs(a)", ta, tb),
                                   ("c_vs_a", "(c)vs(a)", ta, tc),
                                   ("c_vs_b", "(c)vs(b)", tb, tc)):
            r = wer(ref, hyp)
            grp = "24 kHz fixtures" if rate == WIRE_RATE else f"{rate // 1000} kHz Map Task"
            for g in (grp, "all"):
                t = totals.setdefault((g, key), [0, 0, 0])
                t[0] += r["errors"]
                t[1] += r["ref_words"]
                t[2] += 1
            show(lab, r)

print("\n=== pooled")
for (g, key), (e, n, k) in totals.items():
    print(f"  {g:18s} {key}: {e} word diffs / {n} ref words = {e / n * 100:.2f}%  (n={k} inputs)")
