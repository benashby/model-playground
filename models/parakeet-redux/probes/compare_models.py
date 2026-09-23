"""Head-to-head on this CPU: Parakeet Redux against the other ASR checkpoints Photon registers.

Models (all through the same Photon runtime, device="cpu"):
    moondream/parakeet-redux        ternary encoder, 178 MB
    nvidia/parakeet-tdt-0.6b-v3     the base model Redux was compressed from
    moondream/parakeet-ultra        full-precision post-trained sibling
    openai/whisper-large-v3-turbo   the obvious non-Parakeet alternative
    Qwen/Qwen3-ASR-0.6B             the smallest other ASR model Photon registers

Inputs: caller channel 0 of tool_call, interruptions and turn_taking at native
24 kHz, plus two HCRC Map Task channels at 20 kHz (q4ec1 ch0, q1ec1 ch1: longer,
spontaneous, unscripted speech), passed as PCM arrays with
timestamps="segment" (the default mode).

Timing: per model, one open (not reported: the first open downloads), a
warmup call on 1 s of silence plus one full pass over all inputs, both
discarded, then 3 timed passes, inputs in rotating order. RTF = audio
seconds / wall seconds per call. n=3 per model per input, n=15 per model.

Agreement: there is NO ground-truth transcript for these fixtures, so nothing
here ranks accuracy. The probe prints each model's transcript and a pairwise
word-level disagreement matrix (`textnorm.wer`, row = reference, column =
hypothesis, pooled over all inputs), and the word-level differences
between each pair on each fixture, so qualitative differences can be read
directly.

A model that fails to load or run on CPU is reported with its exception.

Run from the repo root (downloads each checkpoint on first use):

    uv run --extra asr python models/parakeet-redux/probes/compare_models.py \
        > models/parakeet-redux/results/compare_models.log
"""

from __future__ import annotations

import functools
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import moondream as md  # noqa: E402

from playground.audio import load_channel  # noqa: E402
from textnorm import errors_only, wer  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

MODELS = [
    "moondream/parakeet-redux",
    "nvidia/parakeet-tdt-0.6b-v3",
    "moondream/parakeet-ultra",
    "openai/whisper-large-v3-turbo",
    "Qwen/Qwen3-ASR-0.6B",
]
# (label, path, channel). The Map Task channels are 20 kHz and peak well
# below full scale (see clipping.py), and add longer, spontaneous speech.
INPUTS = [
    ("tool_call", "audio/tool_call.wav", 0),
    ("interruptions", "audio/interruptions.wav", 0),
    ("turn_taking", "audio/turn_taking.wav", 0),
    ("q4ec1_ch0", "audio/corpora/maptask/q4ec1.mix.wav", 0),
    ("q1ec1_ch1", "audio/corpora/maptask/q1ec1.mix.wav", 1),
]
FIXTURES = [label for label, _, _ in INPUTS]
PASSES = 3

audio = {label: load_channel(Path(path), ch) for label, path, ch in INPUTS}
texts: dict[str, dict[str, str]] = {}
rtf: dict[str, dict[str, list[float]]] = {}

for model in MODELS:
    print(f"=== {model}")
    try:
        with md.photon(model, device="cpu") as sp:
            sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
            for s in FIXTURES:
                x, rate = audio[s]
                sp.transcribe(audio=x, sample_rate=rate, timestamps="segment")
            rtf[model] = {s: [] for s in FIXTURES}
            texts[model] = {}
            for p in range(PASSES):
                order = FIXTURES[p % len(FIXTURES):] + FIXTURES[:p % len(FIXTURES)]
                for s in order:
                    x, rate = audio[s]
                    t0 = time.monotonic()
                    r = sp.transcribe(audio=x, sample_rate=rate, timestamps="segment")
                    el = time.monotonic() - t0
                    rtf[model][s].append(len(x) / rate / el)
                    prev = texts[model].get(s)
                    if prev is not None and prev != r["text"]:
                        print(f"  NOTE: {s} text differs between passes")
                    texts[model][s] = r["text"]
                    if p == 0:
                        print(f"  {s}: language={r.get('language')!r} segments={len(r.get('segments') or [])}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")
        continue
    for s in FIXTURES:
        v = rtf[model][s]
        print(f"  {s:14s} RTF min/med/max {min(v):6.1f}/{statistics.median(v):6.1f}/{max(v):6.1f}x  n={len(v)}")
    allv = [v for s in FIXTURES for v in rtf[model][s]]
    print(f"  ALL            RTF median {statistics.median(allv):.1f}x  range {min(allv):.1f}-{max(allv):.1f}x  n={len(allv)}")

ok = [m for m in MODELS if m in texts]
print(f"\n=== RTF summary (median over n={PASSES * len(FIXTURES)} calls; min-max)")
for m in ok:
    allv = [v for s in FIXTURES for v in rtf[m][s]]
    print(f"  {m:32s} {statistics.median(allv):6.1f}x  ({min(allv):.1f}-{max(allv):.1f})")

print("\n=== transcripts")
for s in FIXTURES:
    print(f"--- {s}")
    for m in ok:
        print(f"  [{m}] {texts[m][s]}")

print("\n=== pairwise disagreement, pooled over fixtures: row = reference, column = hypothesis")
short = {m: m.split("/")[-1] for m in ok}
print("  " + " " * 30 + "".join(f"{short[m][:16]:>18s}" for m in ok))
for a in ok:
    cells = []
    for b in ok:
        e = sum(wer(texts[a][s], texts[b][s])["errors"] for s in FIXTURES)
        n = sum(wer(texts[a][s], texts[b][s])["ref_words"] for s in FIXTURES)
        cells.append(f"{e:>4d}/{n:<4d}{e / n * 100:5.1f}% ")
    print(f"  {short[a][:30]:30s}" + "".join(f"{c:>18s}" for c in cells))

print("\n=== word-level differences per pair and fixture")
for i, a in enumerate(ok):
    for b in ok[i + 1:]:
        for s in FIXTURES:
            r = wer(texts[a][s], texts[b][s])
            errs = errors_only(r["ops"])
            print(f"  {short[a]} -> {short[b]} {s}: {r['errors']} diffs / {r['ref_words']} words")
            for e in errs:
                print(f"      {e}")
