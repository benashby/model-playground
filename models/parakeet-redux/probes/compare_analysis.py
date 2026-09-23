"""Break down the head-to-head transcripts in results/compare_models.log.

Analysis only; runs no model. Reads the "=== transcripts" section of
compare_models.log and prints, per input and per model, the raw and normalised
word counts, and for every pair of models the pooled substitutions, deletions
and insertions (`textnorm.wer`, first model = reference). A deletion here
means a word the reference model transcribed and the other did not.

Then lists every run of 3 or more consecutive words that the base model and
Ultra both transcribed but Redux did not (deleted relative to both), so the
kind of content Redux drops can be read directly.

There is no ground truth. None of this says which model is right.

Run from the repo root:

    python models/parakeet-redux/probes/compare_analysis.py \
        > models/parakeet-redux/results/compare_analysis.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from textnorm import align, normalise, wer  # noqa: E402

LOG = Path("models/parakeet-redux/results/compare_models.log")
text = LOG.read_text()
section = text.split("=== transcripts", 1)[1].split("=== pairwise", 1)[0]
texts: dict[str, dict[str, str]] = {}
cur = None
for line in section.splitlines():
    m = re.match(r"^--- (\S+)", line)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"^  \[([^\]]+)\] (.*)$", line)
    if m and cur:
        texts.setdefault(m.group(1), {})[cur] = m.group(2)

models = list(texts)
inputs = list(next(iter(texts.values())))
short = {m: m.split("/")[-1] for m in models}

print("=== word counts (raw whitespace words / normalised words)")
print("  " + f"{'input':14s}" + "".join(f"{short[m][:22]:>24s}" for m in models))
for s in inputs + ["ALL"]:
    cells = []
    for m in models:
        src = [texts[m][s]] if s != "ALL" else list(texts[m].values())
        raw = sum(len(t.split()) for t in src)
        nrm = sum(len(normalise(t)) for t in src)
        cells.append(f"{raw}/{nrm}")
    print("  " + f"{s:14s}" + "".join(f"{c:>24s}" for c in cells))

print("\n=== pooled S/D/I per ordered pair (reference -> hypothesis), all inputs")
for a in models:
    for b in models:
        if a == b:
            continue
        S = D = I = N = 0
        for s in inputs:
            r = wer(texts[a][s], texts[b][s])
            S += r["S"]
            D += r["D"]
            I += r["I"]
            N += r["ref_words"]
        print(f"  {short[a]:22s} -> {short[b]:22s} S={S:4d} D={D:4d} I={I:4d} "
              f"over {N} ref words = {(S + D + I) / N * 100:.1f}%")

red, base, ultra = "moondream/parakeet-redux", "nvidia/parakeet-tdt-0.6b-v3", "moondream/parakeet-ultra"
if all(m in texts for m in (red, base, ultra)):
    print("\n=== runs of >= 3 words present in base AND ultra, missing from redux")
    total = 0
    for s in inputs:
        for ref_model in (base,):
            ops = align(normalise(texts[ref_model][s]), normalise(texts[red][s]))
            run: list[str] = []
            runs: list[str] = []
            for o, r, _h in ops + [("=", None, None)]:
                if o == "D":
                    run.append(r)
                else:
                    if len(run) >= 3:
                        runs.append(" ".join(run))
                    run = []
            ultra_norm = " ".join(normalise(texts[ultra][s]))
            both = [x for x in runs if x in ultra_norm]
            total += sum(len(x.split()) for x in both)
            for x in both:
                print(f"  {s}: \"{x}\"")
    print(f"  words in such runs: {total}")
