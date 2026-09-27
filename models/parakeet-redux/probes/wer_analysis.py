"""What the errors are: deletion run lengths and the words each runtime drops.

For each named segment set and runtime from nemo_vs_onnx.py, re-scores the
joined predictions with AppTek's score.py (--out gives the normalised
reference and hypothesis per channel), aligns them with jiwer, and reports:

  - deletions grouped by run length (1, 2-3, 4-10, 11-30, 31+ words in a row),
    which separates scattered dropped words from whole stretches going missing;
  - the most deleted words, each with the share of its occurrences in the
    reference that were deleted.

Run from the repo root after nemo_vs_onnx.py score has written the .pred.jsonl files:

    uv run --extra asr --with jiwer --with openai-whisper==20250625 \\
        python models/parakeet-redux/probes/wer_analysis.py \\
        > models/parakeet-redux/results/wer_analysis.log
"""

import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "logs" / "wer" / "nemo_vs_onnx"
APPTEK = ROOT / "audio" / "corpora" / "apptek"

CASES = [
    ("apptek-en-US_General_clean_bench", ["nemo_cuda_graphs", "onnx_fp32_cuda", "onnx_int8_cuda"]),
    ("apptek-en-US_General_g711mu_bench_first12", ["nemo_cuda_graphs", "onnx_fp32", "onnx_int8"]),
    ("hvb_clean_bench", ["nemo_cuda_graphs", "onnx_fp32_cuda", "onnx_int8_cuda"]),
]
RUNS = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 10**9)]


def normalised(tag: str, runtime: str):
    d = OUT / tag
    # delete_on_close=False: on Windows a file still open here cannot be opened by the child.
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete_on_close=False) as tmp:
        tmp.close()
        subprocess.run([sys.executable, str(APPTEK / "score.py"), "--ref", str(d / "ref.jsonl"),
                        "--pred", str(d / f"{runtime}.pred.jsonl"), "--out", tmp.name],
                       cwd=APPTEK, check=True, capture_output=True)
        return [json.loads(l) for l in open(tmp.name, encoding="utf-8")]


def analyse(tag: str, runtime: str) -> None:
    runs, dele, refc = collections.Counter(), collections.Counter(), collections.Counter()
    for o in normalised(tag, runtime):
        r, h = o["ref_norm"], o["pred_norm"]
        if not r.strip():
            continue
        rw = r.split()
        refc.update(rw)
        for c in jiwer.process_words(r, h).alignments[0]:
            if c.type == "delete":
                n = c.ref_end_idx - c.ref_start_idx
                runs[n] += 1
                dele.update(rw[c.ref_start_idx:c.ref_end_idx])
    total = sum(dele.values())
    print(f"\n=== {tag} / {runtime}: {total} deleted words of {sum(refc.values())} reference words")
    for lo, hi in RUNS:
        w = sum(k * v for k, v in runs.items() if lo <= k <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**6 else f"{lo}-{hi}")
        print(f"  runs of {label:>5} words: {sum(v for k, v in runs.items() if lo <= k <= hi):6d} runs, "
              f"{w:6d} words ({w / max(total, 1) * 100:5.1f} % of deletions)")
    top = dele.most_common(12)
    print("  most deleted: " + ", ".join(f"{w} {n} ({n / refc[w] * 100:.0f} %)" for w, n in top))


def main():
    for tag, runtimes in CASES:
        for rt in runtimes:
            analyse(tag, rt)


if __name__ == "__main__":
    main()
