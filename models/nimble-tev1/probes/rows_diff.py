"""Record-by-record difference between two public_suite.py runs of the same records.

    python models/nimble-tev1/probes/rows_diff.py --root $RUNS --left nimble-ollama-format-h100 \
        --right nimble-ollama-format-3090 [--subsets boolq]

For every record both runs scored, the largest absolute difference between
their per-option probabilities, and whether the predicted answer or its
correctness differs. Prints per subset and pooled: how many records agree to
1e-6, the median, 99th percentile and maximum difference, how many predictions
differ, and each run's accuracy. Used for two questions: is a backend's output
the same as another's (two GPUs), and how much does Ollama's unisolated history
move a record against an isolated run of the port. No record text is printed.
"""
import argparse
import json
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
           "civil_comments", "aegis2", "helpsteer2", "summeval-relevance", "summeval-consistency", "pubmedqa"]


def rows(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["id"]] = r["student"]
    return out


def stats(pairs):
    diffs = sorted(d for d, *_ in pairs)
    n = len(diffs)
    return {"n": n, "same_1e-6": sum(d < 1e-6 for d in diffs), "median": diffs[n // 2],
            "p99": diffs[min(n - 1, int(n * 0.99))], "max": diffs[-1],
            "prediction_differs": sum(p for _, p, _, _ in pairs),
            "left_correct": sum(l for *_, l, _ in pairs), "right_correct": sum(r for *_, r in pairs)}


def fmt(label, s):
    return (f"  {label:22} n {s['n']:5d}  same to 1e-6 {s['same_1e-6']:5d}  median {s['median']:.2e}  p99 {s['p99']:.2e}"
            f"  max {s['max']:.4f}  prediction differs {s['prediction_differs']:3d}"
            f"  acc {100 * s['left_correct'] / s['n']:.1f}% vs {100 * s['right_correct'] / s['n']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    a = ap.parse_args()
    print(f"left {a.left}, right {a.right}")
    pooled = []
    for s in a.subsets.split(","):
        left, right = rows(a.root / s / a.left / "rows.jsonl"), rows(a.root / s / a.right / "rows.jsonl")
        pairs = []
        for k in left.keys() & right.keys():
            pl, pr = left[k].get("probabilities"), right[k].get("probabilities")
            if not pl or not pr:
                continue
            pairs.append((max(abs(pl[x] - pr[x]) for x in pl), left[k]["prediction"] != right[k]["prediction"],
                          bool(left[k]["correct"]), bool(right[k]["correct"])))
        pooled += pairs
        print(fmt(s, stats(pairs)))
    if len(a.subsets.split(",")) > 1:
        print(fmt("all records", stats(pooled)))


if __name__ == "__main__":
    main()
