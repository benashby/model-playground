"""One Markdown table from Bespoke's compare_public output, per subset.

    python models/nimble-tev1/probes/compare_table.py --root $RUNS --compare compare-nimble-format-h100

Reads <root>/<subset>/<compare>/comparison.json (written by
nimble.evaluation.compare_public with exactly two runs) and prints, per subset:
each run's accuracy and ECE, how many records only the left or only the right
run got right, and the exact McNemar p on those discordant pairs. p < 0.05 is
marked. With 13 subsets tested, about one in twenty true-null subsets is
expected to be marked by chance; the pooled row is the test to read first.
"""
import argparse
import json
import math
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
           "civil_comments", "aegis2", "helpsteer2", "summeval-relevance", "summeval-consistency", "pubmedqa"]


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--compare", required=True, help="comparison directory name under each subset")
    a = ap.parse_args()
    left = right = None
    tb = tc = 0
    print("| Subset | left acc | right acc | left ECE | right ECE | only left right | only right right | McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SUBSETS:
        path = a.root / s / a.compare / "comparison.json"
        if not path.exists():
            print(f"| {s} | — | — | — | — | — | — | — |")
            continue
        j = json.loads(path.read_text())
        names = list(j["runs"])
        left, right = names
        (pair,) = j["pairs"].values()
        rl, rr = j["runs"][left], j["runs"][right]
        b, c, p = pair["left_only_correct"], pair["right_only_correct"], pair["mcnemar_exact_p"]
        tb, tc = tb + b, tc + c
        print(f"| {s} | {100 * rl['accuracy']:.1f}% | {100 * rr['accuracy']:.1f}% | {rl['ece10']:.3f} | {rr['ece10']:.3f}"
              f" | {b} | {c} | {p:.3g}{' *' if p < 0.05 else ''} |")
    p = mcnemar_exact(tb, tc)
    print(f"| **all records** | | | | | {tb} | {tc} | {p:.3g}{' *' if p < 0.05 else ''} |")
    print(f"\nleft = {left}, right = {right}")


if __name__ == "__main__":
    main()
