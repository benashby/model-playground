"""Per-subset accuracy and error counts for public_suite.py runs, side by side.

    python models/nimble-tev1/probes/suite_table.py --root $RUNS \
        nimble-ollama-format-h100 nimble-nimble-format-h100 tev1-ollama-format-h100 ...

Reads <root>/<subset>/<run>/summary.json, written by Bespoke's runner, and prints
one Markdown table: for each subset and run, correct / count and accuracy, then
the runner's error rows (a request that failed, for example a prompt longer
than the model's context, which the runner counts as incorrect). The macro mean
is the unweighted mean of the 13 subset accuracies, which is how Ollama's blog
and Bespoke's table both average the suite; the micro mean pools every record.
Error rows stay in the denominator, as the runner scores them.
"""
import argparse
import json
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
           "civil_comments", "aegis2", "helpsteer2", "summeval-relevance", "summeval-consistency", "pubmedqa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    table = {}
    for run in a.runs:
        for s in SUBSETS:
            path = a.root / s / run / "summary.json"
            if path.exists():
                j = json.loads(path.read_text())
                al = j["summary"]["all"]
                table[run, s] = (al["correct"], j["count"], j.get("errors", 0))
    print("| Subset | " + " | ".join(a.runs) + " |")
    print("|---|" + "---:|" * len(a.runs))
    for s in SUBSETS:
        cells = []
        for run in a.runs:
            if (run, s) not in table:
                cells.append("—")
                continue
            c, n, e = table[run, s]
            cells.append(f"{100 * c / n:.1f}% ({c}/{n}{f', {e} err' if e else ''})")
        print(f"| {s} | " + " | ".join(cells) + " |")
    macro, micro, errs = [], [], []
    for run in a.runs:
        rows = [table[run, s] for s in SUBSETS if (run, s) in table]
        complete = len(rows) == len(SUBSETS)
        macro.append(f"{100 * sum(c / n for c, n, _ in rows) / len(rows):.1f}%" + ("" if complete else f" ({len(rows)} subsets)"))
        micro.append(f"{100 * sum(c for c, _, _ in rows) / sum(n for _, n, _ in rows):.1f}%")
        errs.append(str(sum(e for _, _, e in rows)))
    print("| **macro mean** | " + " | ".join(macro) + " |")
    print("| micro mean | " + " | ".join(micro) + " |")
    print("| error rows | " + " | ".join(errs) + " |")


if __name__ == "__main__":
    main()
