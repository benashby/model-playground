"""Are the probabilities calibrated, and would one temperature fix them?

    python models/nimble-tev1/probes/temperature.py --root $RUNS --run nimble-ollama-format-h100

Reads the rows.jsonl of one run across Bespoke's 13 public subsets (written by
public_suite.py) and needs no inference: softmax(logit / T) over the answer
codes equals p ** (1 / T) renormalised, so any temperature can be applied to
the stored probabilities afterwards.

Records are split into two halves by a hash of their family, never by record,
so the paired records a family holds (VitaminC's contrastive pairs, PAWS'
paraphrases) are never split between fitting and evaluation. One T is fitted
on the first half by minimising mean negative log-likelihood, and ECE (10
equal-width bins over the top probability), Brier and NLL are reported on the
second half at T = 1 and at the fitted T, pooled and per subset. The halves are
then swapped, so the fitted T can be checked for stability.

Rows the runner recorded as errors (no probabilities, for example a prompt
longer than the context) are counted and left out: a temperature cannot help
them. Accuracy does not depend on T. No record text is printed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
           "civil_comments", "aegis2", "helpsteer2", "summeval-relevance", "summeval-consistency", "pubmedqa"]


def load(root, run):
    rows, errors = [], {}
    for subset in SUBSETS:
        path = root / subset / run / "rows.jsonl"
        errors[subset] = 0
        if not path.exists():
            errors[subset] = None
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            probs = (r.get("student") or {}).get("probabilities")
            if not probs:
                errors[subset] += 1
                continue
            target = r["reference"]["target"]
            key = str(target).lower() if isinstance(target, bool) else str(target)
            half = int(hashlib.sha256(str(r["family"]).encode()).hexdigest(), 16) % 2
            rows.append({"subset": subset, "half": half, "target": key, "labels": list(probs),
                         "p": [max(float(probs[k]), 1e-300) for k in probs]})
    return rows, errors


def tempered(p, t):
    logs = [math.log(x) / t for x in p]
    peak = max(logs)
    e = [math.exp(x - peak) for x in logs]
    s = sum(e)
    return [x / s for x in e]


def metrics(rows, t):
    nll = brier = 0.0
    bins = [[0, 0.0, 0] for _ in range(10)]  # count, confidence sum, correct
    correct = 0
    for r in rows:
        q = tempered(r["p"], t)
        i = r["labels"].index(r["target"])
        nll -= math.log(max(q[i], 1e-300))
        brier += sum((x - (j == i)) ** 2 for j, x in enumerate(q))
        top = max(range(len(q)), key=q.__getitem__)
        b = bins[min(int(q[top] * 10), 9)]
        b[0] += 1
        b[1] += q[top]
        b[2] += top == i
        correct += top == i
    n = len(rows)
    ece = sum(abs(c - k) for _, c, k in bins) / n
    return {"n": n, "accuracy": correct / n, "nll": nll / n, "brier": brier / n, "ece10": ece}


def fit(rows):
    """Golden-section search for the T minimising mean NLL, over log T in [ln 0.1, ln 20]."""
    lo, hi = math.log(0.1), math.log(20)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = metrics(rows, math.exp(a))["nll"], metrics(rows, math.exp(b))["nll"]
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = metrics(rows, math.exp(a))["nll"]
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = metrics(rows, math.exp(b))["nll"]
    return math.exp((lo + hi) / 2)


def line(label, m):
    return (f"  {label:22} n {m['n']:5d}  acc {m['accuracy']:.4f}  nll {m['nll']:.4f}  "
            f"brier {m['brier']:.4f}  ece10 {m['ece10']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="directory holding <subset>/<run>/rows.jsonl")
    ap.add_argument("--run", required=True, help="run directory name, e.g. nimble-ollama-format-h100")
    a = ap.parse_args()
    rows, errors = load(a.root, a.run)
    print(f"run {a.run}: {len(rows)} scored records; error rows per subset "
          + ", ".join(f"{s} {'missing' if e is None else e}" for s, e in errors.items()))
    for fit_half in (0, 1):
        train = [r for r in rows if r["half"] == fit_half]
        test = [r for r in rows if r["half"] != fit_half]
        t = fit(train)
        print(f"\n== fit on half {fit_half} ({len(train)} records), evaluate on half {1 - fit_half}"
              f" ({len(test)} records): fitted T = {t:.3f}")
        print(line("pooled, T = 1", metrics(test, 1.0)))
        print(line(f"pooled, T = {t:.3f}", metrics(test, t)))
        for s in SUBSETS:
            sub = [r for r in test if r["subset"] == s]
            if sub:
                m1, mt = metrics(sub, 1.0), metrics(sub, t)
                print(f"  {s:22} n {m1['n']:5d}  acc {m1['accuracy']:.4f}  ece10 {m1['ece10']:.4f} -> {mt['ece10']:.4f}"
                      f"  nll {m1['nll']:.4f} -> {mt['nll']:.4f}  brier {m1['brier']:.4f} -> {mt['brier']:.4f}")


if __name__ == "__main__":
    main()
