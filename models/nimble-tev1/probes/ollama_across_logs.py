"""Does Ollama give the same answer on two machines? Compares the `ollama` lines of fidelity logs.

    python models/nimble-tev1/probes/ollama_across_logs.py \
        models/nimble-tev1/results/fidelity-nimble-h100.log models/nimble-tev1/results/fidelity-nimble-rtx3090.log

fidelity.py prints Ollama's own distribution for every question of every
fixture (the `<question>: ollama {...}` lines). Given two such logs, from the
same Ollama version and model on different hardware, this prints each
question's largest probability difference and whether the top answer differs,
then the largest over all questions. Both logs must cover the same fixtures.
"""
import argparse
import json
import re
from pathlib import Path


def ollama_lines(path):
    out, fixture = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\r")
        if line.startswith("== "):
            fixture = line.split()[1]
        m = re.match(r"  (\w+): ollama (\{.*\})$", line)
        if m:
            out[fixture, m.group(1)] = json.loads(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    a = ap.parse_args()
    left, right = ollama_lines(a.left), ollama_lines(a.right)
    if left.keys() != right.keys():
        raise SystemExit("the two logs do not cover the same fixture questions")
    print(f"left {a.left.name}, right {a.right.name}")
    worst, flips = 0.0, 0
    for key in left:
        p, q = left[key], right[key]
        d = max(abs(p[x] - q[x]) for x in p)
        flip = max(p, key=p.get) != max(q, key=q.get)
        worst, flips = max(worst, d), flips + flip
        print(f"  {key[0]} {key[1]:10} max|d| {d:.2e}  top answer {'DIFFERS' if flip else 'same'}")
    print(f"  {len(left)} questions: largest |d| {worst:.2e}, top answer differs on {flips}")


if __name__ == "__main__":
    main()
