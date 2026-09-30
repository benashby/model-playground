"""What a decision costs as the state grows and as questions are added.

    python models/nimble-tev1/probes/latency_sweep.py --server http://127.0.0.1:8080 --system ollama-tev1 \
        --state-tokens 64,256,1024,2000 --questions 1,2,4,8,16,32 --repeats 5

For every (state length, question count) pair, each repeat scores the same
request three ways, in this order:

  cold plain   slot displaced first, no primer: every question evaluates its whole prompt
  cold primed  slot displaced first, primer on (Ollama's default): the cost of a new state
  warm         the same request again at once, primer on: the state is already cached

"cold primed" is the streaming case: a transcript that changes every few hundred
milliseconds is a new state each time, and the Ollama format puts the state
before the schema, so nothing after it survives in the cache.

The state is a numbered filler transcript sized to the requested token count
with the server's own tokenizer; the questions are yes/no questions about it.
Their content does not matter for timing, only their count and length.

Two times are printed per run. wall_ms is what the caller waits, including the
port's own /apply-template and /tokenize round trips (one render per question,
one tokenize per question plus two per answer code). server_ms is the sum of
llama-server's own prompt_ms over the primer and every question, which is the
model's cost alone. The gap between them is client plumbing.

START THE SERVER WITH --cache-ram 0, for the reason priming.py gives. A cold run
whose first evaluation covered fewer tokens than the shared prompt is marked
NOT COLD. A combination whose prompt does not fit the server's context is
reported as skipped. The first repeat is warmup and is left out of the summary.
"""
import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import systemone  # noqa: E402

MODES = [("cold plain", True, False), ("cold primed", True, True), ("warm", False, True)]


def filler(server, target):
    """A transcript of numbered lines whose token count first reaches target."""
    lines, text = [], ""
    while len(server.tokenize(text, False)) < target:
        i = len(lines)
        lines.append(f"Line {i}: the caller confirms that item {i} is still pending review.")
        text = "\n".join(lines)
    return text


def questions(k):
    return {f"q{i:02d}": {"type": "noul", "instructions": f"Does the transcript mention item {i * 7}?",
                          "criteria": {"false": "It does not", "true": "It does"}}
            for i in range(k)}


def run(server, state, qs, fmt, system, cold, primed):
    if cold:
        server.post("/completion", {"prompt": "displace " + uuid.uuid4().hex, "n_predict": 1,
                                    "cache_prompt": True})
    t0 = time.perf_counter()
    _, trace = systemone.decide(server, state, qs, fmt, system, use_prime=primed)
    wall = (time.perf_counter() - t0) * 1000
    primer = trace["primer"]
    server_ms = sum(r["prompt_ms"] or 0 for r in trace["rows"]) + (primer["prompt_ms"] if primer else 0)
    evaluated = sum(r["prompt_n"] or 0 for r in trace["rows"]) + (primer["prompt_n"] if primer else 0)
    first = trace["rows"][0]
    not_cold = cold and (primer["prompt_n"] < primer["shared_tokens"] if primer
                         else first["prompt_n"] < first["prompt_tokens"])
    return {"wall_ms": wall, "server_ms": server_ms, "evaluated": evaluated,
            "prompt_tokens": first["prompt_tokens"], "not_cold": not_cold}


def summary(xs):
    return f"{statistics.median(xs):8.1f} [{min(xs):.1f}-{max(xs):.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--format", default="ollama", choices=["ollama", "nimble", "tev1"])
    ap.add_argument("--state-tokens", default="64,256,1024,2000")
    ap.add_argument("--questions", default="1,2,4,8,16,32")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    s = systemone.Server(a.server)
    props = s.props()
    print("host:", hostinfo.describe())
    print(f"server build {props.get('build_info', '?')}, model {Path(props.get('model_path', '?')).name}, "
          f"format {a.format}, system {a.system}, repeats {a.repeats} (first discarded)")
    table = []
    for n in [int(x) for x in a.state_tokens.split(",")]:
        state = {"transcript": filler(s, n)}
        for k in [int(x) for x in a.questions.split(",")]:
            qs = questions(k)
            print(f"\n== state {n} tokens, {k} question(s)", flush=True)
            results = {label: [] for label, _, _ in MODES}
            try:
                for rep in range(a.repeats):
                    for label, cold, primed in MODES:
                        r = run(s, state, qs, a.format, a.system, cold, primed)
                        flag = " NOT COLD" if r["not_cold"] else ""
                        print(f"  rep {rep} {label:11} wall {r['wall_ms']:8.1f} ms  server {r['server_ms']:8.1f} ms"
                              f"  evaluated {r['evaluated']:6d} tok  prompt {r['prompt_tokens']} tok/question{flag}",
                              flush=True)
                        if rep > 0:
                            results[label].append(r)
            except Exception as e:  # a prompt longer than the context: the server refuses or truncates
                print(f"  skipped: {type(e).__name__}: {e}")
                continue
            for label, rs in results.items():
                if rs:
                    table.append((n, k, label, rs[0]["prompt_tokens"], summary([r["wall_ms"] for r in rs]),
                                  summary([r["server_ms"] for r in rs]),
                                  summary([r["wall_ms"] / k for r in rs])))
    print("\n== summary: median [min-max] over repeats after the first")
    print(f"{'state':>5} {'q':>3} {'mode':11} {'tok/q':>6} {'wall ms':>24} {'server ms':>24} {'wall ms per q':>24}")
    for n, k, label, pt, wall, srv, per in table:
        print(f"{n:5d} {k:3d} {label:11} {pt:6d} {wall:>24} {srv:>24} {per:>24}")


if __name__ == "__main__":
    main()
