"""What Ollama's shared-prefix primer buys, and whether it is only a speed change.

    python models/nimble-tev1/probes/priming.py --server http://127.0.0.1:8080 --system ollama-tev1 \
        --repeats 3 models/nimble-tev1/probes/fixtures/ticket-3q.json models/nimble-tev1/probes/fixtures/ticket-8q.json

Ollama's comment on primeSharedPrefix says priming "only changes speed" and that
scores stay the same. Each fixture is scored four ways, in this order, repeated:

  cold plain   cache displaced first, no primer: every question evaluates its whole prompt
  cold primed  cache displaced first, primer on (what Ollama does)
  warm plain   no displacement, no primer: whatever checkpoint the last run left
  warm primed  no displacement, primer on

"Displaced" means one unrelated, never-repeated prompt is evaluated first, so
the slot holds nothing from this fixture. A never-repeated prompt matters: some
llama-server builds abort on an exactly repeated fully cached prompt with this
model family (identical_prompt.py).

START THE SERVER WITH --cache-ram 0. Displacement only empties the slot.
llama-server also keeps a host-memory prompt cache (8192 MiB by default) and
restores a saved state from it, so without that flag every "cold" run after
the first silently resumes from a checkpoint. The first version of this probe
ran that way and reported warm numbers as cold. A cold run is now checked: if
its first question evaluated fewer tokens than its prompt holds, the line is
marked NOT COLD.

Prints per-question prompt_n (tokens actually evaluated), wall time, and the
largest probability difference of each mode from cold plain. The first repeat
is warmup; read the later ones for timing.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import systemone  # noqa: E402
from fidelity import distribution  # noqa: E402

MODES = [("cold plain", True, False), ("cold primed", True, True),
         ("warm plain", False, False), ("warm primed", False, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures", nargs="+", type=Path)
    ap.add_argument("--server", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--format", default="ollama")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    s = systemone.Server(a.server)
    props = s.props()
    print("host:", hostinfo.describe())
    print(f"server build {props.get('build_info', '?')}, model {Path(props.get('model_path', '?')).name}")
    for path in a.fixtures:
        req = json.loads(path.read_text())
        print(f"\n== {path.name}: {len(req['questions'])} questions, format {a.format}")
        for rep in range(a.repeats):
            base = None
            for label, cold, primed in MODES:
                if cold:
                    s.post("/completion", {"prompt": "displace " + uuid.uuid4().hex, "n_predict": 1,
                                           "cache_prompt": True})
                t0 = time.perf_counter()
                body, trace = systemone.decide(s, req["state"], req["questions"], a.format, a.system,
                                               use_prime=primed)
                wall = (time.perf_counter() - t0) * 1000
                dists = {q: distribution(body["answers"][q]) for q in req["questions"]}
                if base is None:
                    base = dists
                diff = max(abs(dists[q][k] - base[q][k]) for q in dists for k in dists[q])
                primer = trace["primer"]
                first_row = trace["rows"][0]
                evaluated = primer["prompt_n"] if primer else first_row["prompt_n"]
                shared = primer["shared_tokens"] if primer else first_row["prompt_tokens"]
                if cold and evaluated < shared:
                    label += " NOT COLD"
                print(f"  rep {rep} {label:11} total {wall:8.0f} ms  per question {wall / len(dists):6.0f} ms"
                      f"  primer {'-' if not primer else str(primer['prompt_n']) + ' tok'}"
                      f"  prompt_n {[r['prompt_n'] for r in trace['rows']]}  max|d| vs cold plain {diff:.2e}")
                first = next(iter(req["questions"]))
                print(f"      {first}: {json.dumps({k: round(v, 6) for k, v in dists[first].items()})}")


if __name__ == "__main__":
    main()
