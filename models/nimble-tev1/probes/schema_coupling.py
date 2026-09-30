"""Does adding an unrelated question change the answer to another one?

    python models/nimble-tev1/probes/schema_coupling.py --server http://127.0.0.1:8080 --system ollama-tev1

The API documents questions as scored independently, and they are: no answer
feeds another. But Ollama's format (and Nimble's) put every question's schema
into every question's prompt, so the prompt for "team" changes when "churn" is
added. This scores the first question of fixtures/ticket-8q.json alone, then
with the first k questions for k = 2..8, cold each time and without the primer
(so priming is not a second variable), and prints its probabilities.
Tev1's own format sends each question alone and should not move.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import systemone  # noqa: E402
from fidelity import distribution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--formats", default="ollama,tev1")
    ap.add_argument("--fixture", type=Path, default=HERE / "fixtures" / "ticket-8q.json")
    a = ap.parse_args()
    s = systemone.Server(a.server)
    print("host:", hostinfo.describe())
    print(f"server build {s.props().get('build_info', '?')}")
    req = json.loads(a.fixture.read_text())
    names = list(req["questions"])
    for fmt in a.formats.split(","):
        system = a.system if fmt == "ollama" else fmt if fmt in systemone.SYSTEM_PROMPTS else a.system
        print(f"\n== format {fmt}, system {system}; question {names[0]!r}")
        base = None
        for k in range(1, len(names) + 1):
            s.post("/completion", {"prompt": "displace " + uuid.uuid4().hex, "n_predict": 1,
                                   "cache_prompt": True})
            qs = {n: req["questions"][n] for n in names[:k]}
            body, trace = systemone.decide(s, req["state"], qs, fmt, system, use_prime=False)
            p = distribution(body["answers"][names[0]])
            base = base or p
            d = max(abs(p[x] - base[x]) for x in p)
            print(f"  with {k} question(s): prompt {trace['rows'][0]['prompt_tokens']:4} tok  "
                  f"{json.dumps({x: round(v, 6) for x, v in p.items()})}  max|d| vs alone {d:.2e}")


if __name__ == "__main__":
    main()
