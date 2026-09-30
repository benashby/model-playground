"""Does systemone.py reproduce Ollama's /v1/systemone, and does the llama.cpp build matter?

    python models/nimble-tev1/probes/fidelity.py --ollama http://127.0.0.1:11434 --model tev1:0.8b \
        --system ollama-tev1 \
        --server ollama-runner=http://127.0.0.1:<port of the llama-server Ollama spawned> \
        --server other-build=http://127.0.0.1:8080 \
        models/nimble-tev1/probes/fixtures/ticket-3q.json

Ollama 0.35.0 runs decision models on a llama-server it spawns itself; its port
is on that process's command line (`ps -eo args | grep [l]lama-server`). Scoring
through that same process isolates the port. Scoring through a separately built
llama-server loaded with the same GGUF isolates the build.

For every fixture and question, prints each source's probabilities and the
largest absolute difference from Ollama. Priming is on, as it is in Ollama.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import systemone  # noqa: E402


def distribution(answer):
    if answer["type"] == "noul":
        return {"false": 1 - answer["noul"], "true": answer["noul"]}
    return {str(k): v for k, v in answer["probabilities"].items()}


def ollama(url, model, req):
    body = json.dumps({"model": model, "state": req["state"], "questions": req["questions"]}).encode()
    r = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=body,
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures", nargs="+", type=Path)
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True, help="systemone SYSTEM_PROMPTS key matching the model")
    ap.add_argument("--server", action="append", default=[], metavar="NAME=URL")
    a = ap.parse_args()
    print("host:", hostinfo.describe())
    servers = [(n, systemone.Server(u)) for n, u in (s.split("=", 1) for s in a.server)]
    for name, s in servers:
        props = s.props()
        print(f"server {name}: build {props.get('build_info', '?')}, "
              f"n_ctx {props.get('default_generation_settings', {}).get('n_ctx', '?')}")
    worst = {name: 0.0 for name, _ in servers}
    for path in a.fixtures:
        req = json.loads(path.read_text())
        ref = ollama(a.ollama, a.model, req)
        runs = {name: systemone.decide(s, req["state"], req["questions"], "ollama", a.system)
                for name, s in servers}
        print(f"\n== {path.name}  (Ollama usage.input_tokens {ref['usage']['input_tokens']})")
        for name, (body, _) in runs.items():
            print(f"  {name}: usage.input_tokens {body['usage']['input_tokens']}")
        for q in req["questions"]:
            p_ref = distribution(ref["answers"][q])
            print(f"  {q}: ollama {json.dumps({k: round(v, 6) for k, v in p_ref.items()})}")
            for name, (body, _) in runs.items():
                p = distribution(body["answers"][q])
                d = max(abs(p[k] - p_ref[k]) for k in p_ref)
                worst[name] = max(worst[name], d)
                same = (max(p, key=p.get) == max(p_ref, key=p_ref.get))
                print(f"    {name}: {json.dumps({k: round(v, 6) for k, v in p.items()})} "
                      f"max|d| {d:.2e} same answer {same}")
    print()
    for name, d in worst.items():
        print(f"largest |probability difference| from Ollama, {name}: {d:.2e}")


if __name__ == "__main__":
    main()
