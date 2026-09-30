"""Bespoke Labs' 13-subset public benchmark, scored against a local backend.

Reuses their runner unchanged (nimble/evaluation/evaluate_public_jev.py at
bespokelabsai/nimble 62076b4): the same records, the same validation and the
same rows.jsonl, summary.json and manifest.json, so their compare_public and
summarize_public_suite work on the output. Only the transport is replaced.

    # Ollama's own /v1/systemone
    python models/nimble-tev1/probes/public_suite.py --nimble-src $NIMBLE_SRC \
        --data $NIMBLE_SRC/data/public/boolq/all.jsonl --output-dir $RUNS/boolq/ollama-nimble \
        ollama --url http://127.0.0.1:11434 --model nimble

    # systemone.py over a bare llama-server, with the prompt format as a variable
    python models/nimble-tev1/probes/public_suite.py --nimble-src $NIMBLE_SRC \
        --data $NIMBLE_SRC/data/public/boolq/all.jsonl --output-dir $RUNS/boolq/port-tev1-native \
        port --server http://127.0.0.1:8080 --model tev1 --format tev1 --system tev1

Probabilities depend on what the server evaluated before (results/public-suite-
smoke.log: one of eight records moved 0.0194 between two servers with different
histories), so for numbers that reproduce, start llama-server with --cache-ram 0
and pass --isolate. Ollama's backend cannot be isolated this way.

The backend description is written into the manifest's api_url field, which
the runner checks on resume, so a resumed run cannot silently change backend.
Keep the run directories outside the repository: rows.jsonl holds benchmark
text under third-party licences. Commit only summaries and comparisons.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import systemone  # noqa: E402


def ollama_transport(url, api_error):
    endpoint = url.rstrip("/") + "/v1/systemone"

    def call(payload):
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            raise api_error(exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise api_error(0) from None
    return call, f"ollama {endpoint}"


def port_transport(a, api_error):
    server = systemone.Server(a.server)
    props = server.props()
    label = (f"systemone.py server={a.server} build={props.get('build_info', '?')} "
             f"gguf={Path(props.get('model_path', '?')).name} format={a.format} system={a.system} "
             f"prime={not a.no_prime} temperature={a.temperature} isolate={a.isolate}")

    def call(payload):
        try:
            if a.isolate:
                # Empty the slot so this record cannot resume from the previous one's
                # checkpoint. Only complete with llama-server --cache-ram 0.
                server.post("/completion", {"prompt": "displace " + uuid.uuid4().hex,
                                            "n_predict": 1, "cache_prompt": True})
            body, _ = systemone.decide(server, payload["state"], payload["questions"], a.format,
                                       a.system, not a.no_prime, a.temperature)
        except urllib.error.HTTPError as exc:
            raise api_error(exc.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            raise api_error(0) from None
        return {"model": payload["model"], **body}
    return call, label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nimble-src", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    sub = ap.add_subparsers(dest="backend", required=True)
    o = sub.add_parser("ollama")
    o.add_argument("--url", required=True)
    o.add_argument("--model", required=True)
    p = sub.add_parser("port")
    p.add_argument("--server", required=True)
    p.add_argument("--model", required=True, help="label recorded as the model name")
    p.add_argument("--format", default="ollama", choices=["ollama", "nimble", "tev1"])
    p.add_argument("--system", required=True, help=f"one of {sorted(systemone.SYSTEM_PROMPTS)}")
    p.add_argument("--no-prime", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--isolate", action="store_true",
                   help="score every record from an empty slot (start llama-server with --cache-ram 0)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.nimble_src.resolve()))
    from nimble.evaluation import evaluate_public_jev as runner

    if a.backend == "ollama":
        transport, label = ollama_transport(a.url, runner.ApiError)
    else:
        transport, label = port_transport(a, runner.ApiError)
    runner.API_URL = label  # recorded in manifest.json and checked on resume
    print("host:", hostinfo.describe(), file=sys.stderr)
    print("backend:", label, file=sys.stderr)
    t0 = time.perf_counter()
    report = runner.run(a.data, a.output_dir, model=a.model, concurrency=1, limit=a.limit,
                        transport=transport,
                        progress=lambda done, total: print(f"  {done}/{total}", file=sys.stderr))
    (a.output_dir / "host.json").write_text(json.dumps(
        {"host": hostinfo.describe(), "backend": label,
         "wall_seconds_this_invocation": round(time.perf_counter() - t0, 1)}, indent=2) + "\n")
    print(json.dumps({"done": report.get("done"), "errors": report.get("errors"),
                      "summary": report.get("summary")}, indent=1))


if __name__ == "__main__":
    main()
