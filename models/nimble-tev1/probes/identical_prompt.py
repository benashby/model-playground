"""Does a llama-server build survive the same fully cached prompt twice, on a Qwen3.5 GGUF?

    python models/nimble-tev1/probes/identical_prompt.py --model tev1-0.8b.gguf \
        --binary distro=/usr/bin/llama-server --binary ollama=<ollama>/lib/ollama/llama-server

For each binary: start it on a free port with the given GGUF, send the same
short prompt twice with cache_prompt, and report both HTTP results, whether the
process is still alive, and any abort line from its log.

The second request finds the whole prompt cached. llama-server must evaluate at
least one token, so it tries to roll the cache back by one position. Recurrent
layers cannot roll back, and one build tested here aborts in
common_context_seq_rm instead of re-evaluating. A decision service retrying a
request sends exactly this.
"""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostinfo  # noqa: E402

PROMPT = "hello world hello world hello world"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return f"no response ({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--binary", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    print("host:", hostinfo.describe())
    print("model:", a.model.name)
    for spec in a.binary:
        name, binary = spec.split("=", 1)
        version = subprocess.run([binary, "--version"], capture_output=True, text=True)
        version = [l for l in (version.stdout + version.stderr).splitlines()
                   if l.startswith(("version:", "built with"))] or ["version unknown"]
        port = free_port()
        with tempfile.TemporaryFile("w+") as log:
            proc = subprocess.Popen([binary, "-m", str(a.model.resolve()), "-c", "2048", "-np", "1",
                                     "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(port)],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    # a crashing build may dump core into its cwd
                                    cwd=tempfile.gettempdir())
            try:
                for _ in range(240):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                            if r.status == 200:
                                break
                    except (urllib.error.URLError, ConnectionError, OSError):
                        pass
                    time.sleep(0.5)
                results = [post(port, {"prompt": PROMPT, "n_predict": 1, "cache_prompt": True})
                           for _ in range(2)]
                time.sleep(1)
                alive = proc.poll() is None
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(30)
            log.seek(0)
            aborts = [l.strip() for l in log if "failed to remove sequence" in l or "GGML_ASSERT" in l]
        print(f"\n== {name}: {' | '.join(version[:1])}")
        print(f"  request 1: {results[0]}   request 2: {results[1]}   alive after: {alive}")
        for line in aborts[:3]:
            print("  log:", line[line.find(" ") + 1:] if line[:1].isdigit() else line)


if __name__ == "__main__":
    main()
