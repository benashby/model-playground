#!/usr/bin/env python3
"""Which field carries K2-Horizon's reasoning trace?

This probe exists because an earlier investigation concluded the trace was
"generated, billed, and thrown away" and built a root-cause diagnosis on top
of that. The conclusion was wrong: the probe read ``reasoning_content`` (the
OpenAI/DeepSeek spelling) while vLLM 0.30.0 emits ``reasoning``. Nothing
errored -- the wrong key simply returned empty, which reads exactly like an
absent feature.

So this probe reads BOTH spellings, on BOTH transports, and reports each
separately. A probe that can only observe one name cannot distinguish
"the server withheld it" from "I asked for the wrong thing".

Stdlib only, so it runs without the project venv.

    PLAYGROUND_HOST=<host> python models/k2-horizon-32b/probes/reasoning_fields.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

HOST = os.environ.get("PLAYGROUND_HOST")
PORT = os.environ.get("K2_PORT", "8080")
MODEL = "k2-horizon-32b"
# Arithmetic with a carry -- cheap, but enough that the model actually thinks.
PROMPT = "A train leaves at 14:05 and arrives at 17:40. How many minutes?"

# The two spellings. vLLM 0.30.0 serves the first; the OpenAI-compatible
# clients most people reach for look for the second.
NAMES = ("reasoning", "reasoning_content")


def post(payload: dict, stream: bool):
    if not HOST:
        sys.exit("PLAYGROUND_HOST is not set (export it, or pass it inline)")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            return json.load(r)
        chunks = []
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunks.append(json.loads(body))
        return chunks


def run(effort: str) -> None:
    base = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "reasoning_effort": effort,
        "max_tokens": 600,
    }

    resp = post(base, stream=False)
    msg = resp["choices"][0]["message"]
    usage = resp["usage"]
    lens = {n: len(msg.get(n) or "") for n in NAMES}
    print(f"  non-streaming effort={effort:<6} "
          + " ".join(f"{n}={lens[n]:>5}" for n in lens)
          + f"  content={len(msg.get('content') or ''):>4}"
          + f"  completion_tokens={usage['completion_tokens']:>4}"
          # vLLM leaves this at 0 even when it DOES return the trace --
          # a real misreport, and the thing that made the trace look absent.
          + f"  usage.reasoning_tokens={(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}")
    if effort == "high":
        print(f"    message keys: {sorted(msg)}")

    chunks = post({**base, "stream": True}, stream=True)
    deltas = [c["choices"][0].get("delta", {}) for c in chunks if c.get("choices")]
    slens = {n: sum(len(d.get(n) or "") for d in deltas) for n in NAMES}
    print(f"  streaming     effort={effort:<6} "
          + " ".join(f"{n}={slens[n]:>5}" for n in slens)
          + f"  content={sum(len(d.get('content') or '') for d in deltas):>4}"
          + f"  chunks={len(deltas)}")
    if effort == "high":
        keys = sorted({k for d in deltas for k in d})
        print(f"    delta keys:   {keys}")


if __name__ == "__main__":
    # NEVER print HOST: probe output is committed, and this repo is public.
    print(f"host=$PLAYGROUND_HOST:{PORT} model={MODEL}  (address redacted by design)")
    print("A field reading 0 means THAT SPELLING was empty -- not that the trace is absent.\n")
    for effort in ("low", "medium", "high"):
        run(effort)
