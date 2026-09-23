#!/usr/bin/env python3
"""K2-Horizon-32B on vLLM: effort tiers, decode rate, prefill cost, tool calls.

This replaces an uncommitted script whose output survives as
results/k2bench.log. That script read only `delta.content`, so its
"reasoning_chars" column is always 0 and its "ttft" was really the time to the
first *answer* token. This version reads `delta.reasoning` as well and reports
both times.

Each section guards against a specific way the number could come out wrong:

  effort     both ways of passing reasoning_effort (top-level field, and
             chat_template_kwargs), since only the first had evidence before.
             The server also sets a default of "high", so omitting it is
             measured too.
  decode     tok/s from the streamed usage block over the time between the
             first and last token, three runs, each prompt made unique.
  prefill    every request starts with a random nonce so the prefix cache can
             never hit, then one request is repeated to measure what a hit
             is worth. Measured out to the served 131k window rather than
             extrapolated.
  tools      each tool_call_format the template accepts, checked for the
             structured result (name and arguments), not just "an answer".
  canary     a distinctive token sent as plain-string content and as an
             OpenAI-style parts list, rendered through /tokenize, to show
             whether the prompt actually reaches the model.

Stdlib only. The host comes from PLAYGROUND_HOST and is never printed.

    PLAYGROUND_HOST=<host> python models/k2-horizon-32b/probes/k2_bench.py \
        | tee models/k2-horizon-32b/results/k2-bench.log
"""

from __future__ import annotations

import json
import os
import random
import secrets
import statistics as st
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("PLAYGROUND_HOST") or sys.exit("PLAYGROUND_HOST is not set")
PORT = os.environ.get("K2_PORT", "8080")
BASE = f"http://{HOST}:{PORT}"
MODEL = "k2-horizon-32b"
REPEATS = 3
TOOL_REPEATS = int(os.environ.get("TOOL_REPEATS", "10"))

TRAIN = "A train leaves at 14:05 and arrives at 17:40. How many minutes is the journey? Give the number only."
PROMPTS = [
    TRAIN,
    "What is 17 multiplied by 23? Give the number only.",
    "Name the capital of Australia in one word.",
]


def nonce() -> str:
    # secrets, not a seeded random: a rerun must never reuse a nonce, or the
    # prefix cache it exists to defeat would hit on the second invocation.
    return secrets.token_hex(6)


def post(path: str, body: dict):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def stream_chat(body: dict) -> dict:
    """Stream one chat completion; return timings, field sizes and usage."""
    body = {**body, "model": MODEL, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.monotonic()
    first_any = first_content = last = None
    reasoning = content = ""
    tool_calls: dict[int, dict] = {}
    usage = finish = None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                now = time.monotonic()
                got = False
                if d.get("reasoning"):
                    reasoning += d["reasoning"]; got = True
                if d.get("content"):
                    content += d["content"]; got = True
                    first_content = first_content or now
                for tc in d.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                    got = True
                if got:
                    first_any = first_any or now
                    last = now
                finish = ch.get("finish_reason") or finish
    return {
        "t_first_any": (first_any - t0) if first_any else None,
        "t_first_content": (first_content - t0) if first_content else None,
        "t_total": time.monotonic() - t0,
        "gen_span": (last - first_any) if first_any and last else None,
        "reasoning": reasoning, "content": content,
        "tool_calls": list(tool_calls.values()), "usage": usage or {}, "finish": finish,
    }


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def effort_section() -> None:
    print("### EFFORT: tokens and timing per tier, both ways of passing it, and the server default")
    print("    t_any = first token of any kind (reasoning or answer); t_ans = first answer token")
    for mode in ("top-level", "chat_template_kwargs", "omitted"):
        for effort in ("low", "medium", "high"):
            if mode == "omitted" and effort != "high":
                continue
            rows = []
            for p in PROMPTS:
                for _ in range(REPEATS):
                    body = {"messages": [{"role": "user", "content": f"[{nonce()}] {p}"}], "max_tokens": 4000}
                    if mode == "top-level":
                        body["reasoning_effort"] = effort
                    elif mode == "chat_template_kwargs":
                        body["chat_template_kwargs"] = {"reasoning_effort": effort}
                    rows.append(stream_chat(body))
            ct = [r["usage"].get("completion_tokens", 0) for r in rows]
            rc = [len(r["reasoning"]) for r in rows]
            no_trace = sum(1 for r in rows if not r["reasoning"])
            label = "server default" if mode == "omitted" else effort
            print(f"  {mode:<21} {label:<14} n={len(rows)}  completion_tokens median {st.median(ct):>5}"
                  f" (min {min(ct)}, max {max(ct)})  reasoning_chars median {st.median(rc):>5}"
                  f"  no-trace {no_trace}/{len(rows)}"
                  f"  t_any {fmt(st.median(r['t_first_any'] for r in rows))}s"
                  f"  t_ans {fmt(st.median(r['t_first_content'] for r in rows if r['t_first_content']))}s")
    # trace-to-answer size, high effort, the train prompt, both transports
    ratios = []
    for _ in range(REPEATS):
        r = stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] {TRAIN}"}],
                         "reasoning_effort": "high", "max_tokens": 4000})
        ratios.append((len(r["reasoning"]), len(r["content"])))
    print("  high effort, train prompt, (reasoning_chars, content_chars):", ratios)
    print("  answers:", [stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] {TRAIN}"}],
                                      "reasoning_effort": "high", "max_tokens": 4000})["content"].strip()
                         for _ in range(REPEATS)])


def decode_section() -> None:
    print("### DECODE: single stream, tok/s = completion_tokens / (last token - first token)")
    rates = []
    for i in range(REPEATS + 1):
        r = stream_chat({"messages": [{"role": "user", "content":
                          f"[{nonce()}] Write about 300 words on the history of the telegraph."}],
                         "reasoning_effort": "low", "max_tokens": 600})
        n = r["usage"].get("completion_tokens", 0)
        rate = (n - 1) / r["gen_span"] if r["gen_span"] else None
        tag = "warmup, discarded" if i == 0 else f"run {i}"
        print(f"  {tag:<18} tokens {n:>4}  span {fmt(r['gen_span'])}s  {fmt(rate, 1)} tok/s")
        if i:
            rates.append(rate)
    print(f"  median {st.median(rates):.1f} tok/s over n={len(rates)}")


def filler(target_tokens: int) -> str:
    # Measured at ~1.0 token per word for this vocabulary (a first guess of 1.3
    # undershot, stopping at 77k when 100k was intended). The exact count is
    # read back from usage, so this only has to land near the target.
    words = ["alpha", "river", "stone", "copper", "window", "garden", "signal", "harbor", "violet", "engine"]
    return " ".join(random.choice(words) for _ in range(int(target_tokens * 0.99)))


def one_token(text: str) -> tuple[float, int]:
    """Non-streaming, max_tokens=1: total time is prefill plus one decode step
    (~15 ms at 66 tok/s). Streaming was tried first and failed: at low effort
    the single generated token can arrive with neither reasoning nor content
    text, so no first-token time was ever recorded."""
    t0 = time.monotonic()
    with post("/v1/chat/completions", {"model": MODEL, "max_tokens": 1, "reasoning_effort": "low",
                                       "messages": [{"role": "user", "content": text}]}) as r:
        u = json.load(r)["usage"]
    return time.monotonic() - t0, u["prompt_tokens"]


def prefill_section() -> None:
    print("### PREFILL: time to first token vs prompt length, prefix cache defeated by a leading nonce")
    print("    time = non-streaming request with max_tokens=1 (prefill + one ~15 ms decode step)")
    targets = [int(x) for x in os.environ.get("PREFILL_TARGETS", "200,2000,16000,32000,64000,96000,128000").split(",")]
    for target in targets:
        body_text = filler(target)
        times, ptoks = [], []
        for _ in range(REPEATS):
            t, u = one_token(f"[{nonce()}] {body_text}\n\nReply with the single word: ok")
            if u < target * 0.5:
                sys.exit(f"prompt rendered to {u} tokens for a ~{target}-token input: the prompt is not reaching the model")
            times.append(t); ptoks.append(u)
        med = st.median(times)
        print(f"  prompt_tokens {st.median(ptoks):>7,.0f}  TTFT median {med:6.2f}s"
              f" (min {min(times):.2f}, max {max(times):.2f}, n={len(times)})"
              f"  {st.median(ptoks) / med:,.0f} prompt tok/s")
    # what a prefix-cache hit is worth: the same 60k prompt twice
    text = f"[{nonce()}] {filler(60_000)}\n\nReply with the single word: ok"
    miss, _ = one_token(text)
    hits = [one_token(text)[0] for _ in range(REPEATS)]
    print(f"  same ~60k prompt again: first {miss:.2f}s, repeats {', '.join(f'{h:.2f}' for h in hits)}s"
          f" (prefix cache hit)")


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}


def tools_section() -> None:
    n = TOOL_REPEATS
    print(f"### TOOLS: each tool_call_format, n={n}, checked for a structured call with the right arguments")
    for fmt_name in ("xml", "json", "xml_typed"):
        ok = 0
        seen = []
        for _ in range(n):
            r = stream_chat({"messages": [{"role": "user", "content":
                              f"[{nonce()}] What's the weather in Oslo, in celsius?"}],
                             "tools": [WEATHER], "tool_choice": "auto", "reasoning_effort": "low",
                             "chat_template_kwargs": {"tool_call_format": fmt_name}, "max_tokens": 800})
            calls = r["tool_calls"]
            good = False
            if calls:
                try:
                    args = json.loads(calls[0]["arguments"] or "{}")
                    good = calls[0]["name"] == "get_weather" and str(args.get("city", "")).lower() == "oslo"
                except json.JSONDecodeError:
                    args = calls[0]["arguments"]
                seen.append((calls[0]["name"], args, r["finish"]))
            else:
                seen.append(("NO TOOL CALL", (r["content"] or "")[:300], r["finish"]))
            ok += good
        print(f"  {fmt_name:<10} structured and correct {ok}/{n}   first: {seen[0]}")
        for item in seen:
            if not (item[0] == "get_weather" and isinstance(item[1], dict)
                    and str(item[1].get("city", "")).lower() == "oslo"):
                print(f"      miss: {item}")


def canary_section() -> None:
    print("### CANARY: does the user's text reach the rendered prompt?")
    c = "ZEBRA" + nonce().upper()
    for label, content in (("plain string", f"Say {c}"),
                           ("parts list", [{"type": "text", "text": f"Say {c}"}])):
        with post("/tokenize", {"model": MODEL, "add_generation_prompt": True,
                                "messages": [{"role": "user", "content": content}]}) as r:
            toks = json.load(r)["tokens"]
        with post("/detokenize", {"model": MODEL, "tokens": toks}) as r:
            text = json.load(r)["prompt"]
        print(f"  {label:<13} tokens {len(toks):>3}  canary present: {c in text}")
    print("  (server runs with --chat-template-content-format string; see results/startup.log)")


def multiturn_section() -> None:
    """Feed a tool result back and check the model uses it. The template raises
    unless the prior assistant turn carries a thinking field (think, reasoning,
    reasoning_content, think_fast or think_faster), so each variant is tried."""
    n = int(os.environ.get("MULTITURN_REPEATS", "6"))
    print(f"### MULTI-TURN TOOLS: call, then feed a result back, n={n} per variant")
    for field in ("reasoning", "reasoning_content", None):
        answered = 0; notes = []
        for _ in range(n):
            first = stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] What's the weather in Oslo, in celsius?"}],
                                 "tools": [WEATHER], "tool_choice": "auto", "reasoning_effort": "low", "max_tokens": 800,
                                 "chat_template_kwargs": {"tool_call_format": "xml_typed"}})
            if not first["tool_calls"]:
                notes.append("no first call"); continue
            call = first["tool_calls"][0]
            asst = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}
            if field:
                asst[field] = first["reasoning"]
            msgs = [{"role": "user", "content": "What's the weather in Oslo, in celsius?"}, asst,
                    {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Oslo", "temp_c": -7, "sky": "snow"})}]
            try:
                second = stream_chat({"messages": msgs, "tools": [WEATHER], "reasoning_effort": "low", "max_tokens": 800,
                                      "chat_template_kwargs": {"tool_call_format": "xml_typed"}})
                text = second["content"] or ""
                used = "-7" in text or ("7" in text and "snow" in text.lower())
                answered += used
                # Record everything, not just content: an empty answer could be a
                # second tool call, an answer inside `reasoning`, or truncation.
                notes.append(("used result" if used else "did not answer from it")
                             + f": finish={second['finish']} content={text.strip()[:70]!r}"
                             + f" tool_calls={[c['name'] for c in second['tool_calls']]}"
                             + f" reasoning={second['reasoning'].strip()[:70]!r}")
            except urllib.error.HTTPError as e:
                notes.append(f"HTTP {e.code}: {e.read().decode()[:160]}")
        print(f"  thinking field {field or '(none)':<18} answered from the tool result {answered}/{n}")
        for n_ in notes:
            print(f"      {n_}")


def concurrency_section() -> None:
    """Aggregate decode throughput with N simultaneous streams. Batch-1 figures
    say nothing about capacity; this measures it directly."""
    import concurrent.futures as cf
    print("### CONCURRENCY: N simultaneous 300-token generations, aggregate and per-stream tok/s")
    for n in (1, 4, 8, 16, 32):
        def one(_):
            return stream_chat({"messages": [{"role": "user", "content":
                                 f"[{nonce()}] Write about 300 words on the history of the telegraph."}],
                                "reasoning_effort": "low", "max_tokens": 300, "ignore_eos": True})
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            rs = list(ex.map(one, range(n)))
        wall = time.monotonic() - t0
        toks = sum(r["usage"].get("completion_tokens", 0) for r in rs)
        per = [(r["usage"].get("completion_tokens", 1) - 1) / r["gen_span"] for r in rs if r["gen_span"]]
        print(f"  N={n:<3} {toks:>6} tokens in {wall:6.2f}s  aggregate {toks / wall:7.1f} tok/s"
              f"  per-stream median {st.median(per):5.1f} tok/s  TTFT median {st.median(r['t_first_any'] for r in rs):.2f}s")


if __name__ == "__main__":
    print(f"host=$PLAYGROUND_HOST:{PORT} model={MODEL} repeats={REPEATS}  (address redacted by design)\n")
    sections = {"canary": canary_section, "effort": effort_section, "decode": decode_section,
                "tools": tools_section, "prefill": prefill_section,
                "multiturn": multiturn_section, "concurrency": concurrency_section}
    for name in (sys.argv[1:] or list(sections)):
        sections[name]()
        print()
