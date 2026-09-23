"""Where does the second-turn answer go when `content` comes back empty?

k2_bench.py's multi-turn section found second turns that end with
finish_reason "stop", no tool call and empty `content`. This probe repeats that
exchange and prints both fields in full, plus whether the tool's result (-7,
snow) appears in either, to tell a parser misroute (answer inside `reasoning`)
from a model that stops without answering. Also compares reasoning efforts.

    PLAYGROUND_HOST=<host> python models/k2-horizon-32b/probes/multiturn_detail.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from k2_bench import WEATHER, nonce, stream_chat

KW = {"chat_template_kwargs": {"tool_call_format": "xml_typed"}}
RESULT = {"city": "Oslo", "temp_c": -7, "sky": "snow"}
N = int(os.environ.get("N", "6"))

print(f"host=$PLAYGROUND_HOST model=k2-horizon-32b  (address redacted by design)")
for effort in ("low", "high"):
    tally = {"answered in content": 0, "result only in reasoning": 0, "no answer anywhere": 0, "no first call": 0}
    for i in range(N):
        q = f"[{nonce()}] What's the weather in Oslo, in celsius?"
        first = stream_chat({"messages": [{"role": "user", "content": q}], "tools": [WEATHER],
                             "reasoning_effort": effort, "max_tokens": 2000, **KW})
        if not first["tool_calls"]:
            tally["no first call"] += 1
            continue
        c = first["tool_calls"][0]
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "content": "", "reasoning": first["reasoning"],
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": c["name"], "arguments": c["arguments"]}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(RESULT)}]
        second = stream_chat({"messages": msgs, "tools": [WEATHER], "reasoning_effort": effort,
                              "max_tokens": 2000, **KW})
        has = lambda t: "-7" in t or "minus 7" in t.lower() or "−7" in t
        if has(second["content"]):
            tally["answered in content"] += 1
        elif has(second["reasoning"]):
            tally["result only in reasoning"] += 1
        else:
            tally["no answer anywhere"] += 1
        if i < 2 or not second["content"].strip():
            print(f"--- effort={effort} run {i+1}: finish={second['finish']} "
                  f"completion_tokens={second['usage'].get('completion_tokens')}")
            print(f"    content  : {second['content'].strip()!r}")
            print(f"    reasoning: {second['reasoning'].strip()!r}")
    print(f"== effort={effort} n={N}: {tally}\n")
