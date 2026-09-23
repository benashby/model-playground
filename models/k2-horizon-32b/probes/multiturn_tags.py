"""Two follow-ups to multiturn_detail.py.

1. Hypothesis: at low effort, a second turn loses its answer into `reasoning`
   because the history's thinking is rendered with <ifm|think> tags (the
   template does that for the `reasoning`/`reasoning_content` fields), while
   the new turn opens with <ifm|think_faster>. The model copies the history's
   closing tag, the parser waits for </ifm|think_faster>, and the answer stays
   in `reasoning`. If that is right, sending the history's thinking under the
   field that matches the effort (`think_faster` for low) should fix it.

2. Tool-call reliability at high effort (the server default), per format,
   printing what came back when no structured call did.

    PLAYGROUND_HOST=<host> python models/k2-horizon-32b/probes/multiturn_tags.py
"""
import json, os, sys, urllib.error
sys.path.insert(0, os.path.dirname(__file__))
from k2_bench import WEATHER, nonce, stream_chat

N = int(os.environ.get("N", "8"))
RESULT = json.dumps({"city": "Oslo", "temp_c": -7, "sky": "snow"})
MATCH = {"low": "think_faster", "medium": "think_fast", "high": "think"}
print("host=$PLAYGROUND_HOST model=k2-horizon-32b  (address redacted by design)\n")

print(f"### 1. second turn after a tool result, by effort and history thinking field, n={N} each")
print("    (vLLM's OpenAI layer drops message keys it does not know, so `think_faster` may never reach the template)")
for effort, field in (("low", "reasoning"), ("low", "think_faster"), ("medium", "reasoning"), ("high", "reasoning")):
    t = {"answer in content": 0, "answer stuck in reasoning": 0, "no answer": 0, "no first call": 0, "http error": 0}
    err = ""
    for _ in range(N):
        q = f"[{nonce()}] What's the weather in Oslo, in celsius?"
        kw = {"tools": [WEATHER], "reasoning_effort": effort, "max_tokens": 4000,
              "chat_template_kwargs": {"tool_call_format": "xml_typed"}}
        first = stream_chat({"messages": [{"role": "user", "content": q}], **{**kw, "reasoning_effort": "low"}})
        if not first["tool_calls"]:
            t["no first call"] += 1; continue
        c = first["tool_calls"][0]
        asst = {"role": "assistant", "content": "", field: first["reasoning"] or "Calling the weather tool.",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}]}
        try:
            second = stream_chat({"messages": [{"role": "user", "content": q}, asst,
                                               {"role": "tool", "tool_call_id": "call_1", "content": RESULT}], **kw})
        except urllib.error.HTTPError as e:
            t["http error"] += 1; err = json.loads(e.read().decode()).get("error", {}).get("message", "")[:110]; continue
        if "-7" in second["content"]:
            t["answer in content"] += 1
        elif "-7" in second["reasoning"]:
            t["answer stuck in reasoning"] += 1
        else:
            t["no answer"] += 1
    print(f"  effort {effort:<6} history field {field:<13} {t}" + (f"\n      error: {err}" if err else ""))

print(f"\n### 2. first-turn tool calls at HIGH effort, n={N} per format")
for fmt in ("xml_typed", "xml", "json"):
    ok, misses = 0, []
    for _ in range(N):
        r = stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] What's the weather in Oslo, in celsius?"}],
                         "tools": [WEATHER], "tool_choice": "auto", "reasoning_effort": "high", "max_tokens": 4000,
                         "chat_template_kwargs": {"tool_call_format": fmt}})
        if r["tool_calls"] and r["tool_calls"][0]["name"] == "get_weather":
            ok += 1
        else:
            tail = r["reasoning"].strip()[-160:]
            misses.append(f"finish={r['finish']} tokens={r['usage'].get('completion_tokens')} "
                          f"content={r['content'].strip()[:120]!r} reasoning_tail={tail!r}")
    print(f"  {fmt:<10} structured call {ok}/{N}")
    for m in misses[:3]:
        print(f"      miss: {m}")
