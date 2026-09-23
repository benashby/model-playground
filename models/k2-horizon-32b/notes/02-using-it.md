# Using it

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

## Serving it

The model needs vLLM 0.30.0 or later. How many GPUs depends on how much context
you want ([how it works](03-how-it-works.md) has the arithmetic):

| GPUs (80 GB each) | Largest `--max-model-len` that fits | Measured here |
|---|---|---|
| 2, `--tensor-parallel-size 2` | about 296,900 tokens | yes, at 131,072 |
| 4, `--tensor-parallel-size 4` | the full native 524,288 | yes, at 524,288, with KV cache for 875,872 tokens |
| 2, FP8 weights (`IFM/K2-Horizon-32B-FP8`) | about 423,000 | yes, at 131,072 |
| 2, BF16 weights with `--kv-cache-dtype fp8` | the full native 524,288 | yes, at 524,288 |

On two GPUs, use the FP8 weights: they decode at 101.4 tok/s against 66.6 for
BF16 and gave the same answers on the checks run here. For the full window,
four GPUs with BF16 found a planted code more reliably near the end of the
window than two GPUs with an FP8 KV cache, though the sample is small. The
comparison is in [results](04-results.md). The FP8 weights take the same
command with the model path changed.

This is the four-GPU command for the native context. It was started and
checked on 2026-09-23: it served with `max_model_len` 524,288, and a canary
reached the rendered prompt (`results/serve-512k-check.log`). The two-GPU setup
used for most of the measurements differs only in `--tensor-parallel-size 2`
and `--max-model-len 131072`; its full flag list is in the "non-default args"
line of `results/startup.log`.

```bash
podman run --rm --name k2-horizon-32b \
  --device nvidia.com/gpu=all --ipc=host -p 8080:8080 \
  -v /opt/models:/models:ro \
  docker.io/vllm/vllm-openai:v0.30.0 \
    /models/K2-Horizon-32B \
    --served-model-name k2-horizon-32b \
    --host 0.0.0.0 --port 8080 \
    --model-impl vllm --trust-remote-code --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 524288 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --max-num-seqs 32 --max-num-batched-tokens 8192 \
    --chat-template-content-format string \
    --reasoning-parser k2_horizon \
    --enable-auto-tool-choice --tool-call-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}'
```

Four flags matter more than the rest:

- `--chat-template-content-format string` makes vLLM hand the chat template
  plain-string message content. The template turns anything else into an
  empty string, so without this flag a client that sends content as a list of
  parts gets answers to an empty prompt, with no error
  ([failures](06-failures.md)). IFM's published recipe does not include it.
- `--reasoning-parser k2_horizon` separates the reasoning trace from the answer
  and returns it in the `reasoning` field.
- `--enable-auto-tool-choice --tool-call-parser k2_horizon` turns tool calls
  into structured `tool_calls` instead of text.
- `--default-chat-template-kwargs` sets the effort a request gets when it does
  not ask for one. The template itself also defaults to `high`.

`--max-model-len 131072` was our choice. The native window is 524,288 tokens,
and IFM's recipe sets no limit. The prefill cost in
[results](04-results.md) is the reason to cap it.

## Calling it

The server speaks vLLM's OpenAI-compatible API, so the OpenAI Python SDK
works. This example is `probes/client_example.py`, run against the server
above with SDK 3.19.0 (output in `results/client-example.log`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<host>:8080/v1", api_key="EMPTY")

r = client.chat.completions.create(
    model="k2-horizon-32b",
    messages=[{"role": "user", "content": "What is 17 multiplied by 23? Give the number only."}],
    temperature=1.0, top_p=0.95, max_tokens=4000,
    reasoning_effort="medium",
)
msg = r.choices[0].message
print(msg.content)                     # the answer
print(getattr(msg, "reasoning", None)) # the trace; vLLM calls this field "reasoning"
print(getattr(msg, "reasoning_content", None))  # None: not the field vLLM uses

stream = client.chat.completions.create(
    model="k2-horizon-32b", stream=True, reasoning_effort="medium", max_tokens=4000,
    messages=[{"role": "user", "content": "Name the capital of Australia in one word."}],
)
for chunk in stream:
    d = chunk.choices[0].delta if chunk.choices else None
    if d:
        print(getattr(d, "reasoning", None) or "", d.content or "", end="")
```

The trace is in `reasoning` (`delta.reasoning` when streaming). Code that reads
`reasoning_content`, the name most OpenAI-style clients use, sees nothing and
no error. That is how an earlier version of this note concluded the trace was
never returned ([the reasoning trace](05-reasoning-trace.md)).

## Sampling

IFM recommends `temperature=1.0`, `top_p=0.95` and `reasoning_effort="high"`
[CLAIM]. The card gives no top-k, min-p or penalty guidance.

## Reasoning effort

`reasoning_effort` takes `"high"`, `"medium"` or `"low"`, and the chat template
rejects anything else with an error. Each level writes a different opening
think tag into the prompt (`results/chat-template-excerpts.txt`). It can be
passed as a top-level request field, as above, or inside
`chat_template_kwargs`; both were measured and behave the same.

Three short questions, three runs each, nine requests per row, passed as a
top-level field (`probes/k2_bench.py`, `results/k2-bench.log`):

| effort | tag in the prompt | completion tokens, median (range) | runs with no trace | first token | first answer token |
|---|---|---|---|---|---|
| `low` | `<ifm\|think_faster>` | 5 (4 to 165) | 8 of 9 | 0.10 s | 0.10 s |
| `medium` | `<ifm\|think_fast>` | 57 (20 to 88) | 0 of 9 | 0.08 s | 0.90 s |
| `high` | `<ifm\|think>` | 113 (22 to 502) | 0 of 9 | 0.08 s | 1.74 s |
| not set (server default) | `<ifm\|think>` | 206 (47 to 721) | 0 of 9 | 0.08 s | 3.15 s |

The first token of any kind arrives in about 0.1 s at every level. Effort
changes how long the model thinks before it starts the answer, and how many
tokens you pay for. High-effort cost varies a lot between runs of the same
question, so budget from the range rather than the median.

An earlier version of this table showed 25, 61 and 260 tokens and a "TTFT" of
0.41 s, 0.91 s and 3.93 s. Those came from one question, and the timing was the
time to the first answer token, because the old probe never read the
`reasoning` field.

## Tool calling

With `--enable-auto-tool-choice --tool-call-parser k2_horizon` on the server, a
standard OpenAI `tools` array returns structured calls:

```
finish_reason: "tool_calls"
tool_calls:    [{"type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Oslo\", \"unit\": \"celsius\"}"}}]
```

The model has three call formats, chosen per request with
`chat_template_kwargs: {"tool_call_format": ...}`. How reliable each one is
depends on the effort level, and the rankings reverse between low and high
(full numbers in [results](04-results.md)):

| `tool_call_format` | low effort | high effort |
|---|---|---|
| `xml` (default) | 9 of 10 | 8 of 8 |
| `json` | 5 of 10 | 8 of 8 |
| `xml_typed` | 10 of 10 | 1 of 8 |

Leave the format at the default, `xml`, which was the only one reliable at both
levels. A failed call does not raise an error: the call's markup comes back as
text in `content` with `finish_reason: "stop"`. Treat any response whose
`content` contains `<ifm|tool_call` as a failed call, not an answer.

An earlier version of this note said all three formats "parsed correctly".
That came from a single low-effort request per format.

### Sending the tool result back

To continue after a call, send the assistant's turn back with its reasoning in
a `reasoning` (or `reasoning_content`) field, then the tool result:

```python
messages = [
    {"role": "user", "content": "What's the weather in Oslo, in celsius?"},
    {"role": "assistant", "content": "", "reasoning": first.reasoning,
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": -7, "sky": "snow"}'},
]
```

Two things go wrong here, both measured:

- Leave out the thinking field and the request fails with HTTP 400, "Assistant
  message is missing a thinking field". The template insists on one. It also
  accepts `think`, `think_fast` and `think_faster`, but vLLM's API drops fields
  it does not know, so only `reasoning` and `reasoning_content` get through.
- Run the follow-up turn at low or medium effort and `content` comes back
  empty: 16 of 16 times here. In the three such turns inspected in full
  (`results/multiturn-detail.log`), two had written a reply inside `reasoning`,
  after a `</ifm|think>` tag, and one ended without writing a reply at all. At
  high effort the reply arrived in `content` 7 of 7 times.

So run the turn after a tool result at `reasoning_effort="high"`. At a lower
effort, when `content` is empty, the text after `</ifm|think>` in `reasoning`
is the reply if it is there, but there may be none. That check is worth doing
on any low-effort request: the same misrouting was seen twice on single-turn
requests too. [How it works](03-how-it-works.md) explains why.

## Context

The server here allows 131,072 tokens of the 524,288-token native window. That
cap was our choice: prefill cost grows faster than prompt length, as measured
in [results](04-results.md).

---

Previous: [Requirements and deployment](01-requirements.md) | [Contents](../README.md#contents) | Next: [How it works](03-how-it-works.md)
