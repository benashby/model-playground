# How it works

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

Most of what this model costs to run, and most of the ways it failed here, can
be worked out from three things: its shape, the memory bandwidth of the GPUs,
and its chat template. This article works each one out and checks the result
against what vLLM reported.

## The shape

From `config.json` and the safetensors headers (`results/weights.log`):

| | |
|---|---|
| Layers | 64 |
| Hidden size | 5120, with an MLP of 26624 |
| Attention | 64 query heads and 8 key/value heads, head_dim 128 |
| Positions | RoPE, `rope_theta` 10,000,000, no scaling, 524,288 positions |
| Weights | 34,779,304,960 parameters, all BF16, 69,558,691,056 bytes |

Eight key/value heads serve 64 query heads, so each KV head is shared by eight
query heads (grouped-query attention). This keeps the long context
affordable, because the KV cache stores only the eight shared heads.

The model was trained up to 512K context and no further. RoPE is used without
any scaling, and `max_position_embeddings` is 524,288. Going past that would
mean extrapolating the position encoding beyond anything in training.

## What the KV cache costs, and why 512K needs four GPUs

Every token in context stores a key and a value vector for every KV head in
every layer:

```
2 (K and V) × 64 layers × 8 KV heads × 128 dims × 2 bytes (BF16) = 262,144 bytes
```

That is 256 KiB per token. With tensor parallelism the cache is split evenly
across the GPUs, so each one holds its share. The arithmetic predicts vLLM's
own numbers closely:

| | Predicted from 256 KiB per token | Reported by vLLM |
|---|---|---|
| Cache for one 524,288-token request, 2 GPUs | 64.0 GiB per GPU | "64.0 GiB KV cache is needed" (`results/recipe-as-published.log`) |
| Tokens that fit in 36.25 GiB per GPU, 2 GPUs | 296,960 | "estimated maximum model length is 296928" |
| Tokens that fit in 52.56 GiB per GPU, 4 GPUs | 861,143 | "GPU KV cache size: 861,184 tokens" (`results/recipe-tp4.log`) |

The small gaps are vLLM allocating in blocks. In practice:

- On two 80 GB GPUs the weights take 32.4 GiB each and leave about 36 GiB for
  the cache. That holds about 297k tokens, so the native 512K window cannot
  fit. The largest usable limit is just under 297k.
- On four GPUs each holds only 16.2 GiB of weights and 52.56 GiB of cache,
  enough for 861k tokens: one full-length request with room for most of a
  second.
- An FP8 KV cache halves the per-token cost to 128 KiB. On two GPUs with
  37.06 GiB of cache each, that predicts 607,191 tokens; vLLM reported 607,200
  (`results/variant-bf16-tp2-kvfp8.log`), enough for the full window.
- FP8 weights halve the weight memory instead, which leaves 51.69 GiB of cache
  per GPU on two GPUs: 423,444 tokens predicted, 423,456 reported
  (`results/variant-fp8-tp2.log`).

## Decode speed is set by memory bandwidth

Generating one token means reading every weight once. At batch size 1 the GPUs
spend most of their time moving weights from memory, not computing, so memory
bandwidth sets a ceiling. NVIDIA rates the H100 SXM at 3.35 TB/s [CLAIM].

| GPUs | Weight bytes each GPU reads per token | Bandwidth ceiling | Measured | Fraction of ceiling |
|---|---|---|---|---|
| 2 | 34.8 GB | 96 tok/s | 66.6 tok/s | 69% |
| 4 | 17.4 GB | 193 tok/s | 108.7 tok/s | 56% |
| 2, FP8 weights | about 17.4 GB | about 193 tok/s | 101.4 tok/s | 53% |

Measured figures are from `results/k2-bench.log`, `results/k2-bench-tp4.log`
and `results/quality-fp8-tp2.log`. FP8 weights halve the bytes read per token
without adding GPUs, so they reach about the same speed as four BF16 GPUs on
two. The FP8 row's byte count is approximate, because `lm_head` stays in BF16.
Doubling the GPUs made decode 1.63× faster, not 2×. Each extra GPU adds an
all-reduce between them in every layer, and that communication does not shrink
as the weights are split further. That explanation fits the numbers, but it was
inferred, not measured.

With 32 simultaneous streams on two GPUs, each stream still ran at 61.7 tok/s
and the aggregate reached 1,897 tok/s (`results/k2-bench-load.log`). The weights read for one token serve every
stream in the batch, so batching is close to free until something else runs
out.

## Prefill cost grows with the square of the prompt

Reading a prompt before the first token (prefill) is compute-bound. Attention
compares every token with every earlier token, so its cost grows with the
square of the prompt length. The rest of the model grows linearly.

A curve of that form fits the two-GPU measurements (`results/k2-bench-load.log`)
to within 0.04 s at every point from 224 to 126,747 tokens:

```
seconds ≈ 0.102 + 60.79e-6 × n + 897.2e-12 × n²
```

At 126,747 tokens the squared term is already 65% of the time. Extrapolating
the fit (not measured) gives about 97 s at 297k tokens and 279 s at 524k on two
GPUs. An earlier version of this note extrapolated linearly and predicted about
11 s at 131k; the measured value at 126,747 tokens is 22.24 s.

On four GPUs, measured directly out to 495,026 tokens (`results/k2-bench-tp4.log`),
the same form fits to within 0.24 s at every point:

```
seconds ≈ 0.106 + 37.33e-6 × n + 451.5e-12 × n²
```

The squared term is almost exactly half the two-GPU one, which is what
splitting the attention work across twice as many GPUs should give. At 495,026
tokens it accounts for 86% of the 129.13 s wait. Using the model's full context
is possible on four GPUs, but a near-full prompt costs over two minutes before
the first token unless the prefix cache already holds it.

A prefix-cache hit skips this work. The same ~60k-token prompt took 6.97 s the
first time and 0.39 s on each repeat on two GPUs. An agent that keeps a long
system prompt fixed pays the prefill cost for it once.

## The chat template

The template ships with the weights as `chat_template.jinja`, 994 lines. Three
parts of it explain the three silent failures found in this investigation. The
relevant lines are in `results/chat-template-excerpts.txt`.

### Content that is not a plain string becomes empty

Lines 932 to 937 keep a message's `content` only if it is a string, and replace
anything else with an empty string. OpenAI-style clients may send content as a
list of parts (`[{"type": "text", "text": ...}]`), and vLLM may convert every
message to that form before rendering.

vLLM decides which form to use per template, and for this one it decides wrong.
The startup log of IFM's own recipe says `Detected the chat template content
format to be 'openai'`, so every message is converted to parts, and every
prompt renders to the same 10 tokens with no user text in them. That includes
plain-string requests. `--chat-template-content-format string` overrides the
detection. See [failures](06-failures.md) for what it looked like from outside.

### Effort is a tag written into the prompt

Lines 983 to 994 end the prompt by opening the assistant's thinking block, with
a different tag for each effort level: `<ifm|think>` for high,
`<ifm|think_fast>` for medium and `<ifm|think_faster>` for low. Anything else
raises an error. The model writes its reasoning, emits the matching closing
tag, then writes the answer. vLLM's `k2_horizon` reasoning parser splits the
output at the closing tag, putting what comes before it in `reasoning` and what
comes after in `content`.

At low effort the model usually closes the block immediately. That is why 8 of
9 low-effort answers came back with no trace at all.

### Past thinking is replayed with the high-effort tag

A tool-calling conversation has to send the assistant's earlier turn back,
including its reasoning. Lines 941 to 961 require one of five fields for it:
`think`, `think_fast`, `think_faster`, `reasoning` or `reasoning_content`.
Each renders with its own tag, and `reasoning` and `reasoning_content` both
render as `<ifm|think>`, the high-effort tag.

vLLM's OpenAI-compatible layer drops message fields it does not recognise, so
only `reasoning` and `reasoning_content` ever reach the template. Sending
`think_faster` gets HTTP 400, "Assistant message is missing a thinking field"
(`results/multiturn-tags.log`).

So a low- or medium-effort follow-up turn shows the model a history closed with
`</ifm|think>`, then opens a new block with a different tag. The model copies
the tag from the history and closes with `</ifm|think>`. vLLM's parser picks
the closing tag to look for from the request's effort level and splits only at
that exact tag (`k2_horizon_reasoning_parser.py`, lines 19 to 21 and 54 to 55,
in `results/reasoning-parser-excerpt.txt`). At low effort it waits for
`</ifm|think_faster>`, never sees it, and files everything, including any
reply, under `reasoning`. Measured on the turn after a tool result, eight
requests each:

| Effort on the follow-up turn | Reply in `content` | `content` empty, tool result only in `reasoning` |
|---|---|---|
| low | 0 | 8 |
| medium | 0 | 8 |
| high | 7 | 0 (one first call failed) |

In three low-effort turns inspected in full (`results/multiturn-detail.log`),
two contained a finished reply after `</ifm|think>`, exactly as the tag
mismatch predicts, and one stopped without writing a reply.

At high effort the tags match and the reply arrives normally. The template
supports matching the history's tag to the effort, but that route is closed
through vLLM's API.

History makes the mismatch almost certain, but it is not the only trigger. In
two single-turn, low-effort requests with no history at all, the model also
closed its thinking with `</ifm|think>` and the answer came back in `reasoning`
with `content` empty (`results/needle-selfcheck-32k.log` and
`results/needle-sweep-bf16-tp4.log`). How often that happens was not measured.

### Tool-call formats

The template writes tool definitions into the prompt and accepts three formats
for the model's calls, chosen with `chat_template_kwargs.tool_call_format`:
`xml` (the default), `json` and `xml_typed`. How reliable each is depends on
the effort level. That is measured in [results](04-results.md) and turned into
advice in [using it](02-using-it.md).

---

Previous: [Using it](02-using-it.md) | [Contents](../README.md#contents) | Next: [Results](04-results.md)
