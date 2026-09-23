# Results

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

Hardware as in [requirements](01-requirements.md): 2 × H100 80GB HBM3, TP=2, NVLink, vLLM 0.30.0, BF16.

## Decode throughput

Single stream, measured as generated tokens over the time between the first
and last streamed token, with a warmup run discarded (`probes/k2_bench.py`,
`results/k2-bench.log`):

| Run | tok/s | tokens |
|---|---|---|
| 1 | 66.6 | 366 |
| 2 | 66.6 | 352 |
| 3 | 66.6 | 350 |
| median | 66.6 | |

The original, uncommitted script measured 65.6 tok/s on three runs
(`results/k2bench.log`). Each set agrees with itself to one decimal place.
For a 32 B dense model in BF16 on two H100s at batch 1, that is the expected
order of magnitude. Decode is memory-bandwidth-bound, and a dense 32 B model
has to move all 65 GiB of weights across the tensor-parallel pair for every
token step.

These are single-stream decode figures, so do not quote them as a capacity
figure. Throughput with concurrent requests is measured separately below.

## Decode on four GPUs

The same measurement on four H100s with `--tensor-parallel-size 4`
(`results/k2-bench-tp4.log`): 108.3, 108.8 and 108.7 tok/s, median 108.7. That
is 1.63 times the two-GPU rate. [How it works](03-how-it-works.md) compares both
with the memory-bandwidth ceiling.

## Throughput under load

Two GPUs, N requests started at once, each generating 300 tokens
(`results/k2-bench-load.log`):

| Simultaneous requests | Aggregate tok/s | Per request, median tok/s | Time to first token, median |
|---|---|---|---|
| 1 | 65.2 | 66.4 | 0.10 s |
| 4 | 252.5 | 64.9 | 0.15 s |
| 8 | 502.9 | 65.1 | 0.16 s |
| 16 | 978.7 | 63.3 | 0.17 s |
| 32 | 1897.0 | 61.7 | 0.20 s |

Aggregate throughput grows almost in proportion to the number of requests,
while each request slows by only 7% at 32. The server was started with
`--max-num-seqs 32`, so 32 is its limit; beyond that requests queue.

## Prefill: time to first token against prompt length

Every request starts with a random string so that no two share a prefix and
the prefix cache cannot help. Time is a non-streaming request for one token,
which is prefill plus one decode step of about 15 ms. Three runs per row.

Two GPUs (`results/k2-bench-load.log`):

| Prompt tokens | Time to first token, median | Prompt tokens per second |
|---|---|---|
| 224 | 0.10 s | 2,238 |
| 2,007 | 0.22 s | 9,048 |
| 15,867 | 1.31 s | 12,068 |
| 31,706 | 2.95 s | 10,765 |
| 63,388 | 7.57 s | 8,375 |
| 95,066 | 13.95 s | 6,815 |
| 126,747 | 22.24 s | 5,700 |

Four GPUs (`results/k2-bench-tp4.log`):

| Prompt tokens | Time to first token, median | Prompt tokens per second |
|---|---|---|
| 2,007 | 0.17 s | 12,105 |
| 31,708 | 1.78 s | 17,824 |
| 126,747 | 12.12 s | 10,458 |
| 253,468 | 38.37 s | 6,606 |
| 380,187 | 79.80 s | 4,764 |
| 495,026 | 129.13 s | 3,834 |

A prompt near the full native window takes a little over two minutes to reach
its first token, even on four GPUs.

The rate peaks around 16k tokens and then falls, because attention's cost grows
with the square of the prompt ([how it works](03-how-it-works.md) fits the
curve). An earlier version of this note measured up to 60,019 tokens and
extrapolated in a straight line to about 11 s at 131k. The measured figure at
126,747 tokens on two GPUs is 22.24 s.

A prefix-cache hit removes nearly all of this cost. On two GPUs the same
~60k-token prompt took 6.97 s the first time and 0.38 to 0.39 s on each of
three repeats. On four GPUs the same ~60k-token prompt took 3.97 s the first
time and 0.36 to 0.39 s on repeats.

## Cheaper configurations: FP8 weights and an FP8 KV cache

IFM's FP8 release (`IFM/K2-Horizon-32B-FP8`, 34.79 GiB, block-quantised FP8
with `lm_head` left in BF16) ships the same chat template and tokenizer as the
BF16 release; only whitespace and a version string differ. It was compared
with the BF16 weights, and separately with BF16 weights plus
`--kv-cache-dtype fp8`, using the same launcher (`probes/serve_variant.sh`)
and the same probes on every configuration.

| | BF16 weights, 4 GPUs | BF16 weights, 2 GPUs | BF16 weights + FP8 KV cache, 2 GPUs | FP8 weights, 2 GPUs |
|---|---|---|---|---|
| Context served | 524,288 | 131,072 | 524,288 | 131,072 |
| KV cache room | 875,872 tokens | 303,968 tokens | 607,200 tokens | 423,456 tokens |
| Decode, one request | 108.7 tok/s | 66.6 tok/s | 66.3 tok/s | 101.4 tok/s |
| Prefill, ~125k tokens | 12.12 s | 22.24 s | 21.41 s | 19.49 s |
| Answers correct (72) | 72 | not run | 71 | 72 |
| Tool calls, low effort; high effort | 10/10; 10/10 | 9/10; 8/8 | 9/10; 10/10 | 8/10; 10/10 |
| Needle at ~126k, three depths | 3 of 3 | not run | 1 of 1 (50% only) | 3 of 3 |
| Needle at ~474k, three depths | 2 of 3 | cannot fit | 1 of 3 | cannot fit |

Sources: `results/quality-bf16-tp4.log`, `results/quality-bf16-tp4-rerun.log`,
`results/needle-126k-bf16-tp4.log` and `results/needle-sweep-bf16-tp4.log`
for the four-GPU column; `results/k2-bench.log`, `results/k2-bench-load.log`
and `results/k2-bench-tools-prefill.log` for the two-GPU BF16 column;
`results/variant-bf16-tp2-kvfp8.log` and `results/quality-bf16-tp2-kvfp8.log`
for the FP8 KV cache; `results/variant-fp8-tp2.log` and
`results/quality-fp8-tp2.log` for the FP8 weights. Every cell with a count is
from a small number of requests, so treat a difference of one as noise.

On the same two GPUs the FP8 weights decode 1.52 times faster than the BF16
weights and leave room for 39% more cached tokens, and they matched the
four-GPU BF16 server on the answers and the needle test. Low-effort tool calls
were 8 of 10 against 10, a gap these counts cannot resolve. The checks are
small enough to catch gross damage and nothing finer, so equal quality is not
shown.

The FP8 KV cache fits the full 524,288-token window on two GPUs, as the
arithmetic predicted, and decodes at the same speed as BF16 on two GPUs. It
also gave the one wrong answer across all three quality runs: 17 × 23 given as
289. Its weakest result is at the far end of the window. At ~474k tokens it
found the code in one of three positions, against two of three for BF16, and
in one of the misses it returned a wrong six-digit number instead of saying it
could not find one. One request per cell cannot separate that from chance, but
it is where a lower-precision cache would be expected to hurt.

## Long context: does it use what it reads?

To check whether the model can find something in a long prompt, a random
six-digit code was planted in one and the model was asked for it, at low effort, one request per cell, on the four-GPU
BF16 server (`probes/k2_quality.py`, `results/needle-sweep-bf16-tp4.log`). Two
kinds of haystack were used: natural prose (Project Gutenberg's *War and
Peace*, public domain) and synthetic filler drawn at random from ten words.

| Prompt tokens | Natural text, code at 50% | Synthetic filler, code at 50% |
|---|---|---|
| ~32,000 | found | found (and at 10% and 90%) |
| ~127,000 | found | found |
| ~254,000 | found | lost: word salad until the token limit |
| ~380,000 | found | lost: word salad until the token limit |
| ~474,000 | found (and at 90%); lost at 10% | lost at 10%, 50% and 90%, word salad |

On natural text the model found the code at every length up to 380,000
tokens, and in two of three positions at 474,000. On synthetic filler it broke
down from about 254,000 tokens: instead of answering it produced unrelated
words ("Village of lot of window village Hills economic Military Housing
Area...") until it reached the 2,000-token limit. The long context is usable on
ordinary text, then, but very repetitive input can make the model degenerate
well short of the window, and real long inputs such as logs or data dumps can
be repetitive.

The one natural-text miss is odd enough to quote. With the code near the start
of a 474,000-token prompt, the model ignored the question and replied "Hello,
I'm Jonathan, an AI developed by Moonshot AI." That is one observation, and
nothing here explains it.

In one natural-text case (~254,000 tokens) the code was found but arrived in
`reasoning` with `content` empty, the same tag mismatch described in
[how it works](03-how-it-works.md), this time on a single-turn request.

## Tool calling

A request for the weather in Oslo with one weather tool offered, counting a
structured call with the right city as a success
(`results/k2-bench-tools-prefill.log` for low effort, `results/multiturn-tags.log`
for high):

| `tool_call_format` | Low effort (n=10) | High effort (n=8) |
|---|---|---|
| `xml` (default) | 9 | 8 |
| `json` | 5 | 8 |
| `xml_typed` | 10 | 1 |

The failures and what they looked like are in [failures](06-failures.md).

## The turn after a tool result

The model called the tool, the tool result was sent back, and the next turn
was checked for an answer that used it (`results/multiturn-tags.log`, eight
requests per row):

| Follow-up turn effort | Reply in `content` | `content` empty, tool result only in `reasoning` |
|---|---|---|
| low | 0 | 8 |
| medium | 0 | 8 |
| high | 7 | 0 |

The high row had one request where the first call failed, so there were seven
follow-ups. Sending the earlier turn without any thinking field returned HTTP
400 every time (`results/k2-bench-multiturn.log`).

## Reasoning quality spot-check

Asked how many minutes a train journey from 14:05 to 17:40 takes, with the
answer as a number only, the model answered 215, which is correct, in all
three runs at high effort (`results/k2-bench.log`). The trace returned in
`reasoning` for an earlier run of the same question (`results/chat2.json`)
works it out four ways, then checks the result against a timeline before
answering.

## Liveness checks

| Check | Result | Evidence |
|---|---|---|
| Model listed | `/v1/models` returns the served id, `max_model_len` 131072, vLLM `/version` 0.30.0 | checked 2026-09-23 |
| Architecture | `Resolved architecture: K2HorizonForCausalLM` | `results/startup.log` |
| VRAM resident | 73,843 MiB per GPU, both cards, TP=2 | `results/node.log` |
| NVLink | FlashInfer allreduce `backend=mnnvl` | `results/startup.log` |
| Tool choice | `"auto" tool choice has been enabled` | `results/startup.log` |
| Prompt actually arrives | a canary string is present in the rendered prompt, sent both as a plain string and as a list of parts | `results/k2-bench.log` |

---

Previous: [How it works](03-how-it-works.md) | [Contents](../README.md#contents) | Next: [The reasoning trace](05-reasoning-trace.md)
