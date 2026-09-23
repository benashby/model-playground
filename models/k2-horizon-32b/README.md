# IFM K2-Horizon-32B

A 32B dense reasoning LLM, deployed and investigated on 2026-09-23.

> Not a harness subject. This is a text-only model with no audio, turns or
> barge-in, so `DuplexSession` cannot drive it and nothing here tries. It is
> documented because it was deployed onto the same CUDA node the speech model
> uses, and getting it to serve surfaced several failures worth recording. One
> of them, a chat template that emptied every prompt, is the clearest case in
> this repository of a server reporting success while doing nothing.

- Weights (BF16): <https://huggingface.co/IFM/K2-Horizon-32B> (Apache-2.0)
- NVFP4 variant, not used: <https://huggingface.co/IFM/K2-Horizon-32B-NVFP4>
- Serving recipes: <https://recipes.vllm.ai/IFM>

## What the model is

A 32B dense decoder-only model from IFM, part of the K2-Horizon family
(0.9B / 3.7B / 7B / 32B / MoVA-36B-A4B / 375B-A23B). The weights are
Apache-2.0. IFM's announcement is titled "Frontier Performance, Radically
Open", and its earlier models (Amber, Crystal, the first K2) were released with
their training datasets, which are public on the Hub.

For this model the openness is so far a promise. The card says the training
data, recipe and code "will be made public" and that intermediate checkpoints
"will be released" [CLAIM]. It lists two datasets,
`IFM/K2-Horizon-Pretrain-Data` and `IFM/K2-Horizon-Midtrain-Data`, and a branch
table marking intermediate checkpoints as available. On 2026-09-23 neither
dataset was publicly reachable (HTTP 401) and the model repository had a single
branch, `main` (`results/hub-availability.log`). An earlier version of this
note said the datasets were already published; that was wrong.

| Property | Value |
|---|---|
| Architecture | `K2HorizonForCausalLM`, dense decoder-only |
| Parameters | 34.78 B (34,779,304,960) |
| Layers | 64 |
| Hidden / intermediate | 5120 / 26624, head_dim 128 |
| Attention heads | 64 query, 8 key/value |
| Vocabulary | 250,624 |
| Native context | 524,288 tokens |
| Weights on disk | 64.78 GiB, all BF16, 579 tensors in 64 safetensors shards |
| Reasoning | returned separately; vLLM 0.30.0 puts it in `reasoning` ([details](notes/05-reasoning-trace.md)) |
| Tool calls | `json`, `xml`, or `xml_typed` (default `xml`) |
| Vendor sampling | `reasoning_effort="high"`, `temperature=1.0`, `top_p=0.95` |

The architecture rows are read from `config.json` and the safetensors headers
(`probes/weights_inventory.py`, `results/weights.log`), not from the card.

The card says this is Stage 1 of the final training run. Its published
benchmarks are mid-training numbers, and Stage 2 is unreleased. In the card's
own comparison table, Stage 1 trails a contemporary 27B dense model by a wide
margin on agentic and coding work (tau3-Banking 22.5 vs 48.0; Terminal-Bench
36.6 vs 79.8) and is closer on knowledge (GPQA Diamond 82.3 vs 90.5).

### Who would pick it

On IFM's own numbers it is not the model to choose for agentic or coding work
today. The card compares it with Qwen3.8-27B, a dense model of similar size
that leads it by a wide margin on both [CLAIM].

Its case rests on two things. The first is licensing and provenance: Apache-2.0
weights from a lab whose earlier models shipped with their training data, which
matters for reproducibility and for checking a benchmark against the training
set. That argument only fully holds once the promised data and intermediate
checkpoints are public, which on 2026-09-23 they were not. The second is its
native 524,288-token context, although prefill cost limits how much of it is
practical to use (see [results](notes/04-results.md)).

It is a research checkpoint from Stage 1 of an unfinished training run, and
should not be expected to stand in for a mature model.

## What was found

- IFM's published vLLM recipe fails on two 80 GB GPUs: without a context
  limit it needs 64.0 GiB of KV cache per GPU and has 36.25 GiB. With four
  GPUs it starts, and then answers an empty prompt on every request, because
  vLLM converts all messages to a form the chat template discards. The
  model's reply to "What is the capital of Australia?" was "Hello! How can I
  assist you today?". One flag, `--chat-template-content-format string`,
  fixes it. ([failures](notes/06-failures.md))
- The full native 524,288-token context fits on four H100s, with KV cache for
  861,184 tokens. Every token costs 256 KiB of cache, which predicts vLLM's own
  memory figures to within a block. ([how it works](notes/03-how-it-works.md))
- Decode runs at 66.6 tok/s on two GPUs and 108.7 on four. Prefill cost grows
  with the square of the prompt: 22.24 s for 126,747 tokens on two GPUs. A
  prefix-cache hit cut a ~60k-token prompt from 6.97 s to 0.39 s. Thirty-two
  simultaneous requests reached 1,897 tok/s. ([results](notes/04-results.md))
- Tool calling works, but which call format is reliable depends on the effort
  level, and the turn after a tool result comes back with empty `content` at
  low and medium effort (16 of 16). The cause is a tag mismatch between the
  chat template and vLLM's parser, traced in both sources.
  ([using it](notes/02-using-it.md), [how it works](notes/03-how-it-works.md))
- On two GPUs, IFM's FP8 weights decode 1.52 times faster than BF16 (101.4
  against 66.6 tok/s) with the same answers on the checks run. With BF16
  weights, an FP8 KV cache fits the full 524,288-token window on two GPUs.
  ([results](notes/04-results.md))
- The long context is usable on natural text: a planted code was found at every
  length tried up to 380,000 tokens. Very repetitive input makes the model
  degenerate into word salad from about 254,000 tokens.
  ([results](notes/04-results.md))
- The reasoning trace is returned in `reasoning`, not `reasoning_content`. An
  earlier version of this note said it was discarded, because the probe read
  the wrong field. ([the reasoning trace](notes/05-reasoning-trace.md))
- The training data and intermediate checkpoints named on the card were not
  public on 2026-09-23.

## Contents

| Article | What it covers |
|---|---|
| [Requirements and deployment](notes/01-requirements.md) | GPUs and context length, why BF16, the vLLM version floor, and settings not copied from a neighbouring model |
| [Using it](notes/02-using-it.md) | The serve command and its key flags, a tested client example, effort levels, tool calls and multi-turn use |
| [How it works](notes/03-how-it-works.md) | KV-cache arithmetic, decode against memory bandwidth, the prefill curve, and the chat template's mechanics |
| [Results](notes/04-results.md) | Decode, load, prefill to 500k tokens, prefix caching, FP8 against BF16, long-context retrieval, tool-call reliability and liveness |
| [The reasoning trace](notes/05-reasoning-trace.md) | Where the trace is returned, and how an earlier conclusion about it went wrong |
| [Failures worth recording](notes/06-failures.md) | A broken image, an invisible process, the recipe that empties every prompt, and tool calls that fail silently |
| [Method and open questions](notes/07-method.md) | The probes, the order to repeat the work in, and what is still unknown |

The probes behind every number are in [`probes/`](probes/) and their raw output
is in [`results/`](results/). A few older logs from before the probes existed
are kept as history; [method](notes/07-method.md) says which. The standard these
notes are held to is in [`models/README.md`](../README.md).
