# Method and open questions

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

## Probes

| Probe | Output | Covers |
|---|---|---|
| `k2_bench.py` | `results/k2-bench.log`, `results/k2-bench-tools-prefill.log`, `results/k2-bench-load.log` | Canary render, effort tiers, decode rate, tool-call reliability per format, prefill cost to the full served window, prefix-cache benefit, multi-turn tool use, throughput under load |
| `client_example.py` | `results/client-example.log` | The usage guide's OpenAI SDK example, run as written |
| `reasoning_fields.py` | `results/reasoning-fields.log` | Which response field carries the reasoning trace, per effort level, streaming and not |
| `weights_inventory.py` | `results/weights.log` | Architecture, parameter count, dtypes and size, read from the files on disk |
| `hub_availability.sh` | `results/hub-availability.log` | What IFM has actually published on the Hub |
| none (log capture) | `results/startup.log`, `results/node.log` | The flags the two-GPU server ran with (its "non-default args" line), startup log, GPU, driver and image |
| `k2_quality.py` | `results/quality-*.log`, `results/needle-*.log` | A 24-question answer check, tool-call reliability at both efforts, and a planted-code (needle) test in natural or synthetic long prompts |
| `serve_variant.sh` | `results/variant-*.log` | Starts the documented serve command with the weights, GPU count, context or KV-cache dtype changed, and reports cache size |
| `recipe_launch.sh` | `results/recipe-as-published.log`, `results/recipe-tp4.log`, `results/recipe-tp4-empty-prompt.log`, `results/recipe-tp4-fixed.log`, `results/k2-bench-tp4.log` | IFM's published recipe as written, on four GPUs, and with the content-format fix |
| none (one-off check) | `results/image-import-check.log`, `results/chat-template-excerpts.txt` | The broken `cu129` image, and the template code the note quotes |
| none kept | `results/k2bench.log`, `results/tok2.json`, `results/chat2.json`, `results/k2resp.json` | Output of the original investigation |

The last row is kept as history. `k2bench.log` came from a script that was
never committed and read only `delta.content`, so its `reasoning_chars` column
is always 0 and its "ttft" is the time to the first answer token. `k2_bench.py`
replaced it, and every figure in these notes now comes from the newer logs
except where the text says otherwise. The first run of `k2_bench.py` crashed in
its prefill section (the traceback is at the end of `results/k2-bench.log`);
prefill was re-measured non-streaming in the later logs.

`results/quality-bf16-tp4.log` is also kept as history. Its answer score,
59 of 72, came from a scoring bug: the model writes a narrow no-break space
between a number and its unit, and the first normaliser deleted it, so "250 cm"
failed to match "250". The rerun with the fixed scorer is
`results/quality-bf16-tp4-rerun.log` (72 of 72). Its needle section used a
200-token limit that was too small to see why replies were empty; the rerun
used 2,000.

## Repeating this investigation

Check the GPU generation against the quantisation format first. Reading one
document can rule out the entire download.

Read the model's minimum engine version. Day-zero support is common now, but
"supported" often means "in a release published this week", and an older
image fails in ways that never mention the version.

Before starting anything, find out what is holding the GPU with the driver's
own per-process query instead of a management tool. Multiple container
runtimes on one host hide work from each other.

Prove the prompt arrives before measuring anything. Send a canary string and
confirm it appears in the rendered prompt via `/tokenize`. If only the chat
endpoint misbehaves, compare against `/v1/completions`. A broken model and a
broken template look identical from the outside, and that comparison tells
them apart.

Render the prompt and read it. `/tokenize` with `add_generation_prompt` plus
`/detokenize` shows exactly what the model is conditioned on. Here it found the
missing user content in one request. A second request showed the template
pre-seeding the think tag, which was taken at the time as the reason the
reasoning trace was missing ([the reasoning trace](05-reasoning-trace.md)).

When output seems to be missing, dump every key of the response and of each
streamed delta before concluding anything. The trace here was never missing: it
was in a field the probe did not read. Running the same prompt through the raw
completion endpoint is still useful for seeing exactly what the model
generated.

Measure prefill and decode separately. They scale differently because
different things limit them: memory bandwidth for decode, compute for prefill.
A single "tokens per second" number mixes the two and predicts neither.

Test tool calling with a real tool array and check the response shape:
`finish_reason == "tool_calls"`, `content is None`, and parsed `arguments`.
Getting an answer back proves little. The wrong parser leaves markup in
`content` and reports success.

Do not inherit a neighbouring model's serve flags, because tuning is measured
against specific weights.

## Open questions

Answered on 2026-09-23, and where:

- Throughput under load: near-linear to 32 simultaneous requests
  ([results](04-results.md)).
- Prefix-cache benefit: a ~60k-token prompt from 6.97 s to 0.39 s
  ([results](04-results.md)).
- Multi-turn tool use: works at high effort; at low and medium the follow-up
  reply goes missing, for a traced reason ([how it works](03-how-it-works.md)).
- Whether IFM's recipe works as published: it does not, in two different ways
  ([failures](06-failures.md)).
- Whether the long context is usable: on natural text the model found a
  planted code up to 380,000 tokens and in two of three positions at 474,000
  ([results](04-results.md)).
- FP8 weights and an FP8 KV cache: both work on two GPUs, compared in
  [results](04-results.md).

Still open:

1. Combining FP8 weights with an FP8 KV cache. It should fit the full
   524,288-token window on two GPUs with room to spare; it was not run.
2. Retrieval near the full window with an FP8 KV cache. At ~474k tokens it
   found the code in one of three positions against two of three for BF16,
   and once returned a wrong code. One request per cell is not enough to tell.
3. At what length repetitive input starts to break the model. On ten-word
   filler it failed from about 254,000 tokens and worked at 127,000; natural
   text worked to 380,000. Neither boundary was mapped closely.
4. The reply "Hello, I'm Jonathan, an AI developed by Moonshot AI" to a
   474k-token prompt. Seen once; not investigated.
5. How often a single-turn, low-effort reply lands in `reasoning` instead of
   `content`. Seen twice ([how it works](03-how-it-works.md)).
6. What role the template's pre-seeded think tag plays in the reasoning trace
   being returned. `reasoning` is populated regardless
   ([the reasoning trace](05-reasoning-trace.md)).
7. Why `xml_typed` tool calls fail at high effort (7 of 8) when they succeed at
   low effort (10 of 10), and why `json` does the reverse.
8. Agentic and coding benchmarks. IFM's own table shows Stage 1 trailing
   badly there, and nothing here tests it.
9. Stage 2 weights, when released, against all of the above.

---

Previous: [Failures worth recording](06-failures.md) | [Contents](../README.md#contents)
