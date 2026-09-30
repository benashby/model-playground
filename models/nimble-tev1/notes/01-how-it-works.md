# How System One scoring works

> Part of the [Nimble 9B and Tev1](../README.md) investigation. See also [all model notes](../../README.md).

Everything on this page was read from source code or from the files Ollama
ships, and each statement names the file or probe it came from.

## The request

A request carries `state` (a string, or an object or array) and 1 to 64 named
questions. A question has a `type`, `instructions` and usually `criteria`:

| Type | Criteria | Answer |
|---|---|---|
| `choice` | 2 to 26 named options, each with a description or `null` | the most probable option, every option's probability, and `confidence` |
| `noul` | optional `false` and `true` descriptions | `noul`, the probability of `true` |
| `score` | 2 to 26 levels, lowest first | the probability-weighted level, per-level probabilities, a legend and `confidence` |

`confidence` is `1 - H(p) / ln(N)`, the concentration of the distribution. The
API reference says it is not calibrated correctness, and the code agrees: it
is computed from the same probabilities and nothing else
(`decision/systemone.go`, ollama/ollama `1abe35e`).

## From question to probabilities

Ollama compiles each question into its own prompt, then reads one next-token
distribution per prompt. Nothing is generated.

1. Every allowed answer gets a one-letter code, `A` upwards. Twenty-six letters
   is where the limit of 26 options comes from.
2. One JSON document holds the whole `state` and the schema of every question.
   Each question's prompt is that document followed by
   `Requested field: "<name>"`, wrapped in the model's chat template with
   thinking disabled and Ollama's system prompt.
3. The prompt is scored through llama-server's ordinary `/completion`
   endpoint with `n_predict` 1, `logit_bias` +100 on every code token,
   `top_k` equal to the number of codes, `n_probs` equal to the number of
   codes and `post_sampling_probs` on. A shared bias lifts the codes above
   every other token without changing their relative logits, so the returned
   probabilities are the softmax over the codes' own logits. If another token
   still outranks a code, the request is retried with +1000
   (`llm/llama_server_score.go`).
4. Ollama takes the logarithm of those probabilities, applies a softmax again
   and builds the answer (`decision/systemone.go`). No temperature is applied.

Ollama 0.35.0 ships its own llama-server binary (`lib/ollama/llama-server` in
the release tarball) built from llama.cpp tag `b11232` plus a compatibility
patch (`LLAMA_CPP_VERSION` and `llama/compat/` in the Ollama source). System
One is therefore llama.cpp scoring, driven by Go. [`probes/systemone.py`](../probes/systemone.py)
ports steps 1 to 4 to Python over any llama-server, and
[the CPU results](02-cpu-results.md) show that it reproduces Ollama to 1.30e-08.

### The primer

Qwen3.5, the base of all three models, is a hybrid: in each group of four
layers, three are recurrent (Gated DeltaNet) and one is full attention
(`qwen35.full_attention_interval = 4` in every GGUF, `results/registry.log`).
A KV cache can be trimmed back to a shared prefix; a recurrent state cannot.
llama-server can only resume from a saved checkpoint, and it saves those a few
tokens before the end of each prompt, which is past the point where the next
question's prompt differs.

Ollama works round this by sending a primer first: the shared tokens plus four
more, with `n_predict` 0, which makes llama-server checkpoint exactly at the
end of the shared part. Each question then evaluates about a dozen tokens of
its own. Ollama's comment says priming "only changes speed" and that scores
stay the same. On this page's evidence it changes both
([CPU results](02-cpu-results.md)).

## Three prompt formats

The weights were trained on particular prompt text, and Ollama sends the same
format to every decision model. `systemone.py` implements three formats, each
meant to be byte-identical to its source, and `probes/prompt_parity.py` checks
two of them against the source code itself (`results/prompt-parity.log`).

| Format | Source | Shape |
|---|---|---|
| `ollama` | `decision/systemone.go` | one JSON document with every question, Go's compact encoding, `<`, `>` and `&` escaped, a description always present |
| `nimble` | `nimble/scoring/parallel_schema.py` (bespokelabsai/nimble `62076b4`) | the same structure with Python's default separators (`", "`, `": "`), only `<` and `>` escaped, a description only where one was given |
| `tev1` | `examples/decide.py`, `build_dataset.py` (togethercomputer/tev1 `1dde778`) | one JSON object per question, `{"state", "question", "options": [{"label", "key", "description"}]}` |

[MEASURED] `systemone.py`'s `ollama` format matches Ollama's own
`decision.Compile` on all four fixtures in `probes/fixtures/prompt-edge-cases.json`,
which cover a plain string state, an object state with all three question
types, HTML-significant characters with U+2028 and U+2029, and structured
instructions. Its `nimble` format matches Bespoke's `prepare_prompts` on the
two fixtures Bespoke's code accepts; it refuses the other two (a `noul` with no
criteria, and an object as instructions). On the plain-string fixture the
Ollama and Nimble prompts are 372 and 397 characters, differing first at the
space after `"context":`.

The system prompts differ too. Ollama's registry layer for Nimble is Bespoke's
prompt with a newline added at each end; for Tev1 it is Together's prompt
re-wrapped onto three lines with a newline at each end (`results/registry.log`
against the constants in each project). Ollama passes the layer through
unchanged, which the port's agreement with Ollama confirms.

The `tev1` format has no Go or Python reference to diff against, because
Together's code sends one question through an OpenAI-style chat request. It
was written from their `payload()` function. Two choices in it are this
repository's: a `noul` becomes the options `no` then `yes`, following Tev1's
BoolQ encoding, and a missing description falls back to the key.

Open: whether the format matters for accuracy. It is the first experiment in
[the GPU handoff](../HANDOFF.md).

## What Ollama ships

Read from the registry by `probes/registry.py`, which streams each GGUF only
as far as the end of its tensor table (`results/registry.log`).

| Tag | GGUF bytes | Parameters | Quantisation | Blocks | Embedding | Ollama `num_ctx` | GGUF `general.name` |
|---|---:|---:|---|---:|---:|---:|---|
| `nimble` | 9,527,501,312 | 8,953,803,264 | Q8_0, norms F32 | 32 | 4096 | 8194 | Bespoke Nimble 9B Merged Current |
| `tev1` | 4,482,403,072 | 4,205,751,296 | Q8_0, norms F32 | 32 | 2560 | 2050 | Tev1 4B Experimental |
| `tev1:0.8b` | 811,843,424 | 752,393,024 | Q8_0, norms F32 | 24 | 1024 | 2050 | Tev1 0.8B Experimental |

Each `num_ctx` is a trained limit (8,192 and 2,048 tokens) plus the two
positions scoring needs. "Merged Current" suggests the Nimble checkpoint is
the one Bespoke's repository calls the latest, published 24 September, which
it says runs at temperature 1.0 with no fitted temperature [CLAIM]. Bespoke
fitted T=2.179 for an earlier revision [CLAIM]; Ollama applies no temperature
to any model, and nothing here checks which revision Ollama merged.

All three carry the chat template in the GGUF and Ollama ships no template
layer, so the template llama-server applies is the one the weights came with.
The Nimble and Tev1 templates differ (SHA-256 prefixes `a4aee8afcf2e0711` and
`2424952d53b6b63c`).

## Licensing

Each artifact in the runtime path, from `results/licences.log` and
`results/registry.log`:

| Artifact | Licence |
|---|---|
| Nimble weights | Apache-2.0: the Bespoke adapter and the Qwen3.5-9B base are both tagged apache-2.0, and Ollama's licence layer says the same |
| Nimble-V3, published 2026-09-30 | CC BY-NC 4.0. Not the file Ollama ships, and not usable commercially |
| Tev1 4B and 0.8B weights | Unsettled. Neither Hugging Face repo has a licence tag, and both cards say the licence for the fine-tuned weights "is being finalized". Ollama's registry ships Apache-2.0 text and an MIT licence for "open-jev contributors" alongside the Tev1 GGUFs. Which one governs is not established here |
| Qwen3.5 bases | Apache-2.0 |
| Ollama | MIT |
| llama.cpp | MIT |
| Bespoke's benchmark data | each subset under its upstream licence, listed in their `docs/PUBLIC_BENCHMARKS.md`; none of it is committed here |

---

[Contents](../README.md#contents) | Next: [CPU results](02-cpu-results.md)
