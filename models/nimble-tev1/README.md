# Nimble 9B and Tev1: System One decision models

Three small open models that answer typed questions about a piece of text in
one forward pass each, with no generation. Investigated from 2026-09-30,
starting the day after Ollama 0.35.0 added a `/v1/systemone` endpoint for them.

> Status: the CPU half. Everything that did not need a GPU has been measured
> and is written up below. Accuracy, calibration and GPU latency are the next
> step, and [HANDOFF.md](HANDOFF.md) is the plan for it.

- Ollama announcement: <https://ollama.com/blog/ollama-now-supports-jev-style-decision-models>
- API reference: <https://docs.ollama.com/api/systemone>
- Nimble: <https://github.com/bespokelabsai/nimble>, <https://huggingface.co/bespokelabs/Bespoke-Nimble-9B>
- Tev1: <https://github.com/togethercomputer/tev1>, <https://huggingface.co/togethercomputer/Tev1-4B-experimental>

## What these models are

TypeSafe AI sells a hosted model called Jev that returns typed decisions: pick
one of these options, is this true, where on this scale does it fall. Each
answer comes with a probability for every allowed option. Two open models
built to do the same job have since been published, and Ollama now serves both
locally:

| Ollama tag | Model | From | Base | Size on disk |
|---|---|---|---|---:|
| `nimble` | Bespoke-Nimble-9B | Bespoke Labs | Qwen3.5-9B | 9.5 GB, Q8_0 |
| `tev1` | Tev1-4B-experimental | Together AI | Qwen3.5-4B | 4.5 GB, Q8_0 |
| `tev1:0.8b` | Tev1-0.8B-experimental | Together AI | Qwen3.5-0.8B | 812 MB, Q8_0 |

A request is some text (`state`) and up to 64 named questions of three kinds:
`choice`, `noul` (yes or no) and `score` (an ordered rubric). Each allowed
answer is given a one-letter code, and the probabilities are the model's
next-token probabilities for those letters and no others. Nothing is
generated, so an answer cannot be malformed, and a question costs one prompt
evaluation.

Why look at them: a component that makes a cheap, typed decision is what a
local pipeline of small models needs between its stages. Route a request, gate
a tool call, decide whether a speaker has finished. A number that says how sure
the model is only helps if it means the same thing from one run to the next.
This investigation is mostly about whether it does.

## What was found so far

All on a 2016 Xeon CPU with `tev1:0.8b`; details, hardware and caveats are in
[the CPU results](notes/02-cpu-results.md).

- System One in Ollama is llama.cpp. Ollama drives its bundled llama-server's
  ordinary `/completion` endpoint with a `logit_bias` trick, and a clean
  llama.cpp build of the tag Ollama pins (`b11232`) reproduces Ollama's
  probabilities to 1.30e-08 through the Python port in
  [`probes/systemone.py`](probes/systemone.py). Ollama is not needed to run
  these models this way ([how it works](notes/01-how-it-works.md)).
- Ollama sends every decision model the same prompt format, and it is exactly
  neither model's own. Its prompt for Nimble differs from Bespoke's scorer in
  JSON spacing and escaping, and Tev1 was trained on a different structure
  altogether ([how it works](notes/01-how-it-works.md#three-prompt-formats)).
- The same request can return different probabilities for four separate
  reasons: the llama.cpp version (up to 3.86 points between `b9190` and
  `b11232`, same answers), Ollama's prefix primer (up to 1.85 points, which its
  own code comment says cannot happen), which other questions are in the
  request (up to 1.33 points), and what the server evaluated just before
  (0.0194 on one of eight benchmark records). Each is deterministic and each is
  measured.
- An older llama.cpp (`b9190`) crashes the second time it sees an identical,
  fully cached prompt with these models. `b11232` does not.
- On this CPU a cold 8-question request takes about 21 s without the primer and
  about 4 s with it; a cached question takes 141 to 161 ms.
- Tev1's weights have no settled licence. Both Hugging Face cards say it "is
  being finalized", while Ollama ships an Apache-2.0 text beside them. Nimble is
  Apache-2.0 ([licensing](notes/01-how-it-works.md#licensing)).

## Using it without Ollama

Start any llama-server on one of the GGUFs (`probes/registry.py --download DIR`
fetches Ollama's and verifies them), then:

```bash
python models/nimble-tev1/probes/systemone.py models/nimble-tev1/probes/fixtures/ticket-3q.json \
    --server http://127.0.0.1:8080 --system ollama-tev1
```

`--format` selects the prompt format (`ollama`, `nimble` or `tev1`) and
`--system` the system prompt. `--serve PORT` exposes a minimal `/v1/systemone`
so TypeSafe's SDK or Bespoke's runner can point at a bare llama-server. It is
standard-library Python and handles one request at a time.

## Contents

| Article | What is in it |
|---|---|
| [01, how it works](notes/01-how-it-works.md) | the scoring mechanism from source, the primer, the three prompt formats, what Ollama ships, licensing |
| [02, CPU results](notes/02-cpu-results.md) | fidelity across builds, the crash, priming, coupling, cache history, the benchmark data |
| [HANDOFF.md](HANDOFF.md) | the GPU plan: accuracy on Bespoke's 13 public subsets, prompt formats, quantisation, calibration, latency |

## Probes

| Probe | What it produces |
|---|---|
| [`systemone.py`](probes/systemone.py) | the port: compiler, scorer, answer maths, three prompt formats, `--serve` |
| [`prompt_parity.py`](probes/prompt_parity.py) | byte comparison with Ollama's Go compiler (via [`parity/main.go`](probes/parity/main.go)) and Bespoke's Python, `results/prompt-parity.log` |
| [`registry.py`](probes/registry.py) | Ollama's manifests, system prompts, parameters, licences and GGUF inventories, `results/registry.log` |
| [`licences.py`](probes/licences.py) | the Hugging Face licence tags and card wording, `results/licences.log` |
| [`fidelity.py`](probes/fidelity.py) | the port and other builds against Ollama, `results/fidelity.log` |
| [`identical_prompt.py`](probes/identical_prompt.py) | the repeated-prompt crash, per build, `results/identical-prompt.log` |
| [`priming.py`](probes/priming.py) | cold, primed and warm timing and probabilities, `results/priming.log` |
| [`schema_coupling.py`](probes/schema_coupling.py) | one question's probabilities as others are added, `results/schema-coupling.log` |
| [`fetch_public_data.py`](probes/fetch_public_data.py) | Bespoke's 13 benchmark subsets rebuilt and checksummed, `results/public-data.log` |
| [`public_suite.py`](probes/public_suite.py) | Bespoke's benchmark runner against Ollama or the port, `results/public-suite-smoke.log` |
| [`build_llama_cpp.sh`](probes/build_llama_cpp.sh) | llama-server at a pinned tag for CUDA, Vulkan, HIP or CPU |
| [`hostinfo.py`](probes/hostinfo.py) | the host line every log starts with |

The fixtures are in [`probes/fixtures/`](probes/fixtures/). Every probe is
standard-library Python except `fetch_public_data.py`, which needs `pyarrow`,
`transformers` and `jinja2`, and `prompt_parity.py`, which needs a Go
toolchain and checkouts of Ollama and Nimble at the pinned commits.

## Repeating it

The order that made the findings appear:

1. Read Ollama's handler before the docs. The API page describes a decision
   model; `server/routes.go` and `llm/llama_server_score.go` show a
   llama-server client, which is what made a port possible at all.
2. Prove the port against Ollama's own runner process before anything else,
   so a later difference can be blamed on a build and not on the port.
3. Compare builds on the same GGUF, then a clean build of Ollama's pinned tag,
   which separates the llama.cpp version from Ollama's patch.
4. Time with the prompt cache disabled (`--cache-ram 0`). Without it every
   "cold" run after the first is warm, and the first version of the priming
   probe reported exactly that.
5. Diff prompt formats against their source code, not against documentation.

## Open questions

- Accuracy and calibration of all three models, per prompt format, on the 13
  human-labelled subsets. The plan is in [HANDOFF.md](HANDOFF.md).
- Whether the build, primer and coupling effects are larger on a GPU, for
  Nimble, or for Tev1 4B.
- Whether probabilities from a CUDA or Vulkan build match the CPU's.
- Which llama.cpp change between `b9190` and `b11232` fixed the repeated-prompt
  crash, and which changed the probabilities.
- Which Nimble revision Ollama merged, and what licence Tev1's weights end up
  under.
