# NVIDIA NemotronLabs VoiceChat 11B

A full-duplex speech-to-speech model. Audio goes in and audio comes out, with no
ASR, LLM and TTS pipeline in between. It has 11.1 B parameters and is served by
NVIDIA's NIM container over the OpenAI Realtime WebSocket dialect.

This is the model the repository was first built around, and the only one
driven by the harness in `src/playground/`.

- Weights: <https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B>
- Inference code: <https://github.com/NVIDIA-NeMo/Speech> (`nemo/collections/speechlm2/`)
- Container: `nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest`

Everything here was measured on a rented 2× H100 80GB HBM3 (NVLink) cloud node,
using one GPU. No number was copied from a model card.

## What it is, and what problem it solves

The usual way to build a voice agent is a cascade: an ASR model transcribes the
caller, an LLM decides what to say, and a TTS model speaks it. It works, it is
easy to reason about, and each stage can be swapped on its own, but some of its
problems can't be tuned away. Each stage waits for the one before it, so
the latency floor is the sum of all three. ASR throws away how something was
said and TTS invents new intonation from text, so hesitation and emphasis are
lost on the way through. And the pipeline has no natural way to represent a
caller talking over the agent, so interruption is handled by separate
voice-activity detection and a kill switch.

A speech-to-speech model replaces the cascade with one model that reads and
writes audio directly. That is what makes full duplex possible: it can listen
and speak at once, because there is no stage boundary where one has to stop for
the other.

This model also carries a dedicated tool-calling output head alongside the
language-modelling head, so tool calls don't compete with speech for the same
output. That head is why it was worth taking apart
([architecture](notes/03-architecture.md)).

The repository was built to answer one question about it: does full-duplex
tool calling hold up when tools are slow and the caller interrupts? It does
fire as designed, but the more useful result was what the model does while a
tool is outstanding ([barge-in results](notes/07-results-barge-in.md)).

## What was found

- Barge-in handling fires once injected tool latency passes a threshold
  between 8 s and 10 s. The threshold was predicted from an older log before
  any run, and it held. Above it the model also starts calling tools nobody
  asked for (`get_news_headlines`, for France, twice a run) and passes 100 to
  `convert_currency` when the caller said twenty-five. The model card says
  tool arguments must be values the user spoke.
  ([barge-in results](notes/07-results-barge-in.md))
- Turn-taking latency comes in whole multiples of the server's 160 ms
  inference budget. All 36 measured turns land within ±13 ms of 3, 4 or 5
  chunks, so the fastest possible reply is 480 ms, against a claimed ~450 ms.
  ([latency results](notes/08-results-latency.md))
- The ~73 GB footprint is set by two environment variables, not by the
  weights. Lowering them took the card from 73,308 MiB to 44,716 MiB and the
  model still served full sessions. The cost showed up as occasional 5-chunk
  replies (2 of 18 turns), which the stock setting never produced.
  ([requirements](notes/01-requirements.md), [latency results](notes/08-results-latency.md))
- The container's LICENSE file is 538 bytes of links to three agreements hosted
  elsewhere. It names a different licence for the model than the Hugging Face
  release does. ([licensing](notes/02-licensing.md))
- The realtime socket runs at 24 kHz in both directions, and the per-tool
  `on_hold_message` field is just text copied into the system prompt.
  ([protocol](notes/05-protocol.md))

## Contents

| Article | What it covers |
|---|---|
| [Requirements and runtime](notes/01-requirements.md) | Hardware, cold start, and where the 80 GB goes |
| [Licensing](notes/02-licensing.md) | Weights, inference code and container, each licensed separately |
| [Architecture and distributions](notes/03-architecture.md) | The tensor inventory, the tool-call head, and proof the HF and NIM downloads are the same weights |
| [Setting it up](notes/04-deployment.md) | Getting the NIM container running, including every step that went wrong |
| [The protocol](notes/05-protocol.md) | The Realtime event vocabulary and where the server disagrees with its docs |
| [The harness and its fixtures](notes/06-harness.md) | How the client drives a session, why each design choice was forced, and the test audio |
| [Results: barge-in under slow tools](notes/07-results-barge-in.md) | The latency sweep, the policy's behaviour, and the model's |
| [Results: latency, headroom, VRAM](notes/08-results-latency.md) | Compute headroom, quantised turn-taking, and the smaller-card test |
| [Failures worth recording](notes/09-failures.md) | Three results that looked clean while measuring nothing |
| [Traps, collected](notes/10-traps.md) | Audio, process, SSH, toolchain and measurement traps met along the way |
| [Probes and method](notes/11-method.md) | Which probe produced which number, and the order to repeat the work in |
| [Open questions](notes/12-open-questions.md) | What is answered, and what still needs node time |

The code behind every number is in [`probes/`](probes/), and its raw output is
in [`results/`](results/). The standard these notes are held to is in
[`models/README.md`](../README.md).
