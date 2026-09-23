---
name: adding-a-model
description: Use when starting a new model investigation in model-playground — any modality: a text LLM behind vLLM or llama.cpp, an ASR or TTS checkpoint, a speech-to-speech model, a NIM container, an offline transformers checkpoint, or the same model moved between the cloud GPU node, the local workstation and the LAN GPU box, or between CUDA and ROCm. Covers the classification questions to answer BEFORE writing code, what a new model note and its probes must contain, licensing triage across weights/code/container, VRAM and precision budgeting, fixture requirements, the checklist for a first honest run, and — for speech models only — the protocol-adapter contract and the refactor from one adapter to many. Start here rather than in the code.
---

# Adding a model

Start here, not in the code. Most of the cost of a model investigation is
decided before the first line — by which questions you answered wrong.

"Adding a model" means adding a directory under `models/`: a note, the probes
that produced its numbers, and their committed output. Most models need no
harness at all — a text LLM or an ASR checkpoint is driven by its own probe.
The standard the note is held to is in `models/README.md`.

## Current reality

Three models are written up; **one** of them is driven by the speech harness.
That one has **one adapter**: `src/playground/protocol.py`, a single module
hardcoded to the OpenAI-Realtime dialect that NVIDIA's NIM container speaks.
There is no registry, no dispatch, no `Backend` base class.

That is correct for one speech model and wrong for two. This skill describes
the classification work that is always required, and — if the new model turns
out to be a second *speech* model — the refactor it should trigger.

## Step 1 — classify before coding

Answer all five. They determine what the investigation costs, and how much
of the existing speech harness (if any) survives.

### 1. What modality shape?

| Shape | Example | Harness fit |
|---|---|---|
| **Full-duplex speech↔speech** | VoiceChat 11B | Direct fit. This is what it was built for. |
| **Half-duplex / turn-based speech** | most voice assistants | Fits, but barge-in machinery is inert — say so rather than reporting zeros. |
| **Cascade** (ASR→LLM→TTS) | a pipeline you assemble | Harness can drive it, but you own the seams; latency attribution is per-stage. |
| **Text-only LLM** | K2-Horizon-32B | In scope for a note, **not** for the harness — audio pacing, channels and barge-in are all dead weight. Drive it from `models/<slug>/probes/` instead. |
| **ASR-only / TTS-only** | Parakeet Redux, a vocoder | In scope for a note, not for the harness. Drive it directly from a probe. |

### 2. What serving stack?

| Stack | Transport | Notes |
|---|---|---|
| **NIM container** | WebSocket / HTTP | Prepackaged, opinionated, fastest to stand up. Licensing caveat below. |
| **vLLM** | OpenAI-compatible HTTP | Great for text; streaming *audio* needs a custom server layer. |
| **llama.cpp** | HTTP (`llama-server`) | GGUF only. CPU or Vulkan/ROCm. See `local-inference` skill in the flake. |
| **Triton** (self-built) | gRPC / HTTP | What NIM wraps. Buildable from a HF checkpoint. |
| **transformers / NeMo offline** | in-process | No streaming. Ground-truth cross-checks and fine-tuning only. |

### 3. Streaming or request/response?

This is the sharpest fork.

- **Streaming**: `DuplexSession` applies. You need an adapter emitting the same
  event vocabulary (below).
- **Request/response**: `DuplexSession` does **not** apply — there is no
  concurrent receive loop to protect, and forcing it through buys nothing.
  Write a simpler driver and reuse `tools.py`, `audio.py` and the scenario
  format only.

Do not bend a request/response backend into the duplex session to "keep things
uniform." The duplex machinery exists to protect a property that backend
doesn't have.

### 4. Which host, and which compute stack?

See the `model-hosts` skill for the full matrix. The short version:

| Host | GPU | Stack | Good for |
|---|---|---|---|
| **cloud-gpu** (rented VM) | 2× H100 80GB, NVLink | **CUDA** | anything large; the only place NIM/TensorRT works. Billed hourly. |
| **workstation** (this desktop) | RDNA4 Radeon (gfx1201) | **ROCm / Vulkan** | llama.cpp, small models, CPU inference; no CUDA ever |
| **lan-gpu** | 2× RTX 3090 | **CUDA** | mid-size models, always-on, no marginal cost |
| **workstation CPU** | — | — | tiny models, tokenizer work, smoke tests, ternary ASR |

**ROCm is not a drop-in for CUDA.** A CUDA-only container (every NIM image, most
TensorRT artifacts) simply will not run on the workstation. Check this *first* — it
eliminates hosts before you spend time on anything else.

### 5. What is the licensing shape?

Enumerate **every artifact in the runtime path separately**. "The model is open"
is not a statement about your deployment. For VoiceChat the three pieces carried
three different licenses, and the encumbered one was the piece nobody thinks of
as "the model":

| Artifact | Ask |
|---|---|
| Weights | commercial use? output restrictions? field-of-use clause? |
| Inference code | usually Apache-2.0/MIT — this is often the escape hatch |
| Serving container | **the usual trap.** Dev/test free, production licensed? |
| Compiled engines | TensorRT output may inherit container terms |

Record the answer in the model's note file (below) with a link to the actual
license text, not a summary of marketing.

## Step 2 — the adapter contract

Whatever backend you add, the session loop needs these capabilities. Today
`protocol.py` provides them as module-level functions; a second model should
turn that into a class or module implementing the same surface.

**Outbound**
- `session_update(instructions, tools) -> event` — configure; carries the agent
  spec and tool schemas
- `audio_append(pcm: bytes) -> event` — one chunk of caller audio
- `function_call_output(call_id, output: str) -> event` — return a tool result
- `session_close() -> event`

**Inbound** — classify a raw message into:
- agent audio bytes
- agent transcript (delta / final)
- caller transcript (delta / final)
- **speech-started** (the barge-in signal)
- **tool call** → `ToolCall(call_id, name, arguments)`
- turn boundary, session end, error

**Constants**
- wire sample rate, chunk duration, chunk bytes, sample format

If a backend cannot supply *speech-started*, barge-in evaluation is impossible
with it — state that plainly rather than emitting empty `barge_in` sections.

## Step 3 — the refactor, when N=2

Do this **when adding the second model**, not before, and not later.

```
src/playground/
├── protocols/
│   ├── __init__.py          # registry: name -> adapter
│   ├── base.py              # the contract above, as a Protocol/ABC
│   ├── nim_realtime.py      # today's protocol.py, moved verbatim first
│   └── <new>.py
```

Sequence that keeps it honest:

1. Move `protocol.py` → `protocols/nim_realtime.py` **unchanged**. Commit.
2. Re-run `tool_call.wav`; confirm the event counts and tool durations match the
   prior run. A pure move must not change behaviour.
3. *Then* extract `base.py` from what the second backend actually needs — not
   from what you imagine a general backend needs.
4. Add `--protocol` (default the incumbent) and a registry lookup.

Extracting the abstraction from one example produces an abstraction shaped like
that one example. Two real implementations is the minimum.

## Step 4 — host and deployment

Capture, in the model's note file:

- exact pull commands (registry, auth, size)
- exact launch command with ports and mounts
- **measured** VRAM at rest, and the per-process breakdown
- cold-start time and when the port actually opens
- warmup behaviour and whether early turns carry a JIT spike

### VRAM budgeting

Measure, never assume. Two things dominate and both surprise people:

- **Precision.** VoiceChat ships **F32** — 4 bytes/param, so 11 B = 44 GB. At
  bf16 it would be 22 GB. Check `dtype` in the safetensors header before
  believing any hardware requirement.
- **Preallocated KV cache.** vLLM reserves aggressively; ~28 GB of VoiceChat's
  73 GB is cache, not weights. Often tunable via `gpu_memory_utilization`.

A stated "80 GB minimum" may be a packaging choice rather than a property of the
model. Read the header:

```python
import json, struct
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
# per-tensor: dtype, shape -> params and bytes/param
```

## Step 5 — fixtures

See `audio-fixtures`. The non-negotiable: **two-party recordings with one
speaker per channel**. Check every file's channel count before trusting it —
2 784 files from the obvious benchmark dataset turned out to be mono stimulus
and were useless.

Check whether the model's own repo ships demo audio. VoiceChat's did, it was the
best material available, and nothing in the model card mentioned it.

## Step 6 — the first honest run

Before reporting anything, see `evaluating`. Minimum bar:

- [ ] Round trip proven — a value produced by *your* handler is spoken back
- [ ] Injected tool latency measured against configured, and they match
- [ ] First turn discarded or warmup controlled for
- [ ] `--speed 1.0`
- [ ] Any inert machinery named explicitly (e.g. "barge-in policy never fired")

That last one is the easiest to skip and the most damaging. A harness that runs
clean while a whole subsystem never executed reads as a passing result.

## Per-model note files

One directory per model at `models/<slug>/`, with `README.md` covering: licensing (all
artifacts), architecture as measured from weights, host and launch, protocol
quirks where it deviates from its own docs, fixtures used, and results with
their caveats.

`models/README.md` states the full standard. `models/nemotron-voicechat-11b/`
is the reference for depth and for separating *measured* from *assumed*, and
for how a long note is split into `notes/` articles with the README as a hub.

The note is not finished until it has been through the humanizer and
`check_rewrite.py`. Load `.claude/skills/writing-notes/SKILL.md` before
writing it.
