# Architecture and distributions

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


We ended up with ~86 GB of model on disk in two formats. This note explains
why, and proves that they contain the same weights.

## Two distributions, one model

### What we downloaded

| | HF checkpoint | NGC NIM artifact |
|---|---|---|
| Source | `huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` | `ngc registry model download-version nim/nvidia/nemotron-labs-voicechat:1.0.0` |
| Size on disk | 42 G (`model.safetensors` = 44.4 GB) | 43.69 GB, 23 files |
| Shape | one fused blob | Triton model repository |
| License | OpenMDW-1.1 | NVIDIA terms |
| Serves | transformers / NeMo, offline batch | the NIM container, realtime WebSocket |

### Why both

The HF pull started on instruction before the serving path was decided. Once the NIM container was chosen, it needed the Triton-shaped
artifact, so a second download followed.

This later turned out to be avoidable: `generate-model-repo.md` shows the
container can convert the HF checkpoint into a Triton repo itself
(`NEMO_CHECKPOINT_PATH=/checkpoint`, ~15 minutes). Either download alone would
have been sufficient.

### Proof they are the same weights

Read the safetensors headers (a JSON prefix, so there is no need to load
tensors) and sum parameters per module.

#### HF checkpoint: 11.10 B params, 1632 tensors

```
stt_model.llm                      7.714 B   339 tensors
tts_model.tts_model                0.797 B   418
stt_model.perception               0.614 B   640
stt_model.embed_tokens             0.587 B     1
stt_model.function_head            0.587 B     1
stt_model.lm_head                  0.587 B     1
tts_model.audio_codec              0.200 B   214
stt_model.rnnt_decoder             0.007 B     9
stt_model.rnnt_joint               0.002 B     6
```

#### NIM artifact: 11.719 B params

```
nano-v2-vllm/model.safetensors     9.475 B   342 tensors
eartts_vllm/model.safetensors      0.834 B   453
perception.safetensors             0.614 B   640
embeddings.safetensors             0.587 B     1
codec.safetensors                  0.200 B   214
rnnt-asr.safetensors               0.009 B    15
```

#### The arithmetic

```
HF  stt_model total       = 10.0981 B
NIM nano + perception + rnnt = 10.0981 B      delta = 0 params
```

The delta is exactly zero, and it holds component by component:

- `perception.safetensors` → 0.614 B / **640** tensors
  ≡ HF `stt_model.perception` → 0.614 B / **640** tensors
- `codec.safetensors` → 0.200 B / **214**
  ≡ HF `tts_model.audio_codec` → 0.200 B / **214**
- `nano-v2-vllm` 9.475 B = `stt_model.llm` (7.714) + three 0.587 B heads

Both are F32 throughout. Neither is quantized or distilled, and it is the same
checkpoint in both.

### Why NIM is 0.6 B larger

`embeddings.safetensors` is a duplicate: one tensor, `embed_tokens.weight`,
shape `[131072, 4480]`, 0.587 B. The same table already sits inside
`nano-v2-vllm`. It is broken out so the TTS stage can read it without loading
the LLM. That redundancy plus ~37 M of TTS-side adapter accounts for the entire
gap.

### Where the two differ

Only the placement of the seams differs. HF gives one fused blob you load into transformers. NIM cuts the same tensors
along pipeline boundaries so each stage can be fed independently while the
others are mid-flight. That decomposition is what makes streaming possible, and
you cannot bolt it onto the fused checkpoint without implementing the
streaming server yourself.

### Which to use

For realtime evaluation, use the NGC artifact, because it is already in Triton
form and skips the 15-minute conversion. It is the identical model, so there is
no quality reason to prefer it.

Keep the HF checkpoint for:
- Offline / batch inference (`offline_voicechat_fc_infer.py`)
- NeMo fine-tuning (starts from HF weights, not the Triton repo)
- Ground truth: if the serving layer misbehaves, running the same input
  offline separates "model does this" from "NIM does this"
- The permissive production path

It also shipped the demo audio (see [the harness and its fixtures](06-harness.md)), which turned out to be
the best test material available.

## Architecture, measured from the weights

Most of what follows came from reading safetensors headers and the container's
own logs rather than the model card.

### The published description

A hybrid Mamba/Transformer: a Fast Conformer speech encoder in front, a
Nemotron Nano v2 LLM backbone, and an NVIDIA TTS decoder behind, plus a
separate output channel that emits tool-call scripts while audio continues.

- 11 B parameters
- Audio in 16 kHz → audio out 22.05 kHz (the WebSocket wire is 24 kHz both
  directions; the server resamples internally)
- ~450 ms smooth turn-taking, ~480 ms on user interruption
- #2 among open full-duplex models on VoiceBench

### What the tensor inventory confirms

The NIM artifact decomposes into exactly the advertised stages:

| File | Params | Role |
|---|---|---|
| `perception.safetensors` | 0.614 B | Fast Conformer audio encoder |
| `rnnt-asr.safetensors` | 0.009 B | Streaming RNNT ASR |
| `nano-v2-vllm/` | 9.475 B | Nemotron Nano v2 LLM backbone |
| `eartts_vllm/` | 0.834 B | TTS decoder |
| `codec.safetensors` | 0.200 B | Audio codec |
| `embeddings.safetensors` | 0.587 B | Shared embedding table (duplicate) |

The tiny RNNT ASR (9 M params) drives `input_audio_buffer.speech_started`. It
is cheap enough to run continuously on the input stream, and that is what makes
barge-in detection viable at all.

### A dedicated tool-call head

From the HF checkpoint's module list:

```
stt_model.lm_head          0.587 B    1 tensor
stt_model.function_head    0.587 B    1 tensor     <---
stt_model.embed_tokens     0.587 B    1 tensor
```

There is a `function_head` the same size as the language-modelling head,
sitting parallel to it. All three are `[131072, 4480]`, which makes them
vocabulary-sized projections.

This is the "separate output channel that emits tool-call scripts while the
audio conversation keeps going." It is a real architectural feature, not prompt
engineering over the text head.

#### Why it matters

In a cascade, or in a single-head model, a tool call and speech generation
compete for the same output channel. That competition is the usual reason
assistants stutter, pause, or go silent when calling a tool: the head is busy
emitting JSON instead of words.

A dedicated `function_head` means tool emission and speech emission are
*structurally* separable. Whether that turns into graceful behaviour under a
slow tool is the question the harness was built to answer.

#### Evidence from a real run

From `logs/tool_call.jsonl`:

```
11.24s   tool.call    generate_random_number({"min": 1, "max": 50})
12.36s   caller.said  "can you generate a random number between one and fifty?"
```

The tool call fires 1.1 seconds before the caller's transcript finalizes. The
model commits to a tool call without waiting for a complete transcript, and the
timeline shows the duplex architecture and the separate head at work.

### The prompt-side tool protocol

The container's own warmup log shows the system prompt it constructs:

```
You can use the following tools to assist the user if required:
<AVAILABLE_TOOLS>[ ...tool JSON... ]</AVAILABLE_TOOLS>

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>

Based on the tool responses, you can call additional tools if needed, correct
tool calls if any errors are found, or just respond to the user.
```

So the `function_head` emits into a `<TOOLCALL>` span, and the WebSocket layer
surfaces that as `response.function_call_arguments.done`.

### Resource profile

- 73.5 GB on a single GPU when loaded (the model card says ~66 GB; measured
  higher)
- Fits one H100 80GB with ~8 GB headroom. The prerequisites state an 80 GB
  minimum, so it sits close to the line, which is why the deploy command pins
  `--gpus '"device=0"'` rather than splitting
- No tensor parallelism, so NVLink is irrelevant here
- ~5 minutes to load, plus warmup

---

Previous: [Licensing](02-licensing.md) | [Contents](../README.md#contents) | Next: [Setting it up](04-deployment.md)
