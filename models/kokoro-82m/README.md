# Kokoro-82M

An 82M-parameter text-to-speech model based on StyleTTS 2 and iSTFTNet. Seeded for investigation 2026-09-27.

- Weights: <https://huggingface.co/hexgrad/Kokoro-82M> (Apache-2.0)
- ONNX release: <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX> (Apache-2.0)
- Upstream repo: <https://github.com/hexgrad/kokoro>
- Upstream architecture: StyleTTS 2 (<https://github.com/yl4579/StyleTTS2>) and iSTFTNet

## What the model is

Kokoro-82M is an open-weights speech synthesis model that takes phonemized or raw text and emits 24 kHz mono audio. hexgrad released it in January 2025.

It pairs a feed-forward acoustic model with an inverse Short-Time Fourier Transform (iSTFTNet) vocoder. Inference has no iterative diffusion sampling steps and no autoregressive loop over audio frames, which sets it apart from token-based speech models (such as AudioCraft or VibeVoice) and diffusion models (such as Grad-TTS or E2/F5-TTS).

Voices are external style embeddings, so a new voice does not need its own full set of weights. The model repository bundles 54 pre-extracted voice vectors across 8 language variants, and voices can be blended by linear interpolation between vectors.

### What it is good for

- Fast synthesis on commodity CPUs. Reported community throughput ranges from 40x to 80x real time on modern x86_64 desktop and server cores [CLAIM].
- A small footprint. The base FP32 checkpoint is roughly 328 MB on disk, FP16 is roughly 164 MB, and quantized INT8 ONNX exports are under 90 MB [CLAIM]. Resident memory during inference stays below 200 MB [CLAIM].
- Embedding in a media server. ONNX exports exist and `sherpa-onnx` provides native C++ and Rust bindings, so the model can run inside a native daemon without a Python runtime or PyTorch.
- Conversational turn-taking. With clause-level or punctuation-level text chunking, time-to-first-audio (TTFA) has been reported between 120 ms and 180 ms on CPU [CLAIM].
- Commercial use. The weights and the reference code are both Apache-2.0.

### What it does not do

- Clone an arbitrary voice from raw audio out of the box. A new voice needs a matching style embedding, extracted through the StyleTTS 2 reference encoder or by tuning adapters.
- Generate conversational filler. It speaks the text it is given; it has no internal language model, turn-taking logic or acoustic backchannel behavior.
- Output telephone sample rates. Output is fixed at 24 kHz, so downsampling to 8 kHz (G.711) or 16 kHz (G.722 / wideband Opus) happens downstream in the media pipeline.

## Licensing

| Artifact | License | Notes |
|---|---|---|
| Model weights (`hexgrad/Kokoro-82M`) | Apache-2.0 | Explicitly declared on the Hugging Face model card. |
| Reference code (`hexgrad/kokoro`) | Apache-2.0 | Upstream repository license. |
| ONNX exports (`onnx-community/Kokoro-82M-v1.0-ONNX`) | Apache-2.0 | Exported by Hugging Face ONNX community. |
| Sherpa-ONNX runtime bindings | Apache-2.0 | Next-gen Kaldi project. |
| Text frontend (`misaki` / `espeak-ng`) | MIT / LGPL-3.0 | `espeak-ng` carries LGPL-3.0. Check dynamic linking requirements when embedding in proprietary daemons. |

## Architecture

| Property | Value |
|---|---|
| Parameters | 82.2 M (82,230,000 approx) |
| Architecture family | StyleTTS 2 decoder + iSTFTNet vocoder |
| Native sample rate | 24,000 Hz (24 kHz), 16-bit mono PCM |
| Input representation | IPA phonemes (generated via `misaki` or `espeak-ng`) |
| Conditioning | 256-dimensional style vector per speaker |
| Shipped voices | 54 voices (American English, British English, Japanese, Mandarin, French, Spanish, Italian, Portuguese) |
| Quantization formats | FP32, FP16, ONNX INT8 / UINT8 |

## Host and launch options

### 1. Embedded C++ / Rust via sherpa-onnx

`sherpa-onnx` supports Kokoro-82M natively, without Python:

- C++ API: `OfflineTtsKokoroModelConfig`
- Rust crate: `sherpa-onnx` feature `kokoro_tts`
- Model files required: `model.onnx`, `voices.bin`, `tokens.txt`, `espeak-ng-data`

### 2. Standalone ONNX Runtime in Python
```bash
pip install kokoro-onnx soundfile
```
```python
import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v1.0.onnx", "voices.bin")
samples, sample_rate = kokoro.create(
    "Testing streaming voice generation for media servers.",
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)
sf.write("output.wav", samples, sample_rate)
```

### 3. PyTorch reference runtime
```bash
pip install kokoro soundfile
```

### 4. GStreamer media server integration

A real-time media pipeline can run the model in process through native ONNX, or in Python.

In-process native code uses ONNX Runtime (through `ort`) inside custom GStreamer elements, on CPU or GPU, with no Python runtime. Through `gst-python` (PyGObject), feature pipelines run as Python GStreamer elements on a tap runtime, which can run the reference PyTorch implementation or custom C-extension engines without converting the weights to ONNX.

In a real-time tap, audio leaves a normalized fanout tee (`S16LE, 16 kHz, mono`) into a downstream leaky queue (`max-size-time=500ms`, `leaky=downstream`) followed by an `errorignore` flow guard (`convert-to=ok`). A slow inference thread drops aged frames instead of stalling the conference, and pipeline errors do not propagate back to the participant tee.

Kokoro outputs 24 kHz audio. Before synthesized audio goes into a conference mixer or participant channel, it has to be resampled to the conference rate (such as 16 kHz mono) or a telephony codec rate (8 kHz / 16 kHz).

## Protocol and interface

- Input: Text string with target speaker style identifier (e.g., `af_heart`, `am_adam`, `bf_emma`).
- Chunking strategy: Kokoro processes complete phoneme sequences. For streaming LLM output, split the text on punctuation (`.`, `,`, `!`, `?`, `;`, `:`) before sending it to inference.
- Output: Float32 or INT16 PCM array at 24 kHz.

## Fixtures to probe

When evaluating Kokoro-82M for telephony and conversational media servers:

1. Short turn-taking phrases: "Yes, I can help with that." (Measure cold vs warm TTFA).
2. Numbers and currencies: "Account balance is $1,240.50 on March 14th, 2026." (Test text normalization frontend).
3. Technical jargon and acronyms: "SIP, RTP, WebRTC, Asterisk, and PBX." (Test phonemizer fallback on OOV terms).
4. Telephony resampling test: Audio downsampled from 24 kHz to 8 kHz (G.711 u-law) to verify intelligibility under PSTN bandwidth limits.
5. Multilingual prompt set: Parallel sentences in English, French, Spanish, and Japanese using matching language style vectors.

## Probe plan

Probes to write under `probes/`:

- `probes/latency_ttfa.py`: Measure time-to-first-audio across single-word, short-phrase, and multi-sentence inputs on CPU.
- `probes/throughput_rtf.py`: Measure real-time factor (RTF) across 1, 2, 4, and 8 CPU threads.
- `probes/concurrency.py`: Measure multi-tenant throughput (simulated concurrent calls) on CPU without GPU assistance.
- `probes/voice_blending.py`: Test linear interpolation between two style vectors and evaluate output stability.
- `probes/resampling_fidelity.py`: Measure PESQ/STOI degradation when downsampling native 24 kHz output to 8 kHz telephony audio.

## Open questions

1. What is the measured CPU throughput gap between PyTorch FP32 and ONNX Runtime INT8 on AVX-512 vs non-AVX-512 silicon?
2. How much time does phonemization in `misaki` / `espeak-ng` take compared with neural inference on short utterances?
3. Does style blending introduce audible artifacts or phase distortion through the iSTFTNet vocoder?
4. Does the LGPL-3.0 license of `espeak-ng` stay separate from proprietary host binaries when it is linked as a shared library?
