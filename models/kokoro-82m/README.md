# Kokoro-82M

An 82M-parameter text-to-speech model based on StyleTTS 2 and iSTFTNet. Seeded for investigation 2026-09-27.

- Weights: <https://huggingface.co/hexgrad/Kokoro-82M> (Apache-2.0)
- ONNX release: <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX> (Apache-2.0)
- Upstream repo: <https://github.com/hexgrad/kokoro>
- Upstream architecture: StyleTTS 2 (<https://github.com/yl4579/StyleTTS2>) and iSTFTNet

## What the model is

Kokoro-82M is an open-weights speech synthesis model that takes phonemized or raw text and emits 24 kHz mono audio. It was released in January 2025 by hexgrad.

Unlike autoregressive token-based speech models (such as AudioCraft or VibeVoice) and diffusion-based models (such as Grad-TTS or E2/F5-TTS), Kokoro uses a feed-forward acoustic model coupled with an inverse Short-Time Fourier Transform (iSTFTNet) vocoder. Inference requires no iterative diffusion sampling steps and no autoregressive loop over audio frames.

Voice characteristics are controlled via external style embeddings rather than separate full-model weights. The model repository bundles 54 pre-extracted voice vectors across 8 language variants, with blending supported via linear interpolation between vectors.

### What it is good for

- Fast synthesis on commodity CPUs. Reported community throughput ranges from 40x to 80x real time on modern x86_64 desktop and server cores [CLAIM].
- Low memory footprint. The base FP32 checkpoint is roughly 328 MB on disk, FP16 is roughly 164 MB, and quantized INT8 ONNX exports are under 90 MB [CLAIM]. Resident memory during inference stays below 200 MB.
- Embedded media server integration. Because ONNX exports exist and `sherpa-onnx` provides native C++ and Rust bindings, the model can be embedded directly into a native daemon without a Python runtime or PyTorch dependency.
- Conversational turn-taking. When paired with clause-level or punctuation-level text chunking, time-to-first-audio (TTFA) has been reported between 120 ms and 180 ms on CPU [CLAIM].
- Permissive commercial distribution. Both the weights and reference code are released under Apache-2.0.

### What it does not do

- Zero-shot arbitrary voice cloning from raw audio out of the box. Generating a new voice requires extracting a matching style embedding through the StyleTTS 2 reference encoder or tuning adapters.
- Dynamic conversational filler generation. It synthesizes text explicitly provided to it; it has no internal language model, turn-taking logic, or acoustic backchannel behavior.
- Telephony-native sampling. Output is fixed at 24 kHz. Downsampling to 8 kHz (G.711) or 16 kHz (G.722 / wideband Opus) must be performed downstream in the media pipeline.

## Licensing

Enumerate each artifact in the serving path:

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
`sherpa-onnx` contains native support for Kokoro-82M without Python dependencies:
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
Real-time media pipelines support both in-process native ONNX and Python execution paths:
- **In-process native via `ort`:** Uses ONNX Runtime in custom GStreamer elements, executing on CPU or GPU with zero Python runtime overhead.
- **Python via `gst-python` (PyGObject):** Feature pipelines can run directly as Python GStreamer elements on a tap runtime. This allows running the reference PyTorch implementation or custom C-extension engines without converting weights to ONNX.
- **Pipeline isolation:** In a real-time tap architecture, audio leaves a normalized fanout tee (`S16LE, 16 kHz, mono`) into a downstream leaky queue (`max-size-time=500ms`, `leaky=downstream`) followed by an `errorignore` flow guard (`convert-to=ok`). A slow inference thread drops aged frames rather than stalling the conference, and pipeline errors do not propagate back to the participant tee.
- **Sample rate handling:** Kokoro outputs 24 kHz audio. When feeding synthesized audio into a conference mixer or participant channel, it must be resampled to the conference standard (such as 16 kHz mono) or telephony codec rate (8 kHz / 16 kHz).

## Protocol and interface

- Input: Text string with target speaker style identifier (e.g., `af_heart`, `am_adam`, `bf_emma`).
- Chunking strategy: Kokoro processes complete phoneme sequences. For streaming LLM output, text should be segmented on punctuation boundaries (`.`, `,`, `!`, `?`, `;`, `:`) before dispatching to inference.
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
2. How severe is the phonemization overhead of `misaki` / `espeak-ng` relative to the neural model inference time on short utterances?
3. Does style blending introduce audible acoustic artifacts or phase distortion through the iSTFTNet vocoder?
4. How cleanly does the LGPL-3.0 license of the underlying `espeak-ng` library isolate from proprietary host binaries when linked as a shared library?
