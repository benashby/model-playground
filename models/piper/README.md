# Piper

A fast, local neural text-to-speech engine based on VITS. Seeded for investigation 2026-09-27.

- Upstream code: <https://github.com/rhasspy/piper> (MIT)
- Voice models repository: <https://huggingface.co/rhasspy/piper-voices> (MIT / Public Domain depending on voice dataset)
- Phonemizer: <https://github.com/rhasspy/piper-phonemize>
- Author: Michael Hansen (Rhasspy / Nabu Casa)

## What the model is

Piper is an end-to-end neural text-to-speech system designed to run locally on low-power devices, single-board computers, and commodity server CPUs.

The underlying architecture is based on VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech), mapping text or phoneme sequences directly to raw waveforms without an intermediate spectrogram representation. Audio generation uses a flow-based decoder coupled with a HiFi-GAN-style adversarial vocoder module.

Piper distributes pre-trained voice models under three distinct quality profiles:
1. `low`: 16,000 Hz (16 kHz) output, roughly 15M parameters. Tuned for embedded devices and high-density telephony.
2. `medium`: 22,050 Hz output, roughly 28M parameters. Default balance between speech naturalness and inference overhead.
3. `high`: 22,050 Hz or 24,000 Hz output, roughly 60M parameters. Enhanced acoustic clarity.

### What it is good for

- High-density CPU synthesis. On server CPUs, Piper regularly exceeds 50x to 100x real time, allowing tens to hundreds of concurrent audio streams per core [CLAIM].
- Ultra-low latency turn-taking. Because the model operates non-autoregressively on phoneme spans, time-to-first-audio (TTFA) on CPU can fall below 50 ms for short phrases [CLAIM].
- Telephony compatibility. The `low` quality tier outputs native 16 kHz audio, which aligns directly with wideband VoIP (G.722 or 16 kHz Opus) without requiring a multi-stage resampling step.
- Minimal resource consumption. Models take between 25 MB and 75 MB of memory at runtime, fitting comfortably in memory-constrained containers or microVMs.
- Standalone native embedding. The core engine is written in C++ and links against ONNX Runtime, avoiding Python or PyTorch dependencies entirely.

### What it does not do

- Zero-shot voice cloning. Models are trained per speaker or for a fixed set of multi-speaker IDs. Adding a new voice requires training or fine-tuning a checkpoint.
- Fine-grained prosody or emotion prompting. Pitch, rate, and sentence silence are adjustable through scalar configuration flags, but emotional inflections cannot be guided by natural language prompts.
- Built-in text normalization for specialized domains. Complex numerical strings, dates, and domain-specific acronyms rely on `espeak-ng` rules unless pre-normalized upstream in text preprocessing.

## Licensing

Enumerate each artifact in the serving path:

| Artifact | License | Notes |
|---|---|---|
| Engine code (`rhasspy/piper`) | MIT | Permissive commercial use. |
| Voice models (`piper-voices`) | MIT / Public Domain / CC0 | Model licenses inherit from training corpora. Most official English voices (such as `en_US-lessac`, `en_US-libritts_r`) are MIT or public domain. Check per-voice metadata JSON. |
| Text frontend (`piper-phonemize`) | MIT | C++ phonemization library. |
| Phoneme backend (`espeak-ng`) | LGPL-3.0 / GPL-3.0 | Dynamically linked in standard builds. Proprietary distributions embedding Piper statically must review LGPL linking rules. |
| Inference engine (ONNX Runtime) | MIT | Upstream Microsoft runtime. |

## Architecture

| Property | Low Profile | Medium Profile | High Profile |
|---|---|---|---|
| Parameters | ~15 M | ~28 M | ~60 M |
| Sample rate | 16,000 Hz | 22,050 Hz | 22,050 Hz or 24,000 Hz |
| Audio format | 16-bit mono PCM | 16-bit mono PCM | 16-bit mono PCM |
| Model size on disk | ~20 MB to ~35 MB | ~50 MB to ~70 MB | ~110 MB to ~140 MB |
| Underlying model | VITS (flow-based decoder) | VITS (standard) | VITS (expanded hidden dimensions) |
| Runtime format | ONNX | ONNX | ONNX |

## Host and launch options

### 1. Standalone C++ binary
```bash
echo "Testing Piper speech synthesis." | piper \
  --model en_US-lessac-medium.onnx \
  --config en_US-lessac-medium.onnx.json \
  --output_file output.wav
```

### 2. Embedded in Rust / C++ via sherpa-onnx
`sherpa-onnx` natively supports Piper ONNX checkpoints through its VITS model loader:
- C++ API: `OfflineTtsVitsModelConfig`
- Rust crate: `sherpa_onnx::OfflineTtsVitsModelConfig`
- Configuration points: `model`, `lexicon`, `tokens`, `data_dir` (for espeak-ng-data)

### 3. Python package
```bash
pip install piper-tts
```
```python
import wave
from piper.voice import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx", config_path="en_US-lessac-medium.onnx.json")
with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize("Welcome to the call center media server.", wav_file)
```

### 4. Synauson media server integration
Synauson supports both in-process Rust ONNX and Python runtime execution:
- **In-process Rust via `ort`:** Runs the ONNX graph directly in Rust using ONNX Runtime 1.24.4 inside custom GStreamer elements (`synauson-onnx`).
- **Python via `gst-python` (PyGObject phase 3):** Python GStreamer elements run directly on the tap runtime as part of a `TapFeature` chain.
- **Native 16 kHz alignment:** Piper's `low` profile outputs 16,000 Hz mono audio, directly matching Synauson's conference format (`S16LE, 16 kHz, mono`, `conference_caps()`, `synauson-core/src/pipeline/element_helpers.rs:142`). This eliminates the resampling stage needed by 22.05 kHz or 24 kHz models before mixing into the conference.
- **Pipeline isolation:** Synauson taps (`synauson-core/src/pipeline/tap.rs`) isolate inference on their own streaming threads via downstream leaky queues (`max-size-time=500ms`) and flow guards (`convert-to=ok`), ensuring model stalls or runtime errors never block participant audio.

## Protocol and interface

- Input: UTF-8 plain text string or pre-computed phoneme sequence.
- Audio streaming: Piper can stream raw PCM chunks directly to `stdout` or an audio callback handler as sentences or clauses are evaluated.
- Length scale / noise parameters:
  - `length_scale`: Controls speaking rate (< 1.0 speeds up, > 1.0 slows down).
  - `noise_scale`: Controls phoneme duration variability.
  - `noise_w`: Controls phoneme pronunciation variance.

## Fixtures to probe

When evaluating Piper for telephony and voicebot media workers:
1. Low vs Medium latency: Compare TTFA between `en_US-lessac-low` (16 kHz) and `en_US-lessac-medium` (22.05 kHz).
2. Direct 16 kHz PSTN output: Compare audio intelligibility and artifact levels of the 16 kHz `low` model against downsampled 22.05 kHz `medium` audio.
3. Multi-speaker indexing: Test speaker switching latency in multi-speaker models (e.g., `en_US-libritts-high`) using the `--speaker` ID argument.
4. Stress concurrency: Measure throughput degradation and jitter when executing 10, 25, and 50 simultaneous synthesis threads on a bounded CPU allocation.
5. Punctuation chunking: Measure latency gains when splitting streaming LLM text by commas and clauses compared to full sentence buffers.

## Probe plan

Probes to write under `probes/`:
- `probes/profile_comparison.py`: Benchmark RTF and TTFA across `low`, `medium`, and `high` model tiers on identical hardware.
- `probes/concurrency_scaling.py`: Measure maximum concurrent real-time channels on 1, 2, 4, and 8 CPU cores.
- `probes/telephony_g711_pipeline.py`: Benchmark full pipeline latency: text input -> Piper synthesis -> 8 kHz u-law transcode -> 20 ms RTP frame packetization.
- `probes/memory_footprint.py`: Measure RSS memory per loaded model instance in C++ and Python.

## Open questions

1. How noticeable is the quality loss of the 16 kHz `low` profile on telephone headsets compared to the `medium` profile?
2. When managing 50 concurrent telephony channels on a single CPU node, does running multiple Piper ONNX sessions share weights in memory, or does each session duplicate tensor storage?
3. How does Piper's TTFA on CPU compare directly with Kokoro-82M on short responses (< 10 words)?
4. What is the CPU cost of `piper-phonemize` when processing heavy numbers, addresses, and symbol sequences?
