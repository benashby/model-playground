# Piper

A fast, local neural text-to-speech engine based on VITS. Seeded for investigation 2026-09-27.

- Upstream code: <https://github.com/rhasspy/piper> (MIT)
- Voice models repository: <https://huggingface.co/rhasspy/piper-voices> (MIT / Public Domain depending on voice dataset)
- Phonemizer: <https://github.com/rhasspy/piper-phonemize>
- Author: Michael Hansen (Rhasspy / Nabu Casa)

## What the model is

Piper is an end-to-end neural text-to-speech system built to run locally on low-power devices, single-board computers and commodity server CPUs.

It is based on VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech), which maps text or phoneme sequences straight to raw waveforms with no intermediate spectrogram. Audio comes from a flow-based decoder with a HiFi-GAN-style adversarial vocoder.

Piper's pre-trained voices come in three quality profiles:

1. `low`: 16,000 Hz (16 kHz) output, roughly 15M parameters. Tuned for embedded devices and high-density telephony.
2. `medium`: 22,050 Hz output, roughly 28M parameters. The default balance between naturalness and inference cost.
3. `high`: 22,050 Hz or 24,000 Hz output, roughly 60M parameters. Clearer audio.

### What it is good for

- Many streams per CPU. On server CPUs, Piper regularly exceeds 50x to 100x real time, allowing tens to hundreds of concurrent audio streams per core [CLAIM].
- Fast turn-taking. The model runs non-autoregressively over phoneme spans, so time-to-first-audio (TTFA) on CPU can fall below 50 ms for short phrases [CLAIM].
- Telephony. The `low` profile outputs 16 kHz audio natively, which matches wideband VoIP (G.722 or 16 kHz Opus) with no resampling step.
- Small memory use. Models take between 25 MB and 75 MB of memory at runtime, small enough for memory-constrained containers or microVMs [CLAIM].
- Native embedding. The core engine is written in C++ and links against ONNX Runtime, with no Python or PyTorch.

### What it does not do

- Zero-shot voice cloning. Models are trained per speaker or for a fixed set of multi-speaker IDs, so a new voice means training or fine-tuning a checkpoint.
- Prosody or emotion prompting. Pitch, rate and sentence silence are scalar configuration flags; there is no natural-language control of emotion.
- Text normalization for specialized domains. Complex numbers, dates and domain acronyms follow `espeak-ng` rules unless they are normalized upstream.

## Licensing

| Artifact | License | Notes |
|---|---|---|
| Engine code (`rhasspy/piper`) | MIT | Permissive commercial use. |
| Voice models (`piper-voices`) | MIT / Public Domain / CC0 | Model licenses inherit from training corpora. Most official English voices (such as `en_US-lessac`, `en_US-libritts_r`) are MIT or public domain. Check per-voice metadata JSON. |
| Text frontend (`piper-phonemize`) | MIT | C++ phonemization library. |
| Phoneme backend (`espeak-ng`) | LGPL-3.0 / GPL-3.0 | Dynamically linked in standard builds. Proprietary distributions embedding Piper statically must review LGPL linking rules. |
| Inference engine (ONNX Runtime) | MIT | Upstream Microsoft runtime. |

## Architecture

| Property | Low profile | Medium profile | High profile |
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

`sherpa-onnx` loads Piper ONNX checkpoints through its VITS model loader:

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

### 4. GStreamer media server integration

A real-time media pipeline can run the model in process through native ONNX, or in Python. In-process native code runs the ONNX graph with ONNX Runtime (through `ort`) inside custom GStreamer elements. Through `gst-python` (PyGObject), Python GStreamer elements run on a tap runtime as part of an audio feature chain.

Its native 16 kHz output fits a media router as is: Piper's `low` profile outputs 16,000 Hz mono audio, the standard wideband format (`S16LE, 16 kHz, mono`), so it needs none of the resampling that 22.05 kHz or 24 kHz models need before mixing into a conference.

A media tap keeps inference on its own streaming thread with a downstream leaky queue (`max-size-time=500ms`, `leaky=downstream`) and a flow guard (`convert-to=ok`), so a stalled model or a runtime error never blocks participant audio.

## Protocol and interface

- Input: UTF-8 plain text string or pre-computed phoneme sequence.
- Audio streaming: Piper can stream raw PCM chunks to `stdout` or an audio callback as it finishes each sentence or clause.
- Length scale / noise parameters:
  - `length_scale`: Controls speaking rate (< 1.0 speeds up, > 1.0 slows down).
  - `noise_scale`: Controls phoneme duration variability.
  - `noise_w`: Controls phoneme pronunciation variance.

## Fixtures to probe

For telephony and voicebot media workers:

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
