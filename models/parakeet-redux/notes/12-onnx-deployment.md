# Deploying parakeet-tdt-0.6b-v3 on ONNX

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

This is a reference for running NVIDIA's `parakeet-tdt-0.6b-v3` in production
through its ONNX export, with sherpa-onnx as the runtime. Every recommendation
comes from a measurement in this investigation, and each links to it. The
measurements themselves are in
[accuracy on telephone speech](11-results-telephone-accuracy.md) and
[CPU and memory for live streams](10-results-cpu-sizing.md). For live use,
where delay matters as much as accuracy, see
[live transcription](13-live-transcription.md), which also covers a true
streaming model.

## The decisions at a glance

| Decision | Choice | Why |
|---|---|---|
| Runtime | sherpa-onnx, fp32 export | As accurate as NVIDIA's NeMo on the same audio (within 0.23 points), CC-BY-4.0 weights, no PyTorch |
| Precision | **fp32, never int8** | int8 more than doubles the error rate on 8 kHz audio and silently drops whole phrases |
| Hardware | CPU or NVIDIA GPU, by cost | Same transcripts on both (99.53 % of segments identical) |
| Streaming | VAD-segmented decoding | The model is offline; sherpa-onnx has no streaming mode for it |
| VAD | Silero v5, 0.5 s silence, 20 s maximum, 0.25 s padding | sherpa-onnx cuts segments with no margin; padding restores clipped onsets |
| Telephone input | Upsample 8 kHz to 16 kHz, one speaker per channel | The model expects 16 kHz; mixing both call legs into one channel mixes both halves of the conversation |
| Concurrency | One recogniser per process, decodes on a thread pool, one VAD per stream | Memory is a fixed 3.0 to 3.2 GB per process for fp32; per-stream state is small |

## 1. What you are deploying

The model is a 600M-parameter FastConformer encoder with a TDT
(Token-and-Duration Transducer) decoder
([how it works](01-how-it-works.md)). It transcribes a finished piece of
audio: there is no streaming version. The ONNX export splits it into three
graphs plus a token list:

| File | Role |
|---|---|
| `encoder.onnx` (+ `encoder.weights`) | audio features to acoustic frames; nearly all the compute |
| `decoder.onnx` | the prediction network (an LSTM over emitted tokens) |
| `joiner.onnx` | combines the two into token and duration scores |
| `tokens.txt` | the token vocabulary |

Use the fp32 export from the sherpa-onnx author,
`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3` on Hugging Face (revision
`1a468a35cbba69418f126de829e75261dea4a4e4` was used here). The int8 export,
`sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` in the sherpa-onnx releases, is
smaller and is the one most examples point to. Do not deploy it for
telephone audio (section 3).

The VAD is Silero v5, `silero_vad_v5.onnx` from the sherpa-onnx `asr-models`
release. It is a small ONNX model run by the same library.

Licensing: NVIDIA's weights are CC-BY-4.0, which permits commercial use with
attribution ([licensing](02-licensing.md#the-weights)). The sherpa-onnx
package and the converted model archive have not been through this
investigation's licence probe. Check both before shipping. Open.

## 2. The pipeline

```
audio source (file, microphone, RTP stream)
  │  decode, one speaker per channel
  ▼
16 kHz mono float32
  │  in chunks of any size
  ▼
Silero VAD v5 (per stream)        speech/silence every 32 ms; a segment ends
  │                               after 0.5 s of silence, or at 20 s
  ▼
padded segment                    the VAD's segment, plus 0.25 s either side,
  │                               cut from a rolling buffer of recent audio
  ▼
decode queue → Parakeet (ONNX)    one worker thread per stream: results in
  │                               spoken order; many streams share a pool
  ▼
text, once per utterance
```

[`examples/parakeet_onnx.py`](../examples/parakeet_onnx.py) implements this
as a small `Transcriber` class: `feed(samples)` returns finished utterances,
`flush()` ends the stream.

```python
from parakeet_onnx import Transcriber

t = Transcriber(model_dir="/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3",
                vad_model="/models/silero_vad_v5.onnx")
for chunk in chunks:                  # 16 kHz mono float32 numpy arrays
    for text in t.feed(chunk):
        print(text)
for text in t.flush():
    print(text)
```

Latency from the end of an utterance to its text is the VAD's 0.5 s of
silence plus the decode: for a single stream, a median of 0.96 s with the
fp32 export on one Zen 2 core and its hyperthread, and 0.73 s with int8 on an
i9-12900KS core ([sizing](10-results-cpu-sizing.md)). A draft of the
utterance in progress is possible by re-decoding the open segment
periodically, at about 2.5 times the CPU cost.

## 3. Precision: fp32 only

On the same segments, with the same runtime
([accuracy](11-results-telephone-accuracy.md#the-int8-export-measured)):

| Audio | fp32 | int8 |
|---|---|---|
| AppTek en-US_General, 16 kHz | 7.01 % | 8.43 % |
| AppTek en-IN, 16 kHz | 11.24 % | 18.39 % |
| AppTek en-US_General through G.711 | 10.06 % | 23.06 % |
| HarperValleyBank, real telephone calls | 16.71 % | 34.88 % |

The int8 export fails by dropping stretches of speech: on real calls, 54.3 %
of its deleted words are in runs of 11 or more, against 19.5 % for fp32. The
transcript reads cleanly with a sentence missing, so nothing downstream
notices. It fails the same way on the CPU and the GPU, and on the GPU it is
also slower (16× to 18× real time against 94× to 101×). This is a property
of this export, not of int8 as such: the int8 export of NVIDIA's streaming
model held up on the same G.711 audio
([live transcription](13-live-transcription.md#a-streaming-model-nemotron-measured)). Its only advantage
measured here is memory, covered under concurrency below.

## 4. Segmentation

The model needs speech cut into utterances, and how it is cut changes the
result.

- Pad the segments. sherpa-onnx's `VoiceActivityDetector` returns each
  segment starting exactly where it detected speech, with no margin, which
  clips the first word. Keep a rolling buffer of recent audio and cut each
  segment again from it with 0.25 s either side, as `parakeet_onnx.py` does.
  On the same segments, padding took en-US_General from 10.50 % to 8.04 % and
  en-GB_SCT from 14.93 % to 13.58 %
  ([accuracy](11-results-telephone-accuracy.md#segmentation-and-padding-measured)).
- Live use: 0.5 s of silence ends an utterance, and 20 s is the maximum.
  Shorter silence cuts sentences in two; longer adds latency.
- Files, where latency does not matter: AppTek's recipe (10 s of
  silence, 30 s maximum) gives longer segments, and on HarperValleyBank it
  scored slightly better than the live settings (16.71 % against 17.53 %).
- Sherpa-onnx's Silero implementation and the silero-vad Python package may
  cut differently with the same settings. Everything here used sherpa-onnx's,
  and the 1.0 to 1.4 point gap to the published figures for this model is
  most likely that difference. Open.

## 5. Telephone audio

- Upsample to 16 kHz. The model expects 16 kHz. Narrowband (8 kHz) audio
  needs nothing more than a resampler: GStreamer's `audioresample`, sherpa-onnx's
  own (`accept_waveform(8000, samples)` resamples internally) or a
  band-limited one. Resampling does not restore the missing band: with the
  band-limited resampler used here, narrowband cost 1.7 to 2.7 points of WER.
  Other resamplers were not compared.
- Codecs. G.711 μ-law and Opus at 12 kbit/s cost about the same. Mobile
  codecs (AMR, GSM) were not tested.
- Packet loss. With the decoder's packet loss concealment, 20 % random
  loss costs another 2.2 to 3.5 points. Bursty loss was not tested. Put a
  jitter buffer in front of the decoder (GStreamer's `rtpjitterbuffer`).
- One speaker per channel. Transcribe each leg of a call separately. A
  mixed-down call gives the model both sides at once, and the transcript
  merges them ([using it through Photon](04-usage.md) shows the same problem
  in another runtime).
- Expect short confirmations to go missing. About a quarter of spoken
  "okay"s and a fifth of "yeah"s are dropped, even on clean audio
  ([which words](11-results-telephone-accuracy.md#which-words-go-missing-measured)).
  If a caller's "yes" or "no" drives a decision, confirm it another way.

## 6. Hardware and runtimes

### Which runtime on which hardware

Every combination below was run on the same audio segments
([accuracy](11-results-telephone-accuracy.md#the-runtime-makes-no-difference-measured)).
Versions are from
[`results/gpu_environment.log`](../results/gpu_environment.log); speeds are
batch-1 throughputs from
[`results/nemo_vs_onnx-timing.log`](../results/nemo_vs_onnx-timing.log).

| Runtime | Hardware | What to install | Versions tested | Speed on AppTek segments | Accuracy |
|---|---|---|---|---|---|
| sherpa-onnx, fp32 | x86 CPU | `pip install sherpa-onnx` | 1.13.8 | 22× real time on a Ryzen 7 3700X with 6 workers | the reference: within 0.23 points of NeMo |
| sherpa-onnx, fp32 | NVIDIA GPU | sherpa-onnx's CUDA wheel, NVIDIA's CUDA 12 and cuDNN 9 pip packages, ALSA | 1.13.8+cuda12.cudnn9, cuDNN 9.26, driver 615.71.09 | 101× on an RTX 3090 Ti with 3 workers | the same as on the CPU |
| NeMo 3.0.0 | NVIDIA GPU | PyTorch for CUDA 12.8, `nemo_toolkit[asr]` | PyTorch 2.11.0+cu128, cuDNN 9.19, driver 615.71.09 | 305× with CUDA graphs, 143× without | the same as ONNX fp32 |
| NeMo 3.0.0 | AMD GPU (ROCm) | PyTorch for ROCm 7.2, `nemo_toolkit[asr]`, three workarounds (section 8) | PyTorch 2.14.0+rocm7.2, HIP 7.2, gfx1201 on Linux 7.1.2 | 68× with one CPU core | the same as NeMo on NVIDIA |
| sherpa-onnx, int8 | CPU or NVIDIA GPU | as above, int8 files | 1.13.8 | 18× on the RTX 3090 Ti | do not use for telephone audio |

Not tested: ONNX on an AMD GPU (ONNX Runtime's ROCm or MIGraphX providers),
TensorRT, NVIDIA's Riva and NIM serving stacks, NeMo on a CPU, and Apple
hardware.

What sets accuracy is the same on every device: the fp32 export, padded
segments, and 16 kHz input with one speaker per channel. The device only
changes speed and cost. Two GPU options that could change the balance were
not measured: sherpa-onnx's fp16 export of the model, and decoding several
segments per call (every figure here decodes one at a time). Open.

### CPU

`pip install sherpa-onnx numpy` gives the CPU build. Use one recogniser per
process with `num_threads=1` per decode and run decodes on a thread pool with
one worker per core: sherpa-onnx releases Python's GIL while decoding, so the
threads run in parallel. Throughput and per-stream cost are in
[sizing](10-results-cpu-sizing.md).

### NVIDIA GPU

sherpa-onnx publishes CUDA builds outside PyPI. The one used here:

```
https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.13.8/sherpa_onnx-1.13.8+cuda12.cudnn9.onnxruntime1.28.2-cp312-cp312-linux_x86_64.whl
```

Pass `provider="cuda"` to `OfflineRecognizer.from_transducer`. It needs the
CUDA 12 and cuDNN 9 runtime libraries on the loader path, and ALSA
(`libasound.so.2`), which the CUDA build links against even though it never
touches audio devices. This image works with only the NVIDIA driver on the
host (3.79 GB, most of it cuDNN and cuBLAS):

```dockerfile
FROM docker.io/library/python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libasound2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy soundfile \
      nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
      "https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.13.8/sherpa_onnx-1.13.8+cuda12.cudnn9.onnxruntime1.28.2-cp312-cp312-linux_x86_64.whl"
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib
```

In it, 20 s of audio decoded in 0.073 s on an RTX 3090 Ti after a one-off
0.64 s on the first call. Check that a GPU deployment really runs on the GPU:
if the CUDA provider cannot load, ONNX Runtime can fall back to the CPU, and
the only symptom is speed. A first-decode time in seconds, then tens of
milliseconds, is the GPU; seconds every time is not.

With podman, pass the GPU with `--device nvidia.com/gpu=all` (CDI). On an
SELinux host (Fedora, RHEL, Bazzite) add `--security-opt label=disable`, or
`:Z` on bind mounts, or the container gets `Permission denied` on mounted
files.

### AMD GPU

ONNX was not tested on an AMD GPU. NVIDIA's NeMo does run on one; see
section 8.

## 7. Concurrency

- Memory is per process, not per stream. A process holding the fp32
  model measured 2,256 MB after loading and 3.0 to 3.2 GB while decoding,
  nearly flat from 1 stream to 25; the int8 model takes about 1.8 GB
  ([sizing](10-results-cpu-sizing.md)). Run many streams in one process
  rather than one process per stream, and do not use a 2 GB instance.
- Never decode on an I/O thread. Decoding takes tens to hundreds of
  milliseconds. On an asyncio loop it stalls every stream; in a GStreamer
  callback it stalls the pipeline, and with a network source the jitter
  buffer then drops packets. Hand segments to a worker thread.
- One decode worker per stream keeps results in order. A shared pool
  across streams is fine, but within one stream decode sequentially or
  reorder the results.
- Short segments cost more per second of audio. Every runtime ran
  several times slower on HarperValleyBank's padded short segments than on
  AppTek's longer ones
  ([throughput](11-results-telephone-accuracy.md#throughput-measured-indicative)).

## 8. NeMo, if you need it

NeMo gives the same accuracy as the fp32 ONNX export and is faster on an
NVIDIA GPU (305× real time with CUDA graphs against 101× for ONNX at batch
size 1), at the cost of PyTorch and a much heavier install.

### On an NVIDIA GPU

These are the packages the measurements used, on a host with only the
NVIDIA driver (615.71.09) and podman's CDI device for it. The measured setup
installed them into a venv on this base image; an image build is the same:

```dockerfile
FROM docker.io/library/python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 libnuma1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir "nemo_toolkit[asr]==3.0.0"
```

Run it with `--device nvidia.com/gpu=all` (or `=0` for one card). PyTorch's
CUDA wheels bring their own CUDA and cuDNN libraries, so nothing else is
needed on the host. Transcribing pre-cut 16 kHz segments, with NeMo's
default greedy TDT decoding:

```python
import nemo.collections.asr as nemo_asr
import torch

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cuda")
model.eval()
with torch.inference_mode():
    hyps = model.transcribe(["segment_0001.wav", "segment_0002.wav"], batch_size=1)
texts = [h.text for h in hyps]
```

The CUDA-graph decoder is on by default on NVIDIA and is the fast path: it
changed no results, and without it throughput halved. The first call
downloads the checkpoint from Hugging Face.

### On an AMD GPU

NeMo also runs on an AMD RX 9070 XT (gfx1201) under ROCm 7.2, with the same
base image and PyTorch's ROCm wheels
(`--index-url https://download.pytorch.org/whl/rocm7.2`), passing
`--device /dev/kfd --device /dev/dri` to the container. It needed three
workarounds:

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libatomic.so.1` on `import torch` | slim Python images lack it; ROCm PyTorch links it | install `libatomic1` (also `libnuma1`) |
| model load fails with `"cuda" is an NVIDIA driver library ... Ensure the NVIDIA display driver is installed`, then `Can't instantiate abstract class ASRModel` | NeMo probes NVIDIA's CUDA library through `cuda-bindings` when the model is built ([NeMo issue 15905](https://github.com/NVIDIA-NeMo/Speech/issues/15905)) | `pip uninstall cuda-bindings cuda-pathfinder` |
| first transcription fails with `RuntimeError: miopenStatusUnknownError` in the LSTM | MIOpen's LSTM kernels on gfx1201 | `torch.backends.cudnn.enabled = False` before transcribing |

Also set `decoding.greedy.use_cuda_graph_decoder = False` on ROCm; CUDA
graphs are NVIDIA-only, and turning them off changed no results on NVIDIA
either. Without them, a batch-1 NeMo decode is limited by the CPU thread
driving the GPU. [`probes/nemo_vs_onnx.py`](../probes/nemo_vs_onnx.py) has the
working NeMo code.

## 9. GStreamer

GStreamer handles capture, networking, jitter, codecs and resampling; the
transcriber needs only 16 kHz mono PCM out of it. There are two ways to
connect them, both in [`examples/`](../examples/) and both tested (below).

### Across a pipe: `gst_stdin.py`

`gst-launch-1.0` runs the pipeline in its own process and writes raw PCM to
stdout, and Python reads it. Nothing in Python links against GStreamer. A
G.711 RTP stream on UDP port 5004, one leg of a phone call:

```bash
gst-launch-1.0 -q udpsrc port=5004 \
    caps="application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0" \
    ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec ! audioconvert ! audioresample \
    ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1 \
  | python gst_stdin.py
```

Always run gst-launch with `-q` here: without it, its status messages go to
stdout and into the audio.

### In process: `gst_appsink.py`

The pipeline runs inside Python through PyGObject and ends in an `appsink`,
from which the script pulls buffers.

PyGObject is how Python reaches GStreamer. GStreamer is a C library built on
GLib's GObject type system, and GObject libraries ship machine-readable
descriptions of their API (GObject Introspection `.typelib` files, in
Debian's `gir1.2-gstreamer-1.0` and `gir1.2-gst-plugins-base-1.0`). PyGObject
(`import gi`) reads them at run time and calls the C functions through libffi,
so `from gi.repository import Gst` exposes GStreamer's whole API without
hand-written bindings. It reaches GObject libraries only. llama.cpp, for
example, is not one; a transcript goes to llama.cpp through its HTTP server
or its own Python bindings.

What runs where in the in-process version:

```
Python main thread     try_pull_sample() → map buffer → numpy → VAD (cheap)
                             │ segments
Python decode thread   ONNX decode (GIL released while in C++)
GStreamer threads      udpsrc → rtpjitterbuffer → rtppcmudepay → mulawdec
                       → audioconvert → audioresample → appsink queue
kernel                 UDP socket (RTP), or ALSA/PipeWire for a microphone
```

The appsink queue is bounded (`max-buffers=50`, `drop=false`), so a slow
consumer slows the pipeline instead of growing memory, and buffers are pulled
on the script's thread rather than in a `new-sample` callback, which would
run on a GStreamer streaming thread.

The pipe keeps GStreamer and Python in separate processes, which is simpler
to deploy and isolates crashes. The in-process version gives access to the
pipeline's state and messages from Python. A third option is a GStreamer
element written in Python with gst-python, which lets the transcriber sit
inside a pipeline and post each utterance as a bus message or route to an
internal sink.

In Synauson (PR #13, 2026-09-27), this is formalized as the tap interface
(`synauson-core/src/pipeline/tap.rs`):
```
participant tee.src_%u -> queue (leaky) -> errorignore -> [feature chain ... ending in a sink]
```
The tap hangs off the participant's normalized fanout tee (`S16LE, 16 kHz, mono`,
via `conference_caps()`). The downstream leaky queue (`max-size-time=500ms`,
`leaky=downstream`) gives the tap its own streaming thread and drops oldest
audio if inference falls behind, preventing backpressure on conference audio.
The `errorignore` flow guard (`convert-to=ok`) catches errors, not-negotiated,
and EOS so downstream model failures cannot propagate back to stop the tee.
While VAD and turn detection currently run in-process in Rust via `ort`
(ONNX Runtime 1.24.4) in `synauson-onnx`, the planned PyGObject phase 3 allows
hosting Python elements via `gst-python` on this same tap runtime.

### Tested [MEASURED]

[`examples/test_gstreamer.sh`](../examples/test_gstreamer.sh) runs both
examples on a 90 s call-centre recording, twice each: read from a WAV file,
and sent as a live G.711 RTP stream over UDP in real time by a second
GStreamer pipeline, the way one leg of a phone call arrives. With GStreamer
1.22.0 on Debian bookworm
([`results/gstreamer-examples.log`](../results/gstreamer-examples.log)):

| Test | Utterances transcribed |
|---|---|
| `gst_stdin.py`, file | 4 |
| `gst_stdin.py`, live RTP | 3 |
| `gst_appsink.py`, file | 4 |
| `gst_appsink.py`, live RTP, stopped with SIGINT | 3 |

The two examples produced byte-identical transcripts in each mode. The RTP
runs lost a short "Good morning." at the start of one utterance and the last
utterance of the clip. The first is the narrowband deletion measured in
[accuracy](11-results-telephone-accuracy.md#which-words-go-missing-measured);
the second may be that or the end of the stream arriving as the receiver
stopped, and one run cannot tell them apart.

Testing found a trap in the in-process version. `appsink`'s pull methods
(`try_pull_sample`, `is_eos`) belong to the separate GstApp library, and
PyGObject adds them to the element only after
`gi.require_version("GstApp", "1.0")` and `from gi.repository import GstApp`,
although the name is never used. Without them the pipeline builds and then fails with `AttributeError:
'GstAppSink' object has no attribute 'try_pull_sample'`. For the pipe version, `-e` on `gst-launch-1.0` makes an
interrupt send end-of-stream through the pipeline, so the last utterance is
flushed rather than lost.

## 10. Before production

- fp32 export, never int8, and a check that the GPU provider actually loaded.
- Segments padded; VAD settings chosen for live or batch use.
- Each call leg transcribed on its own channel.
- A plan for the short confirmations the model drops.
- The licences of sherpa-onnx and the converted model checked.
- CPU sized for the share of speech in your audio, not for its length
  ([sizing](10-results-cpu-sizing.md)).
- Accuracy checked on your own audio. These figures are for English
  call-centre and simulated bank calls; noise and most accents were not
  tested.

---

Previous: [Results: accuracy on telephone speech](11-results-telephone-accuracy.md) | [Contents](../README.md#contents) | Next: [Live transcription](13-live-transcription.md)
