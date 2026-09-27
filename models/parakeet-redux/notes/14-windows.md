# Running it on Windows

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

The rest of this note was measured on Linux. This article covers native
Windows (no WSL): what runs, how to set it up, what behaves differently, and
which Linux results were repeated there. The machine is a Windows 10 Pro 19045
desktop with an Intel i7-9700 (8 cores, AVX2, no AVX-512), 32 GB of RAM and an
NVIDIA GTX 1650 with 4 GB (Turing), driver 617.14
([`results/environment-windows.log`](../results/environment-windows.log)).

## What runs where

| Runtime | Linux | Windows |
|---|---|---|
| Photon (Redux, Ultra), CPU | yes, the note's main path | **no** [MEASURED]: the Windows kernels have no int8 CPU path for ternary weights |
| Photon, NVIDIA GPU | not measured here | yes, with `PHOTON_DEVICE=cuda` [MEASURED] |
| sherpa-onnx, CPU | yes | yes [MEASURED] |
| sherpa-onnx, NVIDIA GPU | yes, `linux_x86_64` CUDA wheel | yes, `win_amd64` CUDA wheel [MEASURED] |
| NeMo, NVIDIA GPU | yes, in a container | yes, in a plain venv, no workarounds [MEASURED] |
| NeMo, AMD GPU (ROCm) | yes, three workarounds | not tested |
| GStreamer examples | yes, Debian's GStreamer 1.22 | yes, GStreamer's own 1.26.7 build [MEASURED] |
| Microphone dictation | `pw-record` | ffmpeg's DirectShow input; not tested, the machine has no microphone |

If you need Parakeet on a Windows CPU, use NVIDIA's original checkpoint on
sherpa-onnx. It is the open-runtime option on Linux too
([deploying on ONNX](12-onnx-deployment.md)).

## Setting it up

### Python 3.13 or later

Use Python 3.13 or later for anything that measures time. On Windows,
Python 3.12's `time.monotonic()` advances in 15.625 ms steps, and it is the
clock asyncio schedules and the probes measure with. See
[timing on Windows](#timing-on-windows-measured) below.

```powershell
winget install --id Python.Python.3.13 --scope machine
winget install --id astral-sh.uv --scope machine
uv venv -p 3.13
```

Every Windows result in this article came from Python 3.13.15, except the
3.12 clock measurements, which show why.

### Photon: GPU only

On Linux the `asr` extra pins torch to its CPU build. On Windows use the
`asr-cuda` extra instead, which resolves torch's CUDA 13.0 build, and tell the
probes to use the GPU:

```powershell
uv sync --extra asr-cuda
$env:PHOTON_DEVICE = "cuda"
uv run --extra asr-cuda python models/parakeet-redux/probes/step0.py
```

The two extras cannot be installed together. A probe run with
`PHOTON_DEVICE` set says so in the first line of its output, so a GPU figure
cannot be mistaken for the CPU figures elsewhere in this note.

On the CPU, Photon refuses to load the model [MEASURED]:

```
NotImplementedError: ternary weights need an int8 matrix-multiply kernel on the CPU (avx512vnni, avxvnni or avx2 on x86; neon-i8mm, neon-dotprod or neon-mull on aarch64); the path in use here is 'scalar', so there is nothing to run them on here
```

The CPU has AVX2, and torch reports it (`get_cpu_capability(): AVX2`). The
Windows build of `kestrel-kernels` 0.7.1 (the locked version) has no AVX2 or
VNNI kernel for ternary weights, only the scalar reference, which Photon will
not use. The newer 0.7.3 was tried in a scratch environment and behaved the
same; that run was not recorded. Open: whether a later release adds the
kernels.

### sherpa-onnx

The CPU build is on PyPI, as on Linux:

```powershell
uv run --with sherpa-onnx python models/parakeet-redux/examples/gst_stdin.py
```

The GPU build comes from the same place as the Linux one, as a `win_amd64`
wheel:

```
https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.13.8/sherpa_onnx-1.13.8+cuda12.cudnn9-cp312-cp312-win_amd64.whl
```

(`cp313` and other Python versions are in the same directory.) It needs the
CUDA 12 and cuDNN 9 DLLs on `PATH`. Here they came from NVIDIA's installers:
CUDA Toolkit 12.9, and the cuDNN 9.26.0 archive for CUDA 12 with its
`bin\12.9\x64` directory added to `PATH`. Three things to know:

- cuDNN's CUDA 12 and CUDA 13 builds use the same file name, `cudnn64_9.dll`,
  so only one of them can be first on `PATH`.
- A shell or service started before `PATH` changed keeps the old one. The
  symptom is `onnxruntime_providers_cuda.dll ... depends on "cublasLt64_12.dll"
  which is missing`, and a probe's worker pool then dies with
  `BrokenProcessPool`.
- Check that decoding really runs on the GPU, as on Linux: the first decode
  takes seconds and later ones a fraction of that; if every decode is slow, the
  CUDA provider did not load.

The models are the same files as on Linux. Windows' `tar -xjf` failed on the
`.tar.bz2` archives here, so extract them with Python:

```powershell
python -c "import tarfile, sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')" model.tar.bz2 $HOME\.cache\sherpa-onnx
```

### NeMo

NeMo installs from pip into a plain venv. PyTorch 2.14 has no CUDA 12.8 build,
so this used its CUDA 13.0 build, which runs on the Turing card:

```powershell
uv venv -p 3.12 nemo-venv
uv pip install -p nemo-venv "nemo_toolkit[asr]==3.0.0" soundfile
uv pip install -p nemo-venv --reinstall-package torch --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0+cu130"
```

Install torch last, from the CUDA index alone. With PyPI as a fallback index,
the resolver picks PyPI's Windows torch, which is CPU-only. None of the ROCm
workarounds are needed. On a 4 GB card, watch memory: the Windows driver lets
CUDA spill into shared system memory instead of failing, so an oversized decode
runs, only slower.

### GStreamer

GStreamer's official MSVC installer ships Python bindings (PyGObject) in its
own `lib\site-packages`, and sets `GSTREAMER_1_0_ROOT_MSVC_X86_64`.
[`examples/gst_appsink.py`](../examples/gst_appsink.py) uses that variable to
find them. `gst_stdin.py` and `nemotron_stream.py` only need
`gst-launch-1.0.exe`. Every element the examples use was present in the
runtime and development packages installed here.

Run the pipes from Python or `cmd`, which pass binary data unchanged. Windows
cannot send SIGINT to a program in the background, so a live receiver is
stopped with Ctrl+Break, which `gst_appsink.py` treats like Ctrl+C.
[`examples/test_gstreamer.py`](../examples/test_gstreamer.py) runs the six
tests of `test_gstreamer.sh` this way, on either system:

```powershell
uv run --with sherpa-onnx python models/parakeet-redux/examples/test_gstreamer.py clip90.wav out
```

### The probes

The probes run on both systems. Where they read `/proc` or used
`sched_getaffinity` on Linux, they now go through
[`probes/hostinfo.py`](../probes/hostinfo.py), which asks Windows for the same
facts. Memory on Windows is the working set, which is close to Linux's RSS
without being the same measure. `taskset` has no Windows equivalent, so
`onnx_concurrency.py` takes `ONNX_CPUS=0,1` to pin itself.
`probes/environment.py` reports the Windows hardware and runs one Photon
transcription on each device. `dictate.py` takes `--mic NAME` for a DirectShow
microphone.

## Timing on Windows [MEASURED]

Two things can make a timing probe wrong on Windows without anything failing.
[`probes/clocks.py`](../probes/clocks.py) runs a 100 ms asyncio heartbeat 100
times and reports its lag twice: as `time.monotonic()` reports it, and as
`time.perf_counter()` does, which is the true figure
([`results/clocks-windows.log`](../results/clocks-windows.log); machine load
1 to 14 % throughout):

| Python | Timer | Lag by `monotonic()`, median | True lag, median | True lag, p99 |
|---|---|---|---|---|
| 3.12.10 | default | 9.00 ms | 8.29 ms | 20.57 ms |
| 3.12.10 | 1 ms | -6.00 ms | 0.52 ms | 0.90 ms |
| 3.13.15 | default | 7.00 ms | 7.00 ms | 21.90 ms |
| 3.13.15 | 1 ms | 0.41 ms | 0.41 ms | 1.98 ms |

Windows' default timer wakes a sleeping program late by several milliseconds
on any Python. `playground.audio`, which paces the live probes, now asks
Windows for 1 ms resolution (`timeBeginPeriod(1)`) when it is imported. The
second problem is the clock itself: before Python 3.13, `time.monotonic()` is
`GetTickCount64`, with a resolution of 15.625 ms, so it reported a negative
lag when the true one was 0.52 ms. `playground.audio` prints a warning on
such a Python. On Linux neither problem exists, and `clocks.py` runs there too.

A loaded machine also makes timing wrong. This machine runs other work, and
with it at full load the same heartbeat's lag grew several-fold even with both
fixes in place. Record the machine's load next to any timing run, as
`clocks.py` does.

## Results repeated on Windows [MEASURED]

The Linux figures are this note's committed results; the Windows figures are
from the logs linked in each row. Same probes, same corpora and model files.

| What | Linux | Windows (GTX 1650, i7-9700) |
|---|---|---|
| fp32 ONNX on the GPU, AppTek en-US_General, first 12 channels, `livepad` segments | 8.04 % WER (S 133, D 261, I 67), RTX 3090 Ti | 8.00 % (S 132, D 260, I 67), 411 segments on both ([log](../results/nemo_vs_onnx-windows.log)) |
| NeMo on the GPU, the same segments | not run on this set | 8.00 % (S 134, D 251, I 74) ([log](../results/nemo_vs_onnx-windows.log)) |
| fp32 ONNX on the GPU, the same channels through G.711 | 10.84 % (S 151, D 404, I 67) | 10.86 % (S 153, D 403, I 67) ([log](../results/nemo_vs_onnx-windows.log)) |
| Nemotron streaming on the CPU, en-US_General, one stream per channel | 5.44 %; first words median 0.58 s, p95 1.30; last words median 0.16 s; 0 revisions | 5.47 %; 0.58 s, p95 1.22; 0.16 s; 0 revisions ([log](../results/streaming_online-windows.log)) |
| GStreamer examples, six tests | 4, 3, 4, 3, 4, 4 lines; stdin and appsink identical | the same ([log](../results/gstreamer-examples-windows.log)) |
| Silero VAD speech share, `speech_density.py` | committed log | byte-identical output |
| The four analysis probes (`stream_analysis`, `channel_speed`, `compare_analysis`, `timing_summary`) | committed logs | byte-identical output |

The WER differences are two or three words in about 5,700, which is the size of
difference the note already found between runtimes on the same audio. The
segment cutting, the G.711 round trip through ffmpeg and the scoring all
behave the same on both systems.

## Not measured on Windows

- Speed and sizing. The machine was running other work during these runs, so
  none of its wall times are throughput figures, and the CPU sizing article
  was not repeated.
- The Photon probes beyond `step0.py` and `environment.py`, on the GPU.
- Photon on the CPU, which cannot run.
- The int8 export, the HarperValleyBank corpus, and the other AppTek accents.
- NeMo on an AMD GPU, and ONNX Runtime's DirectML provider.
- A microphone.

---

Previous: [Live transcription](13-live-transcription.md) | [Contents](../README.md#contents)
