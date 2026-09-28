---
name: model-hosts
description: Use when choosing where to run a model, provisioning a host, or debugging a host-specific failure in model-playground — the cloud GPU node (2x H100 80GB, CUDA), the local workstation (RDNA4 Radeon, ROCm/Vulkan, no CUDA ever), the LAN GPU box (2x RTX 3090, CUDA), the Windows GPU box (GTX 1650, native Windows, CUDA), or CPU. Covers the host matrix and how to pick, running on native Windows (wheels, clocks, cuDNN, GStreamer), the CUDA-vs-ROCm hard boundary that eliminates hosts before anything else, VRAM budgeting from safetensors dtype, the cloud deployment recipe including Docker/NVIDIA-container-toolkit/NGC auth, VM lifecycle and the ephemeral-IP trap, and the SSH gotchas specific to this workstation.
---

# Hosts

**This repository is public. Hosts are described by their *nature* — compute,
reach, cost — never by hostname, address, cloud project, or account.** The
concrete names, addresses and credentials live in the operator's private
configuration; anything here that needs one reads it from the environment
(`PLAYGROUND_HOST` / `PLAYGROUND_PORT`). Keep it that way when editing.

## The matrix

Five roles. Which physical machines fill them is deployment detail.

| Role | Compute | VRAM | Reach | Notes |
|---|---|---|---|---|
| **cloud-gpu** | 2× H100 80GB HBM3, NVLink | 160 GB | rented VM, SSH over public IP | Only role that runs CUDA containers at scale. **Billed hourly while up**, and preemptible. |
| **workstation** | RDNA4 Radeon (gfx1201) | 16 GB | local | **ROCm/Vulkan only. No CUDA, ever.** Where development happens. |
| **lan-gpu** | 2× RTX 3090 | 48 GB | LAN | CUDA. Always-on, no marginal cost. Managed out of band. |
| **workstation CPU** | — | — | local | Tiny models, tokenizers, smoke tests. More capable than it sounds — see the Parakeet note in `models/`. |
| **windows-gpu** | GTX 1650 (Turing, sm_75) + 8-core AVX2 CPU, no AVX-512 | 4 GB | LAN | **Native Windows 10**, CUDA 12.9 + 13.4 and cuDNN 9 installed machine-wide. The Windows platform for every note: the Parakeet stack is verified here (see `models/parakeet-redux/notes/14-windows.md`). Also a CI runner, for this repo's `windows-gpu` workflow and another project's, so check load before timing anything. |

Operator tooling for starting, stopping and selecting models on these hosts is
private and lives outside this repo. This file documents what the *harness*
needs to know: which role can run what, and how each one fails.

## Pick the host in this order

1. **Does it need CUDA?** Any NIM container, any TensorRT engine, most prebuilt
   inference images: **yes**. That eliminates **workstation** immediately.
   Check this before anything else — it is the cheapest question and the one
   most often answered last.
2. **Does it fit in VRAM?** Measure, do not trust the model card (below).
3. **Is it worth cloud cost?** **lan-gpu**'s 48 GB has no marginal cost. Use
   **cloud-gpu** only when the model genuinely needs 80 GB-class cards or
   NVLink.
4. **Local-first for iteration.** A 3 B model on the **workstation** you can
   restart in seconds beats a cloud round trip for protocol and harness work.
   Some work does not need a GPU at all: a ternary ASR model runs at 57× real
   time on this workstation's CPU.

## CUDA vs ROCm — the hard boundary

Not a compatibility layer you can shim. Practical consequences:

- **Containers from `nvcr.io` will not run on the workstation.** Full stop.
- **ROCm builds exist** for llama.cpp and vLLM, but coverage lags and kernels
  differ. Assume a model works on ROCm only once observed.
- `gfx1201` (RDNA4) is newer than much of the ROCm ecosystem expects;
  `HSA_OVERRIDE_GFX_VERSION` is sometimes required. It is not for PyTorch:
  gfx1201 is a native target from ROCm 6.4.1.
- **PyTorch and NeMo do run on the workstation's Radeon**, observed with
  PyTorch's ROCm 7.2 wheels in a rootless podman container
  (`--device /dev/kfd --device /dev/dri`). NeMo ASR needed three workarounds,
  written up in the Parakeet note's ONNX deployment article: `libatomic1` in
  slim images, `cuda-bindings` uninstalled (NeMo otherwise probes NVIDIA's
  `libcuda` and crashes at model load), and `torch.backends.cudnn.enabled =
  False` (MIOpen's LSTM fails on gfx1201). The ROCm wheels are ~16 GB of venv.
- **On the lan-gpu role**, podman reaches the GPUs through CDI
  (`--device nvidia.com/gpu=all`); its OS enforces SELinux, so containers need
  `--security-opt label=disable`. A resident model server there holds most of
  both cards: stop it and confirm with `nvidia-smi --query-compute-apps`
  before running anything else.
- **Vulkan** via llama.cpp is often the more reliable local path than ROCm
  proper, at some throughput cost.
- The project module deliberately declares **no** AI toolkit — setting CUDA env
  vars on a Radeon desktop is actively misleading. Note this also means a
  CPU-only inference path (such as Photon's) needs nothing added; verified by
  `ldd` against the installed wheels, which resolve only glibc and libgcc.

## Windows — the second platform

Every note should say how its model runs on Linux **and** on Windows. What the
Parakeet work established about native Windows (no WSL):

- **Wheels decide it.** Check each native package's PyPI files for `win_amd64`
  before assuming anything. Photon's Windows kernels exist but have **no int8 CPU
  path for ternary weights** (`kestrel_kernels._cpu.gemm_isa_available('avx2')`
  is False on an AVX2 CPU), so Parakeet Redux runs on Windows only with
  `PHOTON_DEVICE=cuda` and the `asr-cuda` extra. sherpa-onnx has CPU and
  `cuda12.cudnn9` Windows wheels; NeMo installs from pip with CUDA torch.
- **Python 3.13 or later on Windows** for anything timed. Before 3.13,
  `time.monotonic()` (asyncio's clock) is `GetTickCount64` with 15.625 ms
  resolution. `playground.audio` also calls `timeBeginPeriod(1)`, because
  Windows' default timer makes every sleep overshoot by up to ~15 ms.
  `probes/clocks.py` measures both.
- **Host facts** (CPU, affinity, RSS) come from `probes/hostinfo.py`, which reads
  `/proc` on Linux and Win32 on Windows. `taskset` has no Windows equivalent on
  the command line; probes that pin take `ONNX_CPUS`.
- **cuDNN's CUDA 12 and CUDA 13 builds share the DLL name** `cudnn64_9.dll`, so
  only one can be on PATH; the machine-wide one is the CUDA 12 build.
- **A shell started before a PATH change keeps the old PATH**: GPU runs then fail
  with `cublasLt64_12.dll ... missing`. Services (a CI runner) need a restart.
- **GStreamer's official MSVC installer ships PyGObject** in its own
  `lib/site-packages`; `examples/gst_appsink.py` finds it through
  `GSTREAMER_1_0_ROOT_MSVC_X86_64`. Live receivers are stopped with Ctrl+Break,
  not SIGINT (`examples/test_gstreamer.py`).
- **Windows `tar -xjf` fails on the sherpa-onnx `.tar.bz2` models** (it shells out
  to a bzip2 that breaks); extract with Python's `tarfile`.
- A 4 GB card does not fail when CUDA overcommits: Windows spills into shared
  system memory and the run silently slows. Watch `max_memory_allocated`.

## VRAM budgeting

Two things dominate, and both surprise people. **Measure both.**

### Precision

Read the safetensors header rather than believing the model card:

```python
import json, struct
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
# per-tensor: dtype, shape -> params and bytes/param
```

A model shipped F32 costs 4 bytes/param; the same weights at bf16 cost 2. An
11 B model can therefore be 44 GB or 22 GB depending only on how it was
packaged. A stated "80 GB minimum" is often a packaging choice, not a property.

### Preallocated KV cache

vLLM reserves aggressively — frequently a third or more of resident VRAM is KV
pool, not weights, and it is tunable via `gpu_memory_utilization`. Budget
weights and pool separately, and confirm with a per-process query:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

The per-process query is the useful one — it shows *which stage* holds what,
and it is how you discover that a container you forgot about is holding 73 GB.

## Cloud-GPU deployment recipe

Verified on Ubuntu 24.04, driver 595.x, glibc 2.39.

```bash
# free the GPUs first -- a resident inference server may hold both cards,
# AND a container started by a DIFFERENT runtime may be invisible to the
# tooling you normally use to stop things. Check both:
podman ps ; sudo docker ps
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# Docker CE from the vendor repo, then:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# prove GPU passthrough BEFORE trusting the config write
sudo docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi -L

# nvcr.io auth (username is the literal string $oauthtoken)
tr -d '\n' < ~/.ngc_key | sudo docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
```

The NGC API key and HF token come from the operator's secret store; the HF token
lands in `~/.cache/huggingface/token` on the host and survives reinstalls of the
`hf` CLI. **No credential belongs in this repository.**

### Known rough edges

- **Two container runtimes on one box will hide work from each other.** A
  server started under Docker is invisible to podman-based tooling and to
  `systemctl --user`, so a "stop everything" command can leave 73 GB resident
  and the next server fails with an unhelpful free-memory error. Always check
  `nvidia-smi --query-compute-apps` rather than trusting a stop command.
- **`nvidia-container-toolkit` metapackage can be held/broken.**
  `nvidia-container-toolkit-base` 1.18.2-1 is sufficient — verified by the
  smoke test above. Newer versions emit a CDI field older podman cannot parse.
- **`hf` can become a dangling symlink** into a removed `uv` tool dir. Symptom
  is confusing: `ls` shows the binary, `command -v` finds nothing, and running
  it reports "No such file or directory" *about its target*. Fix:
  `uv tool install --force --with hf_transfer huggingface_hub`.
- **`huggingface_hub` v1.x dropped the `[cli,hf_transfer]` extras**;
  `HF_HUB_ENABLE_HF_TRANSFER` is deprecated for `HF_XET_HIGH_PERFORMANCE`.

## VM lifecycle

- **It can go TERMINATED without you** if the instance is preemptible/spot.
  Everything on the boot disk survives — models, images, installed packages,
  registry auth. Only VRAM and running containers are lost.
- **Keep weights on the persistent boot disk, never a scratch/local-SSD mount**,
  which is wiped on every stop.
- **The external IP may be ephemeral.** Verify after every start. This is why
  no host is hardcoded in the harness: `PLAYGROUND_HOST`/`PLAYGROUND_PORT` come
  from the environment, so a new address is a change there, not in the code.
- Launch containers with a restart policy so they return with the daemon — but
  confirm with `ps` rather than assuming.
- **Cold start is not instant.** Budget ~6 minutes for a large model: load, then
  warmup, and *the port only opens after warmup*. An early connection refusal is
  not a fault. A 65 GiB BF16 model on 2× H100 took ~4 minutes from unit start to
  a responding `/v1/models`.

## SSH gotchas

- **`ssh -n host 'bash -s' <<EOF` runs an empty script.** `-n` redirects stdin
  from `/dev/null`, so the heredoc never arrives. Exit 0, no output.
- **You cannot pipe a secret *and* a heredoc** into one ssh — one stdin.
  Transfer the secret to a `0600` file first, then run a script that reads it.
- **`pgrep -f "<string>"`** matches the wrapper process containing that string,
  so poll loops can report work that already finished.
- **Nested quoting through `ssh "... --data '{...}'"` silently mangles JSON.**
  A request can arrive structurally valid and semantically empty. Write the
  payload to a file with a quoted heredoc and use `curl --data @file`.
