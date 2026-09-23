# Setting it up

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).

Target: the cloud GPU node, a 2× NVIDIA H100 80GB HBM3 machine with NVLink,
running Ubuntu 24.04.

## Prerequisites (all met)

| Requirement | the node | Notes |
|---|---|---|
| Driver ≥ 580 | **595.71.05** | ✓ |
| glibc ≥ 2.35 | **2.39** | ✓ |
| x86_64 | ✓ | |
| ≥ 80 GB VRAM | 81,559 MiB | ✓ but only 1.5 GB over the floor |
| Docker + NVIDIA Container Toolkit | installed during this work | |

## Steps that worked

```bash
# 1. Free the GPUs (a Qwen vLLM server held 72.7 GB on BOTH cards, TP2)
# stop whatever else is holding the GPUs first

# 2. Repair the hf CLI (see below)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --force --with hf_transfer huggingface_hub

# 3. Docker CE
#    standard docker.com apt repo for noble

# 4. NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. NGC auth
tr -d '\n' < ~/.ngc_key | sudo docker login nvcr.io \
  --username '$oauthtoken' --password-stdin

# 6. Pull container (34.4 GB) and model (43.69 GB)
sudo docker pull nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
ngc registry model download-version nim/nvidia/nemotron-labs-voicechat:1.0.0 \
  --dest /opt/models

# 7. Launch
sudo docker run -d --name=voicechat --restart=unless-stopped \
  --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 -p 9000:9000 \
  -v /opt/models/nemotron-labs-voicechat_v1.0.0:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Endpoint: `ws://$PLAYGROUND_HOST:$PLAYGROUND_PORT/v1/realtime`. It was
reachable from the workstation because the GCP firewall already permitted 9000.

## Rocks in the path

### `hf` was a dangling symlink

```
~/.local/bin/hf -> ~/.local/share/uv/tools/huggingface-hub/bin/hf   [missing]
```

A previous `uv` tool install had been removed and left the symlink behind. The
symptoms were confusing: `ls ~/.local/bin` showed `hf`, but `command -v hf`
found nothing, and `~/.local/bin/hf --version` reported *"No such file or
directory"* about the target, not the link.

`uv tool install` then refused with `Executable already exists: hf` and needed
`--force` to overwrite the dead link. The HF token at
`~/.cache/huggingface/token` had survived, so auth was intact once the binary
was restored.

### `nvidia-container-toolkit` metapackage is held/broken

```
E: Unable to correct problems, you have held broken packages.
```

But `nvidia-container-toolkit-base` 1.18.2-1 *was* installed and is sufficient.
A test container confirmed it:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi -L
# GPU 0: NVIDIA H100 80GB HBM3 (UUID: ...)
# GPU 1: NVIDIA H100 80GB HBM3 (UUID: ...)
```

The root cause is unresolved, and it may resurface on upgrade.

### `ssh -n` eats heredoc stdin

`ssh -n host 'bash -s' <<'EOF'` silently runs an empty script. `-n` redirects
stdin from `/dev/null`, so `bash -s` receives nothing. It exits 0 with no
output, which looks like success.

### `pgrep -f` matches its own command line

```bash
pgrep -f "hf download" && echo "downloading"
```
kept reporting "still downloading" after completion, because the `bash -c`
wrapper containing the string is itself a process that matches. It cost one
confused check.

## Startup sequence and timing

A cold start takes ~6 minutes:

```
Starting Triton server...
[TRITON] loading: nemotron-voicechat:1          # ~5 min
[TRITON] Started GRPCInferenceService at 0.0.0.0:8001
[TRITON] Started HTTPService at 0.0.0.0:8000
Triton server is ready.
Run inference for warm up...                    # then port 9000 opens
```

Port 9000 opens only after warmup has run, which is later than Triton reporting
ready.

### Warmup does not cover everything

```
WARNING [jit_monitor.py:106] Triton kernel JIT compilation during inference:
_zero_kv_blocks_kernel. This causes a latency spike; consider extending warmup
to cover this shape/config.
```

The first turns of a session may therefore carry an unrepresentative JIT
compilation spike. For latency measurement, discard the first exchange or run a
throwaway session before a measured one.

## GPU memory, measured

73,172 MiB (71.5 GiB) on GPU 0, at rest with no session streaming. GPU 1 is
completely free, and utilization is 0% at idle.

Per-process breakdown:

| Process | Memory | What |
|---|---|---|
| `VLLM::EngineCore` | 37,352 MiB | LLM backbone (`nano-v2-vllm`) + KV cache |
| `VLLM::EngineCore` | 31,046 MiB | TTS decoder (`eartts_vllm`) + KV cache |
| `triton_python_backend_stub` | 4,164 MiB | perception / RNNT / codec |
| `tritonserver` | 582 MiB | server overhead |

There are two separate vLLM engines, one for the LLM and one for the TTS
decoder, so each pipeline stage can be driven independently, which streaming
requires.

For scale, 11.1 B params at F32 is ~44 GB of weights, and the remaining ~28 GB
is KV cache that the two engines preallocate. The 31 GB attributed to a
0.834 B TTS model is almost entirely cache, so `gpu_memory_utilization` can be
lowered to fit something alongside, at the cost of concurrent session
capacity.

### Where the 80 GB requirement comes from

The weights are stored as `F32`, verified from the safetensors headers: 1626 of
1632 tensors in the HF checkpoint, and the floating-point tensors in the NIM
components checked. There is no fp16, bf16 or quantization in either download,
so on disk it takes 4 bytes per parameter. An 11 B model at bf16 would be
~22 GB of weights; stored at float32 it is ~44 GB.

The first version of this section concluded that the 80 GB floor was mostly an
artifact of those F32 weights plus generous KV preallocation, and left open
whether NIM exposed precision or `gpu_memory_utilization`. Reading the
container's loader answered it, and changed the conclusion. The runtime already
casts the LLM to bf16, and the footprint is set by two environment variables,
`LLM_GPU_MEM_UTIL` and `TTS_GPU_MEM_UTIL`, which preallocate a fixed fraction
of the card regardless of model size. Lowering them ran the model at
44,716 MiB. The details and the cost are in [requirements](01-requirements.md)
and [latency results](08-results-latency.md).

Self-hosting through the Apache-2.0 `speechlm2` code (see
[licensing](02-licensing.md)) still gives full control over precision and cache
sizing, and avoids the container's per-GPU licensing question.

Consequences as shipped:

- One H100 80GB per instance, with ~8 GB headroom against the stated 80 GB
  minimum
- A second instance fits on GPU 1 and is useful for A/B runs against
  different instructions or tool sets
- At the stock memory settings an A100 40GB will not work, despite "A100"
  appearing in the compatibility list. With the lowered settings the model used
  44,716 MiB, which is still more than 40 GB; nothing below that was tested

## Final resource state

- Container image: 34.4 GB
- Model on GPU 0: 73,172 MiB; GPU 1 idle and available
- Disk: 148 G used of 1.9 T (both model copies plus the image)

---

Previous: [Architecture and distributions](03-architecture.md) | [Contents](../README.md#contents) | Next: [The protocol](05-protocol.md)
