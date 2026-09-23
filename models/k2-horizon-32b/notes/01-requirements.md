# Requirements and deployment

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

## Why BF16 and not NVFP4

IFM publishes the 32B in four forms: BF16 (used here), FP8, NVFP4 and GGUF
(`results/hub-availability.log`). The NVFP4 card says:

> **Serving note**: Requires NVIDIA Blackwell-generation GPUs (B-series) or
> newer with native NVFP4 support.

The available node is 2× H100 (Hopper, SM90), and no pre-Blackwell NVIDIA GPU
has FP4 tensor cores, so NVFP4 is ruled out on this hardware. An earlier
version of this note added that vLLM falls back to a Marlin kernel on SM90
with open upstream reports of garbage output; a search of the vLLM issue
tracker did not find those reports, so that claim is withdrawn.

On this node BF16 costs nothing extra. At TP=2 the 64.78 GiB of weights sit in
about 159 GiB of VRAM (2 × 81,559 MiB), leaving 37.11 GiB per GPU for the KV cache,
and the NVFP4 card's own table scores BF16 higher anyway (89.5 vs 88.4
average) [CLAIM]. A quantised format only helps on GPUs whose tensor cores can
execute it, so check the GPU generation before downloading anything.

H100s do have FP8 tensor cores, so IFM's FP8 build runs here, and it turned out
to be the best option for a two-GPU machine: 101.4 tok/s decode against 66.6
for BF16 on the same two GPUs, the same score on the answer and needle checks,
and tool calls at 8 of 10 on low effort against 9 or 10 for BF16. The other
option, an FP8 KV cache, fits the full 524,288-token window on two GPUs. Both
are compared with BF16 in [results](04-results.md).

## Deployment

### Hardware and software

Most measurements used two GPUs with a 131,072-token limit. The node was later
given four GPUs to serve the full native context.

| | Two GPUs | Four GPUs |
|---|---|---|
| GPUs | 2 × NVIDIA H100 80GB HBM3, NVLink | 4 × NVIDIA H100 80GB HBM3 |
| Parallelism | tensor-parallel size 2 | tensor-parallel size 4 |
| Context served | 131,072 (our choice) | 524,288, the full native window |
| KV cache | 37.11 GiB per GPU, 303,968 tokens | 875,872 tokens |
| Full-length requests the cache can hold at once | 2.32 at 131,072 tokens | 1.67 at 524,288 tokens |
| Resident VRAM | 73,843 MiB per GPU | not recorded |
| Decode, one request | 66.6 tok/s | 108.7 tok/s |
| Interconnect in use | FlashInfer allreduce, `backend=mnnvl` | same engine |
| Warm start | 48 s (weights load 5.27 s, engine init 11.89 s) | not timed |

Common to both: vLLM 0.30.0 from the image `docker.io/vllm/vllm-openai:v0.30.0`,
run under podman at `--gpu-memory-utilization 0.90`, on Ubuntu 24.04.4 LTS with
NVIDIA driver 595.71.05.

Sources: `results/startup.log`, `results/node.log`, `results/weights.log` and
`results/serve-512k-check.log`, with decode figures from `results/k2-bench.log`
and `results/k2-bench-tp4.log`. The first, cold start (about 4 minutes
including compilation) and the 3 m 45 s weight download were seen during the
original deployment but not recorded.

### vLLM version is a hard floor

Model support landed in vLLM pull request #55063, merged on 2026-09-03. The
v0.29.0 release (2026-09-09) was cut from a branch that does not include it, so
v0.30.0 (2026-09-22) is the first release that can load this model. The
startup log line `Resolved architecture: K2HorizonForCausalLM` confirms it
loads through native support, with no fallback.

### Configuration deliberately not inherited

The node already ran a different model under vLLM with a tuned serve command:
FP8 KV cache, DeepGEMM environment variables, a hybrid-attention cache mode,
and YaRN rope overrides. None of it was copied. Those settings were tuned
against an FP8 hybrid-attention model, while this one is dense BF16 and never
touches DeepGEMM, so nobody has evaluated them for this model.

---

[Contents](../README.md#contents) | Next: [Using it](02-using-it.md)
