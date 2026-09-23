# Requirements and runtime

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


## The short version

| | |
|---|---|
| Accelerator | 1 × NVIDIA H100 80GB (CUDA). Not tested on anything smaller, but see below: the 80 GB is mostly *configuration* |
| Measured residency | 73,172 MiB at rest, one GPU, single session |
| Cold start | ~6 min (≈5 min Triton load, then warmup). Port 9000 opens only after warmup |
| Wire format | 24 kHz mono PCM16, both directions, 80 ms / 3840-byte chunks |
| Designed concurrency | `max_batch_size: 8`, `instance_group { count: 1, kind: KIND_GPU }` |
| ROCm / CPU | No. NIM is a CUDA container, and there is no non-CUDA path |

## Where the 80 GB goes

An 11 B model that needs 80 GB looks like it must be storing its weights
inefficiently. The checkpoint *is* F32, verified directly from the safetensors
headers of the deployed artifact:

```
nano-v2-vllm/model.safetensors:  342 tensors, all F32
                                 9.4754 B params, 35.30 GiB, 4.00 bytes/param
```

F32 is not what is resident, though. 9.4754 B × 4 bytes = 37.9 GB, against a
measured LLM engine footprint of 37,352 MiB, so the weights would not fit. vLLM
casts to bf16 at load, per `nano-v2-vllm/config.json` (`"dtype": "bfloat16"`),
which gives ~19 GB of resident weights inside that budget.

Precision does not explain the footprint. Two environment variables in the
container's own loader do:

```python
# /opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py
LLM_GPU_MEM_UTIL = float(os.environ.get("LLM_GPU_MEM_UTIL", "0.45"))
TTS_GPU_MEM_UTIL = float(os.environ.get("TTS_GPU_MEM_UTIL", "0.4"))
```

Each of the two vLLM engines preallocates a fixed fraction of the card, however
large its model is. On an 80 GB H100 that predicts the measured census almost
exactly:

| Engine | Predicted | Measured | Δ |
|---|---|---|---|
| LLM (0.45 × 81,559 MiB) | 36,702 MiB | 37,352 MiB | +650 |
| TTS (0.40 × 81,559 MiB) | 32,624 MiB | 31,046 MiB | −1,578 |
| Triton python backend | | 4,164 MiB | |
| `tritonserver` | | 582 MiB | |
| **total** | **69,325 + 4,746** | **73,172 MiB** | within ~1% |

Three different numbers had been conflated. Kept apart, they are:

| Stage | Precision | Size |
|---|---|---|
| Distribution (on disk) | F32 | ~44 GB all components, 35.30 GiB for the LLM alone |
| Runtime (resident) | bf16 | ~19 GB LLM weights |
| Allocation (configured) | | 85% of the card, by default, irrespective of both |

The earlier version of this note said *"the 80 GB requirement is largely a
packaging choice: bf16 weights would be ~22 GB"*, which pointed at precision
as the lever. The runtime is already bf16, so precision was never the lever.
The lever is `LLM_GPU_MEM_UTIL` / `TTS_GPU_MEM_UTIL`, and they are ordinary
environment variables you can set on `docker run`.

Lowering them has been tested once. At `0.30` / `0.20` the model used
44,716 MiB and still served full sessions, with a cost that shows only in the
latency tail ([latency results](08-results-latency.md)).

> **Open:** how far below that it goes. The weights need ~19 GB plus ~4 GB TTS,
> and everything above that is elective KV cache, so a 40 to 48 GB card looks
> arithmetically feasible. See [open questions](12-open-questions.md).

---

[Contents](../README.md#contents) | Next: [Licensing](02-licensing.md)
