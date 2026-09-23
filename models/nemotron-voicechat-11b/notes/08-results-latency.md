# Results: latency, headroom, VRAM

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).

## Compute headroom [MEASURED]

The container grades its own work per audio chunk. These figures were read from
its logs during this sweep (`probes/inference_budget.py`):

```
samples      : 89
budget       : 160 ms
per-chunk ms : min 60.7  p50 66.1  p95 68.9  max 71.3
mean         : 65.9 ms  (stdev 2.1)
headroom     : 2.42x real-time at p50
over budget  : 0 / 89 chunks     queue depth: max 0
```

This is the server timing itself, so it excludes the network, the client and
the harness. Nothing missed its real-time budget and the queue never grew.

This is single-session headroom. `config.pbtxt` declares
`max_batch_size: 8` with one GPU instance, so reaching 8 concurrent sessions
requires batching to recover most of the remaining 2.42×. Whether it does is
untested ([open questions](12-open-questions.md)).

## Round trip and injection accuracy

Carried forward from the first pass and still true:

- `generate_random_number(1,50)` → local handler returns a fixed 25 → agent
  speaks "twenty-five". `convert_currency(25,USD,EUR)` → 25 × 0.92 → agent
  speaks "twenty-three point zero zero EUR".
- Injected latency is accurate: 0.1005 s and 1.5009 s against 0.1 / 1.5
  configured. This shows the injection works and is not a model measurement.
- The model did not hallucinate an unregistered weather tool. It declined and
  suggested an external source. This was measured at 1.5 s latency, below the
  threshold at which [barge-in results](07-results-barge-in.md) shows fabrication
  beginning, so it should not be quoted as a general property.

## Turn-taking latency is quantised [MEASURED]

NVIDIA's positioning claims ~450 ms. This is the first time the claim has been
testable at all ([failures](09-failures.md)). Probe: `probes/turn_taking.py`, measuring
`caller.speech_stopped` → first agent audio chunk whose RMS clears the silence
floor, over 36 turns across 6 sessions.

The first attempt reported a median and looked noisy: p50 0.636 s in one
sitting, 0.486 s in another, same configuration. Plotting the raw values showed
that the values are discrete, so the spread was not noise:

| multiple of the 160 ms budget | nominal | measured range |
|---|---|---|
| 3 chunks | 0.480 s | 0.467 to 0.492 |
| 4 chunks | 0.640 s | 0.632 to 0.653 |
| 5 chunks | 0.800 s | 0.794 to 0.804 |

Every one of 36 turns falls within ±13 ms of an integer multiple of the
server's own 160 ms inference budget (above). The model responds after a whole
number of processing chunks instead of after a continuously distributed delay.
The apparent variance between sittings was a different mix of 3-chunk and
4-chunk responses.

This has two consequences:

- The right statistic is the histogram. Any p50 quoted for this model is an
  artefact of which mode happened to dominate that run, and will not
  reproduce. This note previously would have quoted 0.636 s.
- ~450 ms is below the floor. The fastest possible response is 3 chunks =
  480 ms, so a 450 ms figure is not reachable in this configuration. It is
  either measured from a different reference point than end-of-caller-speech,
  measured with a different chunk size, or aspirational, and nothing here can
  tell those apart.

> Caveat, because it bounds everything above: `speech_stopped` is the
> server's voice-activity decision, not ground-truth end of speech, so these
> figures include whatever VAD hangover the container applies. They faithfully
> measure client-observable turn-taking, which is what a caller experiences,
> but they are an upper bound on the model's own compute latency.

## How small a card: 39% less VRAM, with a cost in the tail

[requirements](01-requirements.md) predicted that `LLM_GPU_MEM_UTIL` /
`TTS_GPU_MEM_UTIL` set the ~73 GB footprint, not the model. This was tested directly
(`probes/vram_floor.sh`) with the same fixture and the same three-session protocol:

| config | GPU total | LLM engine | TTS engine | starts | serves |
|---|---|---|---|---|---|
| stock `0.45 / 0.40` | 73,308 MiB | 37,352 | 31,046 | ✓ | ✓ |
| **`0.30 / 0.20`** | **44,716 MiB** | 25,192 | 14,830 | ✓ | ✓ |

The reduced configuration saves 28.6 GB, a 39% reduction, and the model still
serves complete sessions. 44.7 GiB fits comfortably on a 48 GB card, which the
stock configuration would have ruled out on a spec sheet.

The cost shows up as a new mode, which the median hides:

| config | 3 chunks (0.48 s) | 4 chunks (0.64 s) | 5 chunks (0.80 s) |
|---|---|---|---|
| stock | 10 / 18 (56%) | 8 / 18 (44%) | none |
| lowmem | 8 / 18 (44%) | 8 / 18 (44%) | **2 / 18 (11%)** |

The stock configuration never produced a 5-chunk response across 18 turns; the
reduced one produced two. That is the KV-cache pressure the probe was written
to look for. It appears as an occasional extra 160 ms quantum rather than as
the gradual slowdown a mean would catch. A median-based
comparison of these two configurations reports them identical (0.486 s vs
0.489 s) and misses it completely.

> Open: 18 turns per configuration is enough to see the mode appear but not
> enough to pin its frequency. The floor is not established either: 0.30/0.20
> is the first step tried, not the lowest that works. Resident weights are
> ~19 GB plus ~4 GB, so there is arithmetic room below this.

## What is still not measured

On-hold message behaviour, missing-argument handling with a genuinely absent
argument, tool-error recovery, `dict` vs `object`, mid-session tool swap, and
concurrency. See [open questions](12-open-questions.md).

---

Previous: [Results: barge-in under slow tools](07-results-barge-in.md) | [Contents](../README.md#contents) | Next: [Failures worth recording](09-failures.md)
