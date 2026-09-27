# Results: CPU and memory for live streams on ONNX

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

This article answers a sizing question: how much CPU and memory do live
transcription streams need, when they all run in one process on a small
cloud VM? It measures NVIDIA's original `parakeet-tdt-0.6b-v3`, the checkpoint
Redux was compressed from, running on ONNX through sherpa-onnx. Redux itself
cannot run there: it needs Photon's kernels ([licensing](02-licensing.md)).
Photon's own figures for the same question are at the end, for comparison.

The short answer, for a 2-vCPU VM with the fp32 export, which is the one to
deploy ([accuracy](11-results-telephone-accuracy.md#the-int8-export-measured)):
one stream uses a small fraction of the machine, five streams of finished
sentences fit with multi-second delays when several people talk at once, and
ten or more do not fit. Somewhere around three should fit comfortably, which
is an estimate between the measured levels. Memory is about
3.2 GB for fp32 (1.8 GB for int8) and hardly changes with the number of
streams.

Most of this article measures the int8 export on an i9-12900KS, because it
was written before int8 was found to fail on telephone audio. The same test
was then repeated with both exports on a second, slower CPU (a Ryzen 7 3700X),
and the cloud estimates use that fp32 measurement.

## What was run

| | |
|---|---|
| **Model** | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`, NVIDIA's `parakeet-tdt-0.6b-v3` exported to int8 ONNX by the sherpa-onnx project; the fp32 export in [the second-CPU section](#fp32-and-int8-on-a-second-cpu-measured) |
| **Runtime** | `sherpa-onnx` 1.13.8, installed from PyPI, which bundles its own ONNX Runtime |
| **Probe** | [`probes/onnx_concurrency.py`](../probes/onnx_concurrency.py) |
| **Machine** | The desktop CPU in [hardware and throughput](06-results-throughput.md#hardware-and-software): i9-12900KS, no AVX-512 |
| **Process** | One Python process, one shared recogniser, decodes on a thread pool, one decode thread each |

Parakeet-TDT is an offline model: it transcribes a finished piece of audio.
sherpa-onnx has no streaming mode for it, so live use means cutting the stream
into utterances with a VAD (Silero, one per stream) and decoding each
utterance once when the speaker pauses. The probe measures two ways of doing
that:

- **Finals only.** Each utterance is decoded once, after 0.5 s of silence.
  You get whole sentences, each shortly after it ends.
- **Finals plus partials.** As above, and every 2.0 s the utterance still in
  progress is decoded again, so a viewer sees a draft of it. This is the
  nearest equivalent to Photon's 2 s streaming snapshots, and it costs more
  CPU because the same audio is decoded several times.

Every stream is fed at real-time speed through the repository's wall-clock
pacer, stream starts are spread over 2.0 s, and CPU is the process's total CPU
time (all threads) divided by wall time. So "1.00 cores" means one core kept
busy for the whole run.

### How much of the audio is speech

A VAD skips silence, and nothing is decoded until someone speaks, so CPU cost
follows how much of the audio is speech.
[`probes/speech_density.py`](../probes/speech_density.py)
([`results/speech_density.log`](../results/speech_density.log)) measured it
for the fixtures [MEASURED]:

| Channel | Speech | Longest utterance |
|---|---|---|
| `turn_taking` ch0, one side of a short scripted call | 14 % | 2.1 s |
| `tool_call` ch0 / ch1 | 24 % / 48 % | 7.1 s / 13.2 s |
| Map Task channels, unscripted conversation, 8 channels | 12 % to 76 % | up to 16.9 s |
| Map Task `q2ec3` ch0, first 60 s, the sizing run | 81 % | 7.2 s |

The sizing figures below come from that last slice, where someone is talking
81 % of the time. That is close to a worst case. One side of a real
two-party conversation will usually be well below it, and will cost
proportionally less.

## Measured on a 2-vCPU equivalent [MEASURED]

A cloud vCPU is one hyperthread, so a 2-vCPU VM is one physical core and its
hyperthread sibling. The probe was pinned to exactly that (`taskset -c 0,1`,
which on this CPU are the two hyperthreads of one core) with two decode workers, and run on the
81 %-speech slice.
Raw output: [`results/onnx_concurrency-dense-2cpu.log`](../results/onnx_concurrency-dense-2cpu.log).

Latency is from the end of an utterance's audio to its text. About 0.5 s of it
is the VAD waiting to be sure the speaker has stopped.

| Streams | Finals: CPU (of 2) | Finals: latency median / p95 | Finals + partials: CPU (of 2) | Finals + partials: latency median / p95 |
|---|---|---|---|---|
| 1 | 0.09 | 0.73 s / 1.04 s | 0.23 | 0.76 s / 1.22 s |
| 5 | 0.58 | 0.74 s / 1.75 s | 1.32 | 1.21 s / 3.80 s |
| 10 | 1.32, and 1.98 in the median second | 2.46 s / 4.51 s | 1.89 | 8.11 s / 10.14 s |
| 25 | 1.89 | 28.62 s / 49.80 s | 1.96 | 39.52 s / 65.10 s |

At 10 streams the two threads were already full for most of the run. At 25 the
work queued: 60.0 s of audio took 119.59 s to get through with finals only and
135.91 s with partials, and the median utterance waited 27.406 s for a free
worker. Each decode still took about 0.5 s; the delay was all queueing.

Overload made results late and left them unchanged: at every level, every
stream's transcript was identical to the single-stream transcript.

With partials, a partial never waits behind another partial from the same
stream; a tick that finds one still running is skipped. At 1 stream none were
skipped and a partial arrived a median 0.25 s after its tick. At 10 streams
133 ticks were skipped and the partials that did arrive were a median 6.59 s
old, which makes them useless as a live preview.

Per stream, on this core, with 81 % speech:

| | CPU per stream |
|---|---|
| Finals only | 0.091 cores (1 stream), 0.116 cores (5 streams) |
| Finals + partials | 0.229 cores (1 stream), 0.264 cores (5 streams) |

### Memory [MEASURED]

| Point in the run | Resident memory (RSS) |
|---|---|
| Recogniser loaded, before any audio | 977 MB |
| After the first 1-stream level | 1787 MB |
| After 25 streams, finals only | 1793 MB |
| Peak for the whole run | 1805 MB |

Memory is a fixed cost of about 1.8 GB for the model and the runtime's
buffers. Each extra stream adds almost nothing, because a stream only holds its
VAD state and the audio of the utterance in progress. This was measured over
60 s streams, and hours-long streams were not tried.

### The same probe with the whole machine, on sparse audio [MEASURED]

Run unpinned with 16 decode workers on `turn_taking` ch0 (14 % speech), the
same process handled up to 100 streams without falling behind
([`results/onnx_concurrency.log`](../results/onnx_concurrency.log)):

| Streams | Finals: CPU | Finals: latency median / p95 | Finals + partials: CPU |
|---|---|---|---|
| 1 | 0.02 | 0.59 s / 0.71 s | 0.04 |
| 25 | 0.52 | 0.60 s / 0.74 s | 0.80 |
| 50 | 1.48 | 0.73 s / 1.01 s | 2.10 |
| 100 | 3.17 | 0.66 s / 1.44 s | not run |

Per stream that is 0.021 to 0.032 cores, against 0.091 to 0.116 cores on the
81 % slice. That ratio, 0.18 to 0.35, is close to the ratio of speech
(14 % against 81 %, 0.17), which supports scaling the dense figures by how much of
your audio is speech. At 100 streams the event loop's p99 lag reached
118.2 ms, so Python itself starts to show at that count even with CPU to
spare.

## fp32 and int8 on a second CPU [MEASURED]

The 2-vCPU test was repeated on a Ryzen 7 3700X (Zen 2, AVX2, no AVX-VNNI or
AVX-512), pinned to CPUs 2 and 10, which are one core and its SMT sibling,
with Silero VAD as before and both exports
([`results/onnx_concurrency-dense-2cpu-zen2-int8.log`](../results/onnx_concurrency-dense-2cpu-zen2-int8.log),
[`results/onnx_concurrency-dense-2cpu-zen2-fp32.log`](../results/onnx_concurrency-dense-2cpu-zen2-fp32.log)):

| Streams | int8: CPU (of 2) | int8: latency median / p95 | fp32: CPU (of 2) | fp32: latency median / p95 |
|---|---|---|---|---|
| 1 | 0.15 | 0.89 s / 1.36 s | 0.16 | 0.96 s / 1.49 s |
| 5 | 1.05 | 1.82 s / 3.47 s | 1.26 | 2.65 s / 4.58 s |
| 10 | 1.80 | 11.14 s / 17.23 s | 1.85 | 17.88 s / 29.94 s |
| 25 | 1.98 | 57.91 s / 110.35 s | 1.98 | 75.62 s / 144.67 s |

Finished sentences only. With partials every 2 s, one fp32 stream used 0.42
cores and five already fell behind (median 6.97 s).

- fp32 costs little more CPU than int8: 0.163 against 0.147 cores for one
  stream, 0.251 against 0.211 at five. On this CPU, which has no AVX-VNNI,
  fp32 is even faster for a single thread (5.2× real time against 4.5×).
- Where fp32 costs more is memory: 2,256 MB after loading and 3,011 to
  3,238 MB while decoding, against 965 MB and 1,776 to 1,820 MB for int8.
- A Zen 2 thread did 1.6 to 1.9 times less work than an i9-12900KS thread
  (0.147 against 0.091 cores per int8 stream; 4.5× against 8.6× real time
  single-threaded), so it saturates sooner: at five streams its latency is
  already a few seconds.
- Every transcript at every level was identical to the single-stream one, on
  both exports.

## Estimates for example cloud VMs

Nothing in this section was run on a cloud VM: these figures are
**estimates**, for the fp32 export. An earlier version of this section scaled
the i9's int8 figures by an assumed 1.5 to 2 times less work per GCP N2 vCPU.
The Zen 2 thread above measured 1.6 to 1.9 times slower than the i9's, inside
that range, so the estimates now use its measured fp32 figures directly, on
the assumption that an N2 vCPU does about as much work as that Zen 2 thread.
That assumption is still mine and still the weakest number here. N2 VMs have
AVX-512, which the Zen 2 machine lacks, so they may do better.

Average CPU on an N2 VM for fp32 streams with 81 % speech (0.163 to 0.251
vCPU per stream for finished sentences, 0.374 to 0.421 with partials):

| Streams | Finals only | Finals + partials |
|---|---|---|
| 1 | 0.16 to 0.25 vCPU | 0.37 to 0.42 vCPU |
| 5 | 0.8 to 1.3 vCPU | 1.9 to 2.1 vCPU |
| 25 | 4.1 to 6.3 vCPU | 9.4 to 10.5 vCPU |

A VM sized to the average will still fall behind in bursts. In the i9 2-vCPU
run, latency stayed low while the process averaged 29 % of its threads (5 streams, finals) and degraded at 66 %
(10 streams, finals); on the Zen 2 machine, five fp32 streams at 63 % already
had a p95 latency of 4.58 s. The probe starts every stream within 2 s on identical
audio, so their utterances end together and the load comes in bursts; mixed
real traffic should be smoother. Sizing so the average is a third to a half of
the VM's vCPUs gives these picks (GCP machine shapes, memory as published for
each shape):

| Streams | Mode | VM | vCPU / memory | Expected fit |
|---|---|---|---|---|
| 1 | either | `n2-standard-2` | 2 / 8 GB | Comfortable |
| 1 to 3 | finals only | `n2-standard-2` | 2 / 8 GB | Comfortable |
| 5 | finals only | `n2-standard-2` | 2 / 8 GB | Tight: p95 latency of several seconds when streams speak at once |
| 5 | finals only | `n2-standard-4` | 4 / 16 GB | Comfortable |
| 5 | finals + partials | `n2-standard-8` | 8 / 32 GB | Comfortable; 2 vCPU cannot keep up and 4 vCPU is marginal |
| 25 | finals only | `n2-highcpu-8` or `n2-standard-8` | 8 / 8 or 32 GB | Tight: covers the average, and bursts will queue |
| 25 | finals only | `n2-highcpu-16` | 16 / 16 GB | Comfortable |
| 25 | finals + partials | `n2-highcpu-32` | 32 / 32 GB | Comfortable; 16 vCPU is marginal |

For memory, the fp32 process (up to 3.2 GB) fits every shape above. It does
not fit an `n2-highcpu-2` (2 vCPU, 2 GB), and an `n2-highcpu-4` (4 GB) leaves
little for anything else. The int8 process (1.8 GB) is smaller, but int8 is
not an option for telephone audio.

For audio with less speech, scale the CPU column down in proportion. A
channel that is 40 % speech needs roughly half the vCPUs above.

## All CPU, and which CPU features

Everything here ran on the CPU only. The test machine has AVX2, FMA, F16C and
AVX-VNNI and no AVX-512 ([hardware](06-results-throughput.md#hardware-and-software)),
and the probe passes no special flags: sherpa-onnx picks its own kernels.
What the ONNX Runtime bundled with sherpa-onnx requires as a minimum was not
tested. The second machine, a Ryzen 7 3700X, has AVX2 and FMA but no
AVX-VNNI or AVX-512, and both exports ran on it; fp32 was the faster of the
two there for a single thread. Whether AVX-512 makes either export faster is
Open, and is the main reason the cloud estimates above might be pessimistic.

## Transcripts

On `turn_taking` ch0, decoding the VAD's utterances one at a time produced
exactly the text Photon's Redux produced on the same channel ("Hi. Do you know
any good cookie recipes? Ooh, maybe a peanut butter one. Yes, please. Thanks
for your help."). Decoding the whole 41.1 s channel as one piece, from the
same 16 kHz audio, did worse: "Hi. Do you know any good cooking recipes maybe a
peanut butter one, please? Thanks for helping." (both in
[`results/onnx_concurrency.log`](../results/onnx_concurrency.log)). So cutting at
pauses helped the text here, on top of being what makes live use possible.
This is one channel, and no word error rate was computed.

## For comparison: Redux on Photon [MEASURED]

[`probes/concurrency.py`](../probes/concurrency.py)
([`results/concurrency.log`](../results/concurrency.log)) ran the same kind of
test on Redux through Photon's own streaming API, on `turn_taking` ch0, on the
whole unpinned machine:

| Streams | CPU | Snapshot interval median / max | Final result after audio ends, median |
|---|---|---|---|
| 1 | 0.47 | 2.01 s / 4.14 s | 1.82 s |
| 5 | 2.31 | 2.01 s / 4.30 s | 2.17 s |
| 10 | 4.18 | 2.02 s / 6.31 s | 5.73 s |
| 25 | 5.17 | 4.75 s / 16.47 s | 11.57 s |

On the same audio, one Photon stream used 0.47 cores against 0.02 for one
ONNX stream with finals only. The difference comes from how the streaming
works: Photon transcribes the audio again for every 2 s snapshot, whether or
not anyone is speaking, while the VAD approach decodes each utterance once and
skips silence.

Photon also stopped keeping up at 25 streams while using only 5.17 of the
machine's 24 threads: 41.1 s of audio took 107.88 s. The event loop was not the
cause (its worst lag was 3.4 ms); Photon took audio from the streams more
slowly than real time. Where that limit lives inside Photon is not visible
from outside. As everywhere, all 25 transcripts matched the single-stream one.

## What was not exercised

- Any cloud VM. The cloud figures are scaled from a desktop core by an assumed
  ratio.
- More than two decode workers on dense speech. The 16-worker run used sparse
  audio.
- Streams longer than 60 s, and streams carrying different audio.
- The fp32 ONNX export, and any ONNX export of Ultra or Redux.
- Licensing of the ONNX path. NVIDIA's weights are CC-BY-4.0
  ([licensing](02-licensing.md#the-weights)), but the sherpa-onnx package and
  the converted model archive were not put through `licensing.py`. Open.

Each level ran once. Treat the figures as good to within a factor of about two,
which is enough to choose a VM size and not enough to promise a latency.

---

Previous: [Method and open questions](09-method.md) | [Contents](../README.md#contents) | Next: [Results: accuracy on telephone speech](11-results-telephone-accuracy.md)
