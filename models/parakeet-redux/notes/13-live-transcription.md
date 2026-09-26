# Live transcription: latency and accuracy

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

This article is about transcribing speech as it happens, for example a phone
call shown as text while the caller talks, where both delay and errors matter.
It measures where the delay comes from, what each setting trades between
delay and accuracy, how a CPU and a GPU compare, and what a true streaming
model changes. The setup it builds on is in
[deploying on ONNX](12-onnx-deployment.md).

Live use has two delays, and they differ:

- **Time to first words:** how long after someone starts speaking the first
  text appears.
- **Time to final:** how long after they stop speaking the finished, stable
  text of what they said is available. This is the one a downstream system
  (a voice agent, a search index, a supervisor's screen) usually waits for.

The main results, [MEASURED] on the hardware named in each section:

- With NVIDIA's `parakeet-tdt-0.6b-v3` (offline, fp32 ONNX) and a VAD, the
  final text of an utterance arrives a median 0.52 s after the speaker stops
  on an RTX 3090 Ti, for up to 50 simultaneous streams, and about 0.95 s on
  the same machine's CPU, for up to 10. Almost all of the GPU's 0.52 s is the
  0.5 s of silence the VAD waits for.
- That silence threshold is the main dial. Below 0.5 s accuracy falls
  quickly (1 to 2.3 points of WER at 0.2 s); above it there is little to
  gain.
- The offline model shows nothing until the speaker pauses. Drafts of the
  utterance in progress, re-decoded every 2 s, fill that gap at about 2.5
  times the compute.
- A true streaming model (NVIDIA's Nemotron streaming, through the same
  library, run as one continuous stream) shows words 0.53 to 0.72 s after
  speech starts, finishes an utterance about 0.2 s after it ends, and was more
  accurate than offline Parakeet on every accent tested, including through
  G.711. It costs several times the CPU, is English only, and has a different
  licence ([details](#a-streaming-model-nemotron-measured)).

## Where the delay comes from

For the offline model, an utterance becomes text in three steps, each with a
cost:

| Step | What happens | Delay |
|---|---|---|
| Wait for the pause | The VAD sees speech stop and waits for `min_silence` of quiet before it closes the segment | the silence setting, 0.5 s here |
| Queue | The segment waits for a free decoder | zero until the machine is saturated, then growing without limit |
| Decode | The model transcribes the whole utterance | a median 0.02 s on the RTX 3090 Ti; on the Ryzen 7 3700X a median 0.45 s, up to 1.37 s for the longest utterances |

Nothing reaches the screen before the first step ends, which is why the
offline model's time to first words and time to final are the same unless
drafts are added. A draft decodes the utterance in progress on a timer (every
2 s here) and replaces the previous draft; the final still comes after the
pause.

## Choosing the silence threshold [MEASURED]

The first 12 AppTek channels of each accent, cut with different silence
settings and decoded on the GPU with the fp32 export
([`results/nemo_vs_onnx-latency-sweep.log`](../results/nemo_vs_onnx-latency-sweep.log)).
The time to final is at least the silence setting.

| Silence before an utterance ends (0.25 s padding) | en-US_General | en-IN | en-GB_SCT |
|---|---|---|---|
| 0.2 s | 9.05 % | 12.33 % | 14.63 % |
| 0.3 s | 8.44 % | 11.31 % | 13.98 % |
| 0.5 s | 8.04 % | 10.06 % | 13.58 % |
| 0.8 s | 8.02 % | 9.41 % | 13.40 % |
| 1.0 s | 7.93 % | 9.36 % | 13.38 % |

A short threshold ends utterances at every hesitation, so sentences are
decoded in pieces without the context around them. From 0.5 s to 0.8 s
there is a small further gain on two accents (0.65 points on en-IN), and
1.0 s adds almost nothing beyond 0.8 s. For live use, 0.5 s is the sensible
low end and 0.8 s a reasonable choice where another 0.3 s of delay is
acceptable.

## Padding [MEASURED]

sherpa-onnx's VAD returns each segment without margin, which clips word
onsets ([accuracy](11-results-telephone-accuracy.md#segmentation-and-padding-measured)).
Padding is cut from audio already buffered, so it adds no delay. At 0.5 s of
silence, same log:

| Padding either side | en-US_General | en-IN | en-GB_SCT |
|---|---|---|---|
| none | 10.50 % | 11.22 % | 14.93 % |
| 0.1 s | 9.33 % | 10.11 % | 14.12 % |
| 0.25 s | 8.04 % | 10.06 % | 13.58 % |
| 0.5 s | 7.43 % | 10.14 % | 13.35 % |

(The unpadded row is from
[`results/nemo_vs_onnx-padding.log`](../results/nemo_vs_onnx-padding.log).)
0.1 s is too little. Between 0.25 s and 0.5 s the result is mixed: better on
two accents, level on en-IN. Either is a sound choice.

## CPU against GPU [MEASURED]

Both runs used the same machine (a Ryzen 7 3700X with an RTX 3090 Ti), the
fp32 export, Silero VAD v5 with 0.5 s of silence, and the same 60 s of
81 %-speech audio for every stream, with starts spread over 2 s
([`results/onnx_concurrency-dense-live-gpu-fp32.log`](../results/onnx_concurrency-dense-live-gpu-fp32.log),
[`results/onnx_concurrency-dense-live-cpu-fp32.log`](../results/onnx_concurrency-dense-live-cpu-fp32.log)).
The GPU run used 4 decode workers, the CPU run 8 on its 16 threads.

Time to final, from the end of an utterance's speech, median / p95:

| Streams | GPU | CPU |
|---|---|---|
| 1 | 0.67 s / 0.72 s | 0.94 s / 1.69 s |
| 5 | 0.52 s / 0.69 s | 0.97 s / 2.00 s |
| 10 | 0.52 s / 0.67 s | 1.05 s / 2.38 s |
| 25 | 0.52 s / 0.63 s | 5.32 s (falling behind) |
| 50 | 0.52 s / 0.86 s | 25.85 s (falling behind) |
| 100 | 10.23 s (falling behind) | not run |

- On the GPU the decode takes about 0.02 s, so the time to final is the VAD's
  0.5 s plus almost nothing, up to 50 streams.
- At 100 streams the GPU was not the limit: each decode still took about
  0.1 s, while the single Python process driving all the streams (their VADs,
  buffering and hand-off) used 2.5 cores. More processes per GPU should carry
  more streams; that was not tested.
- With one stream, GPU decodes took 0.19 s against 0.02 s at five or more,
  probably because each new segment length costs a one-time kernel setup
  when the GPU is otherwise idle. The effect is small and was not
  investigated.
- On the CPU a decode takes a median 0.45 s and up to 1.37 s for long
  utterances, which is what spreads its p95; its queue builds from about 10
  streams, and at 25 it cannot keep up.
- Every stream's transcript was identical to the single-stream transcript
  at every level, on both.

With drafts every 2 s, the same runs:

| Streams | GPU: final median / p95 | GPU: draft age median | CPU: final median / p95 | CPU: draft age median |
|---|---|---|---|---|
| 1 | 0.67 s / 0.72 s | 0.18 s | 0.97 s / 1.71 s | 0.48 s |
| 5 | 0.66 s / 0.74 s | 0.04 s | 1.03 s / 2.18 s | 0.59 s |
| 10 | 0.75 s / 1.27 s | 0.05 s | 2.19 s / 4.11 s | 1.17 s |
| 25 | 1.55 s / 3.43 s | 0.23 s | falling behind | falling behind |

"Draft age" is how old a draft is when it appears, from the tick that asked
for it. The GPU keeps drafts fresh for about 10 streams, the CPU for about 5.
Drafts cost about 2.5 times the compute of finals alone
([sizing](10-results-cpu-sizing.md)).

A second CPU data point: one Zen 2 core and its hyperthread gave 0.96 s for
one stream and a p95 of 4.58 s at five ([sizing](10-results-cpu-sizing.md)).

## A streaming model: Nemotron [MEASURED]

`parakeet-tdt-0.6b-v3` has no streaming mode: NVIDIA's own serving stack lists
it as offline only, and the chunked "buffered" workaround in NeMo has a
published cost of 6.32 % to 9.22 % WER at 2.4 s of delay (see
[other options](#other-options-claim)). NVIDIA's
`nemotron-speech-streaming-en-0.6b` is a different model built for streaming:
a cache-aware FastConformer that emits text in fixed chunks while the speaker
talks. sherpa-onnx ships it as
`sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25`, with
560 ms chunks, int8 only, for its `OnlineRecognizer`.
[`probes/streaming_online.py`](../probes/streaming_online.py) ran it on the
same first 12 AppTek channels per accent as the offline sweep, clean and
through G.711, on the Ryzen 7 3700X's CPU, and scored it the same way.

sherpa-onnx's own endpointing (end an utterance after some trailing
silence, then reset the stream) costs a lot of accuracy, because each reset
discards the model's context
([`results/streaming_online-endpoints.log`](../results/streaming_online-endpoints.log)):

| Clean, first 12 channels | One continuous stream | Endpoint after 1.2 s of silence |
|---|---|---|
| en-US_General | 5.44 % | 8.80 % |
| en-IN | 9.42 % | 11.82 % |
| en-GB_SCT | 12.29 % | 15.28 % |

Run as one continuous stream, it is more accurate than offline Parakeet with
live settings on every accent and on the telephone condition
([`results/streaming_online.log`](../results/streaming_online.log),
[`results/nemo_vs_onnx-g711-livepad.log`](../results/nemo_vs_onnx-g711-livepad.log)):

| First 12 channels | Streaming, continuous | Parakeet offline, 0.5 s silence, 0.25 s padding |
|---|---|---|
| en-US_General, clean | 5.44 % | 8.04 % |
| en-US_General, G.711 | 6.15 % | 10.84 % |
| en-IN, clean | 9.42 % | 10.06 % |
| en-IN, G.711 | 11.49 % | 12.67 % |
| en-GB_SCT, clean | 12.29 % | 13.58 % |
| en-GB_SCT, G.711 | 13.05 % | 16.65 % |

Its delays, measured in audio time against the speech segments the offline
pipeline's VAD finds on the same audio:

| Delay | Median | p95 |
|---|---|---|
| First words, from the start of speech | 0.53 s to 0.72 s | 1.17 s to 1.79 s |
| Last word, from the end of speech | 0.16 s to 0.23 s | 0.64 s to 1.04 s |

(Ranges are across the six accent and condition runs.) Its text grows but is
never revised: 0 of 25,567 text changes altered text already shown. So a word
on screen is final when it appears, and an utterance is complete about 0.2 s
after the speaker stops, against 0.52 s for offline Parakeet on a GPU and
about 0.95 s on a CPU.

Two things weigh against it:

- Compute. The int8 export decoded at 1.3× to 1.4× real time per worker on
  the Ryzen 7 3700X, with 12 workers running at once, so a live stream
  occupies about three quarters of a CPU thread, several times what offline
  Parakeet needs. It was measured on the CPU only. On a GPU, sherpa-onnx ran
  Parakeet's int8 export several times slower than its fp32 export
  ([accuracy](11-results-telephone-accuracy.md#throughput-measured-indicative)),
  so an int8-only model may not gain much there, but that was not measured
  for this one. Nor was how many live streams one machine can carry.
- It is English only, and its licence is NVIDIA's Open Model License, not
  CC-BY-4.0 [CLAIM, from the model card]. Parakeet v3 covers 25 European
  languages [CLAIM].

Its int8 export held up on G.711 audio, unlike Parakeet's int8 export
([accuracy](11-results-telephone-accuracy.md#the-int8-export-measured)), so
Parakeet's collapse belongs to that one export, not to int8 as such. No fp32
export of the streaming model was offered to compare against.

## Other options [CLAIM]

These were read, not measured. Each figure is from the source linked.

| Option | What it is | Published figure |
|---|---|---|
| NeMo buffered streaming of parakeet-tdt-0.6b-v3 ([script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py)) | The offline model run on overlapping chunks; needs NeMo | WER from 6.32 % to 9.22 % at 2.4 s of delay ([arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)) |
| [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | One English model for offline and chunked streaming; sherpa-onnx has it offline only | 5.91 % offline, 6.52 % at 0.56 s chunks, Open ASR sets |
| [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) | Small streaming model that emits an end-of-utterance token; no punctuation; not in sherpa-onnx | 9.30 % Open ASR average; end-of-utterance p50 160 ms |
| NVIDIA NIM streaming ASR ([support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)) | NVIDIA's serving stack; lists parakeet-tdt v2 and v3 as offline only, Nemotron and other Parakeet models as streaming | |

## Recommended setups

| Goal | Setup | Measured here |
|---|---|---|
| Words while the caller speaks, and the best accuracy, in English | Nemotron streaming, one continuous stream per call leg, lines broken at pauses by a separate VAD ([`examples/nemotron_stream.py`](../examples/nemotron_stream.py)) | first words 0.53 s to 0.72 s, complete about 0.2 s after speech ends; 5.44 % to 13.05 % WER; about three quarters of a CPU thread per stream |
| Final text fast, many streams, any of Parakeet's languages | Parakeet v3 fp32 on an NVIDIA GPU, 0.5 s silence (0.8 s if 0.3 s more delay is acceptable), 0.25 to 0.5 s padding ([`examples/parakeet_onnx.py`](../examples/parakeet_onnx.py)) | 0.52 s after speech ends for up to 50 streams in one process; drafts fresh for about 10 |
| Lowest compute per stream, no GPU | The same on a CPU | about 0.95 s after speech ends, up to 10 streams on a 16-thread Zen 2 with 8 workers; 0.16 to 0.25 cores per stream |

### Setting it up

Both setups read 16 kHz mono audio, one speaker per stream, and both take it
from any GStreamer pipeline through a pipe, as in
[deploying on ONNX](12-onnx-deployment.md#9-gstreamer). For a G.711 RTP call
leg:

```bash
gst-launch-1.0 -q -e udpsrc port=5004 \
    caps="application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0" \
    ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec ! audioconvert ! audioresample \
    ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1 \
  | python nemotron_stream.py        # or: python gst_stdin.py, for Parakeet
```

Both examples, over this pipeline and from a file, are exercised by
[`examples/test_gstreamer.sh`](../examples/test_gstreamer.sh)
([`results/gstreamer-examples.log`](../results/gstreamer-examples.log)). On a
90 s call over live G.711 RTP, the streaming model produced all four of the
clip's utterances and offline Parakeet three.

For Parakeet, the latency and accuracy settings are arguments of
`Transcriber` in `parakeet_onnx.py`:

```python
t = Transcriber(provider="cuda",        # or "cpu"
                min_silence_s=0.5,      # the latency dial: 0.5 to 0.8
                pad_s=0.25)             # 0.25 to 0.5; never below 0.25
```

For the streaming model, keep `enable_endpoint_detection=False`: every reset
loses context. Decide where utterances end outside the recogniser, from a VAD
or from the model's punctuation, and never call `reset()` in the middle of a
call.

## What was not measured

- More than one process per GPU, and batching several utterances per GPU
  call.
- Drafts at a shorter interval than 2 s.
- Telephone audio in the CPU-against-GPU latency runs, which used one clean
  60 s clip. (The accuracy sweeps and the streaming model did include G.711.)
- The streaming model's behaviour with many live streams at once, and any
  fp32 or GPU-accelerated version of it.
- The network, capture and playback delays of a real call, which add to
  every figure here.

---

Previous: [Deploying on ONNX](12-onnx-deployment.md) | [Contents](../README.md#contents)
