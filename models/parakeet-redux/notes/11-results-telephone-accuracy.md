# Results: accuracy on telephone speech

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

This article measures how accurately NVIDIA's `parakeet-tdt-0.6b-v3` transcribes
telephone speech, and what changes that accuracy: the runtime (NVIDIA's NeMo
against an ONNX export), the precision of the export, the telephone channel
itself, and how the audio is cut into segments. It is the evidence behind
[deploying the model on ONNX](12-onnx-deployment.md).

The main results, all [MEASURED]:

- The fp32 ONNX export is as accurate as NeMo. On the same audio segments the
  two differ by 0.03 to 0.23 points of word error rate, and ONNX gives the
  same text on a CPU and on an NVIDIA GPU for 99.53 % of segments.
- The int8 ONNX export is not usable for telephone audio. On 8 kHz speech it
  more than doubles the error rate, and most of the extra errors are whole
  phrases that disappear from the transcript without any sign.
- Narrowband telephone audio (G.711 or Opus) costs 1.7 to 2.7 points against
  the same speech at 16 kHz, and 20 % packet loss costs another 2.2 to 3.5 on
  top of the codec.
- On real telephone calls (HarperValleyBank) the model makes about 16.5 %
  errors, roughly twice the 7.95 % of the ASR that the corpus's creator
  built for that telephone platform.
- Every runtime, NeMo included, comes out 1.0 to 1.4 points worse than the
  figures published for this model on the same corpus. The runtime is not the
  cause, and the segmentation is the remaining suspect.

## What was run

### Corpora

| Corpus | What it is | Licence | Used |
|---|---|---|---|
| AppTek Call-Center Dialogues | Role-played call-centre conversations recorded over VoIP at 16 kHz, one speaker per file, professional verbatim transcripts | CC-BY-SA-4.0 | Accents en-US_General (134 channels), en-IN (146) and en-GB_SCT (132) |
| Gridspace-Stanford HarperValleyBank | Simulated bank calls placed over a real telephone network, 8 kHz, one speaker per file, crowd-worker transcripts | CC-BY-4.0 | All 2,892 channels (1,446 calls) |

Both come from pinned upstream revisions through DVC; see
[`audio/corpora/MANIFEST.md`](../../../audio/corpora/MANIFEST.md). AppTek's
paper (arXiv [2604.27543](https://arxiv.org/abs/2604.27543)) publishes
parakeet v3 word error rates for this corpus, which makes it the check on
this pipeline.

### Segment sets

Every channel was cut into speech segments once, by
[`probes/nemo_vs_onnx.py`](../probes/nemo_vs_onnx.py) using the VAD code in
[`probes/wer_telephone.py`](../probes/wer_telephone.py), and the segments were
saved as float32 WAV files. Every runtime then transcribed exactly those
files, so any difference between runtimes comes from the runtime.
[`probes/run_matrix.sh`](../probes/run_matrix.sh) runs the whole matrix.

| Set | Audio | Condition | Segmentation |
|---|---|---|---|
| A | AppTek, all channels of the three accents | as recorded | `bench` |
| B | AppTek, first 12 channels of each accent | as recorded, and five telephone conditions (below) | `bench` |
| C | HarperValleyBank, all channels | as recorded (already 8 kHz telephone audio) | `bench` and `livepad` |

The segmentations are Silero VAD v5 through sherpa-onnx:

- `bench` is AppTek's published recipe: a segment ends after 10.0 s of
  silence, is at least 0.25 s long and at most 30 s, with 30 ms of audio kept
  either side (the default padding of the silero-vad package).
- `livepad` is the live-transcription setup of
  [`dictate.py`](../dictate.py): a segment ends after 0.5 s of silence, is at
  most 20 s long, and keeps 0.25 s either side.

The telephone conditions resample 16 kHz audio to 8 kHz with a band-limited
filter, pass it through a codec, and resample it back to 16 kHz:

| Condition | Codec |
|---|---|
| `g711mu` | G.711 μ-law (ffmpeg `pcm_mulaw`) |
| `opus12` | Opus, VoIP mode, narrowband, 12 kbit/s, 20 ms frames |
| `opus12-lossN` | as `opus12`, with N % of the 20 ms packets dropped at random and replaced by the Opus decoder's packet loss concealment |

ffmpeg's `-packet_loss` option only tells the Opus encoder how much loss to
expect and drops nothing, so the probe drives libopus directly and drops
packets itself.

### Runtimes

| Name | Runtime | Hardware |
|---|---|---|
| `nemo_cuda_graphs` | NeMo 3.0.0, NVIDIA's checkpoint, default greedy TDT decoding including the CUDA-graph decoder | RTX 3090 Ti |
| `nemo_cuda` | the same with the CUDA-graph decoder off | RTX 3090 |
| `nemo_rocm` | the same, on ROCm 7.2 | RX 9070 XT (gfx1201) |
| `onnx_fp32`, `onnx_int8` | sherpa-onnx 1.13.8, fp32 and int8 exports | Ryzen 7 3700X (set B) |
| `onnx_fp32_cuda`, `onnx_int8_cuda` | sherpa-onnx 1.13.8 CUDA build | RTX 3090 Ti |

All decoding is greedy and one segment at a time.

### Scoring

Every figure uses AppTek's own `score.py`, unmodified: Whisper's
`EnglishTextNormalizer`, AppTek's word mappings, and one corpus-level WER per
set from jiwer. HarperValleyBank is scored the same way, with each channel's
human transcript segments joined in time order as the reference. The raw
scores are in [`results/nemo_vs_onnx.log`](../results/nemo_vs_onnx.log).

## Against the published figures

| Accent | NeMo (CUDA graphs) | ONNX fp32 | Published, parakeet v3, Silero segmentation [CLAIM] |
|---|---|---|---|
| en-US_General | 6.86 % | 7.01 % | 5.5 % |
| en-IN | 11.12 % | 11.24 % | 9.7 % |
| en-GB_SCT | 13.09 % | 13.13 % | 12.1 % |

The published figures are from Table 3 of the AppTek paper, which says the
models were run "using their default inference settings" and does not say
more about how. The ranking of the three accents matches theirs, but every
runtime here, NeMo included, is 1.0 to 1.4 points higher. Since NeMo shows
the same gap as ONNX, the runtime is not the cause. What remains is the
segmentation: sherpa-onnx's implementation of Silero VAD against the
silero-vad Python package the paper used, which may place segment boundaries
differently with the same settings. Open.

## The runtime makes no difference [MEASURED]

On the three full accents (set A):

| Accent | `nemo_cuda_graphs` | `nemo_cuda` | `nemo_rocm` | `onnx_fp32_cuda` |
|---|---|---|---|---|
| en-US_General | 6.86 % | 6.86 % | 6.86 % | 7.01 % |
| en-IN | 11.12 % | 11.12 % | 11.12 % | 11.24 % |
| en-GB_SCT | 13.09 % | 13.10 % | 13.10 % | 13.13 % |

- The CUDA-graph decoder, which is only a speed optimisation, changes one to
  three words in tens of thousands. So does moving NeMo from NVIDIA to AMD.
- The fp32 ONNX export follows NeMo to within 0.15 points on these sets, and
  to within 0.23 points on HarperValleyBank (16.48 % against 16.71 %), so the
  conversion to ONNX lost nothing measurable.
- ONNX on a CPU and on an NVIDIA GPU produce the same text for 3,838 of 3,856
  segments (99.53 %) wherever both were run, and their WERs agree to within
  0.02 points on every set.

## The int8 export [MEASURED]

| Set | fp32 | int8 |
|---|---|---|
| en-US_General, all channels | 7.01 % | 8.43 % |
| en-IN, all channels | 11.24 % | 18.39 % |
| en-GB_SCT, all channels | 13.13 % | 16.51 % |
| HarperValleyBank, `bench` | 16.71 % | 34.88 % |
| HarperValleyBank, `livepad` | 17.53 % | 19.65 % |

On the telephone conditions of set B the gap is larger still (ONNX on the
Ryzen CPU; the GPU gives the same within about 1 point):

| First 12 channels | Clean | G.711 | Opus 12k | Opus, 5 % loss | Opus, 10 % loss | Opus, 20 % loss |
|---|---|---|---|---|---|---|
| en-US_General, fp32 | 7.74 % | 10.06 % | 9.66 % | 10.76 % | 10.29 % | 11.85 % |
| en-US_General, int8 | 8.91 % | 23.06 % | 20.62 % | 22.32 % | 21.46 % | 25.21 % |
| en-IN, fp32 | 9.82 % | 11.93 % | 11.53 % | 12.65 % | 13.16 % | 14.92 % |
| en-IN, int8 | 12.23 % | 34.87 % | 32.33 % | 33.24 % | 34.25 % | 40.19 % |
| en-GB_SCT, fp32 | 12.18 % | 14.43 % | 14.87 % | 15.32 % | 16.09 % | 18.36 % |
| en-GB_SCT, int8 | 14.32 % | 34.83 % | 32.79 % | 37.18 % | 40.72 % | 50.26 % |

[`probes/wer_analysis.py`](../probes/wer_analysis.py)
([`results/wer_analysis.log`](../results/wer_analysis.log)) shows how int8
fails, by grouping deleted words by how many are missing in a row:

| Deleted words in runs of 11 or more | fp32 | int8 |
|---|---|---|
| en-US_General, G.711 | 12.1 % of deletions | 55.7 % |
| HarperValleyBank, `bench` | 19.5 % | 54.3 % (175 runs of 31 or more words) |

Most of int8's extra errors are whole phrases and sentences that vanish,
while the transcript still reads cleanly. On clean 16 kHz audio the effect is
much smaller, which makes it easy to miss. The CPU and the GPU produce
identical int8 text for only 53.01 % of segments, yet their int8 error rates
match, so the fault lies in the quantised model whichever runtime executes
it. int8 is also slower on the GPU: 16× to 18× real time against 94× to 101×
for fp32
([`results/nemo_vs_onnx-timing.log`](../results/nemo_vs_onnx-timing.log)).

## What the telephone channel costs [MEASURED]

The fp32 rows of the table above give the cost of the channel, against the
same speech at 16 kHz:

- G.711 and Opus at 12 kbit/s cost about the same, 1.7 to 2.7 points. Both
  are 8 kHz narrowband, and the bandwidth loss is most of the cost.
- Packet loss adds to that. At 20 % loss the total cost is 4.1 to 6.2 points.
- At 5 % and 10 % loss the order sometimes swaps (10.76 % and 10.29 % on
  en-US_General). Each cell is 12 channels, about 2 hours of speech, so
  differences under about half a point are within the noise of the sample.

NeMo gives the same picture to within a few tenths of a point on every cell.

The loss model is simple: each packet is dropped independently. Real networks
lose packets in bursts, which concealment handles worse; bursty loss was not
tested.

## Real telephone calls: HarperValleyBank [MEASURED]

| Runtime | `bench` | `livepad` |
|---|---|---|
| `nemo_cuda_graphs` | 16.48 % | 17.44 % |
| `nemo_rocm` | 16.47 % | 17.43 % |
| `onnx_fp32_cuda` | 16.71 % | 17.53 % |
| `onnx_int8_cuda` | 34.88 % | 19.65 % |
| The corpus's own ASR, Gridspace (no decoding; the corpus's `transcript` field) | 7.95 % | |

The Gridspace figure is from
[`results/wer_telephone-hvb-gridspace.log`](../results/wer_telephone-hvb-gridspace.log).
Two cautions about it. The paper describing the corpus says the human
transcripts were made with Gridspace's own transcription tool, and does not
say whether that tool started from Gridspace's ASR output; if it did, the
comparison favours Gridspace. And the workers were "not instructed to
carefully transcribe word fragments or non-speech noises", so the references
are looser than AppTek's. Allowing for both, a factor of two is still a
large gap: this model is weaker on real narrowband calls than on the AppTek
audio, including the AppTek audio passed through a telephone codec.

Here the shorter `livepad` segments do slightly worse than `bench` for every
fp32 runtime, while for int8 they are far better, since short segments give
int8 less room to drop long stretches.

## Which words go missing [MEASURED]

Most of the errors of the fp32 export and of NeMo are deleted words, and the
two drop the same ones. On all of en-US_General
([`results/wer_analysis.log`](../results/wer_analysis.log)):

| Word | Deleted by NeMo | Share of its occurrences |
|---|---|---|
| ohh | 128 | 60 % |
| okay | 299 | 25 % |
| yeah | 99 | 21 % |
| yes | 64 | 17 % |
| i | 172 | 6 % |
| the | 65 | 2 % |

The words the model skips are mostly short acknowledgements, often spoken
alone while the other person talks. Ordinary words are rarely dropped. Where
a deployment depends on a caller saying "yes", "no" or "okay", check those
words specifically; a quarter of the "okay"s are missing.

## Segmentation and padding [MEASURED]

sherpa-onnx's VAD returns each segment from the first window it judged to be
speech to the last, with no margin, and that clips the first word of an
utterance. To measure the cost, the first 12 AppTek channels of each accent
were cut with the live settings twice: as the VAD returned them (`live`), and
as the same segments with 0.25 s of audio kept either side (`livepad`). The
segment boundaries are identical, so padding is the only difference. From
[`results/nemo_vs_onnx-padding.log`](../results/nemo_vs_onnx-padding.log),
ONNX fp32:

| First 12 channels | `live`, unpadded | `livepad`, 0.25 s | `bench`, AppTek's recipe |
|---|---|---|---|
| en-US_General | 10.50 % | 8.04 % | 7.74 % |
| en-IN | 11.22 % | 10.06 % | 9.82 % |
| en-GB_SCT | 14.93 % | 13.58 % | 12.17 % |

Padding cuts the error rate by 1.2 to 2.5 points, almost all of it deletions
(401 down to 261 on en-US_General). With padding, the live settings end up
0.2 to 1.4 points behind AppTek's long segments, which is the cost of ending
an utterance after 0.5 s of silence rather than 10 s, so a live deployment
using sherpa-onnx's VAD should pad its segments.

## Throughput [MEASURED, indicative]

From [`results/nemo_vs_onnx-timing.log`](../results/nemo_vs_onnx-timing.log),
one segment at a time, model loading excluded:

| Runtime | AppTek segments | HarperValleyBank `livepad` (short segments) |
|---|---|---|
| NeMo, CUDA graphs, RTX 3090 Ti | 305× real time | 52× |
| NeMo, no CUDA graphs, RTX 3090 | 143× | 32× |
| NeMo, ROCm, RX 9070 XT, 1 CPU core | 68× | 34× |
| ONNX fp32, RTX 3090 Ti, 3 workers | 101× | 18× |
| ONNX int8, RTX 3090 Ti, 3 workers | 18× | 16× |
| ONNX fp32, Ryzen 7 3700X, 6 workers | 22× | not run |

These are throughputs of a batch-1 workload, not benchmarks: the two CUDA
NeMo runs shared a host, as did the GPU and CPU ONNX runs. Two patterns hold
despite that. Short segments are several times slower per hour of audio on
every runtime, because per-segment overhead dominates, so a deployment that
decodes many short utterances should batch them. And without CUDA graphs a
batch-1 NeMo decode is limited by the CPU thread driving the GPU, not by the
GPU.

## What was not exercised

- Background noise. The vendor's noise figures are the only ones
  ([how it works](01-how-it-works.md)).
- Eleven of AppTek's fourteen accent groups.
- Mobile codecs (AMR-NB, GSM), whose encoders the local ffmpeg lacks, and
  bursty packet loss.
- Batch sizes above one, and streaming latency on a GPU.
- The silero-vad Python package's segmentation, which is the likeliest
  explanation of the gap to the published figures.

---

Previous: [Results: CPU and memory for live streams on ONNX](10-results-cpu-sizing.md) | [Contents](../README.md#contents) | Next: [Deploying on ONNX](12-onnx-deployment.md)
