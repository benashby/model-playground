# How it works

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

The encoder and decoder are inherited from the base model; the ternary
weights are new.

The encoder is a FastConformer. FastConformer is a Conformer variant that uses self-attention for global context and depthwise convolution for
local acoustic structure. Its distinguishing part is a subsampling front end
that cuts the frame rate by 8× instead of the usual 4× before the expensive
attention stack runs. Attention cost is quadratic in sequence length, so
discarding three quarters of the frames up front should make it cheaper per
second of audio than an encoder that keeps them. That is an argument from the
architecture, and it is untested here: the obvious comparison,
`whisper-large-v3-turbo` in the same runtime, refuses to run without a CUDA
device ([head-to-head](06-results-throughput.md#head-to-head-on-this-cpu-measured)).

The decoder is a TDT (Token-and-Duration Transducer), also inherited. At each
step it emits a token and a duration, meaning how many encoder frames to skip
before the next decision. A classic RNN-T has to step once per frame and emit
blanks through silence; TDT jumps over it. The durations also make word
timestamps native, because they are the alignment, produced by the same
forward pass as the text. Models that add alignment afterwards need a second
model for it. Photon's docs describe
Parakeet's word timestamps as "Native token durations" and Whisper's as
"Alignment" [CLAIM].

The ternary encoder weights are what Redux adds. Every encoder weight is −1, 0,
or +1 ("1.58-bit", log₂(3) bits per weight). The vendor's motivation is
bandwidth: "On CPUs and Apple Silicon, memory bandwidth and not computation is
often the limiter of inference speed. Smaller weights reduce that traffic"
[CLAIM]. The checkpoint shrinks from 1.2 GB to 178 MB (the Redux
`model.safetensors` is 177,774,490 bytes [MEASURED], see
[`results/licensing.log`](../results/licensing.log)), and Photon's kernels
compute directly from the packed representation instead of unpacking to float
first [CLAIM]. The card names the kernels' target instruction sets as "AVX-512
VNNI on x86, NEON on ARM, Metal on Apple GPUs" [CLAIM]. The CPU used here has
no AVX-512 at all, and Redux still ran; which code path Photon takes on it is
not visible to this investigation.

A VAD head sits on the encoder's subsampler, so the model finds its own pauses
without a separate VAD model. Photon uses those pauses "to cut recordings at
pauses into segments of at most 30 seconds" [CLAIM]. It worked that way here:
the 84.5 s `tool_call` channel came back as 11 segments, and a
91.4-minute concatenation of Map Task audio came back in one call with no
segment over 30 s ([long-form results](08-results-input.md#long-form-one-91-minute-call-measured)).

## What the compression cost [CLAIM]

All figures are the vendor's, from the two model cards, re-read on 2026-09-23
by [`probes/vendor_claims.py`](../probes/vendor_claims.py)
([`results/vendor_claims.log`](../results/vendor_claims.log)). Redux and Ultra
ran in Photon on an NVIDIA GPU for these; the original ran in NeMo in bf16.
None of them was measured here.

| Benchmark | parakeet-tdt-0.6b-v3 | parakeet-redux | parakeet-ultra |
|---|---|---|---|
| Open ASR Leaderboard, 7 English sets (WER %) | 6.26 | 6.55 | **5.80** |
| FLEURS, 25 languages (WER %) | 11.62 | 10.56 | **9.55** |
| Business speech, AA-WER style (WER %) | 6.15 | 6.96 | **5.79** |
| Background noise, 9 MUSAN conditions (WER %) | 6.72 | 9.04 | **5.82** |
| TED-LIUM long-form, 11 talks of 10-20 min (WER %) | 2.71 | 2.51 | **1.94** |
| Weights | 1.2 GB | **178 MB** | 1.2 GB |

Quantisation did not degrade the model uniformly: by these figures it improved
the multilingual and long-form results, held roughly level on English, and
cost 35 % relative in noise (9.04 against 6.72). The vendor's explanation is
that "the ternary encoder's acoustic margin is thinner, and at low SNR it
substitutes similar-sounding words more often", and it adds that "Dropped or
invented content is not more frequent than the original's." That claim is
testable and worth testing, because for a voice agent a wrong word is a much
smaller problem than content that was never said. The one relevant
observation here points the other way: on two channels of
conversational Map Task audio, Redux left out about 130 words relative to
either the base model or Ultra ([head-to-head](06-results-throughput.md#how-much-the-three-parakeets-agree)).
With no reference transcript, this is only suggestive.

Why Redux beats the original on long-form audio is not stated by the vendor.
Correction: an earlier version of this note said, without a label, that
"better segmentation outweighs the quantisation loss". That was my inference.
The card only says that Redux ran "through Photon, whose segmenter cuts each
talk at pauses found by the model's VAD head", while "the original runs NeMo's
own long-audio path". The long-form comparison therefore changes the runtime,
the segmentation and the weights at once, and it cannot say which of them
produced the 0.2-point gain. Open.

`parakeet-ultra` is the same architecture at full precision, with further
post-training. Correction: an earlier version of this note said Ultra's "own
leaderboard eval reports 5.32 mean WER, the top of the Open ASR Leaderboard as
of September 2026", next to a table showing 5.80, without explaining the
difference. Both are the vendor's numbers. The self-reported eval file on the
Ultra repository says "mean_wer averages all eight sets" (the seven English
sets plus the TED-LIUM long-form set) and gives 5.32, and "The card's 'Open ASR'
figure (5.80) is the mean of the other seven." The same file gives Redux 6.05
on the eight-set mean against 6.55 on the seven. Neither the card nor the
release post claims the top of the leaderboard, and I did not check the live
leaderboard, so that part is withdrawn.

The vendor's own advice is "We'd use Parakeet Ultra when noise is a concern
and a GPU is available" [CLAIM]. Ultra also runs on this CPU, at 24.9× real
time against Redux's 44.0× in the same test
([head-to-head](06-results-throughput.md#head-to-head-on-this-cpu-measured)).

---

[Contents](../README.md#contents) | Next: [Licensing](02-licensing.md)
