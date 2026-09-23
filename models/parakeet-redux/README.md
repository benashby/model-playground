# Moondream Parakeet Redux

A 1.58-bit ternary speech-to-text model. Investigated 2026-09-23.

- Weights: <https://huggingface.co/moondream/parakeet-redux> (CC-BY-4.0)
- Full-precision sibling: <https://huggingface.co/moondream/parakeet-ultra>
- Base model: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> (NVIDIA)
- Runtime: Photon / Kestrel, <https://docs.moondream.ai/transcription>

## What the model is

The model takes speech in 25 European languages [CLAIM] and returns
punctuated, cased text. It is not a speech-to-speech model: it has no LLM, no
tools and no notion of a turn.

M87 Labs (the Moondream company) announced it in a release post dated
September 22, 2026, as a compression of NVIDIA's `parakeet-tdt-0.6b-v3`; its
self-reported eval results are dated 2026-09-21. Correction: an earlier
version of this note gave 2026-09-21 as the release date. The vendor says the
architecture, tokenizer and output conventions are unchanged [CLAIM]. The
difference is that every encoder weight has been replaced by one of three
values.

[How it works](notes/01-how-it-works.md) covers the ternary compression and
what the vendor says it cost in accuracy.

### What it is good for

- Bulk transcription on a CPU. It ran at a median 43.7× real time over 18 runs
  here, so an hour of audio takes about 80 seconds, with no GPU
  [MEASURED]. A single 91.4-minute file went through in one call in 133.2 s
  [MEASURED].
- Keeping speech recognition off the GPU. The vendor's release post reports
  that "One user has already moved transcription to the CPU to free up GPU
  memory for LLMs" [CLAIM]. Correction: an earlier version called this "the
  vendor's stated motivation". The post's stated motivation is speed and
  footprint on CPUs and Macs; the GPU point is a user's report it quotes.
- Small deployments. The weights are 178 MB [MEASURED], though a full CPU
  install with torch came to 1.30 GB here [MEASURED].
- Word and character timestamps at no measurable cost: every `timestamps`
  mode ran within -1.3 % to +3.1 % of plain text, with identical text
  [MEASURED].
- Long recordings without a separate VAD. Photon cut the 91.4-minute file
  into 902 segments of at most 28.96 s [MEASURED]. Photon only accepts loud
  recordings once they are scaled down, and the wrapper in this repository
  now does that itself, as the input-audio results explain.
- Multilingual transcription across 25 languages from one checkpoint
  [CLAIM]. Only English was tested here.

### What it does not do

It has no diarisation and no speaker labels, so it tells you what was said but
never who said it. Handed a stereo file, it silently merges both speakers
into one transcript and loses words in the mix [MEASURED]. It returns
`language: None` and rejects language forcing, prompts, translation,
temperature and top-p with a `ValueError` [MEASURED]. On this CPU, Photon also
refuses near-full-scale audio at any sample rate other than 16 kHz
[MEASURED].

## When to pick which

Measured here, on one desktop CPU without AVX-512:

- Redux is the fastest Parakeet on a CPU: 44.0× real time against 27.8× for
  NVIDIA's original checkpoint and 24.9× for Ultra, all in the same runtime.
- On short, clear speech the three Parakeets gave the same words, apart from
  Ultra writing "going to" where the others wrote "gonna".
- On unscripted conversational speech (HCRC Map Task), Redux left out about
  130 words relative to either the original or Ultra, out of roughly 890.
  There is no reference transcript, so this counts disagreement between the
  models and is not an error rate. It is still worth watching.
- `whisper-large-v3-turbo` and Qwen3-ASR do not run on a CPU in Photon at
  all; both require CUDA.

Claimed by the vendor, not tested here: Ultra is the most accurate of the
three on every benchmark group they report, and they would "use Parakeet
Ultra when noise is a concern and a GPU is available". Redux is 0.29 WER
points behind the original on English and 2.32 points behind in noise, and
ahead on multilingual FLEURS and long-form audio.

So: pick Redux for CPU-only bulk or live transcription of clear speech where
speed and footprint matter. Pick Ultra, or the original, when you have
conversational or noisy audio and can accept a little over half the speed on
a CPU, or have a GPU. Pick the original if the licence matters: its
CC-BY-4.0 weights run on open runtimes, while Redux runs only on Photon's
proprietary kernels.

## What was found

- Median throughput is 43.7× real time over 18 runs on a desktop CPU without
  AVX-512 (range 40.1 to 48.8), roughly 0.4× the vendor's 113× from an
  AVX-512 server. What causes the gap was not isolated. An earlier single run
  gave 57.1×; its output was never saved, and it never recurred.
  ([throughput results](notes/06-results-throughput.md))
- The 178 MB weights are CC-BY-4.0, but only Photon's kernels can run them.
  One of the eight runtime packages ships a proprietary licence that says you
  have no licence without a separate agreement; the other seven declare none,
  although the vendor says the engine is Apache 2.0 and local use is free.
  ([licensing](notes/02-licensing.md))
- Used as a reference, it scored VoiceChat's own caller transcription at
  0.0 % WER on `interruptions` and 5.8 % on `tool_call` (4 differing words out
  of 69). ([accuracy results](notes/07-results-accuracy.md))
- Streaming previews arrive 4.01 s after the first audio (n=9) and every
  2.00 s after that, as documented. Earlier versions reported 5.2 to 6.6 s
  because the harness's clock included the client's load time. The final
  streamed transcript was byte-identical to batch in all 9 runs, and no
  settled word ever changed, only punctuation and cut-off final words.
  ([accuracy results](notes/07-results-accuracy.md))
- Passing audio at its native rate matters. On 24 kHz fixtures the
  harness's resampler is a no-op, but on 20 kHz Map Task audio its original
  linear version changed 40 of 1357 words, and a naive 16 kHz resample
  changed 85. Correction: an
  earlier version attributed a 2.90 % WER cost to "VoiceChat's 24 kHz wire
  rate"; that figure was 2 dropped filler words from a different path.
  ([accuracy results](notes/07-results-accuracy.md))

## Contents

| Article | What it covers |
|---|---|
| [How it works](notes/01-how-it-works.md) | Ternary weights, the TDT decoder, and the vendor's accuracy claims |
| [Licensing](notes/02-licensing.md) | Permissive weights behind a proprietary runtime, file by file |
| [Setup](notes/03-setup.md) | Installing it with pip or in this repository, and the libraries it needs |
| [Using it through Photon](notes/04-usage.md) | The API, the stereo trap, timestamps, streaming, output shapes and what it rejects |
| [Using it through this repository](notes/05-harness.md) | The repo's wrapper, and why the model is not part of the speech harness |
| [Results: hardware and throughput](notes/06-results-throughput.md) | The CPU, 18-run throughput, load time, and a head-to-head with two other Parakeets |
| [Results: accuracy and streaming](notes/07-results-accuracy.md) | Scoring VoiceChat's transcription, and how streaming transcripts revise themselves |
| [Results: what the input audio does](notes/08-results-input.md) | Resampling, a 91-minute recording in one call, and loud audio |
| [Method and open questions](notes/09-method.md) | The probes, the order to repeat the work in, and what is still unknown |

The code behind the numbers is in [`probes/`](probes/) and its raw output is in
[`results/`](results/). The standard these notes are held to is in
[`models/README.md`](../README.md).
