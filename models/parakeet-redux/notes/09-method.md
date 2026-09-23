# Method and open questions

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

## Probes

Run from the repository root with the `asr` extra installed
(`uv run --extra asr python models/parakeet-redux/probes/<probe>.py`). Each
probe's docstring says what it measures and how to run it.

| Probe | Output | Covers |
|---|---|---|
| `environment.py` | `results/environment.log` | CPU model, cores, ISA flags, RAM, OS, glibc, Python and package versions, torch threads, and whether Photon runs without any loader variables set |
| `licensing.py` | `results/licensing.log` | Declared licence, licence files and sizes for every distribution in the runtime path; PyPI wheel sizes; the quoted clauses located in the shipped LICENSE; HF licence tags |
| `vendor_claims.py` | `results/vendor_claims.log` | Every vendor figure and quotation in the note, re-read from the live model cards, eval files, docs and blog posts |
| `pk_exp.py` | `results/pk_exp.log` | Throughput over 18 runs (section A) and per-channel transcripts (section B). Sections C to E are superseded, see below |
| `channel_speed.py` | `results/channel_speed.log` | The agent-channel slowdown, computed from section A of `pk_exp.log` |
| `timestamps_modes.py` | `results/timestamps_modes.log` | Time and text for each `timestamps` mode, n=5 per mode per fixture |
| `load_footprint.py` | `results/load_footprint.log` | Warm load (n=5), cold load including download (n=3), venv and model-cache size |
| `api_surface.py` | `results/api_surface.log` | Registered models, result shapes per mode, rejected parameters and their errors, stereo-path behaviour, streaming shapes |
| `wer_voicechat.py` | `results/wer_voicechat.log` | VoiceChat's caller transcript scored against Parakeet's, with the old and new normalisers |
| `textnorm.py` | none (self-test) | The shared normaliser and word alignment used by every WER-style figure |
| `pk_stream.py` | `results/pk_stream.log`, `results/asr-*-run*.jsonl` | Streaming at wall-clock rate, 3 runs per fixture, and stream-vs-batch identity |
| `stream_analysis.py` | `results/stream_analysis.log` | Every derived streaming figure: latency, intervals, revision kinds and depths, settled-word accuracy, the quoted excerpt |
| `resample.py` | `results/resample.log`, `results/resample-after-fix.log` | Native rate against the harness's 24 kHz path and a naive 16 kHz resample, on 3 fixtures and 4 Map Task channels |
| `asr_fixes_check.py` | `results/asr-fixes-check.log` | The wrapper fixes: full-scale non-16 kHz audio now transcribes, quiet audio goes through untouched, and streaming is timed from the first audio chunk |
| `clipping.py` | `results/clipping.log` | Which peak levels and sample rates Photon rejects |
| `longform.py` | `results/longform.log` | One 91.4-minute file in one call, against the same audio transcribed file by file |
| `compare_models.py` | `results/compare_models.log` | Redux against the base model, Ultra, Whisper and Qwen3-ASR in the same runtime: RTF and pairwise agreement |
| `compare_analysis.py` | `results/compare_analysis.log` | Word counts and pooled substitutions, deletions and insertions between each pair of models, from `compare_models.log` |
| `step0.py` | none saved | The first smoke test, one run. Its output was never saved, so none of its figures are quoted |

The code of `pk_exp.py` is kept unchanged, apart from its docstring, so its
log stays reproducible. Its section C
(WER) used a partial number normaliser and is replaced by `wer_voicechat.py`;
section D (timestamp modes, one run each) by `timestamps_modes.py`; section E
(one clip, a resample done by the probe itself) by `resample.py`.

The WER comparison scores VoiceChat's `caller.said` events from two September
sessions. Those session logs were only in the gitignored `logs/` directory, so
copies are committed as `results/voicechat-tool_call.jsonl` and
`results/voicechat-interruptions.jsonl`.

`results/asrtest.log`, an early exploratory transcription with no committed
probe, has been deleted. Everything it showed is covered by `pk_exp.py`,
`pk_stream.py` and `api_surface.py`.

## Repeating this investigation

There is no single script for this. The order below is the one in which the
findings appeared, and each step is there because skipping it hides something
specific.

Establish the licence before running anything. Read the actual `LICENSE` and
`METADATA` inside each installed distribution rather than the marketing page.
Here the two disagree, and that disagreement is a finding.

Record the machine with a probe. The CPU and ISA table in an earlier version
of this note had no evidence behind it; `environment.py` now writes it (see
the hardware table in "Results: hardware and throughput"), and it found that
moondream was 2.4.1, not the 2.4.0 the note said.

Check the vector ISA before trusting any throughput claim. The vendor's
figure came from AVX-512 server silicon and this desktop CPU has none. That
is one of several differences between the two machines, and it has not been
isolated as the cause of the gap.

Pin the compute path, then verify that the pin took. torch arrives
transitively, and dependency-source pinning that binds only direct
dependencies silently does not apply to it. Print `torch.__version__` and look
for `+cpu`.

Never report n=1. This note once carried 57.1× from a single run that was
never saved; the median over 18 runs is 43.7×. Three repeats across several
inputs took a couple of minutes and changed the headline number by 30 %.
Discard a warmup call even when it looks negligible.

Vary the content as well as the repeat count. Throughput was 10.2 to 15.4 %
lower on synthetic speech than on human speech of identical duration, and
streaming stability differed 7.6× across three clips. Either would have been
invisible on one fixture.

Prove channel separation by transcribing both channels and reading them. A
stereo file handed to Photon as a path comes back as one fluent
transcript of both speakers, with some words lost in the mix, and nothing
warns you.

Pass audio at its native rate, and measure what the alternatives do. On
24 kHz files the harness's resampler is a no-op. On 20 kHz Map Task audio the
original linear resampler changed up to 9 % of the words on one channel, and
its band-limited replacement still changed up to 6 %. Test real rate changes, not
only files that already match the harness's rate.

Check peak levels. Photon rejects audio that comes near full scale at any
rate other than 16 kHz, with an error that blames the input for being outside
[-1, 1] when it is not.

Time from the event you care about. Timed from before the client was opened,
the first snapshot came at 5.2 to 6.8 s across the reruns (5.2 to 6.6 s as
first published). Timing starts at the first audio chunk now. Measured properly it is 4.01 s, which is the
vendor's documented four seconds.

Pace the stream at wall clock, and reuse the existing pacer. A
`sleep(chunk_ms)` loop accumulates per-iteration send cost and drifts behind
real time, which makes first-snapshot latency look worse than it is.

Log snapshots raw and compute stability afterwards. Report both metrics,
because either one alone misleads, and classify what changed:
here, punctuation and cut-off words, never a settled word.

Save stdout and stderr together, and commit the probe with the log. A
section of the streaming log was empty because a run ended early, and nothing
recorded why.

Score against something independent. The WER numbers in this investigation
exist only because a second, unrelated transcriber was available. Comparing a model to itself
tells you nothing, and comparing models to each other without a reference
tells you agreement, not accuracy.

To repeat this on different silicon, hold the fixtures, device and runtime
versions fixed and change only the CPU. The comparison worth publishing is
AVX-512 against no AVX-512, with core count held equal.

## Open questions

1. Does `parakeet-ultra` load outside Photon (NeMo, ONNX)? If it does, the
   arm the vendor reports as most accurate survives the licensing problem.
2. Is the vendor right that Redux does not drop or invent content more often
   than the original? On two channels of clean but conversational Map Task
   audio, Redux left out about 130 words relative to either the original or
   Ultra. Scoring all three against Map Task's own human transcripts
   would settle it for clean speech; noise is still untested.
3. Why is synthetic speech 10.2 to 15.4 % slower to decode than human speech
   of the same duration? The TDT duration-skipping mechanism is the obvious
   suspect, and it can be checked by comparing emitted token counts per
   second of audio between the two channels.
4. Does the built-in segmenter apply to live streams, or only to file and
   array input? Correction: an earlier version said the segmenter was never
   exercised. It was: the 84.5 s `tool_call` channel came back as 11 segments
   in batch, and the 91.4-minute file was cut into segments of at most 30 s.
   What is still unknown is whether streaming uses the same segmentation.
5. How do NE and UPWR behave on long audio (10 min or more)? Every stability
   figure here comes from clips under 90 s.
6. How much of the gap to the vendor's 113× is AVX-512, and how much is core
   count, clock or memory? Only a second machine can answer it.
7. Why does Photon's resampling reject near-full-scale input, and does it also
   clip or distort loud audio that it accepts?
8. How does Redux compare with Whisper or Qwen3-ASR on a CPU? Neither runs on
   CPU in moondream 2.4.1, so answering it needs a different runtime for them.

---

Previous: [Results: what the input audio does](08-results-input.md) | [Contents](../README.md#contents)
