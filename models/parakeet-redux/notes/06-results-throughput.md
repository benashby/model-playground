# Results: hardware and throughput

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

## Hardware and software

Every number in this investigation came from this one machine. The table is
copied from [`results/environment.log`](../results/environment.log),
which [`probes/environment.py`](../probes/environment.py) wrote.

| | |
|---|---|
| **CPU** | 12th Gen Intel(R) Core(TM) i9-12900KS: 16 cores, 24 threads, 1 socket, max 5500 MHz, 30 MiB L3 |
| **Vector ISA** | AVX, AVX2, FMA, F16C and AVX-VNNI present. No `avx512*` flag at all, and no AMX. torch reports its CPU capability as `AVX2` |
| **Threads used** | `torch.get_num_threads()` = 16; `OMP_NUM_THREADS` and `MKL_NUM_THREADS` unset |
| **RAM** | 62.7 GiB |
| **GPU** | none used: `torch.cuda.is_available()` is `False`, and every call passed `device="cpu"` |
| **OS / libc** | NixOS 26.05, Linux 7.1.2, glibc 2.42 |
| **Python** | CPython 3.12.13, from the Nix store |
| **Runtime** | `moondream` 2.4.1, `kestrel` 0.8.1, `kestrel-native` 0.1.8, `kestrel-kernels` 0.7.1 and its four bundles 0.7.1 |
| **torch** | 2.14.0+cpu |
| **Fixtures** | `audio/{tool_call,interruptions,turn_taking}.wav`: 84.5 s / 30.0 s / 41.1 s, stereo 24 kHz, one speaker per channel |

Correction: an earlier version of this table had no probe behind it and gave
`moondream` 2.4.0. The installed version is 2.4.1. The CPU, ISA, glibc, Python,
`kestrel`, `kestrel-kernels` and torch entries it gave were right.

The vendor's headline figure was produced on "AMD EPYC 9575F (Zen 5, up to
5.0 GHz, AVX-512), 8 physical cores of one chiplet, DDR5-6000" [CLAIM].
Correction: an earlier version of this note said the AVX-512 difference
"accounts for the result". That was never tested. The two machines also
differ in core count, core type, clock and memory, and Photon's kernels are
opaque, so AVX-512 is a plausible cause and nothing more. Open.

## Throughput over 18 runs [MEASURED]

3 fixtures × 2 channels × 3 repeats, with the warmup call discarded, from
[`probes/pk_exp.py`](../probes/pk_exp.py) section A
([`results/pk_exp.log`](../results/pk_exp.log)).

| Fixture | Channel | Audio | RTF min / median / max |
|---|---|---|---|
| tool_call | 0 (human) | 84.5 s | 44.8 / **45.2** / 45.3 |
| tool_call | 1 (agent TTS) | 84.5 s | 40.5 / **40.6** / 42.2 |
| interruptions | 0 (human) | 30.0 s | 47.4 / **47.8** / 48.2 |
| interruptions | 1 (agent TTS) | 30.0 s | 40.8 / **41.9** / 42.5 |
| turn_taking | 0 (human) | 41.1 s | 46.7 / **48.6** / 48.8 |
| turn_taking | 1 (agent TTS) | 41.1 s | 40.1 / **41.1** / 41.9 |
| **all runs** | | | 40.1 / **43.7** / 48.8, n=18 |

The [head-to-head](#head-to-head-on-this-cpu-measured) below re-measured the caller
channels later the same day and got per-fixture medians of 47.7×, 44.0× and
49.2× (n=3 each), in line with the 45.2 to 48.6 medians here.

This corrects an earlier figure in this note. A single run had reported
57.1×, and that run's output was never saved, so the number itself is the only
record of it. Eighteen runs put the median at 43.7×, and nothing near 57× has
appeared in any run since. The honest figure is about 44× real time, ranging
40 to 49× by content.

Against the vendor's 113× on its EPYC server, that is roughly 0.4× of the
advertised throughput on this desktop CPU. The model is still very usable:
44× real time means an hour of audio in about 80 seconds with no GPU, and the
[91.4-minute long-form test](08-results-input.md#long-form-one-91-minute-call-measured)
ran at 41.2×. But the headline figure was not reproduced on this machine, and
should not be quoted as if it applied to one like it.

### Synthetic speech is slower [MEASURED]

The synthetic agent channel was slower to transcribe than the human channel
on every fixture, with the same duration, sample rate and model.
[`probes/channel_speed.py`](../probes/channel_speed.py) computes the gap from
the medians above ([`results/channel_speed.log`](../results/channel_speed.log)):

| Fixture | Human (ch0) median | Agent TTS (ch1) median | Agent slower by |
|---|---|---|---|
| tool_call | 45.2× | 40.6× | 10.2 % |
| interruptions | 47.8× | 41.9× | 12.3 % |
| turn_taking | 48.6× | 41.1× | 15.4 % |

Correction: an earlier version said "~15 % slower". The measured range is
10.2 to 15.4 %, n=3 fixtures. The TDT decoder emits a token plus a duration
per step and skips silence, so decode cost should track the number of emitted
tokens more closely than the wall-clock length, and denser, more continuous TTS
speech leaves it less to skip. That explanation is untested (Open). In
practice, throughput estimates taken on sparse human speech will be optimistic
for synthetic or continuous audio.

## Timestamp modes [MEASURED]

Correction: an earlier version said timestamp modes "cost under 6 %", quoting
1.78 s against 1.89 s on an 84.5 s clip. Those two numbers differ by 6.2 %,
and each mode had been run once.
[`probes/timestamps_modes.py`](../probes/timestamps_modes.py)
([`results/timestamps_modes.log`](../results/timestamps_modes.log)) now runs
every mode 5 times per fixture after a discarded warmup round, interleaving
the modes so drift cannot favour one.

Median time relative to `timestamps="none"`:

| Mode | tool_call (84.5 s) | interruptions (30.0 s) | turn_taking (41.1 s) |
|---|---|---|---|
| `none` | 1.815 s | 0.651 s | 0.854 s |
| `segment` | -0.8 % | -0.2 % | +3.1 % |
| `word` | +0.9 % | -0.1 % | +0.6 % |
| `character` | -1.3 % | +1.6 % | +0.9 % |

Every difference is between -1.3 % and +3.1 %, and the min-to-max spread of
repeats of a single mode is as large (1.799 to 1.872 s for `none` on
`tool_call`), so no mode measurably costs more than another. The text was
byte-identical across all four modes and all repeats on every fixture, so
asking for less timestamp detail than you want saves neither time nor
accuracy.

## Load time and footprint [MEASURED]

From [`probes/load_footprint.py`](../probes/load_footprint.py)
([`results/load_footprint.log`](../results/load_footprint.log)). Each load
ran in a fresh Python process.

| Measurement | Value |
|---|---|
| Warm load, `md.photon(...)` with weights cached | median 2.81 s (2.66 to 2.83), n=5 after 1 discarded |
| `import moondream` | 0.04 s |
| Cold load, empty HF cache, including the download | median 24.37 s (24.12 to 24.38), n=3 |
| Cold minus warm | 21.56 s, attributable to downloading 179.0 MB on this link on this day |
| First call after load (1 s of silence) | 0.04 s (0.04 to 0.05) |
| Model cache on disk | 179.0 MB, of which `model.safetensors` is 177,774,490 bytes |
| Project venv with the `asr` extra | 1.30 GB, of which torch is 737.0 MB and the moondream/kestrel stack about 373 MB |

Corrections: an earlier version gave the cold start as 25.9 s "of which ~20 s
was the 178 MB download" and the warm load as "~2 s", both from a first run
whose output was not saved. Re-measured, cold is 24.37 s and warm is 2.81 s.
The earlier 1.3 GB venv figure holds (1.30 GB); the comment in
`pyproject.toml` that says ~1.2 GB is slightly low.

The warm load matters for streaming: `playground.asr.transcribe_streaming`
opens a new Photon client on every call, so each stream pays it before any
audio is sent.

## Head-to-head on this CPU [MEASURED]

moondream 2.4.1 registers five other speech models besides Redux, and four of
them were tried; `Qwen/Qwen3-ASR-1.7B` was left out as the larger sibling of a
model that would not run.
[`probes/compare_models.py`](../probes/compare_models.py)
([`results/compare_models.log`](../results/compare_models.log)) ran each one
on this CPU, through the same runtime and the same calls: the caller channel
of the three fixtures plus two Map Task channels (q4ec1 ch0, q1ec1 ch1: 20 kHz,
unscripted, mostly Scottish speakers), with a warmup call and one full warmup
pass discarded, then 3 timed passes. n=15 calls per model.

| Model | Weights | Runs on CPU (moondream 2.4.1) | RTF median (min to max), n=15 |
|---|---|---|---|
| `moondream/parakeet-redux` | 178 MB, ternary encoder | yes | **44.0×** (41.0 to 49.6) |
| `nvidia/parakeet-tdt-0.6b-v3` | 1.2 GB | yes | 27.8× (25.4 to 31.1) |
| `moondream/parakeet-ultra` | 1.2 GB | yes | 24.9× (22.1 to 26.2) |
| `openai/whisper-large-v3-turbo` | | no: `ValueError: The optimized Whisper runtime requires a CUDA device` | none |
| `Qwen/Qwen3-ASR-0.6B` | | no: `ValueError: Qwen3-ASR serving requires CUDA bfloat16` | none |

On this CPU Redux is about 1.6× the speed of the checkpoint it was
compressed from, and 1.8× Ultra. Correction: an earlier version of this note
said the original runs "at 28-42× real time" under NeMo or ONNX, with no
source. Those figures are the vendor's ONNX runtime measurements on its EPYC
server (see [licensing](02-licensing.md#the-practical-consequence)). Run in
Photon on this CPU, the original measured 27.8×. Whisper and Qwen3-ASR could
not be compared at all: in this runtime both need an NVIDIA GPU. The docs say
Photon serves them "on supported NVIDIA GPUs" [CLAIM].

### How much the three Parakeets agree

There is no ground-truth transcript for any of these inputs, so none of this
ranks accuracy. [`probes/compare_analysis.py`](../probes/compare_analysis.py)
([`results/compare_analysis.log`](../results/compare_analysis.log)) breaks the
transcripts down.

On the three short, clear fixtures the three models agree almost completely.
After normalisation, Redux and the base model produced the same 117 words;
Ultra differed in 2, writing "going to" where both others wrote "gonna". The
raw text differs more than that: on `tool_call` the base model and Ultra wrote
"between one and fifty" and "Hello.", while Redux wrote "between 1 and 50" and
"Hello!". The vendor says Redux keeps the original's "output conventions
(punctuation, casing, numerals)" [CLAIM]; on this clip it did not.

On the Map Task channels they diverge, and in one direction:

| Normalised words | Redux | Base | Ultra |
|---|---|---|---|
| q4ec1 ch0 | 309 | 337 | 342 |
| q1ec1 ch1 | 348 | 440 | 428 |
| all five inputs | 774 | 894 | 888 |

| Reference, then hypothesis (all inputs) | Substitutions | Deletions | Insertions | Disagreement |
|---|---|---|---|---|
| base, then Redux | 68 | 129 | 9 | 23.0 % |
| Ultra, then Redux | 69 | 132 | 18 | 24.7 % |
| base, then Ultra | 33 | 49 | 43 | 14.0 % |

Base and Ultra disagree with each other in both directions about equally.
Against either of them, Redux leaves out about 130 words and adds 9 to 18.
Reading the transcripts, what goes missing is mostly short turns and
questions: on q4ec1, base and Ultra both have "Is it underneath the rope
bridge or something?", and Redux has only the answer; the base model's "How
far?" and "It's down three steps below or above the machete." are absent too.
Redux also has more word-level slips here: "over the road bridge" where both
others have "rope bridge", "quite a weak distance" for "a wee distance", "we
are at." for "we are a caravan park". It also returns fewer, longer segments
(16 and 38 against the base model's 45 and 69).

The base model and Ultra share an architecture and a training lineage, so
their agreement is weak evidence that Redux is the one in error. It is still
the pattern the vendor says should not happen
("Dropped or invented content is not more frequent than the original's"
[CLAIM]), seen on 2 channels of conversational audio. Open: scoring all three
against Map Task's own human transcripts would settle it.

## Channel separation

This was checked by reading the transcripts, because a mono downmix also
produces a plausible-looking one:

| Fixture | ch0 | ch1 |
|---|---|---|
| tool_call | *"Hello! Can you generate a random number between 1 and 50?…"* | *"Hi, how can I help you today? One moment. Your random number is 20…"* |
| interruptions | *"What is a hurricane? Never mind, what is a tornado?…"* | *"A hurricane is a powerful tropical storm with strong winds…"* |
| turn_taking | *"Hi. Do you know any good cookie recipes? Ooh, maybe a peanut butter one…"* | *"Hi, how can I help you today? I can give you some recipes…"* |

Channel 0 is the caller in all three and channel 1 the agent. That follows from
how these were recorded and is not guaranteed, so check it for each corpus.
What the downmix produces is shown in
[Using it through Photon](04-usage.md#the-minimal-call-and-the-stereo-trap).

---

Previous: [Using it through this repository](05-harness.md) | [Contents](../README.md#contents) | Next: [Results: accuracy and streaming](07-results-accuracy.md)
