# Using it through Photon

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

## The minimal call, and the stereo trap

```python
import moondream as md

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    result = speech.transcribe(audio="call.wav")  # a path, bytes or a file object
    print(result["text"])
```

This works for a mono recording, with one catch: the path must be a regular
file. A symbolic link to a WAV was refused with
`ValueError: could not open encoded audio file "<tmp>/link.wav": Too many levels of symbolic links (os error 40)`
[MEASURED], so resolve links (`Path(p).resolve()`) before passing them.

**For a stereo file it merges both speakers into one transcript.** Photon's docs say so ("Multichannel files are
downmixed to mono from their container metadata") [CLAIM], and the probe
confirmed it [MEASURED]: on the 2-channel `interruptions.wav`, the path call
returned 72 words that interleave the caller's questions with the agent's
answers, and it was byte-identical to transcribing our own mean of the two
channels. The mix also loses words: the agent's "strong winds and heavy rain"
came back as "strong wind", and "in contact with the" disappeared. Nothing
warns you, and the result reads as a fluent transcript of a conversation
nobody had.

For a recording with one speaker per channel, split the channels yourself and
pass each one as PCM with its sample rate:

```python
import moondream as md
import soundfile as sf

audio, rate = sf.read("call.wav", dtype="float32", always_2d=True)

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    for ch in range(audio.shape[1]):
        result = speech.transcribe(audio=audio[:, ch].copy(), sample_rate=rate)
        print(f"channel {ch}: {result['text']}")
```

`.copy()` makes the column contiguous. Pass the file's own `rate`; Photon
resamples internally, and resampling it yourself first changes the
transcript (see [the resampling results](08-results-input.md#sample-rate-what-resampling-first-does-measured)).
If the recording is loud and not 16 kHz, scale it down first, for example
`audio[:, ch] * 0.5`: Photon rejects near-full-scale input that it has to
resample ([loud audio](08-results-input.md#loud-audio-is-rejected-measured)).

## Timestamps

```python
import moondream as md
import soundfile as sf

audio, rate = sf.read("call.wav", dtype="float32", always_2d=True)
caller = audio[:, 0].copy()

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    result = speech.transcribe(audio=caller, sample_rate=rate, timestamps="word")
    for segment in result["segments"]:
        print(f"{segment['start']:6.2f} {segment['end']:6.2f}  {segment['text']}")
        for word in segment["words"][:3]:
            print(f"         {word['start']:6.2f} {word['end']:6.2f}  {word['word']}")
```

`timestamps` takes `"none"`, `"segment"`, `"word"` or `"character"`. The
docs list only the first three [CLAIM]; `"character"` also works for
Parakeet [MEASURED]. The mode does not change the text: across 5 timed repeats
of each mode on each of the three fixtures, every call returned the same
transcript. Timing per mode is in
[the throughput results](06-results-throughput.md#timestamp-modes-measured).

## Streaming

`atranscribe(..., stream=True)` takes an async iterator of 1-D mono PCM
chunks and returns a `PhotonStream`. Iterate it for snapshots, then await
`aresult()` for the final result dict:

```python
import asyncio

import moondream as md
import soundfile as sf

audio, rate = sf.read("call.wav", dtype="float32", always_2d=True)
caller = audio[:, 0].copy()


async def chunks(samples, rate, chunk_s=0.08):
    # stand-in for a microphone: yields 80 ms of mono PCM at a time, in real time
    n = int(rate * chunk_s)
    for i in range(0, len(samples), n):
        yield samples[i : i + n]
        await asyncio.sleep(chunk_s)


async def main():
    with md.photon("moondream/parakeet-redux", device="cpu") as speech:
        updates = await speech.atranscribe(
            audio=chunks(caller, rate), sample_rate=rate, timestamps="segment", stream=True
        )
        async for update in updates:
            print("snapshot:", update["text"])  # replaces the previous snapshot
        final = await updates.aresult()
        print("final:", final["text"])


asyncio.run(main())
```

Each update is a whole replacement transcript, not a delta, so display it in
place of the previous one. The `asyncio.sleep` here is only a stand-in for a
live source and drifts slowly behind real time; for measurement, use the
harness's pacer, as `playground.asr` does.

For a file you already have, the synchronous form gives progress snapshots
while Photon works through it:

```python
import moondream as md

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    updates = speech.transcribe(audio="call.wav", timestamps="segment", stream=True)
    for update in updates:
        print("progress:", update["text"][:60])
    final = updates.result()
    print("final:", final["text"])
```

(With a stereo `call.wav`, this has the same downmix problem as the minimal
call.)

## What comes back [MEASURED]

`transcribe` returns a plain `dict` with these top-level keys, in this order:

| Key | Type | Example (caller channel of `interruptions.wav`) |
|---|---|---|
| `text` | `str` | `'What is a hurricane? Never mind, ...'` |
| `language` | `NoneType` | `None`; always `None` for Parakeet |
| `task` | `str` | `'transcribe'` |
| `duration_seconds` | `float` | `30.04` |
| `source_duration_seconds` | `float` | `30.04` |
| `clip_start_seconds` | `float` | `0.0` |
| `clip_end_seconds` | `float` | `30.04` |
| `segments` | `list` | 5 entries with `timestamps="segment"` |

The entries in `segments` depend on the mode:

| `timestamps=` | Segments | Segment keys | Inner entries |
|---|---|---|---|
| `"none"` | 0 | none | none |
| `"segment"` | 5 | `start`, `end`, `text` | none |
| `"word"` | 5 | `start`, `end`, `text`, `words` | `{'word': 'What', 'start': 0.64, 'end': 0.96}` |
| `"character"` | 2 | `start`, `end`, `text`, `characters` | `{'character': 'W', 'start': 0.64, 'end': 0.8}` |

Character mode groups the text into fewer, longer segments (2 here, against
5). All times are seconds from the start of the source. The docs say word
entries may also carry `probability` "when the selected model reports it"
[CLAIM]; Parakeet's do not.

Each streaming update is a `dict` with the same keys as the result plus
`completed_seconds`, `total_seconds` and `provisional`. `aresult()` returns the
same shape as `transcribe`, and on this clip its text was byte-identical to the
batch transcript.

## What Parakeet rejects [MEASURED]

Each of these raised `ValueError` with the message shown:

| Argument | Message |
|---|---|
| `language="en"` | `Parakeet TDT v3 detects language automatically and does not support language forcing` |
| `initial_prompt="..."` | `Parakeet TDT does not support an initial prompt` |
| `task="translate"` | `Parakeet TDT does not support speech translation` |
| `temperature=0.0` | `Unsupported transcribe option(s): temperature` |
| `top_p=0.9` | `Unsupported transcribe option(s): top_p` |
| `condition_on_previous_text=False` | `Parakeet TDT does not support cross-window text conditioning` |

`task="transcribe"` is accepted, and so is `settings={"temperature": 0.0}`,
which raised nothing. Open: whether Parakeet uses anything in `settings`.

## Other options

`md.photon_models()` in moondream 2.4.1 registers 33 models. Six are speech
models: `moondream/parakeet-redux`, `moondream/parakeet-ultra`,
`nvidia/parakeet-tdt-0.6b-v3`, `openai/whisper-large-v3-turbo`,
`Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`. Correction: an earlier
version of this note said moondream 2.4.0 and listed only the first four.
The installed version is 2.4.1, and the two Qwen3-ASR models were missed.

`device` is `"cpu"`, `"mps"` or `"cuda"`. The vendor says that if you omit it,
Photon prefers CUDA, then Apple silicon, then CPU [CLAIM], so pass it
explicitly if you care which one you measured.

Accepted inputs, per the docs [CLAIM]: a path, raw bytes, or a bounded binary
stream (WAV, FLAC, MP3, Ogg, Opus, M4A, MP4, MOV, WebM); a 1-D mono NumPy
array or CPU torch tensor with `sample_rate=`; or, through `atranscribe`, an
async iterator of such arrays. The docs also say to "Use an asynchronous PCM
iterator for raw audio longer than 30 seconds". In practice `transcribe`
accepted whole arrays of 212.5 to 497.4 s
([`results/resample.log`](../results/resample.log),
[`results/clipping.log`](../results/clipping.log)) and each of the eleven Map
Task channels in turn ([`results/longform.log`](../results/longform.log)),
and returned full transcripts. Encoded files and live sessions are limited
to 24 hours [CLAIM]. `clip_start_seconds` and
`clip_end_seconds` select part of an encoded source [CLAIM]; that was not
tested here.

---

Previous: [Setup](03-setup.md) | [Contents](../README.md#contents) | Next: [Using it through this repository](05-harness.md)
