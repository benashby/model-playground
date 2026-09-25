# Using it through this repository

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

## Usage, through this harness

`src/playground/asr.py` wraps Photon so that the channel and sample-rate traps
described in "Using it through Photon" cannot corrupt the measurement:

```python
import asyncio
from pathlib import Path

from playground.asr import stability, transcribe_file, transcribe_streaming

result = transcribe_file(Path("audio/interruptions.wav"), channel=0)
print(result["text"])

res = asyncio.run(transcribe_streaming(Path("audio/interruptions.wav"), channel=0))
print(res.final["text"])        # .final is Photon's result dict, not a string
print(len(res.snapshots), res.first_snapshot_s)
print(stability(res.snapshots))
```

`transcribe_file(path, *, channel=0, model=..., device="cpu", timestamps="word")`
transcribes one channel at the file's native sample rate and returns Photon's
own result dict. Photon rejects loud audio that it has to resample
([loud audio](08-results-input.md#loud-audio-is-rejected-measured)), so if the
audio peaks above 0.8 and is not at 16 kHz, the function first scales it down
to that peak, raises a warning, and adds an `input_gain` key with the factor
to the result. Audio at 16 kHz, or already peaking below 0.8,
goes through untouched. Three full-scale 20 kHz recordings that failed before
this step was added now transcribe (`results/asr-fixes-check.log`).

`transcribe_streaming(path, *, channel=0, speed=1.0, log=None)` feeds one
channel through the harness's wall-clock pacer into `atranscribe` and records
every snapshot, with its arrival time, in a `Transcript` log. It returns a
`StreamResult` with:

- `.final`: Photon's final result `dict`, the same shape as `transcribe`
  returns. The transcript is `res.final["text"]`. Correction: an earlier
  version of this note listed `.final` as if it were the transcript itself.
- `.snapshots`: a list of `(seconds, text)` pairs, timed from the first audio
  chunk.
- `.audio_seconds`: the length of the channel.
- `.first_snapshot_s`: seconds from the first audio chunk to the first
  snapshot. Over six runs on the fixtures it was 4.00 to 4.02 s
  (`results/asr-fixes-check.log`).
- `.client_open_s`: how long the Photon client took to open (1.39 to 1.55 s
  in the same runs). It is reported on its own so it stays out of the latency.
- `.input_gain`: 1.0 unless the audio had to be scaled down.

Correction: until 2026-09-23 the clock started before the client was opened,
so `first_snapshot_s` included the client's load time and read 5 to 7 s. See
[streaming latency](07-results-accuracy.md#streaming-latency-and-stability-measured).
The log's snapshot events now carry both `since_audio` and `since_open`.

`stability(snapshots)` computes the two churn metrics described in
[accuracy and streaming](07-results-accuracy.md), offline, from that log.

The `speed` parameter compresses the timeline for plumbing checks. Leave it at
`1.0`: any latency number taken at `speed != 1.0` is meaningless.

## Why it is here but not in the harness

The onboarding skill classifies ASR-only models as out of scope for the duplex
harness and names Parakeet among them, and that judgement held up. Parakeet has
no agent, no tool call, no barge-in and no speech-started signal, so it cannot
exercise what the harness exists to measure. Driving it
through `DuplexSession` would leave the tool registry, the interruption policy
and the whole barge-in path inert while the run still looked clean. The
`evaluating` skill names that as the most damaging failure mode available.

`src/playground/asr.py` is therefore a sibling utility, with no protocol adapter, no
`--protocol` flag, no adapter registry and no `policy.py` involvement. The
refactor from `protocol.py` to `protocols/` is still reserved for a second
streaming speech-to-speech backend, and this is not one.

It is in the repository for three reasons:

1. A reference transcript for existing fixtures. The two-party recordings in
   `audio/` have no ground truth, so VoiceChat's own caller transcription had
   nothing to be scored against until now.
2. The ASR leg of a future cascade arm. Cascades are in scope, and comparing a
   full-duplex model against a cascade requires a cascade.
3. A real timing subject. Its streaming interface emits replacement snapshots
   whose earlier text can change. Measuring that trade-off between latency and
   stability needs the harness's wall-clock pacer, which is why this is a
   playground job instead of a standalone script.

## Live dictation from a microphone

[`dictate.py`](../dictate.py) is a small tool for trying the model by talking
to it. It is not a probe and measures nothing. It runs NVIDIA's original
checkpoint on sherpa-onnx with the same Silero VAD setup as
[`probes/onnx_concurrency.py`](../probes/onnx_concurrency.py), so it behaves
like the configuration in [CPU sizing on ONNX](10-results-cpu-sizing.md),
and it does not use Photon or Redux.

```bash
uv run --with sherpa-onnx python models/parakeet-redux/dictate.py
uv run --with sherpa-onnx python models/parakeet-redux/dictate.py --wav audio/tool_call.wav
```

The microphone is captured with `pw-record` at 16 kHz, the rate both models
run at, so nothing is resampled. Each sentence prints as a line once the
speaker has paused for 0.5 s. While someone is still talking, a dim draft of
the current sentence is redrawn in place on the last line about once a second,
cut to the terminal width so that it never wraps. One decode thread handles
both, so sentences always print in the order they were spoken. Ctrl+C stops
the capture and prints the last sentence before exiting.

`--wav` plays channel 0 of a file through `ffmpeg -re` in real time in place
of the microphone. That is how the tool was checked: on the fixtures, under a
pseudo-terminal, including a Ctrl+C sent to the whole process group. It has
not been tried here with a live microphone. Headphones are not needed, because
nothing is played back.

---

Previous: [Using it through Photon](04-usage.md) | [Contents](../README.md#contents) | Next: [Results: hardware and throughput](06-results-throughput.md)
