# Results: what the input audio does

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

## Sample rate: what resampling first does [MEASURED]

Correction: earlier versions of this note said that routing audio "through
VoiceChat's 24 kHz wire rate" and back cost 2.90 % WER. That is not what was
measured. The old probe fed one 24 kHz clip either at its native rate or after
a linear downsample to 16 kHz done by the probe itself, and the 2.90 % was 2
words out of 69. The harness's 24 kHz path was never involved.

[`probes/resample.py`](../probes/resample.py)
([`results/resample.log`](../results/resample.log)) now measures three input
paths for the same channel:

- (a) native: `load_channel`, passed at the file's own rate. Photon resamples
  internally. This is what `playground.asr` does, and the reference.
- (b) harness: `load_pcm16`, which resamples to `WIRE_RATE` (24 kHz) with
  what was then an unfiltered linear interpolator in `audio._resample`, passed
  at 24 kHz. This is what the VoiceChat harness does to every file.
- (c) naive 16 kHz: the same interpolator straight to 16 kHz.

The inputs were the caller channel of the three 24 kHz fixtures and both
channels of two HCRC Map Task dialogues. The Map Task audio is 20 kHz, so path
(b) does change its rate. Transcripts are deterministic: repeating (a)
gave the same text every time, so one transcription per path is the
measurement. The table counts word differences, not errors, because none of
these inputs has a ground truth.

| Input | Rate | Words in (a) | (b) vs (a) | (c) vs (a) | (c) vs (b) |
|---|---|---|---|---|---|
| tool_call ch0 | 24 kHz | 69 | 0 (input identical) | 2 | 2 |
| interruptions ch0 | 24 kHz | 28 | 0 (input identical) | 0 | 0 |
| turn_taking ch0 | 24 kHz | 20 | 0 (input identical) | 0 | 0 |
| q4ec1 ch0 | 20 kHz | 309 | 0 | 4 | 4 |
| q4ec1 ch1 | 20 kHz | 212 | 19 | 52 | 43 |
| q1ec1 ch0 | 20 kHz | 488 | 6 | 7 | 1 |
| q1ec1 ch1 | 20 kHz | 348 | 15 | 22 | 11 |
| Map Task pooled, n=4 | 20 kHz | 1357 | 40 (2.95 %) | 85 (6.26 %) | 59 (4.39 %) |

For files already at 24 kHz, the harness path is a no-op: `_resample` returns
the input unchanged, and the transcript is byte-identical. Resampling to 16 kHz
with the linear interpolator changed 2 words out of 69 on `tool_call`, both of
them dropped fillers ("um" and "uh"), and nothing on the other two fixtures.

For the 20 kHz Map Task audio, both resampling paths changed real words.
Upsampling to 24 kHz through the harness changed 0 to 19 words per channel
(40 of 1357 pooled), and the naive 16 kHz path changed 4 to 52 (85 of 1357).
Among the changes, "as far as" became "this finds", "about a week"
disappeared, and "whereabouts is" became "where about says". One
channel, q4ec1 ch1, accounts for most of it, so the effect depends heavily on
the recording.

With no reference, this cannot say which transcript is closer to what was
said. It does show that the harness's resampler at the time changed the
words, by up to 9 % of them on one channel, which is why
`asr.py` passes audio at its native rate through `load_channel` and never
through `load_pcm16`.

### After replacing the resampler

`audio._resample` now uses a band-limited polyphase filter in place of the
linear interpolator: a Kaiser-windowed sinc that cuts off at the lower of the
two Nyquist frequencies. On test tones it lowers the image that upsampling
leaves from about -6 dB to -39 dB, and the alias that downsampling leaves from
about -3 dB to -90 dB. Rerunning the same probe with the new resampler gave
this ([`results/resample-after-fix.log`](../results/resample-after-fix.log)):

| Input | Words in (a) | (b) vs (a), before | (b) vs (a), after | (c) vs (a), before | (c) vs (a), after |
|---|---|---|---|---|---|
| tool_call ch0 | 69 | 0 | 0 | 2 | 0 |
| interruptions ch0 | 28 | 0 | 0 | 0 | 0 |
| turn_taking ch0 | 20 | 0 | 0 | 0 | 0 |
| q4ec1 ch0 | 309 | 0 | 1 | 4 | 0 |
| q4ec1 ch1 | 212 | 19 | 13 | 52 | 20 |
| q1ec1 ch0 | 488 | 6 | 0 | 7 | 0 |
| q1ec1 ch1 | 348 | 15 | 12 | 22 | 11 |
| Map Task pooled, n=4 | 1357 | 40 (2.95 %) | 26 (1.92 %) | 85 (6.26 %) | 31 (2.28 %) |

The direct 16 kHz path improved most: the two fillers on `tool_call` now
survive, and the pooled Map Task difference fell from 85 words to 31. The
harness path fell from 40 to 26. Neither reached zero, which is what you
would expect: both still change the rate before Photon changes it again, and
on hard recordings such as q4ec1 ch1 the model's output moves with small
changes in its input. The advice is the same as before: pass audio at its
native rate.

## Long form: one 91-minute call [MEASURED]

An earlier version of this note said "You can hand it a 90-minute recording"
with nothing behind it. The vendor's own wording is "You can pass in the whole
recording without setting up a separate VAD model or cutting the audio
yourself" [CLAIM], and the docs limit encoded files to 24 hours [CLAIM].
[`probes/longform.py`](../probes/longform.py)
([`results/longform.log`](../results/longform.log)) now tests it: channel 0 of
all eleven Map Task dialogues concatenated into one 20 kHz WAV of 5482.3 s
(91.4 minutes), passed to `transcribe` as a path.

It failed as recorded. Four of those channels reach full scale, and Photon
rejects near-full-scale audio at any rate other than 16 kHz; see
[the next section](#loud-audio-is-rejected-measured). With the whole file scaled by 0.5
(about 6 dB quieter), one call succeeded:

| Measurement | Value |
|---|---|
| Wall time, one call (n=1) | 133.2 s, 41.2× real time |
| Segments | 902, from 0.24 to 28.96 s long (median 2.80 s), none over 30 s |
| Coverage | first segment starts at 0.00 s, last ends at 5481.40 s of 5482.3 s; start times monotonic |
| Words | 10377 |
| Peak resident memory during the call | 3151 MB |

Transcribing the same eleven (scaled) channels one by one as arrays took
132.2 s, 41.5× real time, so one long call costs nothing in speed. The words
are not the same, though: the long-form transcript differs from the joined
per-file transcripts in 858 of 10406 normalised words (8.25 %: 481
substitutions, 193 deletions, 184 insertions). With no reference transcript
there is no way to say which is closer to the audio, and the two paths differ
in more than length (a file path against arrays, and different segment
boundaries). Open: where the differences fall, and why.

## Loud audio is rejected [MEASURED]

Found while building the long-form test, and mapped by
[`probes/clipping.py`](../probes/clipping.py)
([`results/clipping.log`](../results/clipping.log)). Photon raised
`ValueError: PCM must contain finite samples in [-1, 1]` for input whose
samples were all inside [-1, 1]. It happens only when Photon has to resample:

| Peak of a short pulse | 16 kHz | 20 kHz | 24 kHz | 44.1 kHz | 48 kHz |
|---|---|---|---|---|---|
| 0.90 or below | OK | OK | OK | OK | OK |
| 0.95 | OK | OK | rejected | rejected | rejected |
| 0.99 | OK | rejected | rejected | rejected | rejected |
| 1.00 | OK | rejected | rejected | rejected | rejected |

Real audio is worse than a short pulse. Channel 0 of Map Task q5nc2 (20 kHz,
497.4 s, clipped at full scale) was rejected at gains of 1.0, 0.99, 0.95 and
0.9, as an array or as a 16-bit WAV path, and accepted at 0.8 and 0.5. Four of
the eleven Map Task channel-0 recordings, and three channel-1 recordings, peak
at 1.0000.

The likeliest cause is overshoot in Photon's resampler, which then fails
Photon's own range check. That is an inference, since this investigation cannot
inspect the kernels. In practice, if a recording is loud and is not 16 kHz,
scale it down first (0.5 worked on everything tried here).
Input already at 16 kHz was accepted at full scale in every test, so
resampling to 16 kHz yourself and keeping the result inside [-1, 1] should
also work, but that path was not tested on real audio. None of the three 24 kHz
fixtures is affected, since they peak between 0.13 and 0.59.

The wrapper in this repository now does the scaling itself:
`transcribe_file` and `transcribe_streaming` bring any non-16 kHz input that
peaks above 0.8 down to 0.8, and record the factor. With that change, three
full-scale Map Task channels that Photon had rejected now transcribe, and the
quieter fixtures go through untouched
([`results/asr-fixes-check.log`](../results/asr-fixes-check.log)).

---

Previous: [Results: accuracy and streaming](07-results-accuracy.md) | [Contents](../README.md#contents) | Next: [Method and open questions](09-method.md)
