---
name: audio-fixtures
description: Speech-specific — relevant only to models that take audio, which today means the VoiceChat and Parakeet work. Use when sourcing, preparing, or validating test audio — two-party conversation corpora, channel layout, sample-rate conversion, or judging whether a recording can exercise barge-in. Covers the one-speaker-per-channel requirement and why mono mixdowns are useless, the fixtures already available (NVIDIA's shipped demo audio and HCRC Map Task), corpora evaluated and rejected with reasons, the RMS-overlap method for ranking barge-in value, and ffmpeg conversion rules.
---

# Audio fixtures

**Scope: models that take audio.** The two-channel requirement below is a
full-duplex conversation requirement specifically; an ASR model wants clean
single-speaker material instead.

## The one hard requirement — for full-duplex conversation

**Two-party recordings with one speaker per channel** (stereo, or separate
per-speaker files).

The harness feeds *one* side to the model as the caller and captures the
model's reply. A mono mixdown is useless: you cannot feed one side without also
feeding the other side's speech, and the model would hear its counterpart's
turns as the caller's.

Verify every file before trusting it:

```python
import soundfile as sf
i = sf.info(path)
print(i.samplerate, i.channels, i.duration, i.subtype)   # channels must be >= 2
```

## What is available

### NVIDIA's shipped demo audio — the best material

Inside the HF weights repo, unmentioned in the model card. Already at the
24 kHz wire rate, stereo, **ch0 = caller, ch1 = NVIDIA's reference agent**:

| File | Length | Overlapping speech | Use |
|---|---|---|---|
| `tool_call.wav` | 84.5 s | 1 s | tool calling end to end |
| `interruptions.wav` | 30.0 s | **4 s** | barge-in during *speech* |
| `turn_taking.wav` | 41.1 s | 0 s | clean turn-taking baseline |

Channel 1 is a free **ground truth** — captured agent audio should be roughly
its shape.

**Caveat**: none of these puts a slow tool and an interruption in the same
window, so none can exercise barge-in *during a tool call*.

### HCRC Map Task Corpus — the bulk fixture

University of Edinburgh, free direct HTTP, no signup.
`https://groups.inf.ed.ac.uk/maptask/signals/dialogues/`

11 dialogues fetched (419 MiB), **20 000 Hz, 2-channel, 16-bit PCM**. Best
barge-in material by overlap:

| File | Overlap | Total |
|---|---|---|
| `q4nc3.mix.wav` | **239 s** | 881 s |
| `q1nc1.mix.wav` | **233 s** | 1113 s |
| `q3nc2.mix.wav` | **194 s** | 538 s |

Far denser overlap than NVIDIA's fixtures — map-task dialogue is full of
backchannel and interruption.

> **`.mix.` does not mean mono.** Every file verified 2-channel. There is no
> separate `.g.wav`/`.f.wav` variant published; `.mix.wav` *is* the stereo file.

Local manifest: `audio/corpora/MANIFEST.md`.

## Evaluated and rejected

| Corpus | Why not |
|---|---|
| **Full-Duplex-Bench** (`Ssshangfu/Full-Duplex-Bench-Data`) | Downloaded, all 2 784 files deleted. Every WAV is **mono single-stream stimulus**, not two-sided conversation. The benchmark NVIDIA cites, and the wrong shape for this harness. |
| **Montclair Map Task** | Download behind a Cloudflare JS challenge. Not bypassed. 2-channel with Praat TextGrids — worth revisiting if the gate lifts. |
| **CANDOR** | 1 656 conversations, per-participant tracks, but requires a data-request form. Worth pursuing if this work continues. |

The Full-Duplex-Bench result is the one to remember: **the obvious dataset was
the wrong shape**, and only per-file channel inspection revealed it.

## Ranking barge-in value

Per-second RMS per channel; count seconds where both exceed a threshold:

```python
import soundfile as sf, numpy as np
d, sr = sf.read(path, dtype='float32', always_2d=True)
n = int(len(d) / sr)
a = np.array([np.sqrt((d[i*sr:(i+1)*sr,0]**2).mean()) for i in range(n)]) > 0.01
b = np.array([np.sqrt((d[i*sr:(i+1)*sr,1]**2).mean()) for i in range(n)]) > 0.01
print(f"overlap {int((a&b).sum())}/{n}s")
```

Also identifies **which channel is the caller** — whoever speaks first.

## Conversion

The wire is **24 000 Hz mono PCM16**. `audio.py` resamples on load with a
band-limited polyphase filter (it replaced an unfiltered linear interpolator on
2026-09-23). It is good enough that an image or alias sits 40 to 90 dB down,
but any resampling still changes an ASR transcript slightly (see the Parakeet
note), so pass ASR models their native rate. To match the wire rate exactly
with a reference tool, pre-convert:

```bash
ffmpeg -i in.wav -ar 24000 -ac 1 out.wav              # real polyphase filter
ffmpeg -i stereo.wav -map_channel 0.0.0 -ar 24000 a.wav   # split channel 0
```

Do **not** resample to 16 kHz for the WebSocket path. That instruction belongs
to the offline script; the server resamples internally.

## Storage

`audio/` is gitignored (large, re-fetchable). `audio/corpora/MANIFEST.md`
records source, license/terms, format and overlap per file — keep it current
when adding a corpus, including anything tried and rejected and why.
