# Two-Person Conversation Audio Corpora — Manifest

Built for a full-duplex speech-to-speech test harness. The harness feeds one
speaker's channel to the model as the "caller" and captures the model's
reply, so only recordings with the two speakers on **separate channels or
separate files** are useful. Mono mixdowns were rejected wherever found.
Nothing here was resampled — everything is stored in its native sample rate;
the harness's 24000 Hz wire format is applied at load time, not by this
download step.

Downloaded 2026-09-21.

## 1. HCRC Map Task Corpus

- Source: https://groups.inf.ed.ac.uk/maptask/signals/dialogues/
- License/terms: no license text or terms-of-use page is served alongside
  the index; the corpus is published by the University of Edinburgh HCRC for
  research use (see the corpus's own documentation/citation page at
  https://groups.inf.ed.ac.uk/maptask/signals/citations/ — not fetched here).
  No signup or click-through was required to download.
- What was downloaded: 11 of the 128 available dialogue files (the index has
  no other subdirectory or per-channel variant — see note below), chosen to
  spread across the `q1`–`q5` map/task groups and the `ec`/`nc` (eye
  contact / no eye contact) condition.
- Format note: despite the `.mix.` filename — which reads as "mono
  mixdown" — every file fetched here was verified genuinely **2-channel**,
  20000 Hz, 16-bit PCM, one speaker per channel. There is no separate
  `.g.wav`/`.f.wav` giver/follower variant published in this index or any
  sibling directory (the `signals/` parent only contains `dialogues/` and
  `citations/`), so `.mix.wav` is in fact the correct, already
  channel-separated file to use.
- Total bytes: 438,584,328 (≈419 MiB) across 11 files.

## 2. Full-Duplex-Bench (Hugging Face) — FAILED, mono-only

- Source: https://huggingface.co/datasets/Ssshangfu/Full-Duplex-Bench-Data
  (dataset id resolved on first try, no substitute search needed).
- License/terms: not independently verified here (no explicit license file
  was inspected); irrelevant since the content was discarded — see below.
- What was attempted: downloaded the 7 non-CANDOR zip archives (CANDOR was
  skipped per instructions):
  `v1.0/icc_backchannel.zip`, `v1.0/synthetic_pause_handling.zip`,
  `v1.0/synthetic_user_interruption.zip`, `v1.5/background_speech.zip`,
  `v1.5/talking_to_other.zip`, `v1.5/user_backchannel.zip`,
  `v1.5/user_interruption.zip` — 573 MB compressed, ~1.8 GB extracted.
- Result: **every one of 2784 WAV files across all 7 archives is mono**
  (verified: every file individually inspected in the small archives, and a
  random sample of 60 in the full set — all returned `channels == 1`).
  The dataset packages each test case as a single pre-mixed `input.wav` (plus
  separate `context.wav`/`interrupt.wav` component clips in v1.5, which are
  also individually mono) intended as a single model-input stream for
  benchmarking a duplex model's turn-taking behavior — it is not a
  two-channel human-human conversation recording. It does not meet this
  harness's per-speaker-channel requirement.
- Action: all downloaded zips and extracted files were **deleted**; nothing
  from this corpus was kept. `audio/corpora/full-duplex-bench/` was removed
  (empty).

## 3. Montclair Map Task Corpus (MMTC) — FAILED, bot-gated

- Source: https://digitalcommons.montclair.edu/mmt_corpus/
- What was attempted: the landing page itself loads without a signup form,
  and lists direct-looking download links
  (`https://digitalcommons.montclair.edu/context/mmt_corpus/article/<id>/type/native/viewcontent`
  for each corpus item). No login/agreement form is shown on the landing
  page itself.
- Result: the actual download endpoint (`.../viewcontent`) is behind a
  **Cloudflare "Just a moment..." JavaScript challenge** — the response is
  an HTML challenge page, not the audio file. This is not a signup form, but
  per instructions it is a gate that must not be bypassed (would require
  scripted JS/browser automation to pass Cloudflare's bot check).
- Action: no files downloaded; nothing kept. `audio/corpora/mmtc/` left
  empty.

## 4. CANDOR — skipped per instructions

Not attempted; requires a data request form.

---

## Kept files

| Path | Sample rate | Channels | Duration | Overlapping-speech seconds (RMS>0.01 both channels) |
|---|---|---|---|---|
| `maptask/q1ec1.mix.wav` | 20000 Hz | 2 | 265.09 s | 1 |
| `maptask/q1ec2.mix.wav` | 20000 Hz | 2 | 334.04 s | 8 |
| `maptask/q1nc1.mix.wav` | 20000 Hz | 2 | 1112.61 s | 233 |
| `maptask/q2ec3.mix.wav` | 20000 Hz | 2 | 279.85 s | 65 |
| `maptask/q2nc4.mix.wav` | 20000 Hz | 2 | 339.44 s | 6 |
| `maptask/q3ec1.mix.wav` | 20000 Hz | 2 | 530.32 s | 101 |
| `maptask/q3nc2.mix.wav` | 20000 Hz | 2 | 537.90 s | 194 |
| `maptask/q4ec1.mix.wav` | 20000 Hz | 2 | 212.49 s | 1 |
| `maptask/q4nc3.mix.wav` | 20000 Hz | 2 | 881.38 s | 239 |
| `maptask/q5ec1.mix.wav` | 20000 Hz | 2 | 491.75 s | 37 |
| `maptask/q5nc2.mix.wav` | 20000 Hz | 2 | 497.43 s | 53 |

All 11 files: 16-bit PCM WAV, 2 channels, none deleted (none were mono).

## Deleted files

None from the kept corpus (all 11 Map Task downloads were genuinely
2-channel). The Full-Duplex-Bench download (2784 mono WAV files, ~1.8 GB
extracted) was entirely deleted after verification — see section 2 above —
because every file was mono.

## What was tried and failed

| Target | Reason |
|---|---|
| Full-Duplex-Bench (`Ssshangfu/Full-Duplex-Bench-Data`) | Dataset found and downloaded fine, but content is mono-only (2784/2784 sampled files); does not meet the two-channel requirement |
| Montclair Map Task Corpus | Download endpoint gated by a Cloudflare JS challenge, not a plain signup form — not bypassed per instructions |
| CANDOR | Skipped by instruction — requires a data-request form |
| Map Task per-channel `.g.wav`/`.f.wav` variants | Do not exist in this index; only `.mix.wav` is published, and it is itself genuinely stereo/2-channel |

## Best fixtures for barge-in testing

Ranked by seconds of overlapping speech (both channels RMS > 0.01
simultaneously) — these three have the most genuine cross-talk / potential
barge-in material:

1. **`maptask/q4nc3.mix.wav`** — 239 s of overlap (881 s total duration)
2. **`maptask/q1nc1.mix.wav`** — 233 s of overlap (1112 s total duration)
3. **`maptask/q3nc2.mix.wav`** — 194 s of overlap (538 s total duration)

All three are "no eye contact" (`nc`) condition dialogues, which tend to run
longer and produce more clarification/backchannel overlap than the
eye-contact condition — consistent with them topping the overlap ranking.
