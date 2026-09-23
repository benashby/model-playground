"""Parakeet/Photon ASR: reference transcripts and streaming-stability measurement.

DELIBERATELY NOT A PROTOCOL ADAPTER. Parakeet is ASR-only -- no agent, no tools,
no barge-in, no speech-started signal. Routing it through `DuplexSession` would
leave the tool registry and the entire interruption path inert while the run
still looked clean, which is the single most damaging failure mode this repo
guards against. See `.claude/skills/adding-a-model/SKILL.md`.

What it is good for here is two things the harness could not do before:

  1. A reference transcript for the caller channel, so VoiceChat's own caller
     transcription finally has something to be scored against.
  2. A subject for streaming-latency work. `atranscribe` emits *replacement
     snapshots*, and earlier text can change as more context arrives. How much
     it churns, and how long a word takes to stop moving, is the number that
     decides whether a live agent can act on what it just heard -- and no vendor
     benchmark reports it.

Requires the optional extra: `uv sync --extra asr`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_channel, paced_windows
from .client import Transcript

MODEL_CPU = "moondream/parakeet-redux"
MODEL_GPU = "moondream/parakeet-ultra"

# Photon resamples anything that is not 16 kHz, and its resampler overshoots:
# near-full-scale audio then fails with "PCM must contain finite samples in
# [-1, 1]" although every input sample is inside that range. Measured in
# models/parakeet-redux/results/clipping.log: at 16 kHz any peak works; at
# 20-48 kHz a synthetic pulse fails from 0.95 or 0.99, and real 20 kHz speech
# peaking at 1.0 failed at every gain down to 0.90 and worked at 0.80.
PHOTON_NATIVE_RATE = 16000
SAFE_PEAK = 0.8


def _headroom(samples: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Scale audio down to SAFE_PEAK if Photon will resample it and it is loud.

    Returns (samples, gain). Gain is 1.0, and the samples are untouched, for
    16 kHz input or anything already at or below SAFE_PEAK, which covers every
    fixture shipped with this repository. A gain change is the least invasive
    fix: it leaves the rate alone (resampling first changes transcripts; see
    the Parakeet note) and alters nothing but level.
    """
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if rate == PHOTON_NATIVE_RATE or peak <= SAFE_PEAK:
        return samples, 1.0
    gain = SAFE_PEAK / peak
    warnings.warn(
        f"input peaks at {peak:.3f} at {rate} Hz; scaled by {gain:.3f} so Photon's "
        f"internal resampler does not reject it", stacklevel=3)
    return (samples * gain).astype(np.float32), gain


def _photon(model: str, device: str | None = None) -> Any:
    """Open a Photon client, with a useful error when the extra is not installed.

    Imported lazily and on purpose: `moondream` pulls torch plus ~350 MB of
    proprietary Kestrel kernel bundles, and the core WebSocket harness must not
    pay that cost to import a sibling module.
    """
    try:
        import moondream as md
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "playground.asr needs the optional ASR extra: `uv sync --extra asr`"
        ) from exc
    return md.photon(model, device=device) if device else md.photon(model)


def transcribe_file(
    path: Path,
    *,
    channel: int | None = 0,
    model: str = MODEL_CPU,
    device: str | None = "cpu",
    timestamps: str = "word",
) -> dict[str, Any]:
    """Transcribe ONE channel of a recording at its native sample rate.

    `channel` defaults to 0 -- the caller side of a two-party recording -- and
    that default is the whole reason this wraps Photon rather than calling it
    directly. Photon downmixes a multichannel file to mono from its container
    metadata, which for a two-party conversation feeds the model both halves at
    once. Splitting first and passing raw PCM sidesteps that.

    Audio is passed at its NATIVE rate. Do not route it through `load_pcm16`:
    that resamples to WIRE_RATE, a VoiceChat protocol constant, and Photon then
    resamples again to 16 kHz. Even with the band-limited resampler, that
    double conversion changed 26 of 1357 words on Map Task audio relative to
    native-rate input (models/parakeet-redux/results/resample-after-fix.log).

    Returns Photon's own result dict, plus an `input_gain` key when the audio
    had to be scaled down to get past Photon's range check (see `_headroom`).
    """
    samples, rate = load_channel(path, channel)
    samples, gain = _headroom(samples, rate)
    with _photon(model, device) as speech:
        result = speech.transcribe(audio=samples, sample_rate=rate, timestamps=timestamps)
    if gain != 1.0:
        result["input_gain"] = round(gain, 4)
    return result


@dataclass
class StreamResult:
    """A streaming run: the final transcript plus every snapshot on the way."""

    final: dict[str, Any]
    snapshots: list[tuple[float, str]]   # (seconds since the first audio chunk, text)
    audio_seconds: float
    client_open_s: float = 0.0           # time spent opening the Photon client
    input_gain: float = 1.0

    @property
    def first_snapshot_s(self) -> float | None:
        """Seconds from the first audio chunk to the first transcript snapshot."""
        return self.snapshots[0][0] if self.snapshots else None


async def transcribe_streaming(
    path: Path,
    *,
    channel: int | None = 0,
    model: str = MODEL_CPU,
    device: str | None = "cpu",
    speed: float = 1.0,
    log: Transcript | None = None,
) -> StreamResult:
    """Feed one channel at wall-clock rate and record every transcript snapshot.

    Pacing comes from `audio.paced_windows`, the same loop the WebSocket sender
    uses. Reusing it is not tidiness: a `sleep(chunk_ms)` loop accumulates the
    per-iteration send cost and drifts the stream behind real time, which would
    make the model's first-snapshot latency look worse than it is.

    Times are measured from the moment the first audio chunk is handed to
    Photon, not from before the client is opened. An earlier version started
    the clock first, so every snapshot time included 1.2 to 2.8 s of client
    load and the first preview looked like 5 to 7 s instead of ~4 s. Both
    clocks are logged: `since_audio` (the one that matters) and `since_open`.

    Every snapshot is logged RAW, with its arrival time. The stability metrics
    below are computed from that log afterwards rather than during capture --
    capture is the expensive, unrepeatable half, and baking one definition of
    "settled" into it would be the apparatus deciding the measurement.
    """
    samples, rate = load_channel(path, channel)
    samples, gain = _headroom(samples, rate)
    log = log or Transcript()
    audio_seconds = len(samples) / rate
    if gain != 1.0:
        log.record("asr.input_gain", gain=round(gain, 4))

    first_audio: list[float] = []

    async def chunks():
        async for _, window in paced_windows(samples, rate=rate, speed=speed):
            if not first_audio:
                first_audio.append(time.monotonic())
            yield window

    snapshots: list[tuple[float, str]] = []
    t_open = time.monotonic()

    with _photon(model, device) as speech:
        client_open_s = time.monotonic() - t_open
        updates = await speech.atranscribe(
            audio=chunks(), sample_rate=rate, timestamps="segment", stream=True
        )
        async for update in updates:
            now = time.monotonic()
            since_audio = now - (first_audio[0] if first_audio else now)
            text = update.get("text", "")
            snapshots.append((since_audio, text))
            log.record("asr.snapshot", since_audio=round(since_audio, 3),
                       since_open=round(now - t_open, 3), words=len(text.split()), text=text)
        final = await updates.aresult()

    log.record("asr.final", text=final.get("text", ""), snapshots=len(snapshots))
    return StreamResult(final=final, snapshots=snapshots, audio_seconds=audio_seconds,
                        client_open_s=client_open_s, input_gain=gain)


# ---------------------------------------------------------------------------
# Stability metrics, computed offline from logged snapshots.
#
# Two published readings of "how unstable was that stream", kept separate on
# purpose because they answer different questions and disagree in useful ways:
#
#   normalized_erasure  -- how many tokens had to be TAKEN BACK. Matches what a
#       UI can safely commit to screen. One late revision penalises everything
#       before it, so it conflates "often wrong" with "once badly wrong".
#
#   unstable_word_ratio -- what fraction of words shown were not yet their final
#       value. Answers "could an agent have acted on this word". Forgiving of
#       where the churn happened, so a transcript can score stable while the
#       display still visibly flickers.
#
# Reporting only one of these is a choice, not a simplification. Report both.
# ---------------------------------------------------------------------------


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def stability(snapshots: list[tuple[float, str]]) -> dict[str, float | int]:
    """Normalized erasure and unstable-word ratio over a snapshot sequence."""
    if not snapshots:
        return {"snapshots": 0}

    seqs = [text.split() for _, text in snapshots]
    final = seqs[-1]

    erased = 0
    for prev, nxt in zip(seqs, seqs[1:]):
        erased += len(prev) - _common_prefix(prev, nxt)

    shown = sum(len(s) for s in seqs[:-1])
    unstable = sum(
        1
        for s in seqs[:-1]
        for i, w in enumerate(s)
        if i >= len(final) or final[i] != w
    )

    return {
        "snapshots": len(seqs),
        "final_words": len(final),
        "normalized_erasure": round(erased / len(final), 4) if final else 0.0,
        "unstable_word_ratio": round(unstable / shown, 4) if shown else 0.0,
        "words_shown": shown,
    }
