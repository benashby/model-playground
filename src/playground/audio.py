"""Audio loading, rate conversion, and wall-clock paced chunking."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import soundfile as sf

from .protocol import CHUNK_MS, WIRE_RATE


def load_channel(path: Path, channel: int | None = None) -> tuple[np.ndarray, int]:
    """Load one channel as mono float32 at the file's NATIVE rate.

    Returns (samples, rate). No resampling happens here, which is the point:
    WIRE_RATE is a VoiceChat *protocol* constant, and a consumer that wants a
    different rate (Parakeet is a 16 kHz model) must not be made to round-trip
    through 24 kHz first. Every resampling pass changes the audio slightly, and
    an ASR transcript measurably changes with it, even with the band-limited
    `_resample` below.
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)

    if channel is None:
        return data.mean(axis=1), rate
    if channel >= data.shape[1]:
        raise ValueError(
            f"{path.name} has {data.shape[1]} channel(s); channel {channel} does not exist"
        )
    return np.ascontiguousarray(data[:, channel]), rate


def load_pcm16(path: Path, channel: int | None = None) -> np.ndarray:
    """Load a WAV/FLAC as mono float32 in [-1, 1], resampled to WIRE_RATE.

    `channel` picks one side of a multi-channel recording -- for a two-party
    conversation captured in stereo, channel 0 is typically the caller. Passing
    None downmixes every channel, which for a two-party recording would feed the
    model BOTH halves of the conversation and is almost never what you want.
    """
    mono, rate = load_channel(path, channel)
    return _resample(mono, rate, WIRE_RATE)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Band-limited rational resample (polyphase FIR, Kaiser-windowed sinc).

    This replaced an unfiltered linear interpolator, which left imaging
    artefacts above the source Nyquist when upsampling and aliased when
    downsampling. Fed to an ASR model, that measurably changed transcripts: on
    one Map Task channel at 20 kHz, 19 of 212 words differed from the same audio
    passed at its native rate (see the Parakeet note). The filter here cuts off
    at the lower of the two Nyquist frequencies, with 16 zero crossings each
    side and a Kaiser beta of 8.6, which puts the stopband near -90 dB.

    Implemented with numpy only, as `up` short convolutions (one per polyphase
    branch) rather than one long one over a zero-stuffed signal. The output is
    aligned to the input (the filter delay is removed) and has
    round(len(x) * dst / src) samples, the same length the old code produced.
    """
    x = np.asarray(x, dtype=np.float64)
    if src == dst:
        return x.astype(np.float32)
    g = math.gcd(int(src), int(dst))
    up, down = dst // g, src // g
    half = 16 * max(up, down)
    k = np.arange(-half, half + 1)
    cutoff = 1.0 / max(up, down)
    h = up * cutoff * np.sinc(cutoff * k) * np.kaiser(2 * half + 1, 8.6)

    n_out = int(round(len(x) * dst / src))
    j = np.arange(n_out, dtype=np.int64) * down + half   # index in the upsampled, filtered signal
    a, phase = np.divmod(j, up)
    y = np.zeros(n_out, dtype=np.float64)
    for p in range(up):
        sel = phase == p
        if not sel.any():
            continue
        branch = np.convolve(x, h[p::up])                # y_up[up*a + p] == branch[a]
        idx = a[sel]
        ok = idx < len(branch)
        vals = np.zeros(idx.shape, dtype=np.float64)
        vals[ok] = branch[idx[ok]]
        y[sel] = vals
    return y.astype(np.float32)


def to_bytes(x: np.ndarray) -> bytes:
    """float32 [-1, 1] -> PCM16 little-endian bytes, with clipping."""
    clipped = np.clip(x, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def write_wav(path: Path, x: np.ndarray, rate: int = WIRE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, rate, subtype="PCM_16")


async def paced_windows(
    samples: np.ndarray,
    *,
    rate: int = WIRE_RATE,
    speed: float = 1.0,
) -> AsyncIterator[tuple[int, np.ndarray]]:
    """Yield (index, float32 window) at wall-clock rate, one chunk per CHUNK_MS.

    Pacing is the entire point. A full-duplex model interleaves listening and
    speaking against real elapsed time; blasting the file as fast as the socket
    accepts it destroys every timing property we want to measure and will make
    the model's turn-taking look broken when it is not.

    Deadlines are absolute against a monotonic clock rather than `sleep(0.08)`
    per iteration, because per-iteration sleeps accumulate the cost of the send
    itself -- roughly 1-3 ms each here, which over a 60-second clip drifts the
    stream a second or more behind real time and silently skews latency numbers.

    `speed > 1.0` compresses the timeline for quick smoke tests. Any latency
    figure taken at speed != 1.0 is meaningless; it exists to check plumbing.
    """
    n = max(1, int(round(rate * CHUNK_MS / 1000.0)))
    period = (CHUNK_MS / 1000.0) / speed
    start = time.monotonic()

    total = (len(samples) + n - 1) // n
    for i in range(total):
        window = samples[i * n : (i + 1) * n]
        if len(window) < n:
            window = np.pad(window, (0, n - len(window)))

        yield i, window

        deadline = start + (i + 1) * period
        drift = deadline - time.monotonic()
        if drift > 0:
            await asyncio.sleep(drift)


async def paced_chunks(
    samples: np.ndarray,
    *,
    speed: float = 1.0,
) -> AsyncIterator[tuple[int, bytes]]:
    """PCM16-bytes view of `paced_windows` at WIRE_RATE, for the wire protocol.

    A view, not a second implementation. There must only ever be one pacing loop
    in this repo; a second one drifts differently from the first and the two
    disagree in a way nothing catches.
    """
    async for i, window in paced_windows(samples, rate=WIRE_RATE, speed=speed):
        yield i, to_bytes(window)
