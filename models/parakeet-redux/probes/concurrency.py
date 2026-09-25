"""CPU cost of live streaming, one stream and many, in one process.

For each concurrency level N in LEVELS, one Photon client (device="cpu") runs N
concurrent `atranscribe(..., stream=True)` streams of the same channel
(turn_taking ch0, 41.1 s, 24 kHz native rate), each fed at wall clock through
`playground.audio.paced_windows`. Stream starts are staggered evenly over
2.0 s, the snapshot cadence, so their snapshots do not all land at once.

Measured per level:

  - cores: process CPU time (user + system, all threads, from os.times())
    divided by wall time, over the whole level and sampled every second;
  - event-loop lag: how late a 100 ms heartbeat sleep wakes up. If Photon
    computed on the asyncio thread, this would grow and the pacer would feed
    audio late, making the model look slow for a reason that is the client's;
  - feed lag: for each stream, when its last chunk was handed over against
    when the pacer's schedule said it should be;
  - first snapshot, snapshot intervals, and final lag (last chunk handed over
    to `aresult()` returning), per stream;
  - whether every stream's final text is byte-identical to the N=1 final.

One client is opened for the whole probe and warmed with a 1-stream pass that
is discarded, so load time and first-call cost stay out of every level.

Run from the repo root (about 5 minutes, most of it real-time audio):

    uv run --extra asr python models/parakeet-redux/probes/concurrency.py \
        > models/parakeet-redux/results/concurrency.log 2>&1

Optional arguments (path channel start_s duration_s levels) select another
fixture; the committed log is the default run above.
"""

import asyncio
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from playground.asr import _headroom, _photon  # noqa: E402
from playground.audio import load_channel, paced_windows  # noqa: E402

# Fixture: argv[1:] = path channel [start_s duration_s] [levels,comma,separated].
ARGS = sys.argv[1:]
FIXTURE = ROOT / (ARGS[0] if ARGS else "audio/turn_taking.wav")
CHANNEL = int(ARGS[1]) if len(ARGS) > 1 else 0
START_S = float(ARGS[2]) if len(ARGS) > 2 else 0.0
DUR_S = float(ARGS[3]) if len(ARGS) > 3 else None
LEVELS = [int(n) for n in ARGS[4].split(",")] if len(ARGS) > 4 else [1, 5, 10, 25]
STAGGER_S = 2.0
HEARTBEAT_S = 0.1


def out(*a):
    print(*a, flush=True)


def cpu_s() -> float:
    t = os.times()
    return t.user + t.system


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


async def one_stream(speech, samples, rate, delay):
    await asyncio.sleep(delay)
    rec = {"first_audio": None, "last_audio": None, "sched_last": None, "snaps": []}
    n_chunks = 0

    async def chunks():
        nonlocal n_chunks
        start = time.monotonic()
        async for i, window in paced_windows(samples, rate=rate):
            now = time.monotonic()
            if rec["first_audio"] is None:
                rec["first_audio"] = now
            rec["last_audio"] = now
            rec["sched_last"] = start + i * 0.08
            n_chunks += 1
            yield window

    updates = await speech.atranscribe(
        audio=chunks(), sample_rate=rate, timestamps="segment", stream=True
    )
    async for _ in updates:
        rec["snaps"].append(time.monotonic())
    final = await updates.aresult()
    rec["final_at"] = time.monotonic()
    rec["text"] = final.get("text", "")
    return rec


async def level(speech, samples, rate, n):
    stop = asyncio.Event()
    lags, cpu_samples = [], []

    async def heartbeat():
        while not stop.is_set():
            t = time.monotonic()
            await asyncio.sleep(HEARTBEAT_S)
            lags.append(time.monotonic() - t - HEARTBEAT_S)

    async def sampler():
        c0, t0 = cpu_s(), time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(1.0)
            c1, t1 = cpu_s(), time.monotonic()
            cpu_samples.append((c1 - c0) / (t1 - t0))
            c0, t0 = c1, t1

    hb, sm = asyncio.create_task(heartbeat()), asyncio.create_task(sampler())
    c_start, t_start = cpu_s(), time.monotonic()
    recs = await asyncio.gather(*(
        one_stream(speech, samples, rate, STAGGER_S * i / n) for i in range(n)
    ))
    c_end, t_end = cpu_s(), time.monotonic()
    stop.set()
    await asyncio.gather(hb, sm)
    return recs, (c_end - c_start) / (t_end - t_start), t_end - t_start, lags, cpu_samples


def report(n, recs, cores, wall, lags, cpu_samples, reference):
    audio_s = None
    first = [r["snaps"][0] - r["first_audio"] for r in recs if r["snaps"]]
    intervals = [b - a for r in recs for a, b in zip(r["snaps"], r["snaps"][1:])]
    feed_lag = [r["last_audio"] - r["sched_last"] for r in recs]
    final_lag = [r["final_at"] - r["last_audio"] for r in recs]
    same = sum(r["text"] == reference for r in recs)
    out(f"\n=== N={n} concurrent streams")
    out(f"  wall for level            : {wall:.2f} s")
    out(f"  cores used, whole level   : {cores:.2f}  ({cores / n:.3f} per stream)")
    if cpu_samples:
        out(f"  cores used, 1 s samples   : median {st.median(cpu_samples):.2f}, "
            f"max {max(cpu_samples):.2f}, n={len(cpu_samples)}")
    out(f"  event-loop lag (100 ms hb): median {st.median(lags) * 1000:.1f} ms, "
        f"p99 {q(lags, 0.99) * 1000:.1f} ms, max {max(lags) * 1000:.1f} ms, n={len(lags)}")
    out(f"  feed lag at last chunk    : median {st.median(feed_lag) * 1000:.1f} ms, "
        f"max {max(feed_lag) * 1000:.1f} ms")
    if first:
        out(f"  first snapshot            : median {st.median(first):.2f} s, "
            f"min {min(first):.2f}, max {max(first):.2f}, streams with one {len(first)}/{n}")
    if intervals:
        out(f"  snapshot interval         : median {st.median(intervals):.2f} s, "
            f"p95 {q(intervals, 0.95):.2f}, max {max(intervals):.2f}, n={len(intervals)}")
    out(f"  final lag after last chunk: median {st.median(final_lag):.2f} s, "
        f"max {max(final_lag):.2f} s")
    out(f"  final text == N=1 final   : {same}/{n}")
    return audio_s


async def main():
    samples, rate = load_channel(FIXTURE, CHANNEL)
    samples = samples[int(START_S * rate):][: int(DUR_S * rate) if DUR_S else None]
    samples, gain = _headroom(samples, rate)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    out(f"fixture: {FIXTURE.name} ch{CHANNEL} from {START_S:.1f} s, {len(samples) / rate:.1f} s at {rate} Hz, gain {gain:.3f}")
    out(f"cpu: os.cpu_count()={os.cpu_count()}, affinity={len(os.sched_getaffinity(0))}")
    try:
        import torch
        out(f"torch threads: {torch.get_num_threads()}, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    except ImportError:
        pass
    out(f"levels: {LEVELS}, stagger {STAGGER_S} s, heartbeat {HEARTBEAT_S * 1000:.0f} ms")

    t = time.monotonic()
    with _photon("moondream/parakeet-redux", "cpu") as speech:
        out(f"client open: {time.monotonic() - t:.2f} s")
        out("warmup: 1 stream, discarded")
        warm, *_ = await level(speech, samples, rate, 1)
        reference = warm[0]["text"]
        out(f"reference final ({len(reference.split())} words): {reference[:120]}...")

        idle_c, idle_t = cpu_s(), time.monotonic()
        await asyncio.sleep(5.0)
        out(f"\nidle cores, client open, 5 s: {(cpu_s() - idle_c) / (time.monotonic() - idle_t):.3f}")

        for n in LEVELS:
            report(n, *await level(speech, samples, rate, n), reference)


if __name__ == "__main__":
    asyncio.run(main())
