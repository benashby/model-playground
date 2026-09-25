"""CPU cost of live transcription with NVIDIA's parakeet-tdt-0.6b-v3 on ONNX (sherpa-onnx).

The counterpart of `concurrency.py`, which measured Redux on Photon. Same default
fixture (turn_taking ch0, 41.1 s), same wall-clock pacer (`playground.audio.paced_windows`),
same 2.0 s start stagger, same event-loop heartbeat, same CPU accounting
(os.times() user + system over all threads, divided by wall time).

What differs, and has to: Parakeet-TDT is an offline model, and sherpa-onnx has
no streaming mode for it. The live pattern it supports is VAD-segmented
decoding, so each stream here gets its own Silero VAD, and every utterance the
VAD closes is decoded once ("final"). Two modes:

  finals    only the VAD-closed utterances are decoded;
  partials  as finals, plus every 2.0 s of stream time the utterance still in
            progress (`vad.current_segment`) is re-decoded, which is the closest
            analogue of Photon's 2 s replacement snapshots. A stream never has
            more than one partial in flight; a tick that finds one still
            running is counted as skipped rather than queued, because a stale
            partial is worthless.

Decodes never run on the event loop: each is submitted to a ThreadPoolExecutor
of WORKERS threads sharing one OfflineRecognizer with NUM_THREADS intra-op
threads, and the stream keeps feeding while it runs. The VAD does run on the
loop, and the heartbeat would show it if that were expensive.

Silero VAD needs 16 kHz and the fixtures are 20 or 24 kHz, so the channel is resampled
once with `playground.audio._resample` (band-limited) and both the VAD and the
recognizer see the same 16 kHz audio. Photon resampled internally instead, so
transcripts are not expected to match Photon's byte for byte.

Measured per mode and level: cores; event-loop lag; final latency, from the
end of an utterance's audio (its last sample's wall-clock deadline) to its
text being ready, which includes the VAD's silence hangover; the same latency
minus the hangover (from the VAD handing the segment over); partial interval
skips and age (tick to decoded text, the staleness a viewer sees); wall for the level; and whether every stream's concatenated final
text is identical to the N=1 text for that mode.

Models are not in the repository. Fetch them from the sherpa-onnx releases:

    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

and point SHERPA_MODELS at the directory holding both (default
~/.cache/sherpa-onnx). Run from the repo root. The whole machine (16 decode
workers) on the sparse default fixture, about 12 minutes:

    uv run --with sherpa-onnx python models/parakeet-redux/probes/onnx_concurrency.py \
        > models/parakeet-redux/results/onnx_concurrency.log 2>&1

The default fixture is 14 % speech by this VAD, and Map Task q2ec3 ch0 is 76 %,
with utterances up to 15.4 s. VAD-gated decoding costs roughly in proportion to
speech, so the sparse run is a best case and the dense run the one to size by.

To emulate a 2-vCPU cloud VM (one physical core and its hyperthread sibling),
pin the process and match the worker count, on the first 60 s of the dense
fixture (about 11 minutes):

    ONNX_WORKERS=2 ONNX_LEVELS=1,5,10,25 taskset -c 0,1 \
        uv run --with sherpa-onnx python models/parakeet-redux/probes/onnx_concurrency.py \
        audio/corpora/maptask/q2ec3.mix.wav 0 0 60 \
        > models/parakeet-redux/results/onnx_concurrency-dense-2cpu.log 2>&1
"""

import asyncio
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sherpa_onnx as so

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from playground.audio import _resample, load_channel, paced_windows  # noqa: E402

MODELS = Path(os.environ.get("SHERPA_MODELS", Path.home() / ".cache" / "sherpa-onnx"))
PARAKEET = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
# Fixture: argv[1:] = path channel [start_s duration_s]; default turn_taking ch0, all of it.
ARGS = sys.argv[1:]
FIXTURE = ROOT / (ARGS[0] if ARGS else "audio/turn_taking.wav")
CHANNEL = int(ARGS[1]) if len(ARGS) > 1 else 0
START_S = float(ARGS[2]) if len(ARGS) > 2 else 0.0
DUR_S = float(ARGS[3]) if len(ARGS) > 3 else None
RATE = 16_000
NUM_THREADS = 1          # intra-op threads per decode
# concurrent decodes; 16 = physical cores on the test CPU. ONNX_WORKERS and
# ONNX_LEVELS ("1,5,25") override, for emulating a small VM under taskset.
WORKERS = int(os.environ.get("ONNX_WORKERS", 16))
_lv = os.environ.get("ONNX_LEVELS")
_lv = [int(n) for n in _lv.split(",")] if _lv else None
LEVELS = {"finals": _lv or [1, 5, 10, 25, 50, 100], "partials": _lv or [1, 5, 10, 25, 50]}
STAGGER_S = 2.0
PARTIAL_S = 2.0
HEARTBEAT_S = 0.1
VAD_MIN_SILENCE_S = 0.5
VAD_MAX_SPEECH_S = 20.0


def out(*a):
    print(*a, flush=True)


def cpu_s() -> float:
    t = os.times()
    return t.user + t.system


def rss_mb() -> tuple[float, float]:
    """(current RSS, peak RSS) of this process in MB, from /proc/self/status."""
    f = dict(line.split(":", 1) for line in open("/proc/self/status"))
    return int(f["VmRSS"].split()[0]) / 1024, int(f["VmHWM"].split()[0]) / 1024


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def make_recognizer():
    return so.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET / "encoder.int8.onnx"),
        decoder=str(PARAKEET / "decoder.int8.onnx"),
        joiner=str(PARAKEET / "joiner.int8.onnx"),
        tokens=str(PARAKEET / "tokens.txt"),
        num_threads=NUM_THREADS,
        model_type="nemo_transducer",
    )


def make_vad():
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = VAD_MIN_SILENCE_S
    cfg.silero_vad.max_speech_duration = VAD_MAX_SPEECH_S
    cfg.sample_rate = RATE
    cfg.num_threads = 1
    return so.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def decode(rec, samples):
    t0 = time.monotonic()
    s = rec.create_stream()
    s.accept_waveform(RATE, samples)
    rec.decode_stream(s)
    return s.result.text, t0, time.monotonic()


async def one_stream(rec, pool, samples, delay, partials):
    await asyncio.sleep(delay)
    loop = asyncio.get_running_loop()
    vad = make_vad()
    window = vad.config.silero_vad.window_size
    finals, partial_times = [], []
    rec_ = {"final_lat": [], "final_lat_vad": [], "decode_s": [], "queue_s": [], "skipped": 0,
            "partial_age": []}
    partial_task = None
    t0 = None
    pending = np.zeros(0, dtype=np.float32)

    async def run_final(seg_samples, speech_end_wall, emitted):
        submitted = time.monotonic()
        text, d0, d1 = await loop.run_in_executor(pool, decode, rec, seg_samples)
        rec_["final_lat"].append(d1 - speech_end_wall)
        rec_["final_lat_vad"].append(d1 - emitted)
        rec_["decode_s"].append(d1 - d0)
        rec_["queue_s"].append(d0 - submitted)
        return text

    async def run_partial(seg_samples):
        ticked = time.monotonic()
        await loop.run_in_executor(pool, decode, rec, seg_samples)
        partial_times.append(time.monotonic())
        rec_["partial_age"].append(partial_times[-1] - ticked)

    def drain():
        now = time.monotonic()
        while not vad.empty():
            seg = vad.front
            end_wall = t0 + (seg.start + len(seg.samples)) / RATE
            finals.append(asyncio.create_task(
                run_final(np.array(seg.samples, dtype=np.float32), end_wall, now)))
            vad.pop()

    next_partial = PARTIAL_S
    async for i, win in paced_windows(samples, rate=RATE):
        if t0 is None:
            t0 = time.monotonic()
        pending = np.concatenate([pending, win])
        while len(pending) >= window:
            vad.accept_waveform(pending[:window])
            pending = pending[window:]
        drain()
        if partials and (i + 1) * 0.08 >= next_partial:
            next_partial += PARTIAL_S
            if vad.is_speech_detected():
                if partial_task and not partial_task.done():
                    rec_["skipped"] += 1
                else:
                    cur = np.array(vad.current_segment.samples, dtype=np.float32)
                    if len(cur):
                        partial_task = asyncio.create_task(run_partial(cur))
    vad.flush()
    drain()
    texts = await asyncio.gather(*finals)
    if partial_task:
        await partial_task
    rec_["text"] = " ".join(t.strip() for t in texts if t.strip())
    rec_["partial_intervals"] = [b - a for a, b in zip(partial_times, partial_times[1:])]
    rec_["partials"] = len(partial_times)
    return rec_


async def level(rec, pool, samples, n, partials):
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
        one_stream(rec, pool, samples, STAGGER_S * i / n, partials) for i in range(n)
    ))
    wall = time.monotonic() - t_start
    cores = (cpu_s() - c_start) / wall
    stop.set()
    await asyncio.gather(hb, sm)
    return recs, cores, wall, lags, cpu_samples


def report(mode, n, recs, cores, wall, lags, cpu_samples, reference, audio_s):
    fl = [x for r in recs for x in r["final_lat"]]
    flv = [x for r in recs for x in r["final_lat_vad"]]
    dec = [x for r in recs for x in r["decode_s"]]
    qu = [x for r in recs for x in r["queue_s"]]
    out(f"\n=== {mode}, N={n} concurrent streams")
    out(f"  wall for level            : {wall:.2f} s  (audio {audio_s:.1f} s + stagger {STAGGER_S * (n - 1) / n:.2f} s)")
    out(f"  cores used, whole level   : {cores:.2f}  ({cores / n:.3f} per stream)")
    cur, peak = rss_mb()
    out(f"  RSS after level           : {cur:.0f} MB (process peak so far {peak:.0f} MB)")
    out(f"  cores used, 1 s samples   : median {st.median(cpu_samples):.2f}, "
        f"max {max(cpu_samples):.2f}, n={len(cpu_samples)}")
    out(f"  event-loop lag (100 ms hb): median {st.median(lags) * 1000:.1f} ms, "
        f"p99 {q(lags, 0.99) * 1000:.1f} ms, max {max(lags) * 1000:.1f} ms, n={len(lags)}")
    out(f"  final latency, speech end : median {st.median(fl):.2f} s, p95 {q(fl, 0.95):.2f}, "
        f"max {max(fl):.2f}, n={len(fl)}")
    out(f"  final latency, VAD handoff: median {st.median(flv):.2f} s, p95 {q(flv, 0.95):.2f}, "
        f"max {max(flv):.2f}")
    out(f"  final decode time         : median {st.median(dec):.2f} s, max {max(dec):.2f}")
    out(f"  final queue wait          : median {st.median(qu):.3f} s, max {max(qu):.2f}")
    if any(r["partials"] for r in recs):
        pa = [x for r in recs for x in r["partial_age"]]
        out(f"  partial age (tick to text): median {st.median(pa):.2f} s, p95 {q(pa, 0.95):.2f}, "
            f"max {max(pa):.2f}")
        pi = [x for r in recs for x in r["partial_intervals"]]
        out(f"  partials                  : {sum(r['partials'] for r in recs)} decoded, "
            f"{sum(r['skipped'] for r in recs)} ticks skipped (previous still running)")
        if pi:
            out(f"  partial interval (incl. gaps between utterances): median {st.median(pi):.2f} s, p95 {q(pi, 0.95):.2f}, "
                f"max {max(pi):.2f}, n={len(pi)}")
    same = sum(r["text"] == reference for r in recs)
    out(f"  final text == N=1 text    : {same}/{n}")


async def main():
    x, src_rate = load_channel(FIXTURE, CHANNEL)
    x = x[int(START_S * src_rate):][: int(DUR_S * src_rate) if DUR_S else None]
    samples = np.ascontiguousarray(_resample(x, src_rate, RATE), dtype=np.float32)
    audio_s = len(samples) / RATE
    out(f"fixture: {FIXTURE.name} ch{CHANNEL}, from {START_S:.1f} s, {audio_s:.1f} s, "
        f"{src_rate} Hz resampled to {RATE} Hz")
    out(f"sherpa-onnx {so.__version__}; model {PARAKEET.name}")
    out(f"cpu: os.cpu_count()={os.cpu_count()}, affinity={len(os.sched_getaffinity(0))}")
    out(f"num_threads per decode {NUM_THREADS}, decode workers {WORKERS}, "
        f"VAD min silence {VAD_MIN_SILENCE_S} s, max speech {VAD_MAX_SPEECH_S} s")
    out(f"levels {LEVELS}, stagger {STAGGER_S} s, partial every {PARTIAL_S} s")

    t = time.monotonic()
    rec = make_recognizer()
    out(f"recognizer load: {time.monotonic() - t:.2f} s, RSS {rss_mb()[0]:.0f} MB")
    out(f"cpus usable (affinity): {len(os.sched_getaffinity(0))}")

    # Batch reference: the whole channel in one decode, 3 timed after 1 discarded.
    decode(rec, samples)
    ts = []
    for _ in range(3):
        text, d0, d1 = decode(rec, samples)
        ts.append(d1 - d0)
    out(f"batch, whole channel, 1 thread: RTF median {audio_s / st.median(ts):.1f}x "
        f"(n=3), text: {text}")

    idle_c, idle_t = cpu_s(), time.monotonic()
    await asyncio.sleep(5.0)
    out(f"idle cores, recognizer loaded, 5 s: {(cpu_s() - idle_c) / (time.monotonic() - idle_t):.3f}")

    with ThreadPoolExecutor(WORKERS) as pool:
        for mode, levels in LEVELS.items():
            partials = mode == "partials"
            out(f"\nwarmup ({mode}): 1 stream, discarded")
            warm, *_ = await level(rec, pool, samples, 1, partials)
            reference = warm[0]["text"]
            out(f"reference text ({mode}): {reference}")
            for n in levels:
                report(mode, n, *await level(rec, pool, samples, n, partials), reference, audio_s)


if __name__ == "__main__":
    asyncio.run(main())
