"""Streaming Parakeet Redux at wall-clock rate: latency, stability, and stream-vs-batch.

For each of 3 passes, for each fixture (caller channel 0 of tool_call,
interruptions, turn_taking; 24 kHz native rate, speed 1.0):

  - open a Photon client (timed), exactly as `playground.asr.transcribe_streaming`
    does per call;
  - feed the channel through `playground.audio.paced_windows` (the harness's
    one pacing loop, absolute monotonic deadlines) into
    `atranscribe(..., timestamps="segment", stream=True)`;
  - log every snapshot RAW with two clocks: seconds since the client was
    opened (what `transcribe_streaming` reports as `first_snapshot_s`, because
    its timer starts before the client is opened) and seconds since the first
    audio chunk was handed to Photon (the latency a live caller experiences);
  - compute normalized erasure and unstable-word ratio with
    `playground.asr.stability`.

After the streaming passes, the same channels are transcribed in batch with the
same `timestamps="segment"` and with `timestamps="none"`, and every streamed
final (`aresult()["text"]`) is compared byte for byte with both.

Section F is the per-run table; F2 summarises first-snapshot latency over the
3 passes (n=3 per fixture, no warmup needed because each pass opens a fresh
client, but pass 1 is shown separately in case it differs); G is stream vs batch.

Raw snapshot logs go to models/parakeet-redux/results/asr-<fixture>-run<k>.jsonl;
`stream_analysis.py` computes every derived streaming figure from them.

Run from the repo root (about 8 minutes, most of it real-time audio):

    uv run --extra asr python models/parakeet-redux/probes/pk_stream.py \
        > models/parakeet-redux/results/pk_stream.log

History: the previous version of this probe printed section G's header and
nothing under it in the committed log. Re-running that version unchanged, with
stderr captured, completed G normally (identical=True on all three fixtures),
so the empty section came from a run that ended before G finished, not from a
fault in G. This version flushes every line, and its committed log was captured
together with stderr, so a crash would show in the log.
"""

from __future__ import annotations

import asyncio
import functools
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import moondream as md  # noqa: E402

from playground.asr import stability  # noqa: E402
from playground.audio import load_channel, paced_windows  # noqa: E402
from playground.client import Transcript  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

FIXTURES = ["tool_call", "interruptions", "turn_taking"]
PASSES = 3
MODEL = "moondream/parakeet-redux"
OUT = Path("models/parakeet-redux/results")


async def stream_one(stem: str, run: int) -> dict:
    samples, rate = load_channel(Path(f"audio/{stem}.wav"), 0)
    log = Transcript()
    first_audio: list[float] = []

    async def chunks():
        async for i, window in paced_windows(samples, rate=rate, speed=1.0):
            if i == 0:
                first_audio.append(time.monotonic())
            yield window

    snaps: list[tuple[float, str]] = []
    t_open0 = time.monotonic()
    with md.photon(MODEL, device="cpu") as speech:
        t_opened = time.monotonic()
        log.record("asr.opened", open_s=round(t_opened - t_open0, 3))
        updates = await speech.atranscribe(
            audio=chunks(), sample_rate=rate, timestamps="segment", stream=True
        )
        async for u in updates:
            now = time.monotonic()
            text = u.get("text", "")
            since_open = now - t_open0
            since_audio = now - first_audio[0]
            snaps.append((since_audio, text))
            log.record("asr.snapshot", since_open=round(since_open, 3),
                       since_audio=round(since_audio, 3), words=len(text.split()), text=text)
        final = await updates.aresult()
        t_end = time.monotonic()
    log.record("asr.final", since_audio=round(t_end - first_audio[0], 3),
               text=final.get("text", ""), snapshots=len(snaps))
    log.save(OUT / f"asr-{stem}-run{run}.jsonl")
    first_open = log.events[1]["since_open"] if len(log.events) > 2 else None
    return {
        "audio_s": len(samples) / rate,
        "open_s": t_opened - t_open0,
        "first_since_open": first_open,
        "first_since_audio": snaps[0][0] if snaps else None,
        "final": final.get("text", ""),
        "stab": stability(snaps),
    }


async def main() -> None:
    results: dict[tuple[str, int], dict] = {}
    print("### F. STREAMING, caller channel, speed 1.0, one fresh client per stream")
    for run in range(1, PASSES + 1):
        for stem in FIXTURES:
            r = await stream_one(stem, run)
            results[(stem, run)] = r
            s = r["stab"]
            print(f"  run{run} {stem:14s} audio={r['audio_s']:5.1f}s open={r['open_s']:4.2f}s "
                  f"snaps={s['snapshots']:3d} first_since_open={r['first_since_open']:5.2f}s "
                  f"first_since_audio={r['first_since_audio']:5.2f}s "
                  f"NE={s['normalized_erasure']:.3f} UPWR={s['unstable_word_ratio']:.4f} "
                  f"final_words={s['final_words']}")

    print("\n### F2. first snapshot latency over passes (n per fixture = %d)" % PASSES)
    for stem in FIXTURES:
        so = [results[(stem, k)]["first_since_open"] for k in range(1, PASSES + 1)]
        sa = [results[(stem, k)]["first_since_audio"] for k in range(1, PASSES + 1)]
        op = [results[(stem, k)]["open_s"] for k in range(1, PASSES + 1)]
        print(f"  {stem:14s} since_open min/med/max {min(so):.2f}/{statistics.median(so):.2f}/{max(so):.2f}s"
              f" | since_audio {min(sa):.2f}/{statistics.median(sa):.2f}/{max(sa):.2f}s"
              f" | client open {min(op):.2f}/{statistics.median(op):.2f}/{max(op):.2f}s")
    allsa = [r["first_since_audio"] for r in results.values()]
    allso = [r["first_since_open"] for r in results.values()]
    print(f"  ALL: since_audio {min(allsa):.2f}-{max(allsa):.2f}s median {statistics.median(allsa):.2f}s;"
          f" since_open {min(allso):.2f}-{max(allso):.2f}s median {statistics.median(allso):.2f}s;"
          f" n={len(allsa)}")
    stab_same = all(
        results[(stem, k)]["stab"] == results[(stem, 1)]["stab"]
        for stem in FIXTURES for k in range(1, PASSES + 1)
    )
    print(f"  stability metrics identical across passes: {stab_same}")

    print("\n### G. STREAM-vs-BATCH (streamed aresult()['text'] vs batch transcribe)")
    with md.photon(MODEL, device="cpu") as sp:
        for stem in FIXTURES:
            x, rate = load_channel(Path(f"audio/{stem}.wav"), 0)
            b_seg = sp.transcribe(audio=x, sample_rate=rate, timestamps="segment")["text"]
            b_none = sp.transcribe(audio=x, sample_rate=rate, timestamps="none")["text"]
            for k in range(1, PASSES + 1):
                st = results[(stem, k)]["final"]
                print(f"  {stem:14s} run{k} identical_to_batch_segment={st == b_seg} "
                      f"identical_to_batch_none={st == b_none} words={len(st.split())}")
                if st != b_seg:
                    print(f"     batch : {b_seg}")
                    print(f"     stream: {st}")


asyncio.run(main())
