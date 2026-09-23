"""Print Photon's transcription API surface for Parakeet Redux, as it actually behaves.

Measures no timing. For the installed moondream release it prints:

  A. the models `md.photon_models()` registers;
  B. the top-level keys of a transcription result, with the type and value of
     each scalar (so `language` is shown, not assumed);
  C. the keys of a segment, a word and a character entry for every
     `timestamps` mode, with one example of each;
  D. what happens when each parameter the note says Parakeet rejects is passed
     (`language`, `initial_prompt`, `task="translate"`, `temperature`,
     `top_p`, `condition_on_previous_text`): the exception type and message,
     or "ACCEPTED" if it did not raise;
  E. what happens when a stereo file path is passed directly, against each
     channel transcribed separately (does Photon downmix?);
  F. the streaming-update shapes from `atranscribe(..., stream=True)`, and
     the type of `aresult()`;
  G. the fields of `playground.asr.StreamResult`;
  H. whether a path that is a symbolic link to a WAV is accepted.

Uses the caller channel of `audio/interruptions.wav` (30 s, 24 kHz stereo).

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/api_surface.py \
        > models/parakeet-redux/results/api_surface.log
"""

from __future__ import annotations

import asyncio
import importlib.metadata as im
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import moondream as md

sys.path.insert(0, str(Path(__file__).parent))
from textnorm import wer  # noqa: E402

WAV = Path("audio/interruptions.wav")
MODEL = "moondream/parakeet-redux"


def short(v, n=70):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


print(f"moondream {im.version('moondream')}, kestrel {im.version('kestrel')}")

print("\n### A. md.photon_models()")
models = md.photon_models()
print(f"  {len(models)} models registered")
for m in models:
    print(f"  {m}")
speech_like = [m for m in models if any(k in m.lower() for k in ("parakeet", "whisper", "asr"))]
print(f"  speech models (name contains parakeet/whisper/asr): {speech_like}")

data, rate = sf.read(str(WAV), dtype="float32", always_2d=True)
ch0 = np.ascontiguousarray(data[:, 0])
ch1 = np.ascontiguousarray(data[:, 1])
print(f"\nfixture: {WAV.name}, {data.shape[0] / rate:.1f} s, {rate} Hz, {data.shape[1]} channels")

with md.photon(MODEL, device="cpu") as speech:
    print("\n### B. top-level result keys (timestamps='segment', caller channel)")
    r = speech.transcribe(audio=ch0, sample_rate=rate, timestamps="segment")
    print(f"  type(result) = {type(r).__name__}")
    for k in r:
        v = r[k]
        if isinstance(v, (list, dict)):
            print(f"  {k:26s} {type(v).__name__} (len {len(v)})")
        else:
            print(f"  {k:26s} {type(v).__name__:8s} {short(v)}")

    print("\n### C. entry shapes per timestamps mode")
    for mode in ("none", "segment", "word", "character"):
        try:
            r = speech.transcribe(audio=ch0, sample_rate=rate, timestamps=mode)
        except Exception as e:  # noqa: BLE001
            print(f"  {mode:9s} RAISED {type(e).__name__}: {e}")
            continue
        segs = r.get("segments") or []
        print(f"  {mode:9s} segments={len(segs)}"
              + (f" segment_keys={sorted(segs[0].keys())}" if segs else ""))
        if segs:
            s0 = segs[0]
            print(f"            first segment: start={s0['start']} end={s0['end']} text={short(s0['text'], 50)}")
            if "words" in s0:
                w0 = s0["words"][0]
                print(f"            word_keys={sorted(w0.keys())} first word={w0}")
            if "characters" in s0:
                c0 = s0["characters"][0]
                print(f"            character_keys={sorted(c0.keys())} first char={c0}"
                      f" n_chars_in_seg0={len(s0['characters'])}")

    print("\n### D. parameters the docs say Parakeet rejects")
    probes = [
        ("language='en'", {"language": "en"}),
        ("initial_prompt='hurricane'", {"initial_prompt": "hurricane"}),
        ("task='translate'", {"task": "translate"}),
        ("task='transcribe'", {"task": "transcribe"}),
        ("temperature=0.0", {"temperature": 0.0}),
        ("top_p=0.9", {"top_p": 0.9}),
        ("condition_on_previous_text=False", {"condition_on_previous_text": False}),
        ("settings={'temperature': 0.0}", {"settings": {"temperature": 0.0}}),
    ]
    for label, kw in probes:
        try:
            speech.transcribe(audio=ch0, sample_rate=rate, timestamps="none", **kw)
            print(f"  {label:34s} ACCEPTED (no exception)")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:34s} {type(e).__name__}: {e}")

    print("\n### E. stereo path passed directly vs per-channel")
    t_path = speech.transcribe(audio=str(WAV), timestamps="none")["text"]
    t0 = speech.transcribe(audio=ch0, sample_rate=rate, timestamps="none")["text"]
    t1 = speech.transcribe(audio=ch1, sample_rate=rate, timestamps="none")["text"]
    tm = speech.transcribe(audio=data.mean(axis=1).astype(np.float32), sample_rate=rate,
                           timestamps="none")["text"]
    print(f"  path (stereo file): {t_path}")
    print(f"  channel 0 only    : {t0}")
    print(f"  channel 1 only    : {t1}")
    print(f"  our own mean(ch0,ch1) as PCM identical to path result: {tm == t_path}")
    w_path = len(t_path.split())
    print(f"  words: path={w_path} ch0={len(t0.split())} ch1={len(t1.split())}")
    print(f"  WER(path vs ch0 as reference) = {wer(t0, t_path)['wer'] * 100:.1f}%")
    print(f"  WER(path vs ch1 as reference) = {wer(t1, t_path)['wer'] * 100:.1f}%")

    print("\n### F. streaming through Photon directly")

    async def stream():
        async def chunks():
            n = rate // 10  # 100 ms chunks, pushed as fast as possible (shape check only)
            for i in range(0, len(ch0), n):
                yield ch0[i:i + n]

        updates = await speech.atranscribe(audio=chunks(), sample_rate=rate,
                                           timestamps="segment", stream=True)
        print(f"  type(atranscribe(..., stream=True)) = {type(updates).__name__}")
        n_up, first = 0, None
        async for u in updates:
            if first is None:
                first = u
            n_up += 1
        final = await updates.aresult()
        print(f"  updates received: {n_up}")
        if first is not None:
            print(f"  type(update) = {type(first).__name__}, keys = {sorted(first.keys())}")
        print(f"  type(aresult()) = {type(final).__name__}, keys = {sorted(final.keys())}")
        print(f"  aresult()['text'] == batch ch0 text: {final['text'] == t0}")

    asyncio.run(stream())

    print("\n### H. a path that is a symlink to the fixture")
    import os  # noqa: E402
    import tempfile  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp) / "link.wav"
        os.symlink(WAV.resolve(), link)
        try:
            speech.transcribe(audio=str(link), timestamps="none")
            print("  symlinked path: accepted")
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(str(link), "<tmp>/link.wav")
            print(f"  symlinked path: {type(e).__name__}: {msg}")

    print("\n### G. playground.asr.StreamResult")
    sys.path.insert(0, "src")
    from playground.asr import StreamResult  # noqa: E402
    import dataclasses  # noqa: E402

    for f in dataclasses.fields(StreamResult):
        print(f"  field {f.name}: {f.type}")
    print("  property first_snapshot_s")
