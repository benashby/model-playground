"""First smoke test: one cold load and one transcription of tool_call.wav channel 0.

A single run (n=1), kept as the record of the first contact with the model.
Its output was never saved, so none of its figures are quoted in the note;
load_footprint.py and pk_exp.py replaced it.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/step0.py
"""
import time, numpy as np, soundfile as sf, moondream as md

WAV = "audio/tool_call.wav"  # run from the repo root
data, rate = sf.read(WAV, dtype="float32", always_2d=True)
ch0 = np.ascontiguousarray(data[:, 0])
dur = len(ch0) / rate
print(f"audio: {dur:.1f}s @ {rate} Hz, channel 0 of {data.shape[1]}")

t0 = time.monotonic()
speech = md.photon("moondream/parakeet-redux", device="cpu")
t_load = time.monotonic() - t0
print(f"cold start (load+compile): {t_load:.1f}s")

t0 = time.monotonic()
r = speech.transcribe(audio=ch0, sample_rate=rate, timestamps="segment")
t_tx = time.monotonic() - t0
print(f"transcribe: {t_tx:.2f}s  ->  RTF {dur/t_tx:.1f}x real time")
print(f"language: {r.get('language')!r}")
print("--- text ---")
print(r["text"][:600])
print("--- segments ---")
for s in r["segments"][:4]:
    print(f"  {s['start']:6.2f} {s['end']:6.2f}  {s['text'][:70]}")
speech.close()
