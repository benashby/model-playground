"""Check the three src/playground fixes that came out of this investigation.

1. Loud audio: transcribe_file on the Map Task channels that peak at 1.0 at
   20 kHz. Before the fix every one raised Photon's range error
   (results/clipping.log). Now they should transcribe, with `input_gain` set.
2. Quiet audio is untouched: the NVIDIA fixtures peak far below 0.8, so no
   gain is applied and no `input_gain` key appears.
3. Streaming clock: `first_snapshot_s` is measured from the first audio chunk,
   so it should now read ~4.0 s (the vendor's documented figure), with the
   client's load time reported separately as `client_open_s`.

    uv run --extra asr python models/parakeet-redux/probes/asr_fixes_check.py
"""
import asyncio, sys, warnings
from pathlib import Path

sys.path.insert(0, "src")
from playground.asr import transcribe_file, transcribe_streaming  # noqa: E402

warnings.simplefilter("ignore")
MAP = Path("audio/corpora/maptask")

print("### 1. loud 20 kHz audio (peak 1.0), which used to raise")
for name, ch in (("q5nc2.mix.wav", 0), ("q3nc2.mix.wav", 1), ("q2ec3.mix.wav", 0)):
    try:
        r = transcribe_file(MAP / name, channel=ch, timestamps="segment")
        print(f"  {name} ch{ch}: OK, {len(r['text'].split())} words, input_gain={r.get('input_gain')}")
    except Exception as e:
        print(f"  {name} ch{ch}: FAILED {type(e).__name__}: {e}")

print("\n### 2. quiet 24 kHz fixtures: no gain should be applied")
for name in ("tool_call.wav", "interruptions.wav", "turn_taking.wav"):
    r = transcribe_file(Path("audio") / name, channel=0, timestamps="segment")
    print(f"  {name}: {len(r['text'].split())} words, input_gain key present: {'input_gain' in r}")

print("\n### 3. streaming clock, caller channel, three runs each")
for name in ("interruptions.wav", "turn_taking.wav"):
    for rep in range(3):
        res = asyncio.run(transcribe_streaming(Path("audio") / name, channel=0))
        print(f"  {name} run {rep+1}: first_snapshot_s={res.first_snapshot_s:.2f}  "
              f"client_open_s={res.client_open_s:.2f}  snapshots={len(res.snapshots)}")
