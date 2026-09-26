"""How much of each fixture channel is speech, by the Silero VAD `onnx_concurrency.py` uses.

VAD-gated decoding costs roughly in proportion to speech, so a CPU figure from
a mostly silent channel is a best case. This measures, per channel: length,
the share of samples inside VAD speech segments, the number of segments, and
the longest one (which bounds how expensive a partial re-decode can get).

Same VAD settings as `onnx_concurrency.py` (imported from it), same 16 kHz
band-limited resample. Run from the repo root (under a minute):

    uv run --with sherpa-onnx python models/parakeet-redux/probes/speech_density.py \\
        > models/parakeet-redux/results/speech_density.log 2>&1
"""

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_concurrency import RATE, VAD_MAX_SPEECH_S, VAD_MIN_SILENCE_S, make_vad  # noqa: E402
from playground.audio import _resample, load_channel  # noqa: E402

FILES = [("audio/tool_call.wav", c) for c in (0, 1)]
FILES += [("audio/turn_taking.wav", 0), ("audio/interruptions.wav", 0)]
FILES += [(str(Path(f).relative_to(ROOT)), c)
          for f in sorted(glob.glob(str(ROOT / "audio/corpora/maptask/*.wav")))[:4] for c in (0, 1)]

print(f"VAD: silero, min silence {VAD_MIN_SILENCE_S} s, max speech {VAD_MAX_SPEECH_S} s, {RATE} Hz")
# The slice onnx_concurrency-dense-2cpu.log was measured on.
FILES += [("audio/corpora/maptask/q2ec3.mix.wav", 0, 60.0)]

for f, c, *dur in FILES:
    x, r = load_channel(ROOT / f, c)
    if dur:
        x = x[: int(dur[0] * r)]
        f = f"{f} (first {dur[0]:.0f} s)"
    y = _resample(x, r, RATE)
    v = make_vad()
    w = v.config.silero_vad.window_size
    for i in range(0, len(y) - w, w):
        v.accept_waveform(y[i:i + w])
    v.flush()
    speech = n = longest = 0
    while not v.empty():
        L = len(v.front.samples)
        speech, n, longest = speech + L, n + 1, max(longest, L)
        v.pop()
    print(f"{f} ch{c}: {len(y) / RATE:.1f} s at {r} Hz, speech {speech / len(y) * 100:.0f} %, "
          f"{n} segments, longest {longest / RATE:.1f} s", flush=True)
