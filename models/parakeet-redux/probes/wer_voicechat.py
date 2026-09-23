"""Score VoiceChat's own caller transcription against Parakeet Redux's.

Reference: Parakeet Redux, batch, caller channel 0 at native rate.
Hypothesis: VoiceChat's `caller.said` events from the two committed September
session logs (results/voicechat-{tool_call,interruptions}.jsonl), joined with
spaces in log order.

Both sides go through `textnorm.normalise` (see its docstring for the exact
method). The WER is printed three ways:

  - with the OLD normaliser that used to be in `pk_exp.py` (copied verbatim
    below), to reproduce the previously published figures;
  - with the new normaliser, fillers kept (the headline figure);
  - with the new normaliser, fillers ("um", "uh", ...) removed on both sides.

Every non-matching alignment step is printed, so a reader can see what the
errors are rather than only how many.

Neither side is ground truth; this measures agreement between two ASR systems.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/wer_voicechat.py \
        > models/parakeet-redux/results/wer_voicechat.log
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import moondream as md  # noqa: E402

from playground.audio import load_channel  # noqa: E402
from textnorm import errors_only, normalise, wer  # noqa: E402

RESULTS = Path("models/parakeet-redux/results")


def old_norm(s):  # verbatim from the previous pk_exp.py
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    nums = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twenty": "20", "fifty": "50",
            "hundred": "100", "thirty": "30", "forty": "40"}
    return [nums.get(w, w) for w in s.split()]


def old_wer(ref, hyp):  # verbatim from the previous pk_exp.py
    r, h = old_norm(ref), old_norm(hyp)
    if not r:
        return None, 0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / len(r), len(r)


with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    for stem in ["tool_call", "interruptions"]:
        x, rate = load_channel(Path(f"audio/{stem}.wav"), 0)
        ref = sp.transcribe(audio=x, sample_rate=rate, timestamps="none")["text"]
        lines = (RESULTS / f"voicechat-{stem}.jsonl").read_text().splitlines()
        said = [json.loads(ln).get("text", "") for ln in lines if '"caller.said"' in ln]
        hyp = " ".join(s for s in said if s)

        print(f"=== {stem}")
        print(f"  parakeet  (ref): {ref}")
        print(f"  voicechat (hyp): {hyp}")
        print(f"  normalised ref : {' '.join(normalise(ref))}")
        print(f"  normalised hyp : {' '.join(normalise(hyp))}")
        ow, on = old_wer(ref, hyp)
        print(f"  OLD normaliser : WER {ow * 100:.1f}% over {on} ref words")
        for strip in (False, True):
            r = wer(ref, hyp, strip_fillers=strip)
            label = "fillers removed" if strip else "fillers kept   "
            print(f"  NEW, {label}: WER {r['wer'] * 100:.1f}% = {r['errors']} errors "
                  f"(S={r['S']} D={r['D']} I={r['I']}) over {r['ref_words']} ref words")
            for e in errors_only(r["ops"]):
                print(f"      {e}")
