"""Parakeet experiment suite: repeats, all fixtures, WER vs VoiceChat.

Produced results/pk_exp.log. Kept unchanged so that log stays reproducible.

Still current: section A (throughput, 3 runs x 3 fixtures x 2 channels, warmup
discarded) and section B (per-channel transcripts).

Superseded, and kept only as the record of what was previously published:
  C (WER vs VoiceChat): its `norm` maps only some number words, cannot produce
    11-19 and turns "twenty-five" into "20 5". Replaced by wer_voicechat.py,
    which uses textnorm.py.
  D (timestamp modes): one run per mode. Replaced by timestamps_modes.py
    (n=5 per mode per fixture, warmup discarded).
  E (sample rate): one clip, and it compared native input against a linear
    downsample to 16 kHz done here, not the harness's 24 kHz path. Replaced
    by resample.py.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/pk_exp.py
"""
import json, time, statistics, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from playground.audio import load_channel
import moondream as md

FIX = [Path(f"audio/{n}.wav") for n in ["tool_call","interruptions","turn_taking"]]

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    nums = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6",
            "seven":"7","eight":"8","nine":"9","ten":"10","twenty":"20","fifty":"50",
            "hundred":"100","thirty":"30","forty":"40"}
    return [nums.get(w,w) for w in s.split()]

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r: return None, 0
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.int32)
    d[:,0] = np.arange(len(r)+1); d[0,:] = np.arange(len(h)+1)
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(r[i-1]!=h[j-1]))
    return d[len(r),len(h)]/len(r), len(r)

print("### A. THROUGHPUT: 3 runs x 3 fixtures x both channels (warmup discarded)")
with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    t=time.monotonic(); sp.transcribe(audio=np.zeros(16000,dtype=np.float32), sample_rate=16000)
    print(f"  warmup discard: {time.monotonic()-t:.2f}s")
    results={}
    for f in FIX:
        for ch in (0,1):
            x, rate = load_channel(f, ch); dur=len(x)/rate
            rtfs=[]
            for i in range(3):
                t0=time.monotonic()
                r=sp.transcribe(audio=x, sample_rate=rate, timestamps="segment")
                el=time.monotonic()-t0; rtfs.append(dur/el)
            results[(f.stem,ch)]=(r["text"], rtfs, dur)
            print(f"  {f.stem:14s} ch{ch} {dur:5.1f}s  RTF {min(rtfs):5.1f}/{statistics.median(rtfs):5.1f}/{max(rtfs):5.1f} (min/med/max)")
    allr=[v for _,(_,rr,_) in results.items() for v in rr]
    print(f"  ALL RUNS: median {statistics.median(allr):.1f}x  spread {min(allr):.1f}-{max(allr):.1f}x  n={len(allr)}")

    print("\n### B. TRANSCRIPTS per channel (is ch0 always the caller?)")
    for (stem,ch),(txt,_,_) in results.items():
        print(f"  {stem:14s} ch{ch}: {txt[:95]}")

    print("\n### C. WER: VoiceChat's own caller transcript vs Parakeet reference")
    for stem in ["tool_call","interruptions"]:
        lp=Path(f"models/parakeet-redux/results/voicechat-{stem}.jsonl")  # committed copy of the Sept VoiceChat session logs
        if not lp.exists(): continue
        said=[json.loads(l).get("text","") for l in lp.read_text().splitlines()
              if '"caller.said"' in l]
        vc=" ".join(s for s in said if s)
        ref=results[(stem,0)][0]
        w,nw = wer(ref, vc)
        print(f"  {stem}: VoiceChat WER vs Parakeet ref = {w*100:.1f}% over {nw} ref words")
        print(f"     parakeet : {ref[:110]}")
        print(f"     voicechat: {vc[:110]}")

    print("\n### D. TIMESTAMP GRANULARITY (does mode change the text?)")
    x,rate = load_channel(FIX[0], 0)
    for mode in ["none","segment","word","character"]:
        try:
            t0=time.monotonic(); r=sp.transcribe(audio=x, sample_rate=rate, timestamps=mode)
            segs=r.get("segments") or []
            first=segs[0] if segs else {}
            keys=sorted(first.keys()) if first else []
            print(f"  {mode:10s} {time.monotonic()-t0:5.2f}s segments={len(segs):3d} seg_keys={keys} text_identical={r['text']==results[('tool_call',0)][0]}")
        except Exception as e:
            print(f"  {mode:10s} ERROR {type(e).__name__}: {str(e)[:90]}")

    print("\n### E. SAMPLE-RATE SENSITIVITY (does feeding 24k vs 16k change WER?)")
    x24,_ = load_channel(FIX[0], 0)
    ref = results[("tool_call",0)][0]
    for sr_label, arr, sr in [("native 24k", x24, 24000)]:
        r=sp.transcribe(audio=arr, sample_rate=sr, timestamps="none")
        print(f"  {sr_label}: identical to reference = {r['text']==ref}")
    idx=np.linspace(0,len(x24)-1,int(len(x24)*16000/24000))
    x16=np.interp(idx,np.arange(len(x24)),x24).astype(np.float32)
    r16=sp.transcribe(audio=x16, sample_rate=16000, timestamps="none")
    w,_=wer(ref, r16["text"])
    print(f"  linear-resampled to 16k then declared 16k: WER vs native = {w*100:.2f}%")
