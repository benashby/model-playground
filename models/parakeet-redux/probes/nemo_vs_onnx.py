"""NeMo against sherpa-onnx on identical audio segments: is the WER gap the runtime?

wer_telephone.py's fp32 ONNX pipeline scores 1.0 to 1.5 points above the AppTek
paper's parakeet v3 figures, and the extra errors are mostly short deleted
words ("okay", "yeah"). This probe takes segmentation out of the question: it
cuts each channel once with wer_telephone.py's own VAD code, saves every
segment as a 16 kHz WAV, and has both runtimes transcribe exactly those files.

  cut      host: segment the first N channel files of an AppTek accent with the
           named segmentation, write logs/wer/nemo_vs_onnx/<tag>/segments/*.wav
           and a manifest (file_name, segment index, path, seconds)
  onnx     host: decode every segment with the sherpa-onnx export (int8 or
           fp32, PARAKEET_PRECISION) and write onnx_<precision>.jsonl
  nemo     inside a NeMo environment (see below): decode every segment with
           nvidia/parakeet-tdt-0.6b-v3 via NeMo's ASRModel.transcribe and write
           nemo.jsonl. Needs only this file, not the repository's packages.
  score    host: join each runtime's segment texts per channel in order and
           score them with AppTek's score.py (same scorer as wer_telephone.py),
           then list the words each runtime deletes most

Run from the repo root:

    uv run --extra asr --with sherpa-onnx --with jiwer --with openai-whisper==20250625 \\
        python models/parakeet-redux/probes/nemo_vs_onnx.py cut apptek:en-US_General clean bench 0
    PARAKEET_PRECISION=fp32 uv run ... nemo_vs_onnx.py onnx <tag>
    <NeMo environment> python nemo_vs_onnx.py nemo <tag dir> --name nemo_rocm [--graphs]
    uv run ... nemo_vs_onnx.py score <tag>

<tag> is printed by `cut`, e.g. apptek-en-US_General_clean_bench. `cut` takes the
same corpus, condition and segmentation names as wer_telephone.py, and N=0 for
every channel. Each runtime's output is <tag dir>/<name>.jsonl; `score` scores
every one it finds.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Inside the NeMo container this file is mounted on its own, outside the repo;
# the `nemo` step takes its directory as an argument and never uses ROOT/OUT.
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE
OUT = ROOT / "logs" / "wer" / "nemo_vs_onnx"


def out(*a):
    print(*a, flush=True)


def _cut_one(job):
    """Condition and segment one channel; write its segments. Runs in a worker process."""
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(HERE))
    import wer_telephone as wt

    fi, fn, wav, cond, seg, d = job
    x, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    y = wt.condition(x[:, 0], rate, cond, fi)      # same per-file seed as wer_telephone.py
    rows = []
    for si, s in enumerate(wt._segments(y, seg)):
        p = d / "segments" / f"{fi:04d}_{si:04d}.wav"
        sf.write(str(p), np.asarray(s, dtype=np.float32), wt.RATE, subtype="FLOAT")
        rows.append({"file_name": fn, "seg": si, "path": p.name, "seconds": len(s) / wt.RATE})
    return rows, len(y) / wt.RATE


def cmd_cut(corpus: str, cond: str, seg: str, n: int, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(HERE))
    import wer_telephone as wt

    its = sorted(wt.items(corpus), key=lambda t: t[0])[: n or None]
    tag = f"{corpus.replace(':', '-')}_{cond}_{seg}" + (f"_first{n}" if n else "")
    d = OUT / tag
    (d / "segments").mkdir(parents=True, exist_ok=True)
    with open(d / "ref.jsonl", "w", encoding="utf-8") as ref:
        for fn, _, text in its:
            ref.write(json.dumps({"file_name": fn, "text": text}) + "\n")
    manifest, chan_s = [], 0.0
    jobs = [(fi, fn, wav, cond, seg, d) for fi, (fn, wav, _) in enumerate(its)]
    with ProcessPoolExecutor(workers) as ex:
        for rows, secs in ex.map(_cut_one, jobs):
            manifest += rows
            chan_s += secs
    with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    seg_s = sum(m["seconds"] for m in manifest)
    out(f"tag {tag}: {len(its)} channels ({chan_s / 3600:.2f} h), {len(manifest)} segments "
        f"({seg_s / 3600:.2f} h), float32 WAV so every runtime reads identical samples")


def _manifest(d: Path):
    return [json.loads(l) for l in open(d / "manifest.jsonl", encoding="utf-8") if l.strip()]


def _onnx_init():
    sys.path.insert(0, str(HERE))
    import wer_telephone as wt
    wt._init()


def _onnx_one(path: str) -> str:
    import soundfile as sf
    import wer_telephone as wt
    y, _ = sf.read(path, dtype="float32")
    st = wt._rec.create_stream()
    st.accept_waveform(wt.RATE, y)
    wt._rec.decode_stream(st)
    return st.result.text.strip()


def _onnx_name(wt) -> str:
    """onnx_fp32 / onnx_int8 on the CPU; a suffix names any other provider (onnx_fp32_cuda)."""
    return f"onnx_{wt.PRECISION}" + ("" if wt.PROVIDER == "cpu" else f"_{wt.PROVIDER}")


def cmd_onnx(tag: str, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(HERE))
    import wer_telephone as wt

    d = OUT / tag
    man = _manifest(d)
    t0 = time.monotonic()
    with ProcessPoolExecutor(workers, initializer=_onnx_init) as ex, \
            open(d / f"{_onnx_name(wt)}.jsonl", "w", encoding="utf-8") as f:
        paths = [str(d / "segments" / m["path"]) for m in man]
        for m, text in zip(man, ex.map(_onnx_one, paths, chunksize=4)):
            f.write(json.dumps({**m, "text": text}) + "\n")
    secs = sum(m["seconds"] for m in man)
    out(f"{_onnx_name(wt)}: {len(man)} segments, {secs / 3600:.2f} h in {time.monotonic() - t0:.0f} s "
        f"with {workers} workers")


def cmd_nemo(d: Path, device: str, batch: int, graphs: bool, name: str) -> None:
    """Runs inside the NeMo environment; imports nothing from the repository."""
    import torch
    import nemo
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    out(f"nemo {nemo.__version__}, torch {torch.__version__}, device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cuda" and torch.version.hip:
        # ROCm on gfx1201 (RX 9070 XT, ROCm 7.2): MIOpen's LSTM kernels fail on the
        # TDT prediction network's LSTM with miopenStatusUnknownError. Disabling the
        # vendor library makes PyTorch use its generic kernels for the LSTM. The
        # encoder has no LSTM, and the numbers are the same maths either way.
        torch.backends.cudnn.enabled = False
        out("ROCm: torch.backends.cudnn (MIOpen) disabled for the LSTM")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location=device)
    model.eval()
    # NeMo's default TDT greedy decoding, except CUDA graphs: they are a CUDA-only
    # speed path and change no results, and they are not available on ROCm.
    dec = model.cfg.decoding
    with open_dict(dec):
        dec.greedy.use_cuda_graph_decoder = graphs
    model.change_decoding_strategy(dec)
    out(f"decoding strategy: {dec.strategy}, greedy: {dict(dec.greedy)}")
    man = _manifest(d)
    paths = [str(d / "segments" / m["path"]) for m in man]
    t0 = time.monotonic()
    with torch.inference_mode():
        hyps = model.transcribe(paths, batch_size=batch, verbose=False)
    texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for m, t in zip(man, texts):
            f.write(json.dumps({**m, "text": t.strip()}) + "\n")
    secs = sum(m["seconds"] for m in man)
    out(f"{name}: {len(man)} segments, {secs / 3600:.2f} h in {time.monotonic() - t0:.0f} s")


def cmd_score(tag: str) -> None:
    import collections
    import jiwer
    sys.path.insert(0, str(HERE))
    import wer_telephone as wt

    d = OUT / tag
    for hyp in sorted(d.glob("*.jsonl")):
        if hyp.name in ("manifest.jsonl", "ref.jsonl") or hyp.name.endswith((".pred.jsonl", ".log")):
            continue
        per = collections.defaultdict(list)
        for l in open(hyp, encoding="utf-8"):
            o = json.loads(l)
            per[o["file_name"]].append((o["seg"], o["text"]))
        pred = d / f"{hyp.stem}.pred.jsonl"
        with open(pred, "w", encoding="utf-8") as f:
            for fn, segs in per.items():
                f.write(json.dumps({"file_name": fn,
                                    "text": " ".join(t for _, t in sorted(segs) if t)}) + "\n")
        s = wt.score(d / "ref.jsonl", pred)
        out(f"{hyp.stem:<12} WER {float(s['WER']) * 100:6.2f} %  "
            f"(S {s['Substitutions']}, D {s['Deletions']}, I {s['Insertions']}, hits {s['Hits']})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut"); c.add_argument("corpus"); c.add_argument("cond"); c.add_argument("seg")
    c.add_argument("n", type=int, help="first N channels; 0 = all"); c.add_argument("--workers", type=int, default=12)
    o = sub.add_parser("onnx"); o.add_argument("tag"); o.add_argument("--workers", type=int, default=12)
    n = sub.add_parser("nemo"); n.add_argument("dir", type=Path)
    n.add_argument("--device", default="cuda"); n.add_argument("--batch", type=int, default=1)
    n.add_argument("--graphs", action="store_true",
                   help="keep NeMo's default CUDA-graphs decoder (NVIDIA CUDA only)")
    n.add_argument("--name", default="nemo", help="output name, e.g. nemo_rocm, nemo_cuda_graphs")
    s = sub.add_parser("score"); s.add_argument("tag")
    a = p.parse_args()
    if a.cmd == "cut":
        cmd_cut(a.corpus, a.cond, a.seg, a.n, a.workers)
    elif a.cmd == "onnx":
        cmd_onnx(a.tag, a.workers)
    elif a.cmd == "nemo":
        cmd_nemo(a.dir, a.device, a.batch, a.graphs, a.name)
    else:
        cmd_score(a.tag)


if __name__ == "__main__":
    main()
