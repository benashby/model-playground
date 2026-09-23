"""Measure Parakeet Redux's load time (cold and warm) and its disk footprint.

  A. Warm load: time `md.photon("moondream/parakeet-redux", device="cpu")`
     to return, with the weights already in the HuggingFace cache, then the
     first transcription call (1 s of silence), then close. 6 repeats, each
     in a FRESH Python process so no in-process state carries over; the first
     is discarded as warmup (it pays the OS page cache). n=5 reported.
  B. Cold load: the same, but with HF_HUB_CACHE pointed at an empty temporary
     directory, so the 178 MB of weights, config, tokenizer and ternary map are
     downloaded from huggingface.co inside the timed call. 3 repeats, each
     with a new empty cache. The time depends on this network link on this day
     and is reported as that, not as a property of the model. The temporary
     caches are deleted afterwards; the real cache is never touched.
  C. Disk footprint: the project venv (apparent size of regular files, symlinks
     not followed), the subset of it that the `asr` extra adds (torch and the
     moondream/kestrel stack), and the resolved size of the cached model
     snapshot.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/load_footprint.py \
        > models/parakeet-redux/results/load_footprint.log

Prints sizes and times only, never a path under /home.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r"""
import time, numpy as np
t0 = time.monotonic()
import moondream as md
t_import = time.monotonic() - t0
t1 = time.monotonic()
sp = md.photon("moondream/parakeet-redux", device="cpu")
t_open = time.monotonic() - t1
t2 = time.monotonic()
sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
t_first = time.monotonic() - t2
sp.close()
print(f"RESULT {t_import:.3f} {t_open:.3f} {t_first:.3f}")
"""


def run_child(env: dict) -> tuple[float, float, float]:
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"child failed with exit {p.returncode}")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")][-1]
    a, b, c = (float(x) for x in line.split()[1:])
    return a, b, c


def summary(xs: list[float]) -> str:
    return (f"min {min(xs):.2f} / median {statistics.median(xs):.2f} / max {max(xs):.2f} s"
            f"  n={len(xs)}")


def tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for f in files:
            p = Path(dirpath) / f
            if not p.is_symlink():
                total += p.stat().st_size
    return total


print("### A. warm load (weights cached), fresh process per repeat")
env = dict(os.environ)
rows = [run_child(env) for _ in range(6)]
for i, (a, b, c) in enumerate(rows):
    tag = "  (warmup, discarded)" if i == 0 else ""
    print(f"  run {i}: import moondream {a:.2f} s | md.photon() {b:.2f} s | first transcribe(1 s) {c:.2f} s{tag}")
kept = rows[1:]
print(f"  md.photon() open      : {summary([r[1] for r in kept])}")
print(f"  import moondream      : {summary([r[0] for r in kept])}")
print(f"  first transcribe(1 s) : {summary([r[2] for r in kept])}")
print(f"  import + open         : {summary([r[0] + r[1] for r in kept])}")

print("\n### B. cold load (empty HF cache per repeat; includes the download)")
cold = []
for i in range(3):
    with tempfile.TemporaryDirectory() as tmp:
        e = dict(os.environ, HF_HUB_CACHE=tmp)
        a, b, c = run_child(e)
        size = tree_bytes(Path(tmp))
        cold.append((a, b, c))
        print(f"  run {i}: md.photon() {b:.2f} s (downloaded {size / 1e6:.1f} MB into the empty cache)"
              f" | first transcribe(1 s) {c:.2f} s")
print(f"  md.photon() open, cold: {summary([r[1] for r in cold])}")
print(f"  cold minus warm median: {statistics.median([r[1] for r in cold]) - statistics.median([r[1] for r in kept]):.2f} s"
      " (attributable to the download and first-time file handling)")

print("\n### C. disk footprint")
venv = Path(sys.prefix)
site = next(venv.glob("lib/python*/site-packages"))
print(f"  venv total (regular files, symlinks not followed): {tree_bytes(venv) / 1e9:.2f} GB")
asr_pkgs = ["torch", "torchgen", "functorch", "moondream", "kestrel", "kestrel_native",
            "kestrel_kernels", "kestrel_kernels_bundle_a", "kestrel_kernels_bundle_b",
            "kestrel_kernels_bundle_c", "kestrel_kernels_bundle_d", "tokenizers",
            "safetensors", "triton", "sympy", "networkx"]
asr_total = 0
for name in asr_pkgs:
    p = site / name
    if p.exists():
        b = tree_bytes(p)
        asr_total += b
        print(f"    {name:26s} {b / 1e6:8.1f} MB")
print(f"  of which the packages above: {asr_total / 1e9:.2f} GB")

from huggingface_hub import scan_cache_dir  # noqa: E402

for repo in scan_cache_dir().repos:
    if repo.repo_id == "moondream/parakeet-redux":
        for rev in repo.revisions:
            print(f"  HF cache moondream/parakeet-redux revision {rev.commit_hash[:12]}:"
                  f" {rev.size_on_disk / 1e6:.1f} MB on disk, {len(rev.files)} files")
            for f in sorted(rev.files, key=lambda f: -f.size_on_disk):
                print(f"    {f.file_name:22s} {f.size_on_disk:>11d} bytes")
