"""Record the hardware and software every Parakeet number in this note came from.

Measures nothing about the model. It captures the CPU model, core and thread
counts, the vector-ISA flags that decide which Photon kernels can run (AVX2,
AVX-VNNI, any AVX-512 or AMX), total RAM, the kernel and libc, Python, and the
versions of every package in the runtime path, plus torch's own view of its
build and thread pool. Finally it runs one real (1 s of silence) Parakeet
transcription in a child process with LD_LIBRARY_PATH and the nix-ld variables
removed, to check whether the prebuilt wheels need any NixOS loader help,
then adds back only the library directories the failures name, one at a time,
and reports which libraries were needed.

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/environment.py \
        > models/parakeet-redux/results/environment.log

Prints no hostname, user name or path under /home.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys

print("### CPU")
cpuinfo = open("/proc/cpuinfo").read()
model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
print(f"  model name (/proc/cpuinfo): {model.group(1).strip() if model else '?'}")

lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
for key in ("Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
            "Socket(s)", "CPU max MHz", "CPU min MHz", "L2 cache", "L3 cache"):
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", lscpu, re.M)
    print(f"  {key}: {m.group(1).strip() if m else '?'}")
print(f"  os.cpu_count(): {os.cpu_count()}")
print(f"  sched_getaffinity: {len(os.sched_getaffinity(0))} CPUs usable by this process")

flags = set()
fm = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M)
if fm:
    flags = set(fm.group(1).split())
print("\n### Vector ISA flags (from /proc/cpuinfo)")
for f in ("sse4_2", "avx", "avx2", "fma", "f16c", "avx_vnni", "avx_vnni_int8",
          "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16", "amx_tile",
          "amx_int8"):
    print(f"  {f:14s} {'yes' if f in flags else 'no'}")
avx512 = sorted(f for f in flags if f.startswith("avx512"))
print(f"  any avx512* flag present: {bool(avx512)} {avx512 if avx512 else ''}")

print("\n### Memory")
mem = re.search(r"^MemTotal:\s*(\d+) kB", open("/proc/meminfo").read(), re.M)
print(f"  MemTotal: {int(mem.group(1)) / 1024 / 1024:.1f} GiB")

print("\n### OS")
print(f"  kernel: {platform.system()} {platform.release()}")
osr = dict(
    line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line
)
print(f"  distribution: {osr.get('PRETTY_NAME', '?').strip(chr(34))}")
print(f"  platform.libc_ver(): {platform.libc_ver()}")
ldd = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()
print(f"  ldd --version: {ldd[0] if ldd else '?'}")

print("\n### Python")
print(f"  {platform.python_implementation()} {platform.python_version()}")

print("\n### Packages in the runtime path (importlib.metadata)")
for dist in ("moondream", "kestrel", "kestrel-native", "kestrel-kernels",
             "kestrel-kernels-bundle-a", "kestrel-kernels-bundle-b",
             "kestrel-kernels-bundle-c", "kestrel-kernels-bundle-d",
             "torch", "numpy", "soundfile", "tokenizers", "safetensors", "huggingface-hub"):
    try:
        print(f"  {dist:26s} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist:26s} not installed")

print("\n### torch")
import torch  # noqa: E402

print(f"  torch.__version__: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
cap = getattr(torch.backends.cpu, "get_cpu_capability", None)
print(f"  torch.backends.cpu.get_cpu_capability(): {cap() if cap else '?'}")
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PLAYGROUND_HOST"):
    print(f"  env {var}: {'set' if os.environ.get(var) else 'unset'}")
sys.stdout.flush()

print("\n### Loader: does Photon run with the Nix loader helpers removed?")
# The shell environment may export LD_LIBRARY_PATH and nix-ld variables. Re-run a real
# transcription in a child process with those removed, to test the claim that
# the prebuilt manylinux wheels need nothing added on NixOS.
child = (
    "import numpy as np, moondream as md\n"
    "with md.photon('moondream/parakeet-redux', device='cpu') as sp:\n"
    "    r = sp.transcribe(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)\n"
    "print('OK transcribed, text =', repr(r['text']))\n"
)
stripped = {k: v for k, v in os.environ.items()
            if k not in ("LD_LIBRARY_PATH", "NIX_LD", "NIX_LD_LIBRARY_PATH", "LD_PRELOAD")}
for var in ("LD_LIBRARY_PATH", "NIX_LD", "NIX_LD_LIBRARY_PATH"):
    print(f"  environment {var}: {'set' if os.environ.get(var) else 'unset'} (removed for the child)")
p = subprocess.run([sys.executable, "-c", child], env=stripped, capture_output=True, text=True)
print(f"  child exit code: {p.returncode}")
for line in p.stdout.splitlines():
    print(f"  child stdout: {line}")
if p.returncode != 0:
    for line in p.stderr.splitlines()[-5:]:
        print(f"  child stderr: {line}")
# Then add back, one at a time, only the LD_LIBRARY_PATH directory that
# provides the library the previous attempt could not find (still no nix-ld),
# until the transcription runs or no provider is found. The libraries this
# needs are what Python wheels in the venv (numpy, torch, Photon) need from
# the system on NixOS.
dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
chosen: list[str] = []
needed: list[str] = []
err = p.stderr
for _ in range(10):
    m = re.search(r"(lib[\w.+-]+\.so[.\d]*): cannot open shared object file", err)
    if not m:
        break
    lib = m.group(1)
    provider = next((d for d in dirs if os.path.exists(os.path.join(d, lib))), None)
    if provider is None or provider in chosen:
        print(f"  missing {lib}: no provider found in the environment's LD_LIBRARY_PATH")
        break
    needed.append(f"{lib} (from {os.path.basename(os.path.dirname(provider)).split('-', 1)[-1]})")
    chosen.append(provider)
    p = subprocess.run([sys.executable, "-c", child],
                       env=dict(stripped, LD_LIBRARY_PATH=":".join(chosen)),
                       capture_output=True, text=True)
    err = p.stderr
print(f"  libraries that had to be added back: {needed or 'none'}")
print(f"  final attempt, LD_LIBRARY_PATH = {len(chosen)} dir(s), no nix-ld: exit code {p.returncode}")
for line in p.stdout.splitlines():
    print(f"  child stdout: {line}")
if p.returncode != 0:
    for line in p.stderr.splitlines()[-3:]:
        print(f"  child stderr: {line}")
print(f"  interpreter is from the Nix store: {os.path.realpath(sys.executable).startswith('/nix/store/')}")
