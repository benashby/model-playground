# Setup

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

Every code block in this article and the next two was run on 2026-09-23
against moondream 2.4.1 on the CPU described in
[the throughput results](06-results-throughput.md). Output shapes and error
messages are copied from [`results/api_surface.log`](../results/api_surface.log),
which [`probes/api_surface.py`](../probes/api_surface.py) produced.

## With plain pip

Outside this repository, two commands are enough. On a machine without an
NVIDIA GPU, install the CPU build of torch first, or pip resolves the default
CUDA build, which is several GB you cannot use. `soundfile` is only needed for
the per-channel examples in the next article:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install moondream soundfile
```

In a fresh venv on this machine those installed torch 2.14.0+cpu and
moondream 2.4.1, and the next article's per-channel example then ran
unchanged. The CPU path needed no GPU, CUDA toolkit or API key. (On NixOS, see
the note on system libraries below.)

The weights (178 MB) download on first use into the standard HuggingFace
cache.

## In this repository

The repository keeps this behind an optional extra, because the ASR stack pulls
in torch plus about 330 MB of proprietary kernel wheels (363 MB installed,
from [`results/licensing.log`](../results/licensing.log)), none of which the
core WebSocket client needs:

```
uv sync --extra asr
```

Two settings in `pyproject.toml` matter and are easy to get wrong.

First, torch must be pinned to the CPU wheel index. The default index resolves
a CUDA build, multiple GB that a Radeon or CPU-only machine cannot use. The pin
is an `[[tool.uv.index]]` entry plus a `[tool.uv.sources]` mapping.

Second, torch must be listed explicitly in the extra, even though `moondream`
pulls it in transitively via `kestrel`. `[tool.uv.sources]` binds only direct
dependencies. Without the explicit line the CPU index is silently ignored and
you get the CUDA build anyway. That happened here once. Always verify:

```
uv run --extra asr python -c "import torch; print(torch.__version__)"
# must print a '+cpu' suffix
```

It printed `2.14.0+cpu` here ([`results/environment.log`](../results/environment.log)).

On NixOS the prebuilt manylinux wheels need two system libraries and nothing
else. [`probes/environment.py`](../probes/environment.py) runs a
transcription in a child process with any `LD_LIBRARY_PATH` and nix-ld
variables removed, then adds back only what the failures ask for. With
nothing added, `import numpy` fails on `libstdc++.so.6`. With the directories
for `libstdc++.so.6` (gcc-15.2.0-lib) and `libz.so.1` (zlib-1.3.2) and no
nix-ld, the transcription ran. No `autoPatchelf` was needed.

Correction: an earlier version said you need no `LD_LIBRARY_PATH` or nix-ld
additions at all, and listed the libraries `ldd` reported for the native
modules. That `ldd` output was never saved, and the first claim is wrong for
this setup: the Nix-store Python used here cannot even import numpy without
`libstdc++`. Photon's own modules needed nothing beyond what numpy did.

---

Previous: [Licensing](02-licensing.md) | [Contents](../README.md#contents) | Next: [Using it through Photon](04-usage.md)
