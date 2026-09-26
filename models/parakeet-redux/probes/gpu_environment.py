"""Versions of everything in a GPU runtime environment, for the deployment notes.

Run inside each environment that produced results (the NeMo CUDA container,
the NeMo ROCm container, the sherpa-onnx CUDA image); prints the Python,
PyTorch, CUDA or HIP, NeMo and sherpa-onnx versions it finds, and the GPUs it
sees. The host's NVIDIA driver version comes from nvidia-smi, run outside.
No host names are printed.

    python gpu_environment.py            # inside the container
"""

import importlib.metadata as md
import platform
import sys


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "not installed"


print(f"python {platform.python_version()} ({sys.platform})")
for dist in ("torch", "nemo_toolkit", "sherpa-onnx", "onnxruntime", "cuda-bindings",
             "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
    print(f"  {dist:<26} {version(dist)}")
try:
    import torch
    print(f"torch.version.cuda {torch.version.cuda}, torch.version.hip {torch.version.hip}, "
          f"cudnn {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        arch = p.gcnArchName if torch.version.hip else f"sm_{p.major}{p.minor}"
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}, {arch}, {p.total_memory / 2**30:.1f} GiB")
except ImportError:
    print("torch not installed")
