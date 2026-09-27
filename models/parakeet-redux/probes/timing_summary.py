"""Throughput per runtime, from the nemo_vs_onnx run logs in results/.

Each runtime's log has one line per segment set: "<name>: N segments, H h in S s".
This sums them per runtime and segment set group, and prints real-time factors.
Caveat, and it matters: the runs shared machines (the two CUDA NeMo runs ran at
the same time on one host, the GPU and CPU ONNX runs likewise, and the ROCm run
was capped at one CPU core), so these are indicative throughputs of a batch-1
decode, not benchmarks. Model loading per set is excluded; each line times only
the transcription.

    python models/parakeet-redux/probes/timing_summary.py > models/parakeet-redux/results/nemo_vs_onnx-timing.log
"""

import re
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
LINE = re.compile(r"^(?:== (\S+) .*)$|^(\S+): (\d+) segments, ([\d.]+) h in (\d+) s")

HARDWARE = {
    "nemo_cuda_graphs": "NeMo 3.0.0, RTX 3090 Ti, CUDA graphs on, batch 1",
    "nemo_cuda": "NeMo 3.0.0, RTX 3090, CUDA graphs off, batch 1",
    "nemo_rocm": "NeMo 3.0.0, RX 9070 XT (ROCm 7.2), graphs off, 1 CPU core, batch 1",
    "onnx_fp32_cuda": "sherpa-onnx fp32, RTX 3090 Ti, 3 workers",
    "onnx_int8_cuda": "sherpa-onnx int8, RTX 3090 Ti, 3 workers",
    "onnx_fp32": "sherpa-onnx fp32, Ryzen 7 3700X, 6 workers",
    "onnx_int8": "sherpa-onnx int8, Ryzen 7 3700X, 8 workers",
}


def group(tag: str) -> str:
    if tag.startswith("hvb") and "livepad" in tag:
        return "HarperValleyBank livepad (short segments)"
    if tag.startswith("hvb"):
        return "HarperValleyBank bench"
    return "AppTek bench"


def main():
    acc = defaultdict(lambda: [0, 0.0, 0.0])        # (runtime, group) -> segments, hours, seconds
    for f in sorted(R.glob("nemo_vs_onnx-run-*.log")):
        tag = None
        for line in open(f, encoding="utf-8"):
            m = LINE.match(line.strip())
            if not m:
                continue
            if m.group(1):
                tag = m.group(1)
            elif tag:
                a = acc[(m.group(2), group(tag))]
                a[0] += int(m.group(3)); a[1] += float(m.group(4)); a[2] += int(m.group(5))
    print(f"{'runtime':<16} {'segment set':<42} {'segments':>8} {'hours':>7} {'seconds':>8} {'x real time':>11}")
    for (rt, g), (n, h, s) in sorted(acc.items()):
        print(f"{rt:<16} {g:<42} {n:>8} {h:>7.2f} {s:>8.0f} {h * 3600 / max(s, 1):>11.0f}")
    print()
    for rt, hw in HARDWARE.items():
        print(f"{rt}: {hw}")


if __name__ == "__main__":
    main()
