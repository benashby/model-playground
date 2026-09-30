"""The host, described by its nature: silicon, usable CPUs, memory. Never a name or address."""
import os
import platform
from pathlib import Path


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def cpu_quota():
    """cgroup v2 CPU limit, which os.cpu_count() does not see."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def describe():
    flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    isa = [f for f in ("avx512f", "avx2", "avx_vnni") if f" {f}" in flags]
    try:
        mem = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith("MemTotal")) / 2**20
    except (OSError, StopIteration):
        mem = float("nan")
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    # "cpu cores" is the physical core count of the socket as the kernel reports it.
    # In a container it describes the host, not what this process may use, and
    # Ollama sizes its thread pool from it.
    cores = next((l.split(":")[1].strip() for l in flags.splitlines() if l.startswith("cpu cores")), "?")
    return (f"{cpu_model()}; isa {'+'.join(isa) or 'none detected'}; os.cpu_count {os.cpu_count()}, "
            f"affinity {usable}, cgroup quota {cpu_quota()}, /proc/cpuinfo cpu cores {cores}; "
            f"{mem:.1f} GiB; {platform.system()} {platform.machine()}")


if __name__ == "__main__":
    print(describe())
