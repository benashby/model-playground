"""The host, described by its nature: silicon, usable CPUs, memory. Never a name or address."""
import os
import platform
import sys
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


def describe_windows():
    """The same facts from Win32: there is no /proc, cgroup or `cpu cores` line."""
    import ctypes
    import winreg
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
        model = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40, PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
    isa = [n for n, f in (("avx512f", 41), ("avx2", 40)) if k32.IsProcessorFeaturePresent(f)]

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MemStatus()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(sysm))
    return (f"{model}; isa {'+'.join(isa) or 'none detected'}; os.cpu_count {os.cpu_count()}, "
            f"affinity {bin(proc.value).count('1')}; {m.ullTotalPhys / 2**30:.1f} GiB; "
            f"{platform.system()} {platform.release()} {platform.version()} {platform.machine()}")


def describe():
    if sys.platform == "win32":
        return describe_windows()
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
