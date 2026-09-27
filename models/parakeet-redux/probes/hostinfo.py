"""Host facts the probes print, on Linux and on Windows.

The probes were written on Linux and read these from /proc and
os.sched_getaffinity. On Linux this module does exactly that, so a Linux log
is unchanged. On Windows it asks Win32 for the same facts through ctypes:

  usable_cpus()  the process affinity mask (GetProcessAffinityMask)
  cpu_model()    the registry's ProcessorNameString
  rss_mb()       working set and peak working set (GetProcessMemoryInfo),
                 the nearest Windows equivalent of VmRSS and VmHWM
  pin_cpus()     SetProcessAffinityMask, the stand-in for `taskset -c`

The working set is not the same accounting as Linux RSS (it can be trimmed
under memory pressure, and shared pages count differently), so compare memory
figures across the two systems with that in mind.
"""

from __future__ import annotations

import os
import sys

WINDOWS = sys.platform == "win32"

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(ctypes.c_size_t)]
    _k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def usable_cpus() -> list[int]:
    """CPU indices this process may run on (os.sched_getaffinity on Linux)."""
    if not WINDOWS:
        return sorted(os.sched_getaffinity(0))
    proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
    if not _k32.GetProcessAffinityMask(_k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(sysm)):
        raise ctypes.WinError(ctypes.get_last_error())
    return [i for i in range(proc.value.bit_length()) if proc.value >> i & 1]


def pin_cpus(cpus: list[int]) -> None:
    """Restrict this process to `cpus`, like `taskset -c` (os.sched_setaffinity on Linux)."""
    if not WINDOWS:
        os.sched_setaffinity(0, cpus)
        return
    mask = sum(1 << c for c in cpus)
    if not _k32.SetProcessAffinityMask(_k32.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())


def cpu_model() -> str:
    """The CPU's marketing name (/proc/cpuinfo `model name` on Linux)."""
    if not WINDOWS:
        with open("/proc/cpuinfo") as f:
            return next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "?")
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        return "?"


def rss_mb() -> tuple[float, float]:
    """(current, peak) resident memory of this process in MB.

    Linux: VmRSS and VmHWM from /proc/self/status. Windows: working set and
    peak working set.
    """
    if not WINDOWS:
        with open("/proc/self/status") as f:
            d = dict(line.split(":", 1) for line in f)
        return int(d["VmRSS"].split()[0]) / 1024, int(d["VmHWM"].split()[0]) / 1024
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20


_announced = False


def photon_device() -> str:
    """The device Photon probes run on: PHOTON_DEVICE, default "cpu".

    Every Photon number in the note was measured on a CPU. Photon's Windows
    kernels have no int8 path for the ternary weights on x86 (only a scalar
    reference, which Photon refuses to run), so on Windows Redux runs only
    with PHOTON_DEVICE=cuda. A run on any other device says so in its own log,
    once, so its numbers cannot be read as CPU figures.
    """
    global _announced
    device = os.environ.get("PHOTON_DEVICE", "cpu")
    if device != "cpu" and not _announced:
        print(f"# PHOTON_DEVICE={device}: Photon ran on {device}, not on the CPU the note measured",
              flush=True)
        _announced = True
    return device


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (ru_maxrss on Linux)."""
    if not WINDOWS:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rss_mb()[1]
