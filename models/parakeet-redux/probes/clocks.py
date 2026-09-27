"""Can this Python, on this OS, time the live probes? The asyncio clock and the sleep it schedules.

Every live-timing probe here (onnx_concurrency.py, concurrency.py, pk_stream.py,
streaming_online.py's pacing) paces audio with asyncio and reads latencies from
time.monotonic(), which is also asyncio's own clock. Two things can make that
apparatus wrong without anything failing:

  the clock's resolution  before Python 3.13, time.monotonic() on Windows is
                          GetTickCount64 and advances in 15.625 ms steps;
  the sleep's precision   Windows' default timer fires every ~15.6 ms, so a
                          sleep overshoots unless the process asks for 1 ms
                          (timeBeginPeriod, which playground.audio now does).

This runs a 100 ms asyncio heartbeat 100 times and reports its lag twice, as
time.monotonic() sees it and as time.perf_counter() (a high-resolution clock on
every platform here) sees it. The second is the true lag; a difference between
them is the clock misreporting. On Windows it runs once with the default timer
and once with 1 ms; Linux has no such setting, so it runs once. Machine load is
reported alongside, because a busy machine also delays wake-ups.

Run from the repo root with each interpreter to compare (no dependencies):

    python models/parakeet-redux/probes/clocks.py >> models/parakeet-redux/results/clocks-<os>.log
"""

import asyncio
import os
import platform
import statistics as st
import sys
import time

WINDOWS = sys.platform == "win32"
BEATS = 100
PERIOD_S = 0.1


def machine_load() -> str:
    if not WINDOWS:
        return f"loadavg(1 min) {os.getloadavg()[0]:.2f} on {os.cpu_count()} CPUs"
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue"],
                         capture_output=True, text=True).stdout.strip()
    return f"CPU {float(out.replace(',', '.')):.0f} % (Processor(_Total)) on {os.cpu_count()} CPUs"


async def heartbeat() -> tuple[list[float], list[float]]:
    mono, perf = [], []
    for _ in range(BEATS):
        m0, p0 = time.monotonic(), time.perf_counter()
        await asyncio.sleep(PERIOD_S)
        mono.append((time.monotonic() - m0 - PERIOD_S) * 1000)
        perf.append((time.perf_counter() - p0 - PERIOD_S) * 1000)
    return mono, perf


def p99(xs: list[float]) -> float:
    return sorted(xs)[min(len(xs) - 1, round(0.99 * (len(xs) - 1)))]


def main() -> None:
    info = time.get_clock_info("monotonic")
    print(f"## Python {platform.python_version()} on {platform.system()} {platform.release()}; "
          f"time.monotonic() = {info.implementation}, resolution {info.resolution * 1000:.3f} ms")
    timers = ["default", "1ms"] if WINDOWS else ["default"]
    for timer in timers:
        if timer == "1ms":
            import ctypes
            ctypes.WinDLL("winmm").timeBeginPeriod(1)
        load_before = machine_load()
        mono, perf = asyncio.run(heartbeat())
        print(f"  timer {timer:7s}: lag as time.monotonic() sees it: median {st.median(mono):6.2f} ms, "
              f"p99 {p99(mono):6.2f} | true lag (perf_counter): median {st.median(perf):6.2f} ms, "
              f"p99 {p99(perf):6.2f} | n={BEATS}; before: {load_before}; after: {machine_load()}")


if __name__ == "__main__":
    main()
