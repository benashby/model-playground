"""End-to-end tests of the GStreamer examples, file and live G.711 RTP, on Linux or Windows.

The same six tests as test_gstreamer.sh, driven from Python so they also run on
Windows, where there is no `timeout -s INT` and a shell pipe between two native
programs is not reliably binary-clean. Each gst-launch-1.0 | python pipe here is
two processes joined by an OS pipe, and a live receiver is stopped the way an
operator would stop it: SIGINT on Linux, Ctrl+Break to its process group on
Windows (gst_appsink.py maps that to KeyboardInterrupt).

Needs gst-launch-1.0 on PATH, or GSTREAMER_1_0_ROOT_MSVC_X86_64 set (the
official Windows installer sets it), PyGObject for tests 3 and 4, sherpa-onnx in
the running interpreter, and MODEL_DIR, VAD_MODEL and MODELS_DIR as for the
examples. From the repo root:

    uv run --with sherpa-onnx python models/parakeet-redux/examples/test_gstreamer.py \\
        logs/clip90.wav logs/gst-out

The committed log was made from the first 90 s of AppTek
en_US_General_Agriculture_1586590_channel1.wav, at 16 kHz mono.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"


def gst_launch() -> str:
    exe = shutil.which("gst-launch-1.0")
    root = os.environ.get("GSTREAMER_1_0_ROOT_MSVC_X86_64")
    if exe is None and root:
        exe = str(Path(root) / "bin" / "gst-launch-1.0.exe")
    if exe is None or not Path(exe).exists():
        sys.exit("gst-launch-1.0 not found: put it on PATH or set GSTREAMER_1_0_ROOT_MSVC_X86_64")
    return exe


GST = gst_launch()
CLIP = Path(sys.argv[1]).resolve().as_posix()
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LIVE_S = 97          # how long a live receiver runs before it is stopped, as in test_gstreamer.sh

CAPS = "caps=application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0"
SEND = (f"filesrc location={CLIP} ! wavparse ! audioconvert ! audioresample ! audio/x-raw,rate=8000,channels=1"
        " ! mulawenc ! rtppcmupay ! udpsink host=127.0.0.1 port=5004")
TAIL = "! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1"
RTP = f"udpsrc port=5004 {CAPS} ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec {TAIL}"
FILE = f"filesrc location={CLIP} ! decodebin {TAIL}"

# Windows: stdout of the python side must be UTF-8 whatever the console code page.
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
GROUP = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}


def gst_args(pipeline: str, eos: bool = False) -> list[str]:
    # gst-launch parses its own argv, so split on spaces exactly as the shell script did.
    return [GST, "-q"] + (["-e"] if eos else []) + pipeline.split()


def interrupt(p: subprocess.Popen) -> None:
    if WINDOWS:
        p.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(p.pid, signal.SIGINT)


def stdin_pipe(pipeline: str, script: str, out: Path, live: bool) -> subprocess.Popen:
    """gst-launch ... ! fdsink fd=1 | python script > out; returns the python process."""
    src = subprocess.Popen(gst_args(f"{pipeline} ! fdsink fd=1", eos=live), stdout=subprocess.PIPE, **GROUP)
    dst = subprocess.Popen([sys.executable, str(HERE / script)], stdin=src.stdout,
                           stdout=out.open("wb"), env=ENV)
    src.stdout.close()                     # the python side now holds the only read end
    dst.src = src
    return dst


def send_realtime() -> float:
    t0 = time.monotonic()
    subprocess.run(gst_args(SEND), check=True)
    return time.monotonic() - t0


def stop_after(p: subprocess.Popen, started: float, target: subprocess.Popen | None = None) -> None:
    time.sleep(max(0.0, LIVE_S - (time.monotonic() - started)))
    interrupt(target or p)
    p.wait()


def lines(path: Path, nonblank: bool = False) -> int:
    text = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for l in text if l.strip()) if nonblank else len(text)


def main() -> int:
    print("== 1 gst_stdin.py, file", flush=True)
    stdin_pipe(FILE, "gst_stdin.py", OUT / "1.txt", live=False).wait()
    print(lines(OUT / "1.txt"), flush=True)

    print("== 2 gst_stdin.py, live G.711 RTP (sent in real time, 90 s)", flush=True)
    t = time.monotonic()
    p = stdin_pipe(RTP, "gst_stdin.py", OUT / "2.txt", live=True)
    time.sleep(3)
    print(f"   sender ran {send_realtime():.0f} s", flush=True)
    stop_after(p, t, target=p.src)
    print(lines(OUT / "2.txt"), flush=True)

    print("== 3 gst_appsink.py, file", flush=True)
    with (OUT / "3.txt").open("wb") as f:
        subprocess.run([sys.executable, str(HERE / "gst_appsink.py"), f"{FILE} ! appsink name=sink"],
                       stdout=f, env=ENV, check=True)
    print(lines(OUT / "3.txt"), flush=True)

    print("== 4 gst_appsink.py, live G.711 RTP, stopped with " + ("Ctrl+Break" if WINDOWS else "SIGINT"),
          flush=True)
    t = time.monotonic()
    p = subprocess.Popen([sys.executable, str(HERE / "gst_appsink.py"), f"{RTP} ! appsink name=sink"],
                         stdout=(OUT / "4.txt").open("wb"), env=ENV, **GROUP)
    time.sleep(5)
    send_realtime()
    stop_after(p, t)
    print(lines(OUT / "4.txt"), flush=True)

    print("== 5 nemotron_stream.py (streaming model), file", flush=True)
    stdin_pipe(FILE, "nemotron_stream.py", OUT / "5.txt", live=False).wait()
    print(lines(OUT / "5.txt", nonblank=True), flush=True)

    print("== 6 nemotron_stream.py, live G.711 RTP (sent in real time, 90 s)", flush=True)
    t = time.monotonic()
    p = stdin_pipe(RTP, "nemotron_stream.py", OUT / "6.txt", live=True)
    time.sleep(3)
    send_realtime()
    stop_after(p, t, target=p.src)
    print(lines(OUT / "6.txt", nonblank=True), flush=True)

    same = lambda a, b: (OUT / a).read_bytes() == (OUT / b).read_bytes()  # noqa: E731
    print(f"== identical: file stdin vs appsink: {'yes' if same('1.txt', '3.txt') else 'no'}; "
          f"RTP stdin vs appsink: {'yes' if same('2.txt', '4.txt') else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
