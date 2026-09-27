"""Transcribe a GStreamer pipeline in-process, pulling audio from an appsink.

For applications that already run GStreamer (PyGObject) and want the
transcriber inside the same process. The pipeline's last elements must
deliver 16 kHz mono S16LE into an appsink named "sink":

    python gst_appsink.py 'filesrc location=call.wav ! decodebin ! audioconvert ! audioresample
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! appsink name=sink'

    python gst_appsink.py 'udpsrc port=5004
        caps="application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0"
        ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec ! audioconvert ! audioresample
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! appsink name=sink'

Buffers are pulled on this script's own thread with try_pull_sample rather
than handled in a new-sample callback: decoding must never run on a GStreamer
streaming thread, which would stall the pipeline, and with a network source
back up the jitter buffer until it drops packets. The appsink is bounded
(max-buffers, drop=false) so a slow consumer applies back-pressure instead of
growing memory.

Needs PyGObject with GStreamer's introspection data (Debian/Ubuntu:
python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0; the last one
provides GstApp), plus
parakeet_onnx.py and its MODEL_DIR / VAD_MODEL environment variables.
"""

import os
import signal
import sys

# Windows: GStreamer's MSVC installer ships PyGObject in its own lib/site-packages.
# Its bin directory is needed twice over: Python does not search PATH for an
# extension module's DLLs (add_dll_directory), and GObject Introspection loads the
# libraries a typelib names through the ordinary search, which ignores
# add_dll_directory and does use PATH.
_GST_ROOT = os.environ.get("GSTREAMER_1_0_ROOT_MSVC_X86_64")
if sys.platform == "win32" and _GST_ROOT:
    _bin = os.path.join(_GST_ROOT, "bin")
    os.add_dll_directory(_bin)
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    sys.path.append(os.path.join(_GST_ROOT, "lib", "site-packages"))
# Windows cannot send SIGINT to a process in the background; Ctrl+Break is the
# signal it can send to a process group, so treat it the same way as Ctrl+C.
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

import gi  # noqa: E402
import numpy as np  # noqa: E402

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
# GstApp must be imported even though it is never named: appsink's pull methods
# (try_pull_sample, is_eos) live in the GstApp library, and PyGObject only adds
# them to the element once that library's typelib is loaded.
from gi.repository import Gst, GstApp  # noqa: E402,F401

from parakeet_onnx import Transcriber  # noqa: E402


def main() -> int:
    Gst.init(None)
    pipeline = Gst.parse_launch(sys.argv[1])
    sink = pipeline.get_by_name("sink")
    sink.set_property("max-buffers", 50)
    sink.set_property("drop", False)
    sink.set_property("sync", False)          # files: as fast as possible; live sources pace themselves
    bus = pipeline.get_bus()
    t = Transcriber()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        while True:  # Ctrl+C (SIGINT) stops a live source; the open utterance is still flushed below
            sample = sink.try_pull_sample(100 * Gst.MSECOND)
            if sample is not None:
                buf = sample.get_buffer()
                ok, info = buf.map(Gst.MapFlags.READ)
                if ok:
                    pcm = np.frombuffer(info.data, "<i2").astype(np.float32) / 32768.0
                    buf.unmap(info)
                    for text in t.feed(pcm):
                        print(text, flush=True)
            msg = bus.pop_filtered(Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if msg is not None:
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"gstreamer error: {err.message} ({dbg})", file=sys.stderr)
                    return 1
                break
            if sample is None and sink.is_eos():
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
    for text in t.flush():
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
