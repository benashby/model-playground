"""Transcribe whatever a gst-launch-1.0 pipeline writes to stdout.

The simplest GStreamer integration: GStreamer does the capture, depayloading,
decoding and resampling in its own process, and hands this script raw 16 kHz
mono signed 16-bit PCM on a pipe. Nothing here links against GStreamer.

End every pipeline with

    ... ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1

and run gst-launch-1.0 with -q, so its status messages do not go to stdout.

A file:

    gst-launch-1.0 -q filesrc location=call.wav ! decodebin ! audioconvert ! audioresample \\
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1 | python gst_stdin.py

A G.711 mu-law RTP stream on UDP port 5004 (a SIP/RTP phone leg):

    gst-launch-1.0 -q udpsrc port=5004 \\
        caps="application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0" \\
        ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec ! audioconvert ! audioresample \\
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1 | python gst_stdin.py

audioresample upsamples 8 kHz telephone audio to the 16 kHz the model expects;
that is all the model needs from narrowband input. Uses parakeet_onnx.py (same
directory) and its MODEL_DIR / VAD_MODEL environment variables.
"""

import sys

import numpy as np

from parakeet_onnx import Transcriber

CHUNK = 16_000 * 2 // 10          # 100 ms of s16 mono at 16 kHz


def main() -> int:
    t = Transcriber()
    src = sys.stdin.buffer
    while True:
        raw = src.read(CHUNK)
        if not raw:
            break
        raw = raw[: len(raw) // 2 * 2]                    # whole samples only
        for text in t.feed(np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0):
            print(text, flush=True)
    for text in t.flush():
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
