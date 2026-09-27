"""Words as they are spoken: NVIDIA's streaming model, run as one continuous stream.

The lowest-latency setup measured in ../notes/13-live-transcription.md.
nvidia/nemotron-speech-streaming-en-0.6b (a cache-aware streaming FastConformer,
English only) emits text in 560 ms chunks while the speaker is still talking,
and never revises a word once it has emitted it. It is run here without
sherpa-onnx's endpoint-and-reset cycle, because each reset discards the model's
context and cost 2.4 to 3.4 points of WER in the measurements. Utterance
boundaries come from a separate Silero VAD instead, used only to break lines.

Reads 16 kHz mono signed 16-bit PCM from stdin, like gst_stdin.py, and prints
each new piece of text as it appears, starting a new line at each pause:

    gst-launch-1.0 -q filesrc location=call.wav ! decodebin ! audioconvert ! audioresample \\
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! fdsink fd=1 | python nemotron_stream.py

Needs `pip install sherpa-onnx numpy` and, in MODELS_DIR, the sherpa-onnx export
sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 (from the
sherpa-onnx asr-models releases) and silero_vad_v5.onnx. The model's licence is
NVIDIA's Open Model License, not CC-BY; check it before shipping.
"""

import os
import sys

import numpy as np
import sherpa_onnx

RATE = 16_000
CHUNK = RATE * 2 // 10 * 2        # 200 ms of s16 mono, in bytes


def main() -> int:
    d = os.environ["MODELS_DIR"]
    m = f"{d}/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{m}/tokens.txt", encoder=f"{m}/encoder.int8.onnx", decoder=f"{m}/decoder.int8.onnx",
        joiner=f"{m}/joiner.int8.onnx", num_threads=1, sample_rate=RATE, feature_dim=80,
        decoding_method="greedy_search", enable_endpoint_detection=False)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{d}/silero_vad_v5.onnx"
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.sample_rate = RATE
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    window = cfg.silero_vad.window_size

    stream = rec.create_stream()
    shown = ""                     # everything printed so far; the model only ever appends
    line_start = True
    pending = np.zeros(0, dtype=np.float32)
    src = sys.stdin.buffer
    while True:
        raw = src.read(CHUNK)
        if not raw:
            break
        x = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768.0
        stream.accept_waveform(RATE, x)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        if len(text) > len(shown):
            new = text[len(shown):]
            print(new.lstrip() if line_start else new, end="", flush=True)
            shown, line_start = text, False
        # The VAD only marks pauses for display; the recogniser keeps its context.
        pending = np.concatenate([pending, x])
        while len(pending) >= window:
            vad.accept_waveform(pending[:window])
            pending = pending[window:]
        if not vad.empty():
            while not vad.empty():
                vad.pop()
            if not line_start:
                print(flush=True)
                line_start = True
    stream.accept_waveform(RATE, np.zeros(int(0.66 * RATE), dtype=np.float32))   # flush the last chunk
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    new = rec.get_result(stream)[len(shown):]
    print(new.lstrip() if line_start else new, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
