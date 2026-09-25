"""Talk into the microphone, watch Parakeet transcribe it live (sherpa-onnx, CPU).

    uv run --with sherpa-onnx python models/parakeet-redux/dictate.py
    uv run --with sherpa-onnx python models/parakeet-redux/dictate.py --wav audio/tool_call.wav

Finished sentences print as lines; the sentence in progress is a dim draft
that rewrites in place. Ctrl+C stops and flushes the last sentence.

Same model and VAD setup as probes/onnx_concurrency.py (fetch the models as its
docstring says; SHERPA_MODELS points at them). --wav plays a file's channel 0
in real time through ffmpeg instead of the microphone, for checking without a
mic. Not a probe: nothing here is measured, and nothing is written to results/.
"""

import argparse
import asyncio
import os
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sherpa_onnx as so

MODELS = Path(os.environ.get("SHERPA_MODELS", Path.home() / ".cache" / "sherpa-onnx"))
PARAKEET = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
RATE = 16_000                       # Parakeet and Silero both run at 16 kHz: no resampling
CHUNK_BYTES = RATE * 80 // 1000 * 2  # 80 ms of s16 mono
DRAFT_S = 1.0

MIC_CMD = ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16",
           "--raw", "--latency", "20ms", "-"]   # --raw, not --container raw; see CLAUDE.md


def wav_cmd(path: str) -> list[str]:
    # -re paces at wall clock like a live source; pan picks channel 0 rather
    # than downmixing both speakers into one.
    return ["ffmpeg", "-loglevel", "error", "-re", "-i", path, "-af", "pan=mono|c0=c0",
            "-ar", str(RATE), "-f", "s16le", "-"]


def make_recognizer(threads: int):
    return so.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
        joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
        num_threads=threads, model_type="nemo_transducer")


def make_vad():
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.max_speech_duration = 20.0
    cfg.sample_rate = RATE
    return so.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def decode(rec, samples: np.ndarray) -> str:
    s = rec.create_stream()
    s.accept_waveform(RATE, samples)
    rec.decode_stream(s)
    return s.result.text.strip()


class Display:
    """Finished sentences scroll; the draft owns the last line and never wraps."""

    def __init__(self):
        self.tty = sys.stdout.isatty()

    def final(self, text: str) -> None:
        if text:
            print(("\r\033[K" if self.tty else "") + text, flush=True)
        elif self.tty:
            print("\r\033[K", end="", flush=True)

    def draft(self, text: str) -> None:
        if not self.tty or not text:
            return
        width = shutil.get_terminal_size().columns - 1
        tail = text if len(text) <= width else "…" + text[-(width - 1):]
        print(f"\r\033[K\033[2m{tail}\033[0m", end="", flush=True)


async def amain(args: argparse.Namespace) -> int:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGINT, stop.set)

    rec, vad, show = make_recognizer(args.threads), make_vad(), Display()
    window = vad.config.silero_vad.window_size
    # One worker: finals complete in the order they were spoken.
    pool = ThreadPoolExecutor(1)
    sentence = 0          # bumped per final; a draft from an older sentence is dropped
    draft_task: asyncio.Task | None = None
    finals: list[asyncio.Task] = []
    tail = asyncio.Lock()  # prints finals in submission order

    async def run_final(samples: np.ndarray) -> None:
        text_f = loop.run_in_executor(pool, decode, rec, samples)
        async with tail:
            show.final(await text_f)

    async def run_draft(samples: np.ndarray, gen: int) -> None:
        text = await loop.run_in_executor(pool, decode, rec, samples)
        if gen == sentence:
            show.draft(text)

    def drain() -> None:
        nonlocal sentence
        while not vad.empty():
            sentence += 1
            finals.append(asyncio.create_task(run_final(np.array(vad.front.samples, dtype=np.float32))))
            vad.pop()

    cmd = wav_cmd(args.wav) if args.wav else MIC_CMD
    # stderr inherited, never swallowed: a bad capture flag must be visible.
    # Own session: a terminal Ctrl+C reaches only us, and we stop the capture ourselves.
    src = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                               start_new_session=True)
    assert src.stdout is not None
    print(f"{'playing ' + args.wav if args.wav else 'listening'}; Ctrl+C to stop\n", file=sys.stderr)

    pending = np.zeros(0, dtype=np.float32)
    since_draft = 0.0
    read = asyncio.ensure_future(src.stdout.readexactly(CHUNK_BYTES))
    stopper = asyncio.ensure_future(stop.wait())
    try:
        while True:
            done, _ = await asyncio.wait({read, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if stopper in done:
                read.cancel()
                break
            try:
                raw = read.result()
            except asyncio.IncompleteReadError:
                break                      # source ended (file finished, or pw-record died)
            read = asyncio.ensure_future(src.stdout.readexactly(CHUNK_BYTES))

            pending = np.concatenate([pending, np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0])
            while len(pending) >= window:
                vad.accept_waveform(pending[:window])
                pending = pending[window:]
            drain()

            since_draft += 0.08
            if since_draft >= DRAFT_S and vad.is_speech_detected():
                since_draft = 0.0
                if draft_task is None or draft_task.done():
                    cur = np.array(vad.current_segment.samples, dtype=np.float32)
                    if len(cur):
                        draft_task = asyncio.create_task(run_draft(cur, sentence))
    finally:
        if src.returncode is None:
            src.terminate()
        await src.wait()

    vad.flush()
    drain()
    await asyncio.gather(*finals)
    if draft_task:
        await draft_task
    show.final("")
    pool.shutdown()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Live microphone transcription with Parakeet on sherpa-onnx.")
    p.add_argument("--wav", help="play this file's channel 0 in real time instead of the microphone")
    p.add_argument("--threads", type=int, default=2, help="decode threads (default 2)")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
