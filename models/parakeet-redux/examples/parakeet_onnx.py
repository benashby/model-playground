"""A small live transcriber for parakeet-tdt-0.6b-v3 on ONNX (sherpa-onnx).

Feed it 16 kHz mono float32 audio in chunks of any size; it hands back each
utterance's text once the speaker pauses. The defaults are the configuration
that measured best in this repository's telephone WER study (see
../notes/12-onnx-deployment.md for the numbers behind each choice):

  - the fp32 export, never int8: on 8 kHz telephone audio int8 roughly doubles
    or triples the word error rate, mostly by dropping whole phrases;
  - Silero VAD v5, 0.5 s of silence ends an utterance, 20 s maximum;
  - 0.25 s of audio kept either side of each VAD segment, because sherpa-onnx
    returns segments with no margin and clips word onsets;
  - one decode worker thread, so utterances come back in the order spoken.

Needs `pip install sherpa-onnx numpy` and the model files:

    https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3   (fp32)
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad_v5.onnx

MODEL_DIR points at the fp32 model directory, VAD_MODEL at silero_vad_v5.onnx.
Pass provider="cuda" with sherpa-onnx's CUDA build to decode on an NVIDIA GPU;
the transcripts are the same as on the CPU.

Example:

    t = Transcriber()
    for chunk in chunks_of_16k_float32_audio:
        for text in t.feed(chunk):
            print(text)
    for text in t.flush():
        print(text)
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import sherpa_onnx

RATE = 16_000


class Transcriber:
    def __init__(self, model_dir: str | None = None, vad_model: str | None = None,
                 provider: str = "cpu", threads: int = 2, pad_s: float = 0.25,
                 min_silence_s: float = 0.5, max_speech_s: float = 20.0):
        model_dir = model_dir or os.environ["MODEL_DIR"]
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.onnx", decoder=f"{model_dir}/decoder.onnx",
            joiner=f"{model_dir}/joiner.onnx", tokens=f"{model_dir}/tokens.txt",
            num_threads=threads, model_type="nemo_transducer", provider=provider)
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = vad_model or os.environ["VAD_MODEL"]
        cfg.silero_vad.min_silence_duration = min_silence_s
        cfg.silero_vad.max_speech_duration = max_speech_s
        cfg.sample_rate = RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        self.window = cfg.silero_vad.window_size
        self.pad = int(pad_s * RATE)
        # Recent audio, so each segment can be cut again with a margin. 60 s bounds memory.
        self.keep = 60 * RATE
        self.hist = np.zeros(0, dtype=np.float32)
        self.hist_start = 0                       # absolute sample index of hist[0]
        self.pending = np.zeros(0, dtype=np.float32)
        self.pool = ThreadPoolExecutor(1)         # one worker: results in spoken order
        self.queue: list[Future] = []

    def _decode(self, samples: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(RATE, samples)
        self.rec.decode_stream(s)
        return s.result.text.strip()

    def _cut(self, start: int, n: int) -> np.ndarray:
        lo = max(start - self.pad, self.hist_start) - self.hist_start
        hi = min(start + n + self.pad, self.hist_start + len(self.hist)) - self.hist_start
        return self.hist[max(lo, 0):max(hi, 0)].copy()

    def _drain_vad(self) -> None:
        while not self.vad.empty():
            seg = self.vad.front
            self.queue.append(self.pool.submit(self._decode, self._cut(seg.start, len(seg.samples))))
            self.vad.pop()

    def _ready(self, block: bool) -> list[str]:
        out = []
        while self.queue and (block or self.queue[0].done()):
            text = self.queue.pop(0).result()
            if text:
                out.append(text)
        return out

    def feed(self, samples: np.ndarray) -> list[str]:
        """Add 16 kHz mono float32 audio; return any utterances finished so far."""
        samples = np.asarray(samples, dtype=np.float32)
        self.hist = np.concatenate([self.hist, samples])
        if len(self.hist) > self.keep:
            drop = len(self.hist) - self.keep
            self.hist, self.hist_start = self.hist[drop:], self.hist_start + drop
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= self.window:
            self.vad.accept_waveform(self.pending[:self.window])
            self.pending = self.pending[self.window:]
        self._drain_vad()
        return self._ready(block=False)

    def flush(self) -> list[str]:
        """End of stream: close the open utterance and wait for every result."""
        self.vad.flush()
        self._drain_vad()
        texts = self._ready(block=True)
        self.pool.shutdown()
        return texts
