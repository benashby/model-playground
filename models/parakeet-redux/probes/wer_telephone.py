"""Word error rate of parakeet-tdt-0.6b-v3 (int8 ONNX, sherpa-onnx) on telephone speech.

Two corpora, both with human reference transcripts and one speaker per channel:

  apptek:<accent>  AppTek Call-Center Dialogues, CC-BY-SA-4.0, role-played call-centre
                   conversations recorded over VoIP at 16 kHz, 14 accent groups
                   (https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues).
                   Its paper (arXiv 2604.27543) publishes parakeet v3 WER per accent with
                   Silero segmentation, which makes it the round-trip check for this probe.
  hvb              Gridspace-Stanford HarperValleyBank, CC-BY-4.0, simulated bank calls
                   placed over a real telephone network, 8 kHz
                   (https://github.com/cricketclub/gridspace-stanford-harper-valley).

Telephone conditions are applied per channel, to the audio the model hears:

  clean            the corpus audio as distributed (AppTek 16 kHz, HVB 8 kHz native)
  g711mu           16 kHz -> 8 kHz -> G.711 mu-law (ffmpeg pcm_mulaw) -> 8 kHz -> 16 kHz
  opus12           16 kHz -> 8 kHz -> Opus, VOIP application, narrowband, 12 kbit/s,
                   20 ms frames -> 16 kHz
  gridspace        HVB only: no decoding; scores the corpus's own machine transcript
                   (Gridspace's ASR, the `transcript` field) as a difficulty reference
  opus12-lossN     as opus12, with N % of 20 ms packets dropped (Bernoulli, seeded per
                   file) and the missing frames produced by the Opus decoder's own packet
                   loss concealment. ffmpeg's libopus -packet_loss option only tells the
                   encoder what to expect and drops nothing, so this probe drives libopus
                   directly through ctypes.

All resampling uses `playground.audio._resample` (band-limited). Every condition,
including clean, reaches the VAD and the recogniser at 16 kHz.

Segmentation, with sherpa-onnx's Silero VAD v5 over the whole channel:

  bench    AppTek's published recipe: min silence 10.0 s, min speech 0.25 s, max speech
           30 s, plus the 30 ms per side that the silero-vad package pads by default
  live     the live-use setup of dictate.py and onnx_concurrency.py: min silence 0.5 s,
           min speech 0.25 s, max speech 20 s, no padding (sherpa-onnx adds none)
  livepad  live, with 0.25 s of padding each side
  live-sX  livepad with X s of silence to end an utterance (0.2, 0.3, 0.8, 1.0)
  live-pX  livepad with X s of padding each side (0.1, 0.5)
  none     no VAD: the whole channel in one decode (short calls only)

Each segment is decoded separately and the channel's hypothesis is the segments
joined in order. Scoring runs AppTek's own `score.py` from the downloaded corpus,
unmodified (Whisper EnglishTextNormalizer from openai-whisper 20250625, their word
mappings, jiwer, corpus-level WER), for both corpora, so the two use identical
normalisation. For HVB the reference is the channel's `human_transcript` segments
in time order.

Predictions (model output on the corpora's audio) and the matching references are
written to logs/wer/, which is gitignored, not to results/; this probe prints only
aggregate figures.

The corpora come from the DVC stages in dvc.yaml (`uv run --extra corpora dvc repro`,
or `dvc pull` from a mirror; see audio/corpora/MANIFEST.md). Run from the repo root:

    uv run --extra asr --with sherpa-onnx --with jiwer --with openai-whisper==20250625 \\
        python models/parakeet-redux/probes/wer_telephone.py RUNSPEC [RUNSPEC ...] \\
        [--limit N] [--workers N]

RUNSPEC is corpus/condition/segmentation, e.g. apptek:en-IN/clean/bench or
hvb/clean/live. --limit N takes the first N channel files (sorted by name) of
each corpus, for sweeps. SHERPA_MODELS points at the model directory (default
~/.cache/sherpa-onnx); OPUS_LIB at libopus if ctypes cannot find it.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from playground.audio import _resample  # noqa: E402

MODELS = Path(os.environ.get("SHERPA_MODELS", Path.home() / ".cache" / "sherpa-onnx"))
# PARAKEET_PRECISION=fp32 selects the fp32 export
# (huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3), same author.
PRECISION = os.environ.get("PARAKEET_PRECISION", "int8")
PARAKEET = MODELS / ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3" + ("-int8" if PRECISION == "int8" else ""))
_SUF = ".int8.onnx" if PRECISION == "int8" else ".onnx"
# PARAKEET_PROVIDER=cuda runs the same ONNX graph on an NVIDIA GPU (needs sherpa-onnx's
# CUDA build and CUDA/cuDNN libraries on the loader path); the default is the CPU.
PROVIDER = os.environ.get("PARAKEET_PROVIDER", "cpu")
VAD_MODEL = MODELS / "silero_vad_v5.onnx"
CORPORA = ROOT / "audio" / "corpora"
RATE = 16_000
PHONE_RATE = 8_000
SEG = {  # min_silence, min_speech, max_speech, padding each side (s); None = no VAD
    # AppTek's recipe, with the 30 ms padding the silero-vad package's
    # get_speech_timestamps applies by default (speech_pad_ms=30).
    "bench": (10.0, 0.25, 30.0, 0.03),
    # dictate.py and onnx_concurrency.py: sherpa-onnx's VAD segments, unpadded.
    "live": (0.5, 0.25, 20.0, 0.0),
    "livepad": (0.5, 0.25, 20.0, 0.25),
    # The live-latency sweep: silence needed to end an utterance, and padding.
    "live-s0.2": (0.2, 0.25, 20.0, 0.25),
    "live-s0.3": (0.3, 0.25, 20.0, 0.25),
    "live-s0.8": (0.8, 0.25, 20.0, 0.25),
    "live-s1.0": (1.0, 0.25, 20.0, 0.25),
    "live-p0.1": (0.5, 0.25, 20.0, 0.10),
    "live-p0.5": (0.5, 0.25, 20.0, 0.50),
    # The whole channel as one decode, for short calls only.
    "none": None,
}


def out(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Corpora: each yields (file_name, wav_path, reference_text).

def apptek_items(accent: str):
    d = CORPORA / "apptek" / "test" / accent
    for line in open(d / "metadata.jsonl", encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            yield o["file_name"], d / o["file_name"], o["text"]


def hvb_items():
    d = CORPORA / "harper_valley" / "data"
    for tf in sorted((d / "transcript").glob("*.json")):
        segs = json.load(open(tf, encoding="utf-8"))
        for role in ("agent", "caller"):
            wav = d / "audio" / role / f"{tf.stem}.wav"
            mine = sorted((s for s in segs if s.get("speaker_role") == role),
                          key=lambda s: s["offset_ms"])
            ref = " ".join(s["human_transcript"].strip() for s in mine if s.get("human_transcript"))
            if wav.exists() and ref:
                yield f"{role}/{tf.stem}.wav", wav, ref


def hvb_gridspace(file_name: str) -> str:
    """HVB's own machine transcript (Gridspace ASR, the `transcript` field) for one channel."""
    role, name = file_name.split("/")
    segs = json.load(open(CORPORA / "harper_valley" / "data" / "transcript" / name.replace(".wav", ".json"),
                          encoding="utf-8"))
    mine = sorted((s for s in segs if s.get("speaker_role") == role), key=lambda s: s["offset_ms"])
    return " ".join(s["transcript"].strip() for s in mine if s.get("transcript", "").strip())


def items(corpus: str):
    if corpus.startswith("apptek:"):
        return list(apptek_items(corpus.split(":", 1)[1]))
    if corpus == "hvb":
        return list(hvb_items())
    raise SystemExit(f"unknown corpus {corpus}")


# ---------------------------------------------------------------------------
# Telephone conditions.

def _opus():
    path = os.environ.get("OPUS_LIB") or ctypes.util.find_library("opus")
    if not path:  # fall back to the libopus the local ffmpeg is linked against
        ff = subprocess.run(["sh", "-c", "ldd $(readlink -f $(command -v ffmpeg))"],
                            capture_output=True, text=True).stdout
        m = re.search(r"libopus\.so\.\d+ => (\S+)", ff)
        path = m.group(1) if m else None
    if not path:
        raise SystemExit("libopus not found; set OPUS_LIB")
    lib = ctypes.CDLL(path)
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encode_float.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                      ctypes.c_char_p, ctypes.c_int32]
    lib.opus_decode_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    return lib


def opus_roundtrip(x8: np.ndarray, bitrate: int, loss_pct: float, seed: int) -> np.ndarray:
    lib = _opus()
    err = ctypes.c_int()
    enc = lib.opus_encoder_create(PHONE_RATE, 1, 2048, ctypes.byref(err))   # OPUS_APPLICATION_VOIP
    assert err.value == 0, err.value
    lib.opus_encoder_ctl(ctypes.c_void_p(enc), 4002, ctypes.c_int(bitrate))  # OPUS_SET_BITRATE
    lib.opus_encoder_ctl(ctypes.c_void_p(enc), 4008, ctypes.c_int(1101))     # OPUS_SET_BANDWIDTH narrowband
    dec = lib.opus_decoder_create(PHONE_RATE, 1, ctypes.byref(err))
    assert err.value == 0, err.value
    frame = PHONE_RATE // 50                                                  # 20 ms
    n = (len(x8) + frame - 1) // frame
    x = np.zeros(n * frame, dtype=np.float32)
    x[: len(x8)] = x8
    y = np.zeros_like(x)
    rng = np.random.default_rng(seed)
    buf = ctypes.create_string_buffer(4000)
    pcm_out = (ctypes.c_float * frame)()
    lost = 0
    for i in range(n):
        f = np.ascontiguousarray(x[i * frame:(i + 1) * frame])
        nb = lib.opus_encode_float(enc, f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), frame, buf, 4000)
        assert nb > 0, nb
        if loss_pct and rng.random() < loss_pct / 100:
            got = lib.opus_decode_float(dec, None, 0, pcm_out, frame, 0)       # PLC
            lost += 1
        else:
            got = lib.opus_decode_float(dec, buf.raw[:nb], nb, pcm_out, frame, 0)
        assert got == frame, got
        y[i * frame:(i + 1) * frame] = np.frombuffer(pcm_out, dtype=np.float32)
    lib.opus_encoder_destroy(ctypes.c_void_p(enc))
    lib.opus_decoder_destroy(ctypes.c_void_p(dec))
    return y[: len(x8)]


def g711_roundtrip(x8: np.ndarray) -> np.ndarray:
    raw = np.clip(x8, -1, 1).astype("<f4").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(PHONE_RATE),
                          "-ac", "1", "-i", "-", "-c:a", "pcm_mulaw", "-f", "mulaw", "-"],
                         input=raw, capture_output=True, check=True)
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "mulaw", "-ar", str(PHONE_RATE),
                          "-ac", "1", "-i", "-", "-f", "f32le", "-"],
                         input=enc.stdout, capture_output=True, check=True)
    return np.frombuffer(dec.stdout, dtype="<f4")


def condition(x: np.ndarray, rate: int, cond: str, seed: int) -> np.ndarray:
    """Return 16 kHz float32 audio after the named telephone condition."""
    if cond == "clean":
        return _resample(x, rate, RATE)
    x8 = _resample(x, rate, PHONE_RATE)
    if cond == "g711mu":
        y8 = g711_roundtrip(x8)
    elif (m := re.fullmatch(r"opus(\d+)(?:-loss(\d+))?", cond)):
        y8 = opus_roundtrip(x8, int(m.group(1)) * 1000, float(m.group(2) or 0), seed)
    else:
        raise SystemExit(f"unknown condition {cond}")
    return _resample(y8, PHONE_RATE, RATE)


# ---------------------------------------------------------------------------
# Transcription, one process per worker.

_rec = None


def _init():
    global _rec
    import sherpa_onnx as so
    _rec = so.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET / f"encoder{_SUF}"), decoder=str(PARAKEET / f"decoder{_SUF}"),
        joiner=str(PARAKEET / f"joiner{_SUF}"), tokens=str(PARAKEET / "tokens.txt"),
        num_threads=1, model_type="nemo_transducer", provider=PROVIDER)


def _segments(y: np.ndarray, seg: str):
    import sherpa_onnx as so
    if SEG[seg] is None:
        return [y]
    min_sil, min_sp, max_sp, pad_s = SEG[seg]
    pad = int(pad_s * RATE)
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(VAD_MODEL)
    cfg.silero_vad.min_silence_duration = min_sil
    cfg.silero_vad.min_speech_duration = min_sp
    cfg.silero_vad.max_speech_duration = max_sp
    cfg.sample_rate = RATE
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(y) / RATE + 60)
    w = cfg.silero_vad.window_size
    spans = []

    def take():
        while not vad.empty():
            seg_ = vad.front
            spans.append((seg_.start, seg_.start + len(seg_.samples)))
            vad.pop()

    for i in range(0, len(y) - w + 1, w):
        vad.accept_waveform(y[i:i + w])
        take()
    vad.flush()
    take()
    # Cut from the original audio so padding can extend each span; adjacent
    # padded spans may overlap slightly, as in silero-vad's own tooling.
    return [y[max(0, a - pad):min(len(y), b + pad)] for a, b in spans]


def transcribe(job):
    file_name, wav, cond, seg, seed = job
    x, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    assert x.shape[1] == 1, f"{wav} has {x.shape[1]} channels"
    t0 = time.monotonic()
    y = condition(x[:, 0], rate, cond, seed)
    t1 = time.monotonic()
    texts = []
    for s in _segments(y, seg):
        st = _rec.create_stream()
        st.accept_waveform(RATE, s)
        _rec.decode_stream(st)
        texts.append(st.result.text.strip())
    t2 = time.monotonic()
    return file_name, " ".join(t for t in texts if t), len(y) / RATE, t1 - t0, t2 - t1


# ---------------------------------------------------------------------------
# Scoring with AppTek's score.py, unmodified.

def score(ref_path: Path, pred_path: Path) -> dict:
    d = CORPORA / "apptek"
    r = subprocess.run([sys.executable, str(d / "score.py"), "--ref", str(ref_path), "--pred", str(pred_path)],
                       cwd=d, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit("score.py failed")
    return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in r.stdout.splitlines() if ":" in l)}


def run(spec: str, limit: int | None, workers: int):
    corpus, cond, seg = spec.split("/")
    its = sorted(items(corpus), key=lambda t: t[0])[:limit]
    tag = (spec.replace("/", "_").replace(":", "-") + (f"_first{limit}" if limit else "")
           + ("" if PRECISION == "int8" else f"_{PRECISION}"))
    # Not inside the corpus directories: those are DVC outputs holding only upstream files.
    pred_dir = ROOT / "logs" / "wer"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ref_path, pred_path = pred_dir / f"{tag}.ref.jsonl", pred_dir / f"{tag}.pred.jsonl"
    with open(ref_path, "w", encoding="utf-8") as f:
        for fn, _, ref in its:
            f.write(json.dumps({"file_name": fn, "text": ref}) + "\n")

    t = time.monotonic()
    if cond == "gridspace":  # score the corpus's own ASR output; nothing is decoded
        assert corpus == "hvb", "gridspace transcripts exist only in HVB"
        with open(pred_path, "w", encoding="utf-8") as f:
            for fn, _, _ in its:
                f.write(json.dumps({"file_name": fn, "text": hvb_gridspace(fn)}) + "\n")
        s = score(ref_path, pred_path)
        out(f"{spec + ' ' + PRECISION:<39} files {len(its):>4}  WER {float(s['WER']) * 100:6.2f} %  "
            f"(S {s['Substitutions']}, D {s['Deletions']}, I {s['Insertions']}, hits {s['Hits']})  "
            f"HVB's own Gridspace ASR output, not this model")
        return
    jobs = [(fn, wav, cond, seg, i) for i, (fn, wav, _) in enumerate(its)]
    audio_s = cond_s = dec_s = 0.0
    with ProcessPoolExecutor(workers, initializer=_init) as ex, open(pred_path, "w", encoding="utf-8") as f:
        for fn, text, a, c, d in ex.map(transcribe, jobs, chunksize=1):
            f.write(json.dumps({"file_name": fn, "text": text}) + "\n")
            audio_s, cond_s, dec_s = audio_s + a, cond_s + c, dec_s + d
    wall = time.monotonic() - t
    s = score(ref_path, pred_path)
    out(f"{spec + ' ' + PRECISION:<39} files {len(its):>4}  audio {audio_s / 3600:6.2f} h  WER {float(s['WER']) * 100:6.2f} %  "
        f"(S {s['Substitutions']}, D {s['Deletions']}, I {s['Insertions']}, hits {s['Hits']})  "
        f"decode RTF {audio_s / dec_s:5.1f}x/worker  wall {wall:6.0f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    import sherpa_onnx
    out(f"sherpa-onnx {sherpa_onnx.__version__}, model {PARAKEET.name}, VAD {VAD_MODEL.name}, "
        f"workers {a.workers}, limit {a.limit}")
    for spec in a.specs:
        run(spec, a.limit, a.workers)


if __name__ == "__main__":
    main()
