"""A true streaming model against VAD + offline Parakeet, on the same telephone audio.

nvidia/nemotron-speech-streaming-en-0.6b is a cache-aware streaming
FastConformer-RNNT: it emits words while the speaker is still talking, where
parakeet-tdt-0.6b-v3 (offline) produces nothing for an utterance until the VAD
has seen the pause. This probe runs its sherpa-onnx export
(sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25, 560 ms
chunks, int8; the only export offered) through OnlineRecognizer with endpoint
detection, on the channels and conditions wer_telephone.py uses, and reports:

  - WER: each channel's endpointed utterances joined in order, scored with
    AppTek's score.py, exactly like the offline results;
  - latency in audio time, against speech segments found by the same Silero
    VAD v5 (min silence 0.5 s) the offline pipeline uses:
      first words: from a VAD segment's start to the first partial text that
        changes inside it (how soon a viewer sees the utterance begin);
      endpoint:    from a VAD segment's end to the next endpoint the
        recogniser declares (when the utterance is final);
      last words:  from a VAD segment's end to the last text change before
        the next segment starts (when its last word appears);
    and revisions: text changes that do not extend the previous text, i.e.
    words already shown being changed;
    audio is fed in 80 ms chunks as fast as the recogniser takes it, so these
    are latencies of the model and its endpointing, without compute time;
  - compute: decode seconds per second of audio (real-time factor), on the
    provider used.

Endpoint rules: rule1 2.4 s of silence with nothing decoded, rule2 ENDPOINT_S
(default 0.5 s, to match the offline VAD's silence) after something was
decoded, rule3 20 s maximum utterance. ENDPOINT_S=0 disables endpointing: the
channel is one continuous stream, which isolates the model from the endpointer.

Run on the host with the models (SHERPA_MODELS) and corpora, e.g.:

    python models/parakeet-redux/probes/streaming_online.py apptek:en-US_General clean 12 --workers 6
    PARAKEET_PROVIDER=cuda python ... --workers 2

Predictions go to logs/wer/streaming/<tag>.pred.jsonl and are scored in place.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wer_telephone as wt  # noqa: E402

MODELS = Path(os.environ.get("SHERPA_MODELS", Path.home() / ".cache" / "sherpa-onnx"))
STREAMING = MODELS / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
PROVIDER = os.environ.get("PARAKEET_PROVIDER", "cpu")
ENDPOINT_S = float(os.environ.get("ENDPOINT_S", "0.5"))
RATE = 16_000
CHUNK = RATE * 80 // 1000
OUT = wt.ROOT / "logs" / "wer" / "streaming"

_rec = None


def _init():
    global _rec
    import sherpa_onnx as so
    _rec = so.OnlineRecognizer.from_transducer(
        tokens=str(STREAMING / "tokens.txt"), encoder=str(STREAMING / "encoder.int8.onnx"),
        decoder=str(STREAMING / "decoder.int8.onnx"), joiner=str(STREAMING / "joiner.int8.onnx"),
        num_threads=1, sample_rate=RATE, feature_dim=80, decoding_method="greedy_search",
        enable_endpoint_detection=ENDPOINT_S > 0, rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=ENDPOINT_S or 1.2, rule3_min_utterance_length=20.0, provider=PROVIDER)


def vad_spans(y: np.ndarray):
    """Speech segments (start_s, end_s) by the offline pipeline's VAD, unpadded."""
    import sherpa_onnx as so
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(wt.VAD_MODEL)
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 20.0
    cfg.sample_rate = RATE
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(y) / RATE + 60)
    w = cfg.silero_vad.window_size
    spans = []
    for i in range(0, len(y) - w + 1, w):
        vad.accept_waveform(y[i:i + w])
        while not vad.empty():
            spans.append((vad.front.start / RATE, (vad.front.start + len(vad.front.samples)) / RATE))
            vad.pop()
    vad.flush()
    while not vad.empty():
        spans.append((vad.front.start / RATE, (vad.front.start + len(vad.front.samples)) / RATE))
        vad.pop()
    return spans


def run_channel(job):
    fi, fn, wav, cond = job
    x, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    y = wt.condition(x[:, 0], rate, cond, fi)
    s = _rec.create_stream()
    finals, changes, endpoints = [], [], []      # changes: (audio_s, text) when the partial changes
    last = ""
    t0 = time.monotonic()
    for i in range(0, len(y), CHUNK):
        s.accept_waveform(RATE, y[i:i + CHUNK])
        while _rec.is_ready(s):
            _rec.decode_stream(s)
        now = min(i + CHUNK, len(y)) / RATE
        text = _rec.get_result(s).strip()
        if text != last:
            changes.append((now, text))
            last = text
        if _rec.is_endpoint(s):
            if text:
                finals.append(text)
            endpoints.append(now)
            _rec.reset(s)
            last = ""
    s.accept_waveform(RATE, np.zeros(int(0.66 * RATE), dtype=np.float32))   # tail padding, as sherpa-onnx's examples do
    s.input_finished()
    while _rec.is_ready(s):
        _rec.decode_stream(s)
    tail = _rec.get_result(s).strip()
    if tail:
        finals.append(tail)
    compute = time.monotonic() - t0
    spans = vad_spans(y)
    first, endp, tail_l = [], [], []
    for k, (a, b) in enumerate(spans):
        c = next((t for t, txt in changes if a <= t and txt), None)
        if c is not None and c <= b + 2.0:
            first.append(c - a)
        e = next((t for t in endpoints if t >= b), None)
        if e is not None and e - b <= 5.0:
            endp.append(e - b)
        # The utterance's last word: the last text change after its start and before the next
        # utterance starts (capped 3 s after its end), relative to the end of its speech.
        nxt = spans[k + 1][0] if k + 1 < len(spans) else float("inf")
        last_c = [t for t, _ in changes if a <= t <= min(nxt, b + 3.0)]
        if last_c:
            tail_l.append(last_c[-1] - b)
    # A revision is a text change that is not an extension of the text before it.
    revisions = sum(1 for (_, p), (_, q) in zip(changes, changes[1:]) if p and not q.startswith(p))
    return fn, " ".join(finals), len(y) / RATE, compute, first, endp, tail_l, revisions, len(changes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus"); p.add_argument("cond"); p.add_argument("n", type=int)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    its = sorted(wt.items(a.corpus), key=lambda t: t[0])[: a.n or None]
    tag = f"{a.corpus.replace(':', '-')}_{a.cond}_first{a.n}_endpoint{ENDPOINT_S}_{PROVIDER}"
    OUT.mkdir(parents=True, exist_ok=True)
    ref, pred = OUT / f"{tag}.ref.jsonl", OUT / f"{tag}.pred.jsonl"
    with open(ref, "w", encoding="utf-8") as f:
        for fn, _, text in its:
            f.write(json.dumps({"file_name": fn, "text": text}) + "\n")
    audio = comp = 0.0
    first, endp, tail_l = [], [], []
    revs = nchanges = 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex, open(pred, "w", encoding="utf-8") as f:
        for fn, text, au, co, fi_, en, ta, rv, nc in ex.map(run_channel, [(i, fn, w, a.cond) for i, (fn, w, _) in enumerate(its)]):
            f.write(json.dumps({"file_name": fn, "text": text}) + "\n")
            audio, comp, first, endp, tail_l = audio + au, comp + co, first + fi_, endp + en, tail_l + ta
            revs, nchanges = revs + rv, nchanges + nc
    sc = wt.score(ref, pred)
    q = lambda v, p_: sorted(v)[min(len(v) - 1, int(round(p_ * (len(v) - 1))))] if v else float("nan")
    print(f"{tag}: WER {float(sc['WER']) * 100:.2f} % (S {sc['Substitutions']}, D {sc['Deletions']}, "
          f"I {sc['Insertions']}, hits {sc['Hits']}); {len(its)} channels, {audio / 3600:.2f} h; "
          f"compute {audio / comp:.1f}x real time per worker; "
          f"first words after speech start: median {q(first, .5):.2f} s, p95 {q(first, .95):.2f} (n={len(first)}); "
          f"endpoint after speech end: median {q(endp, .5):.2f} s, p95 {q(endp, .95):.2f} (n={len(endp)}); "
          f"last words after speech end: median {q(tail_l, .5):.2f} s, p95 {q(tail_l, .95):.2f} (n={len(tail_l)}); "
          f"revisions {revs} of {nchanges} text changes", flush=True)


if __name__ == "__main__":
    main()
