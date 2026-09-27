#!/usr/bin/env bash
# The full NeMo-against-ONNX matrix: cut every segment set once, then have each
# runtime transcribe exactly those files, then score them all with one scorer.
#
#   run_matrix.sh cut             segment sets A, B, C (CPU, 12 workers)
#   run_matrix.sh onnx            ONNX fp32 then int8 on every set (CPU, 12 workers)
#   run_matrix.sh nemo NAME [--graphs] [--device cuda|cpu]
#                                 NeMo on every set, via $NEMO_RUN (see below)
#   run_matrix.sh score           score every runtime found on every set
#
# Segment sets (see nemo_vs_onnx.py; conditions and segmentations are wer_telephone.py's):
#   A  AppTek en-US_General, en-IN, en-GB_SCT: all channels, clean, bench
#   B  the same accents, first 12 channels: clean, g711mu, opus12, opus12-loss{5,10,20}, bench
#   C  HarperValleyBank, all channels, clean: bench and livepad
#
# NEMO_RUN is the command that runs nemo_vs_onnx.py inside a NeMo environment,
# given the segment directory and the remaining arguments. For ROCm it is a
# podman command (see notes on running NeMo on AMD GPUs); for a plain venv,
# NEMO_RUN="python models/parakeet-redux/probes/nemo_vs_onnx.py nemo".
# Run from the repo root.
set -euo pipefail

PROBE=models/parakeet-redux/probes/nemo_vs_onnx.py
UVRUN=(uv run -q --extra asr --with sherpa-onnx --with jiwer --with openai-whisper==20250625 python "$PROBE")
OUT=logs/wer/nemo_vs_onnx
ACCENTS=(en-US_General en-IN en-GB_SCT)
CONDS_B=(clean g711mu opus12 opus12-loss5 opus12-loss10 opus12-loss20)

tags() {
  for a in "${ACCENTS[@]}"; do echo "apptek-$a""_clean_bench"; done
  for a in "${ACCENTS[@]}"; do for c in "${CONDS_B[@]}"; do echo "apptek-$a""_$c""_bench_first12"; done; done
  echo hvb_clean_bench
  echo hvb_clean_livepad
}

case "${1:-}" in
  cut)
    for a in "${ACCENTS[@]}"; do "${UVRUN[@]}" cut "apptek:$a" clean bench 0; done
    for a in "${ACCENTS[@]}"; do for c in "${CONDS_B[@]}"; do "${UVRUN[@]}" cut "apptek:$a" "$c" bench 12; done; done
    "${UVRUN[@]}" cut hvb clean bench 0
    "${UVRUN[@]}" cut hvb clean livepad 0
    ;;
  onnx)
    for p in fp32 int8; do for t in $(tags); do
      echo "== $t onnx_$p"; PARAKEET_PRECISION=$p "${UVRUN[@]}" onnx "$t"
    done; done
    ;;
  nemo)
    shift; name=$1; shift
    : "${NEMO_RUN:?set NEMO_RUN to the command that runs nemo_vs_onnx.py nemo}"
    for t in $(tags); do
      echo "== $t $name"; $NEMO_RUN "$OUT/$t" --name "$name" "$@"
    done
    ;;
  score)
    for t in $(tags); do echo "== $t"; "${UVRUN[@]}" score "$t"; done
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
