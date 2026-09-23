#!/usr/bin/env bash
# How small a card can this actually run on?
#
# Section 2 of the note establishes that the ~73 GB residency is set almost
# entirely by two environment variables, not by the model:
#
#     LLM_GPU_MEM_UTIL   default 0.45
#     TTS_GPU_MEM_UTIL   default 0.4
#
# Each is a vLLM gpu_memory_utilization fraction, applied per engine, and both
# are read from the environment by the container's own loader. That predicts the
# 80 GB requirement is a default rather than a floor -- resident weights are
# ~19 GB (LLM, bf16) plus ~4 GB (TTS), and everything above that is elective KV
# cache.
#
# This probe tests the prediction by walking the fractions down and recording,
# at each step, whether the server (a) starts, (b) warms up, and (c) still
# serves a session. Run it from the node.
#
# The failure mode to watch for is NOT out-of-memory. vLLM will happily start
# with a KV cache too small for the workload and then thrash -- which shows up
# as latency and dropped real-time budget, not as an error. So every step
# re-runs a fixture and checks the server's own inference budget afterwards.
#
#   ./vram_floor.sh 0.30 0.25      # LLM frac, TTS frac
set -euo pipefail

LLM_FRAC="${1:?usage: vram_floor.sh <llm_frac> <tts_frac>}"
TTS_FRAC="${2:?usage: vram_floor.sh <llm_frac> <tts_frac>}"
IMAGE="nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest"
MODELS="/opt/models/nemotron-labs-voicechat_v1.0.0"

echo "=== tearing down any existing container ==="
sudo docker rm -f voicechat >/dev/null 2>&1 || true
# The engines do not always release VRAM promptly; wait for the card to settle
# rather than measuring a footprint that includes the previous run's tail.
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 2000 ] && break
  sleep 2
done
echo "    GPU0 settled at ${used} MiB"

echo "=== starting with LLM_GPU_MEM_UTIL=$LLM_FRAC TTS_GPU_MEM_UTIL=$TTS_FRAC ==="
sudo docker run -d --name=voicechat \
  --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -e LLM_GPU_MEM_UTIL="$LLM_FRAC" \
  -e TTS_GPU_MEM_UTIL="$TTS_FRAC" \
  -p 9000:9000 \
  -v "$MODELS":/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  "$IMAGE" >/dev/null

echo "=== waiting for warmup (port 9000 opens only after it completes) ==="
ok=0
for i in $(seq 1 60); do
  if sudo docker logs voicechat 2>&1 | grep -q "Application startup complete"; then ok=1; break; fi
  if ! sudo docker ps --format '{{.Names}}' | grep -q voicechat; then
    echo "    CONTAINER DIED after ${i}0s"
    sudo docker logs --tail 30 voicechat 2>&1 | tail -30
    exit 1
  fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "    TIMED OUT waiting for startup"; exit 1; }

sleep 5
echo "=== residency at rest ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i 0
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 0
echo
echo "started OK at ${LLM_FRAC}/${TTS_FRAC}."
echo "NOW RUN A FIXTURE and re-check inference_budget.py -- starting is not serving."
