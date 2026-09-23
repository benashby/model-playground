#!/usr/bin/env bash
# Start the documented K2 serve command (notes/02-using-it.md) with one or two
# things changed, to compare configurations. Everything not overridden is
# identical to the command in the note.
#
#   MODEL_DIR  weights directory under /opt/models  (default K2-Horizon-32B)
#   TP         tensor-parallel size                  (default 4)
#   MAX_LEN    --max-model-len                       (default 524288)
#   extra args are appended, e.g. --kv-cache-dtype fp8
#
#   ssh <gpu-node> 'MODEL_DIR=K2-Horizon-32B-FP8 TP=2 MAX_LEN=131072 bash -s' < serve_variant.sh
set -u
MODEL_DIR=${MODEL_DIR:-K2-Horizon-32B}; TP=${TP:-4}; MAX_LEN=${MAX_LEN:-524288}
NAME=k2-variant
podman rm -f "$NAME" >/dev/null 2>&1
mkdir -p "${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
podman run -d --name "$NAME" --device nvidia.com/gpu=all --security-opt=label=disable \
  --ipc=host --ulimit stack=67108864 -p 8080:8080 \
  -v /opt/models:/models:ro -v "${VLLM_CACHE_DIR:-$HOME/.cache/vllm}:/cache:Z" \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/hf -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  docker.io/vllm/vllm-openai:v0.30.0 \
  "/models/$MODEL_DIR" --served-model-name k2-horizon-32b --host 0.0.0.0 --port 8080 \
  --model-impl vllm --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --chat-template-content-format string \
  --reasoning-parser k2_horizon --enable-auto-tool-choice --tool-call-parser k2_horizon \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  "$@" >/dev/null
echo "config: MODEL_DIR=$MODEL_DIR TP=$TP MAX_LEN=$MAX_LEN extra=${*:-none}"
t0=$(date +%s)
while :; do
  podman logs "$NAME" 2>&1 | grep -q "Application startup complete" && { echo "RESULT: serving after $(( $(date +%s) - t0 )) s"; break; }
  podman ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "RESULT: exited after $(( $(date +%s) - t0 )) s"; break; }
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "RESULT: still starting after 1200 s"; break; }
  sleep 5
done
podman logs "$NAME" 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|model weights took|Loading weights took|ValueError|quantization|kv_cache_dtype" \
  | grep -v '^\s*INFO:' | sed 's/^(\([A-Za-z_0-9]*\) pid=[0-9]*) //' | cut -c1-300 | sort -u
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
