#!/usr/bin/env bash
# Launch IFM's published vLLM recipe for K2-Horizon-32B (from the model card,
# also at recipes.vllm.ai/IFM) on the GPU node, plus any extra flags given, and
# report whether it serves.
#
# The recipe's own flags are reproduced exactly. The only additions are the
# ones any local launch needs: a local weights path instead of the Hub id,
# a served name, and host/port. `--revision main` is dropped because it only
# applies to Hub downloads.
#
#   bash recipe_launch.sh                                   # recipe as published
#   bash recipe_launch.sh --tensor-parallel-size 4          # one change
#
# Run on the node (it drives podman directly); pipe it over ssh from here:
#   ssh <gpu-node> 'bash -s -- --tensor-parallel-size 4' < models/k2-horizon-32b/probes/recipe_launch.sh
set -u
# VLLM_CACHE_DIR keeps compiled kernels between launches; any writable directory works.
NAME=k2-recipe
podman rm -f "$NAME" >/dev/null 2>&1
mkdir -p "${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
podman run -d --name "$NAME" --device nvidia.com/gpu=all --security-opt=label=disable \
  --ipc=host --ulimit stack=67108864 -p 8080:8080 \
  -v /opt/models:/models:ro -v "${VLLM_CACHE_DIR:-$HOME/.cache/vllm}:/cache:Z" \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/hf -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  docker.io/vllm/vllm-openai:v0.30.0 \
  /models/K2-Horizon-32B --served-model-name k2-horizon-32b --host 0.0.0.0 --port 8080 \
  --model-impl vllm \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --reasoning-parser k2_horizon \
  --enable-auto-tool-choice \
  --tool-call-parser k2_horizon \
  "$@" >/dev/null
echo "extra flags: ${*:-(none, recipe as published)}"
t0=$(date +%s)
while :; do
  if podman logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "RESULT: serving after $(( $(date +%s) - t0 )) s"; break
  fi
  if ! podman ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "RESULT: exited after $(( $(date +%s) - t0 )) s"; break
  fi
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "RESULT: still starting after 900 s"; break; }
  sleep 5
done
podman logs "$NAME" 2>&1 | grep -E "non-default args|max_model_len|Available KV cache memory|GPU KV cache size|Maximum concurrency|ValueError|estimated maximum model length|chat template content format" \
  | grep -v '^\s*INFO:' | sed 's/^(\([A-Za-z_0-9]*\) pid=[0-9]*) //' | cut -c1-400 | sort -u
