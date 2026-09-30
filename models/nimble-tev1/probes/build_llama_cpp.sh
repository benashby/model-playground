#!/usr/bin/env bash
# Build llama-server from a pinned llama.cpp tag, for one GPU backend.
#
#   models/nimble-tev1/probes/build_llama_cpp.sh cuda            # lan-gpu (RTX 3090s)
#   models/nimble-tev1/probes/build_llama_cpp.sh vulkan          # workstation (RDNA4)
#   models/nimble-tev1/probes/build_llama_cpp.sh hip             # workstation, ROCm
#   models/nimble-tev1/probes/build_llama_cpp.sh cpu b9190       # any tag
#
# The default tag, b11232, is the one Ollama 0.35.0 pins (LLAMA_CPP_VERSION at
# ollama/ollama 1abe35e). Ollama also applies its own compat patch on top, so a
# clean build of the same tag is how to tell "the version" from "Ollama's patch"
# when probabilities differ (fidelity.py).
#
# Builds outside the repository, in $LLAMA_BUILD_ROOT (default ~/.cache/llama-builds),
# and prints the binary path and its --version line last.
set -euo pipefail
backend=${1:?usage: build_llama_cpp.sh cuda|vulkan|hip|cpu [tag]}
tag=${2:-b11232}
root=${LLAMA_BUILD_ROOT:-$HOME/.cache/llama-builds}
src=$root/llama.cpp-$tag
build=$src/build-$backend

case $backend in
  cuda)   flags=(-DGGML_CUDA=ON) ;;
  vulkan) flags=(-DGGML_VULKAN=ON) ;;
  # gfx1201 is RDNA4. Override with AMDGPU_TARGETS for another card.
  hip)    flags=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx1201}") ;;
  cpu)    flags=() ;;
  *) echo "unknown backend $backend" >&2; exit 2 ;;
esac

mkdir -p "$root"
if [ ! -d "$src/.git" ]; then
  git clone --quiet --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp "$src"
fi
echo "llama.cpp $tag at $(git -C "$src" rev-parse HEAD)" >&2
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF "${flags[@]}" >&2
cmake --build "$build" --target llama-server -j "$(nproc)" >&2
echo "$build/bin/llama-server"
"$build/bin/llama-server" --version 2>&1 | grep -E '^(version|built with)'
