# Handoff: what is left of the Nimble / Tev1 investigation

Agent-facing. The CPU half and the GPU half are done and written up in
`notes/02-cpu-results.md` and `notes/03-gpu-results.md`; this file is the
record of how they were run and what is still open. Read `CLAUDE.md`, the
`evaluating` and `model-hosts` skills, and both result articles before
running anything, and the `writing-notes` skill before editing a note.

## Status, 2026-09-30

Done: fidelity (A), cross-backend numerics (B), the public suite (C), prompt
formats (D), calibration and temperature (E) and latency (F), on the
cloud-gpu role (one H100 per job, CUDA 12.9, Linux), the windows-3090 role
(RTX 3090, CUDA 13.4, native Windows) and the workstation (RDNA4 over Vulkan,
for B), with CPU builds of the same tag on all three. The suite ran for all
three models in the Ollama format on the H100 and the RTX 3090, and in each
model's own format on the H100: nine runs of 3,880 records, zero error rows.
Nothing measured contradicts the CPU half.

Left, in order of value:

1. **Quantisation (G).** What Q4_K_M or Q6_K costs in accuracy and
   calibration. For Tev1 4B, `bartowski/togethercomputer_Tev1-4B-experimental-GGUF`
   has quantisations made from BF16 [CLAIM, not inspected]; verify tensor types
   with `registry.py`'s GGUF reader before trusting the name. For Nimble there
   is no BF16 GGUF, so the only route is requantising Ollama's Q8_0
   (`llama-quantize --allow-requantize`), which compounds two quantisations;
   say so if you do it. A few subsets are enough.
2. **Own-format runs on a second GPU.** Tev1 0.8B's gain in its own format
   (p 2.14e-05) was measured on the H100 only. Repeating D on the RTX 3090
   would show whether it survives the backend drift.
3. **The per-question floor.** Warm or primed questions cost 40 to 90 ms each
   on either GPU. The notes infer that llama-server restoring the
   recurrent-state checkpoint is the cost; nothing measured it.

Not in scope without asking the owner: llama.cpp's open System One pull request
(#29321; building it runs unmerged contributor code), any other model, and any
fine-tuning.

## Setup

Keep everything large outside the checkout:

```bash
export WORK=$HOME/nimble-tev1-work NIMBLE_SRC=$HOME/nimble-tev1-work/nimble
git clone https://github.com/bespokelabsai/nimble $NIMBLE_SRC && git -C $NIMBLE_SRC checkout 62076b4f2d365b5879dafcf7f6dd072a1fe76df7
git clone https://github.com/ollama/ollama $WORK/ollama && git -C $WORK/ollama checkout 1abe35e6e6e777e858bbfbba283667ee8d516801
git clone https://github.com/togethercomputer/tev1 $WORK/tev1 && git -C $WORK/tev1 checkout 1dde7782382c9f49d627153759b8d1deab426ce0
git config core.hooksPath .githooks          # the leak check; it needs the private pattern file
```

- **llama.cpp `b11232`** (Ollama 0.35.0's pin, commit `6f767fe`):
  `probes/build_llama_cpp.sh <cuda|vulkan|hip|cpu>` on Linux (`CUDA_ARCH=90` on
  an H100; its header has the container recipe for a host with no CUDA
  toolkit), `probes/build_llama_cpp.cmd <cuda|cpu>` on native Windows.
- **Weights:** `probes/registry.py --download $WORK/gguf`. It is single-stream,
  and the registry gives one connection 3 to 6 MB/s; a parallel ranged download
  (`aria2c -x 16 --checksum=sha-256=<manifest digest>`) is several times faster.
  Re-run `registry.py` without `--download` and diff against
  `results/registry.log` first: a changed digest means Ollama republished.
- **Ollama 0.35.0**, only for cross-checks: the release tarball or zip, checked
  against GitHub's digest, with `OLLAMA_MODELS` and `OLLAMA_HOST` of its own.
  Hard-linking the verified GGUFs into `$OLLAMA_MODELS/blobs/sha256-<digest>`
  first makes `ollama pull` fetch only the small layers.
- **Data:** `uv run --no-project --with pyarrow --with transformers --with jinja2
  python models/nimble-tev1/probes/fetch_public_data.py --nimble-src $NIMBLE_SRC`
  must end `all subsets match Bespoke's manifests`.

**Serving a model for the port**, one model per GPU:

```bash
$LLAMA_SERVER -m $WORK/gguf/nimble-latest.gguf -c 8194 -np 1 -ngl 99 \
    -b 1024 -ub 1024 --cache-ram 0 --host 127.0.0.1 --port 8080
# tev1 and tev1:0.8b: -c 2050 -b 512 -ub 512
```

`-c` is Ollama's `num_ctx`. The batch is the one Ollama's scheduler picks
(`automaticGenerationBatch` in `server/sched.go`): 1024 for Nimble's 8,194-token
context when VRAM allows, 512 for Tev1. It moves Nimble's probabilities by
6.12e-03, so on a card with little free VRAM read the batch from Ollama's
runner command line instead of assuming. `--cache-ram 0` is required for
anything timed or isolated. Before measuring, confirm `/health` is `ok`,
`/props` reports the GGUF you just started, and a scored question takes tens
of milliseconds rather than the hundreds a CPU takes.

## Committing

Run directories hold third-party benchmark text in `rows.jsonl`: never commit
them. Commit summaries and comparisons, with absolute paths in manifests
replaced by `$NIMBLE_SRC` or `$WORK`, and LF line endings on anything copied
from Windows. Grep what you add for record text and for any host name or
address. Humanize commit messages, and new or changed notes go through the
`writing-notes` procedure.

## Pitfalls already paid for

- **Cold is not cold without `--cache-ram 0`.** llama-server's host prompt
  cache restores a state even after the slot was displaced.
- **Repeated identical prompts crash `b9190`.** Run `identical_prompt.py` on any
  build other than `b11232` first.
- **The Ollama and Nimble formats couple questions**; the Tev1 format does not.
  Keep the question set fixed when comparing times or probabilities.
- **History moves probabilities.** Use `--isolate` for every scored run; Ollama
  itself cannot be isolated.
- **Batch size moves probabilities**, and Ollama chooses it from free VRAM.
  Match it or the port will not reproduce Ollama.
- **A port can still be held by the previous server.** A script that died before
  stopping its servers left them running, the next model's servers failed to
  bind, and three fidelity runs measured the wrong model without an error. The
  `/props` check above is what caught it.
- **Windows logs have CRLF endings.** A parser comparing the last field against
  `True` counted every line as a changed answer until `\r` was stripped.
- **Windows kills a session's children when SSH disconnects.** Start long jobs
  with `Win32_Process.Create` (see the `model-hosts` skill), and read their logs.
- **`pkill -f` and `pgrep -f`** match their own command line. Kill by PID.
- **Never print a host or address from a probe.** The probes take URLs as
  arguments and log only what the server reports about its build.
