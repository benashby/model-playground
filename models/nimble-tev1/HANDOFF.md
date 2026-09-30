# Handoff: the GPU half of the Nimble / Tev1 investigation

## Status, 2026-09-30 (end of the GPU half)

Done and written up in `notes/03-gpu-results.md`:

- A, B, C, D, E and F, on three GPU hosts: the cloud-gpu role (one H100 per
  job, CUDA 12.9, Linux), the windows-3090 role (RTX 3090, CUDA 13.4, native
  Windows), and the workstation (RDNA4 over Vulkan, for B). CPU builds of the
  same tag on all three machines for B. Ollama 0.35.0 installed on the VM and
  the Windows desktop for the cross-checks.
- The full suite ran for all three models in the Ollama format on the H100 and
  on the RTX 3090, and in each model's own format on the H100: nine runs of
  3,880 records, zero error rows.
- Committed: every probe log under `results/`, one JSON per suite run and per
  comparison under `results/public-suite/` (summary, manifest with paths
  redacted, host), the generated tables, and new probes `latency_sweep.py`,
  `temperature.py`, `suite_table.py`, `compare_table.py`, `rows_diff.py`, plus a
  Windows branch in `hostinfo.py`. No `rows.jsonl`. The per-subset files the
  plan asked for are merged into one file per run.

Not done: G (quantisation); the own-format runs on the RTX 3090.

Contradicts this plan: the serving recipe below uses `-b 512 -ub 512` for every
model. Ollama picks the batch from the context and free VRAM, and gives Nimble
1024, which moves its probabilities by 6.12e-03. Match Ollama's batch (read it
from the runner's command line) or the port will not reproduce Ollama for
Nimble. Nothing measured contradicts the CPU half.

Agent-facing. Written 2026-09-30 by the agent that did the CPU half, for an
agent on a GPU host picking it up. Delete this file, or cut it down to a
record of what was handed over, once its work is written into the notes.

Read before doing anything, in this order:

1. `CLAUDE.md` at the repository root. The governing rule, the public-repo
   rules, and the leak check all apply to you.
2. `.claude/skills/evaluating/SKILL.md` and `.claude/skills/model-hosts/SKILL.md`.
3. `models/nimble-tev1/README.md` and both articles in `notes/`. They are what
   is already known; this file is what is not.
4. `.claude/skills/writing-notes/SKILL.md` before you edit any note.

## Where things stand

The branch is `systemone-decision-models`. It is not merged; `main` is
linear-history only, so it will be rebased or squashed onto `main` when the
note is finished. Push your work to this branch, never to `main`.

Done and committed, all on a CPU with `tev1:0.8b` only:

- `probes/systemone.py`: a standard-library port of Ollama 0.35.0's
  `/v1/systemone` over any llama-server, with three prompt formats (`ollama`,
  `nimble`, `tev1`), the prefix primer, a temperature, and `--serve`.
- It reproduces Ollama to 1.30e-08, and a clean llama.cpp `b11232` build
  reproduces Ollama's bundled one to 1.30e-08. **You do not need Ollama to get
  Ollama's numbers**; you need llama.cpp `b11232` and this port.
- Prompt formats are byte-identical to their sources (`prompt_parity.py`).
- Four deterministic ways the same request's probabilities move: the
  llama.cpp version (3.86e-02, `b9190` against `b11232`), the primer
  (1.85e-02), other questions in the request (1.33e-02), and the server's
  cache history (0.0194 on one record). `--isolate` with `--cache-ram 0`
  removes the last one.
- Bespoke's 13 public benchmark subsets rebuild byte-identical from upstream
  (`fetch_public_data.py`, 3,880 records), and Bespoke's own runner drives
  either backend (`public_suite.py`), checked on 8 records.

Nothing about accuracy, calibration, GPU numerics or GPU latency is known.
That is your job.

## Which host

Use the roles in the `model-hosts` skill. Describe hosts by role in anything
you commit, never by name or address.

Confirm with the owner before running anything; this is the proposal.

- **windows-3090 (RTX 3090 24 GB, CUDA, native Windows)** for the CUDA
  numbers. All three models fit at Q8_0. It is shared: the desktop holds
  part of the card, and it runs CI and interactive use, so check both before
  any timed run and record load. `build_llama_cpp.sh` is bash; build there
  with CMake and Ninja in a VS developer shell, same tag and flags.
- **workstation (RDNA4, 16 GB)** is the second backend, via Vulkan, for
  experiment B's cross-backend question. All three models fit in 16 GB.
- **cloud-gpu** only if the owner approves the cost, for example to separate
  a Windows/WDDM effect from a CUDA one on Linux.

Run models one at a time on one card, so the numbers are single-GPU.

If only one host is available, do everything on it and say which in every log.

## Setup

Work outside the repository for everything large. Suggested layout, adapt as
needed but keep it out of the checkout:

```bash
export WORK=$HOME/nimble-tev1-work          # GGUFs, builds, data, run dirs
export REPO=<this checkout>                  # on branch systemone-decision-models
export NIMBLE_SRC=$WORK/nimble OLLAMA_SRC=$WORK/ollama TEV1_SRC=$WORK/tev1
mkdir -p $WORK && cd $WORK

git clone https://github.com/bespokelabsai/nimble $NIMBLE_SRC && git -C $NIMBLE_SRC checkout 62076b4f2d365b5879dafcf7f6dd072a1fe76df7
git clone https://github.com/ollama/ollama $OLLAMA_SRC     && git -C $OLLAMA_SRC checkout 1abe35e6e6e777e858bbfbba283667ee8d516801
git clone https://github.com/togethercomputer/tev1 $TEV1_SRC && git -C $TEV1_SRC checkout 1dde7782382c9f49d627153759b8d1deab426ce0

cd $REPO
git config core.hooksPath .githooks          # the leak check; see "Committing" below
```

These are upstream repositories you are only reading or running their data
converters from; the pinned commits are the ones the CPU work was checked
against. If a pin has been force-pushed away, stop and report it.

**llama.cpp.** Build the tag Ollama pins, for your backend:

```bash
LLAMA_BUILD_ROOT=$WORK/llama-builds models/nimble-tev1/probes/build_llama_cpp.sh cuda     # or vulkan
```

The last two lines printed are the binary path and its version. Record both.
The CPU half built `b11232` clean as commit `6f767fe`. For experiment B also
build `cpu` on the same host.

**Weights.** Ollama's GGUFs, SHA-256 verified against the manifest:

```bash
python models/nimble-tev1/probes/registry.py --download $WORK/gguf > /dev/null
```

That writes `nimble-latest.gguf`, `tev1-latest.gguf`, `tev1-0.8b.gguf` (about
14.8 GB). Re-run `registry.py` without `--download` and `diff` against
`results/registry.log`: if a manifest digest changed, Ollama republished a
model, and that must be recorded before anything is compared with the CPU
numbers.

**Ollama**, only for experiment A's cross-check. Install 0.35.0 exactly (the
release tarball for your platform). Run it with `OLLAMA_MODELS=$WORK/ollama-models`
on a port of its own, and `ollama pull nimble tev1 tev1:0.8b`. Do not upgrade
it mid-investigation.

**Data**:

```bash
uv run --no-project --with pyarrow --with transformers --with jinja2 \
    python models/nimble-tev1/probes/fetch_public_data.py --nimble-src $NIMBLE_SRC
```

It must end `all subsets match Bespoke's manifests`. If a subset differs,
upstream data changed; do not score that subset, and record it.

**Serving a model** for the port, one model at a time:

```bash
$LLAMA_SERVER -m $WORK/gguf/nimble-latest.gguf -c 8194 -np 1 -ngl 99 \
    -b 512 -ub 512 --cache-ram 0 --host 127.0.0.1 --port 8080
# tev1 and tev1:0.8b: -c 2050
```

`-c` matches Ollama's `num_ctx` for each model. `--cache-ram 0` is required for
anything timed or for `--isolate`; the priming article says why. Wait for
`/health` to return `ok`, then confirm the GPU is actually used: the load log
must say all layers were offloaded, and a single scored question should take
tens of milliseconds, not the hundreds it takes on a CPU. The evaluating skill
has a section on GPU runs that silently ran on the CPU.

## The experiments

Run them in this order. Each one says what it answers, how to run it, and
what to commit. Rows files (`rows.jsonl`) contain third-party benchmark text:
keep run directories under `$WORK`, and commit only `summary.json`,
`manifest.json` and `host.json` copies, and comparisons (see "What to commit").

### A. Reproduce the CPU findings on the GPU (small, do first)

Answers: does the port still match Ollama on a GPU build, and how big are the
build, primer and coupling effects for each of the three models?

For each model in turn (`tev1:0.8b`, `tev1`, `nimble`, with system prompts
`ollama-tev1`, `ollama-tev1`, `ollama-nimble`):

```bash
P=models/nimble-tev1/probes; F="$P/fixtures/ticket-3q.json $P/fixtures/ticket-8q.json"
python $P/fidelity.py --ollama http://127.0.0.1:$OLLAMA_PORT --model <tag> --system <sys> \
    --server port-cuda-b11232=http://127.0.0.1:8080 $F
python $P/priming.py --server http://127.0.0.1:8080 --system <sys> --repeats 5 $F
python $P/schema_coupling.py --server http://127.0.0.1:8080 --system <sys> --formats ollama,tev1
```

For `fidelity.py`, Ollama's own runner is also worth a `--server` entry: find
its port on the process command line (`ps -eo args | grep [l]lama-server`).
Ollama on a GPU picks its own backend, so check its log for which one.

Also run `identical_prompt.py` with the GPU build and `--model` pointing at
`tev1-0.8b.gguf`. It starts its own server with the build's default offload
and passes no `-ngl`, which is fine for a crash test.

Save each as `results/<probe>-<model>-<backend>.log`, with a first line in
the style of the existing logs saying what ran. Before running anything, read
the existing logs so you know what "the same" looks like.

Stop and report if the port disagrees with Ollama's own runner by more than
1e-6 on a GPU: something in the port or the build is wrong, and every later
number depends on it.

### B. Cross-backend numerics

Answers: are CUDA (or Vulkan) probabilities the CPU's probabilities?

Same GGUF, same tag, two builds on the same host (`cuda` and `cpu`, or
`vulkan` and `cpu`). Run `fidelity.py` with both as `--server` entries against
Ollama. Report the largest difference and whether any answer changed. If the
difference is of the order of the 3.86e-02 build effect, that is a finding in
its own right: a threshold does not carry between backends.

### C. The public suite: accuracy and calibration (the main result)

Answers: how accurate and how calibrated is each model on 3,880 human-labelled
decisions, through a backend that reproduces Ollama?

For each model, each of the 13 subsets, backend `port`, format `ollama`,
isolated:

```bash
SUBSETS="vitaminc-dev massive-en-US massive-de-DE boolq squad2 paws multinli civil_comments aegis2 helpsteer2 summeval-relevance summeval-consistency pubmedqa"
for d in $SUBSETS; do
  python models/nimble-tev1/probes/public_suite.py --nimble-src $NIMBLE_SRC \
    --data $NIMBLE_SRC/data/public/$d/all.jsonl --output-dir $WORK/runs/$d/<model>-ollama-format \
    port --server http://127.0.0.1:8080 --model <model> --format ollama --system <sys> --isolate
done
python -m nimble.evaluation.summarize_public_suite --root $WORK/runs --runs <model>-ollama-format   # run from $NIMBLE_SRC
```

The runner resumes an interrupted run and refuses a changed backend.

Two of Tev1's constraints matter here. Its context is 2,050 tokens and some
records are longer (`helpsteer2` reaches 1,933 prompt tokens with Qwen3.5-9B's
tokenizer in the Nimble format). A too-long prompt becomes an error row, which
the runner counts as incorrect: report error counts per subset, never only
accuracy. And MASSIVE has 18 options, inside Tev1's trained range of 2 to 24.

Compare with:

- Ollama's published means on this suite [CLAIM]: Nimble 75.7%, Tev1 4B 73.3%,
  Tev1 0.8B 63.5%. The blog calls it the mean across the 13 subsets.
- Bespoke's per-subset table in `$NIMBLE_SRC/docs/PUBLIC_BENCHMARKS.md`
  [CLAIM], which was BF16 transformers, not a Q8_0 GGUF, and possibly a
  different Nimble revision. A gap to that table is expected; a gap to
  Ollama's number is a finding.

Cross-check one subset (`boolq` is quick) with backend `ollama` against
Ollama itself. Accuracy should agree within the history effect already seen;
report the per-record differences.

### D. Prompt format (the experiment most likely to be new)

Answers: does sending each model its own training format change accuracy or
calibration?

Repeat C for:

- `tev1` and `tev1:0.8b` with `--format tev1 --system tev1`
- `nimble` with `--format nimble --system nimble`

Then pair each against its `ollama`-format run with Bespoke's comparison tool,
which reports accuracy with Wilson intervals, ECE and an exact McNemar test:

```bash
python -m nimble.evaluation.compare_public \
  --runs ollama=$WORK/runs/<d>/<model>-ollama-format/rows.jsonl native=$WORK/runs/<d>/<model>-native-format/rows.jsonl \
  --output-dir $WORK/runs/<d>/compare-<model>-format
```

Two cautions. The `nimble` format refuses a `noul` with no criteria and
`null` descriptions, because Bespoke's own code does; the benchmark records
always carry criteria, so this should not arise, but count refusals. And the
`tev1` format's handling of `noul` (options `no` then `yes`) and of missing
descriptions is this repository's choice, documented in the how-it-works
article; say so next to its numbers.

### E. Calibration and temperature (offline, no GPU)

Answers: are the probabilities calibrated, and would a temperature help?

No new inference is needed. `rows.jsonl` holds each record's probabilities,
and `softmax(logit / T)` equals `p^(1/T)` renormalised, so any temperature can
be applied afterwards. Fit one T per model on half of each subset's families
(families, not records, so pairs are never split) and evaluate ECE, Brier and
NLL on the other half. Write the fitting code as a committed probe
(`probes/temperature.py`) reading the run directories. Bespoke fitted T=2.179
for an older Nimble revision [CLAIM]; the current checkpoint is uncalibrated
by its authors' account [CLAIM].

### F. Latency on the GPU

Answers: what does a decision cost on a GPU, and how does it grow?

`priming.py` on each model gives per-question cost at two question counts.
Add a probe (`probes/latency_sweep.py`) that varies state length (for example
64, 256, 1024 and 2000 tokens of filler around a fixed question) and question
count (1, 2, 4, 8, 16, 32), cold primed and warm, five repeats, first
discarded. The streaming question in the notes (a transcript that grows every
few hundred milliseconds) is the one this answers: how long a cold prefill of
a new state takes, since the Ollama format puts the state before the schema and
so a new state invalidates everything after it.

### G. Quantisation (only if time allows)

Answers: what does Q4_K_M or Q6_K cost in accuracy and calibration?

For Tev1 4B, `bartowski/togethercomputer_Tev1-4B-experimental-GGUF` has
quantisations made from BF16 [CLAIM, not inspected here]; verify its tensor
types with `registry.py`'s GGUF reader before trusting its name. For Nimble
there is no BF16 GGUF; the only option is requantising Ollama's Q8_0
(`llama-quantize --allow-requantize`), which compounds two quantisations. Say
so if you do it. Run C's suite on a few subsets, not all thirteen.

### Not in scope without asking

- llama.cpp's open System One pull request (#29321, a `llama-system-one`
  tool). It is unmerged contributor code; building it runs that code. Read
  the diff and ask the owner before building it.
- Any model beyond these three, and any fine-tuning.

## What to commit

Into `models/nimble-tev1/`:

- `results/`: the probe logs from A, B and F; for C and D, one
  `results/public-suite/<model>-<format>/<subset>.summary.json` per run (copy
  `summary.json`, `manifest.json` merged in or alongside, and `host.json`), the
  `summarize_public_suite` output, and the `compare_public` `comparison.json`
  and `REPORT.md` files. Never `rows.jsonl`: it contains benchmark text under
  third-party licences, and Bespoke's repository keeps it out for the same
  reason. Before committing, grep the files you are adding for record text: a
  summary should contain no sentence from any dataset.
- Redact before committing: the `dataset` field of a manifest is an absolute
  path on your host. Replace the directory part with `$NIMBLE_SRC`. Same for
  any path in a log.
- `probes/`: every new probe, runnable as committed.
- `notes/`: a new article, `03-gpu-results.md`, in the style of the CPU one
  (hardware first, `[MEASURED]` and `[CLAIM]` kept apart, a "Not exercised"
  list), and updates to the README hub's findings, contents and open
  questions. Put numbers from C and D in tables per subset, with error counts.
  Remove from `README.md`'s status box the claim that GPU work is pending, once
  it is not.
- Keep the CPU article as it is, except to correct something shown wrong, and
  if you do, say what was wrong and how you know (a correction record).

Then run the humanizer and `check_rewrite.py` on every changed note, per the
writing-notes skill, and humanize your commit messages.

## Committing and pushing

The leak check (`.githooks/leak-check`) needs the private pattern list at
`$(git rev-parse --git-common-dir)/info/leak-patterns`. That file is never in
the repository and the CPU half did not have it. If your clone lacks it, the
hooks fail closed: stop and ask the owner for it. Do not bypass the check with
`--no-verify`.

Before pushing, confirm the base has not moved:

```bash
git fetch --prune origin
git merge-base --is-ancestor origin/main HEAD || echo "main moved: rebase is the owner's call"
```

Push with `git push origin systemone-decision-models`. Never force-push this
branch; if it diverged, report it.

## Pitfalls already paid for

- **Cold is not cold without `--cache-ram 0`.** llama-server's host prompt
  cache restores a state even after the slot was displaced. A first version of
  `priming.py` reported warm runs as cold because of it.
- **Repeated identical prompts crash `b9190`.** If you use any build other than
  `b11232`, run `identical_prompt.py` on it first.
- **The Ollama format couples questions.** Benchmark records have one question
  each, so C and D are unaffected; F with several questions is not. Keep the
  question set fixed when comparing times.
- **History moves probabilities.** Use `--isolate` for every scored run. The
  Ollama backend cannot be isolated this way, which is why the port is the
  primary backend for C.
- **Ollama's thread count.** On the CPU host Ollama appeared to start more
  threads than there were usable CPUs. Irrelevant on a GPU, but do not time
  Ollama itself; time the port.
- **`pkill -f` and `pgrep -f`** match their own command line. Kill servers by
  PID from `ps`.
- **Never print a host or address from a probe.** The probes take URLs as
  arguments and log only what the server reports about its build; keep it so.

## Reporting back

When you stop, whether finished or blocked, leave the branch pushed and put a
short status at the top of this file: what ran, on which role and backend,
what is committed, what is left, and anything that contradicts the CPU half.
