# CLAUDE.md

A **personal research notebook for exploring AI models**, and the code that
makes the exploring honest.

The loop is always the same: pick a model, get it actually running on hardware
that can be named, understand it properly, stress it until it shows its edges,
then write it up — both as a practical usage guide and as a deeper account of
what it is and why it behaves the way it does.

## What this is not

- **Not a product, and not a library.** Nobody clones this and runs it. There
  is no support, no stability promise, no install guide, no API surface anyone
  else depends on.
- **Not connected to any employer, job, client or commercial project.** This is
  a personal project, done for curiosity. Nothing here is work output, nothing
  here reflects a workplace, and no scenario in it describes a real business.
  `scenarios/support_desk.py` is a support desk for the same reason a tutorial
  is a to-do app: it gives tools an obvious reason to exist. Nothing about the
  measurement is domain-specific, and nothing about it should be written as
  though it were.
- **Never name a company or product this work may inform**, in files, commit
  messages, workflow names or result logs. A technique such a product
  might use is fine, described generically (a media router's tap, a call
  centre's audio); the name is not.
- **Not general.** See *Current state* below. Do not write code or docs that
  imply a generality this repo does not have.

## The governing rule

The value of a note is that its numbers can be trusted, so:

> **The measurement apparatus must not distort the measurement.**

Every non-obvious design decision here exists because a naive alternative
produced *confident wrong numbers* rather than visible breakage. Code that
fabricates a finding costs more than code that crashes — read
`.claude/skills/evaluating/SKILL.md` before changing anything that touches
timing, buffering, or concurrency.

This rule has a documentation corollary and a code corollary:

- **A reader must always be able to tell what was measured from what was merely
  read.** Hence the `[MEASURED]` / `[CLAIM]` / `Open` separation in every note.
- **Every number in a note must have committed code that produced it.** A
  finding without its method is a rumour. This was learned the expensive way:
  a claim that a model discarded its reasoning trace stood for a day because
  the probe behind it — which had simply read the wrong field name — was never
  committed where anyone could look at it.

## Writing for people

Every human-facing document (model notes and their articles,
`models/README.md`, the root `README.md`, `audio/corpora/MANIFEST.md`) goes through the **humanizer** skill
before it is finished, and then through
`.claude/skills/writing-notes/check_rewrite.py`, which fails if the rewrite
touched code, links, tags or numbers. A note that reads like chatbot output
makes readers doubt its numbers. Load `.claude/skills/writing-notes/SKILL.md`
before writing or editing any of these files; it also covers splitting a long
note into navigable articles. Agent-facing files (this one, the skills) are
exempt. Commit messages are human-facing too: humanize them before
pushing.

## Layout

| Path | Owns |
|---|---|
| `models/<slug>/README.md` | **the deliverable**: what the model is, what it solves, how to use it, what it needs, what was measured. For a long note, this is the hub and the content lives in `notes/` |
| `models/<slug>/notes/` | a long note split into articles that link to each other, for reading in the GitHub web UI |
| `models/<slug>/probes/` | the code that produced every number in that note — committed, re-runnable |
| `models/<slug>/results/` | raw probe output, committed as evidence |
| `models/<slug>/examples/` | runnable integration code a reader can copy (Parakeet: an ONNX `Transcriber` and GStreamer integrations, with a test script) |
| `audio/corpora/` | test audio. The telephone WER corpora are DVC stage outputs (`dvc.yaml`, `dvc.lock`, `audio/corpora/fetch.py`) pinned to upstream revisions; `MANIFEST.md` records every corpus, its licence and what was rejected |
| `models/README.md` | the rubric every note is held to |
| `src/playground/` | shared instrument library (below) |
| `scenarios/*.py` | agent specs: instructions + tools, each exporting `SPEC` |
| `logs/*.jsonl` | scratch run logs — gitignored; promote to `results/` to keep |
| `.github/workflows/` | CI on self-hosted GPU runners: push to `main` and manual dispatch only (see *Repository posture*) |
| `.githooks/` | the leak check, run on commit and push; patterns are per clone, never in the tree |
| `logs/wer/` | WER segment sets and per-runtime predictions — gitignored, because they are model output on third-party audio; `results/` gets aggregates only |

**Probe output is committed, so probe output is public.** Never print a
hostname or address from a probe; take it from `PLAYGROUND_HOST` and redact it
in anything written to `results/`.

### The instrument library

Currently specific to the speech-to-speech work; other models are driven by
their own probes and need none of it.

| Module | Owns | Model-specific? |
|---|---|---|
| `protocol.py` | wire events, `ToolCall` parsing, rate/chunk constants | **yes** — the only adapter |
| `client.py` | `DuplexSession`: concurrent send + receive, tool dispatch, barge-in | mostly no |
| `audio.py` | load, resample, channel split, wall-clock pacing | no |
| `tools.py` | `@registry.tool`, schema export, latency injection | no |
| `policy.py` | barge-in disposition — the decision under study, not plumbing | no |
| `live.py` | microphone client via `pw-record`/`pw-play` | no |
| `asr.py` | sibling utility, **not** part of the session loop: reference transcripts + streaming-stability metrics. Optional `asr` extra. | no |

**The containment rule:** everything backend-specific lives in `protocol.py`.
If you reach for a backend detail in `client.py`, `tools.py`, a scenario, or
`policy.py`, that is the signal to extract an adapter instead — see
`.claude/skills/adding-a-model/SKILL.md`.

## Commands

```bash
uv sync                    # resolve deps into .venv

# drive a recorded conversation
playground scenarios/nvidia_demo.py --audio audio/tool_call.wav

# talk to it with a microphone (HEADPHONES REQUIRED — see below)
python -m playground.live scenarios/nvidia_demo.py

ffmpeg -i in.wav -ar 24000 -ac 1 out.wav   # correct pre-conversion
jq -c 'select(.kind|startswith("tool"))' logs/session.jsonl

# evaluation corpora: from upstream, or from a DVC remote configured with --local
uv run --extra corpora dvc repro   # then `git diff dvc.lock` must be clean
uv run --extra corpora dvc pull
```

On Windows (PowerShell), the same commands with Windows paths; Photon needs the
GPU there, and the GStreamer examples have a cross-platform driver:

```powershell
uv sync --extra asr-cuda --extra corpora     # CUDA torch; `asr` and `asr-cuda` conflict
$env:PHOTON_DEVICE = "cuda"; uv run --extra asr-cuda python models/parakeet-redux/probes/step0.py
uv run --with sherpa-onnx python models/parakeet-redux/examples/test_gstreamer.py clip.wav out
```

A DVC remote belongs in `.dvc/config.local` (`dvc remote add --local`), which
is not committed, and should name an SSH alias rather than a host.

Host comes from the **`PLAYGROUND_HOST` / `PLAYGROUND_PORT`** environment
variables, or `--host`. No endpoint is hardcoded: this repo is public and an
address does not belong in it.

## Current state — read this before assuming generality

Six models so far: three investigated, two seeded with a note but no probes
yet, and one half-done. Only one is driven by the harness:

- **NemotronLabs VoiceChat 11B** (speech-to-speech), served by NVIDIA's NIM
  container on a rented 2× H100 80GB (NVLink) cloud node, over the OpenAI
  Realtime WebSocket dialect at `/v1/realtime`.
- **Parakeet Redux** (ternary ASR) — a sibling utility, deliberately *not* in
  the session loop. The same note also measures NVIDIA's original
  `parakeet-tdt-0.6b-v3` for deployment on ONNX (sherpa-onnx) and NeMo:
  telephone WER, CPU sizing, GPU runtimes, GStreamer, and live latency
  against NVIDIA's Nemotron streaming model. That work is driven by its own
  probes (`nemo_vs_onnx.py`, `wer_telephone.py`, `onnx_concurrency.py`,
  `streaming_online.py`) and never touches the harness.
- **K2-Horizon-32B** (text-only reasoning) — no harness involvement at all.
- **Kokoro-82M** (text-to-speech) — a seeded note: vendor claims and a probe
  plan, nothing measured yet.
- **Piper** (text-to-speech) — a seeded note, like Kokoro's.
- **Nimble 9B and Tev1** (typed decision models, text-only) — CPU half measured
  with the smallest model; GPU half pending, planned in `models/nimble-tev1/HANDOFF.md`.
  Driven by its own probes over a bare llama-server; no harness involvement.

**Two platforms, Linux and Windows.** The notes were written on Linux; the
Parakeet stack is also verified on native Windows (a GTX 1650 box), in
`models/parakeet-redux/notes/14-windows.md`. Keep both working and documented:
Linux code paths and output stay unchanged, Windows branches are gated on
`sys.platform == "win32"`, opt-in variables (`PHOTON_DEVICE`, `ONNX_CPUS`) default
to the old behaviour, OS-specific host facts go through
`models/parakeet-redux/probes/hostinfo.py`, and a note says which platforms its
numbers came from. On Windows use Python 3.13+ for anything timed (see
`evaluating`).

There is **no multi-model abstraction** — no adapter registry, no host
registry, no protocol dispatch. `protocol.py` is a single module, not a
package. Adding a second *speech* model is the moment to build that seam.

## Invariants

These govern the speech harness specifically.

- **Pace at wall clock, with absolute monotonic deadlines.** Never
  `sleep(chunk_ms)` in a loop — it accumulates send cost and drifts the stream
  behind real time, making a punctual model look late.
- **Never `await` a tool inside the receive loop.** Dispatch with
  `create_task`. Blocking the receiver stops draining agent audio, which
  manufactures silence that never happened.
- **Save artifacts before cleanup.** A group SIGTERM reaps child processes
  first; a `terminate()` that raises then escapes the `finally` and destroys
  the run's only record.
- **Never swallow subprocess stderr.** A `DEVNULL` here turned a one-line
  audio-flag error into a multi-step misdiagnosis of the model.
- **One speaker per channel.** Downmixing a two-party recording feeds the model
  both halves of its own conversation.
- **`--speed != 1.0` invalidates every latency number.** It exists to check
  plumbing.

## Environment traps

- **Headphones are mandatory** for `playground.live`. The mic otherwise captures
  the agent's own voice, which a full-duplex model hears as a barge-in and talks
  over, escalating until killed.
- **`pw-play` needs `--raw`, not `--container raw`.** `pw-record` accepts both,
  so capture works while playback silently fails.
- **`ssh -n host 'bash -s' <<EOF` runs an empty script** — `-n` redirects stdin
  from `/dev/null`. Exit 0, no output, looks like success.
- **`pgrep -f "<string>"` matches its own wrapper process**, so a poll loop can
  report work that finished. **`pkill -f "<string>"` is worse:** if the string
  appears anywhere in the command that runs it, including later in the same
  shell line, it kills that shell. Kill by PID from `ps`, or with a pattern
  like `"[r]un_matrix"` that cannot match itself.
- **Rootless podman** refuses a bind mount whose host directory does not exist
  (`statfs ... no such file or directory`), may lack the cgroup `cpuset`
  controller (pin with `taskset` inside the container instead of
  `--cpuset-cpus`), and on an SELinux host needs `--security-opt
  label=disable` or `:Z`, or mounted files give `Permission denied`.

## Deeper reference — invoke as needed

Opt-in skills in `.claude/skills/`, not loaded by default:

| Skill | Scope | Reach for it when |
|---|---|---|
| `adding-a-model` | any model | **onboarding any new model, backend, or protocol** — start here, not in the code |
| `evaluating` | any model | designing a measurement, writing a scenario, analysing results, or judging whether a finding is trustworthy |
| `model-hosts` | any model | choosing or provisioning a host; CUDA vs ROCm; local vs remote; VRAM budgeting |
| `realtime-protocol` | VoiceChat only | touching `protocol.py`, debugging events, tool-call plumbing, or the wire format |
| `harness-internals` | VoiceChat only | changing `client.py`/`audio.py`/`live.py`, or adding a capability to the session loop |
| `audio-fixtures` | speech models | sourcing test audio, channel/format questions, corpus licensing |
| `writing-notes` | any note | **writing or editing any human-facing markdown**: the humanizer requirement, what a rewrite must never change, splitting long notes |

Nothing in this repository should ever hardcode a hostname, address, cloud
project or credential, or mention the owner's personal tooling or environment.
The endpoint arrives as `PLAYGROUND_HOST` / `PLAYGROUND_PORT`, and hosts are
described in `.claude/skills/model-hosts/SKILL.md` by their nature rather than
their name. Private notes about the local environment live in
`CLAUDE.local.md`, which is gitignored; read it if it exists, and never copy
its contents into a tracked file.

## Repository posture

Public, read-only, one author. Pull requests, issues, the wiki and projects are
disabled on GitHub; forks are allowed. Finished work is pushed straight to
`main`, which a ruleset keeps linear and protects from force-push and deletion.

- **Never add a `pull_request` or `pull_request_target` trigger** to a
  workflow. CI runs on self-hosted GPU machines, and from a public repository
  those triggers hand the runner to anyone who forks. Push to `main` and
  `workflow_dispatch` only.
- **Actions are restricted to GitHub-owned actions, pinned by commit SHA** (the
  repository setting requires it). Pin as `uses: owner/action@<sha>  # vX.Y.Z`.
- **The leak check.** `.githooks/leak-check` blocks a commit or push whose added
  lines, file paths or commit messages match a private pattern. The patterns
  live in `.git/info/leak-patterns`, never in the tree, because the list is
  itself private; a clone without that file fails closed. Enable per clone with
  `git config core.hooksPath .githooks`. Do not bypass it with `--no-verify`
  to get a commit through: a match means something is about to be published.
- **Two licences.** The write-ups (Markdown under `models/`, the root
  `README.md`, `audio/corpora/MANIFEST.md`) are CC BY 4.0 in
  `LICENSE-CC-BY-4.0`; code and everything else is MIT in `LICENSE`.
