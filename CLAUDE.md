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
- **Never name a company or product this work may inform**, in files, commit
  messages, PR text, workflow names or result logs. A technique such a product
  might use is fine, described generically (a media router's tap, a call
  centre's audio); the name is not.
- **Not general.** Do not write code or docs that imply a generality this repo
  does not have.

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
exempt. Commit messages and PR text are human-facing too: humanize them before
pushing.

## Layout

| Path | Owns |
|---|---|
| `models/<slug>/README.md` | **the deliverable**: what the model is, what it solves, how to use it, what it needs, what was measured. For a long note, this is the hub and the content lives in `notes/` |
| `models/<slug>/notes/` | a long note split into articles that link to each other, for reading in the GitHub web UI |
| `models/<slug>/probes/` | the code that produced every number in that note — committed, re-runnable |
| `models/<slug>/results/` | raw probe output, committed as evidence |
| `models/README.md` | the rubric every note is held to |

**Probe output is committed, so probe output is public.** Never print a
hostname or address from a probe; take it from `PLAYGROUND_HOST` and redact it
in anything written to `results/`.

## Deeper reference — invoke as needed

Opt-in skills in `.claude/skills/`, not loaded by default:

| Skill | Scope | Reach for it when |
|---|---|---|
| `adding-a-model` | any model | **onboarding any new model, backend, or protocol** — start here, not in the code |
| `evaluating` | any model | designing a measurement, writing a scenario, analysing results, or judging whether a finding is trustworthy |
| `model-hosts` | any model | choosing or provisioning a host; CUDA vs ROCm; local vs remote; VRAM budgeting |
| `writing-notes` | any note | **writing or editing any human-facing markdown**: the humanizer requirement, what a rewrite must never change, splitting long notes |

Nothing in this repository should ever hardcode a hostname, address, cloud
project or credential, or mention the owner's personal tooling or environment.
The endpoint arrives as `PLAYGROUND_HOST` / `PLAYGROUND_PORT`, and hosts are
described in `.claude/skills/model-hosts/SKILL.md` by their nature rather than
their name. Private notes about the local environment live in
`CLAUDE.local.md`, which is gitignored; read it if it exists, and never copy
its contents into a tracked file.
