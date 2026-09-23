---
name: evaluating
description: Use when designing a measurement for ANY model in model-playground — a probe, a scenario, a benchmark — or when analysing its output and deciding whether the result can be trusted. Applies to every note in models/, not only the speech work. Covers choosing what is worth measuring rather than what is easy to measure, proving the round trip before believing a number, how to build speech scenarios with injected tool latency as the independent variable, jq recipes for the event log, and the specific ways a run can look clean while measuring nothing — inert subsystems, JIT warmup spikes, speed!=1.0, client-caused silence, and reporting a passing result for a code path that never executed.
---

# Evaluating

This applies to every model written up in `models/`, whatever its modality. The
general parts come first; the sections on scenarios, the event log and pacing
are specific to the speech harness, and are marked where they begin.

## What is worth measuring

Measure the thing that would change your mind about the model, not the thing
the harness makes easy to print. The easy number is usually throughput; the
interesting number is usually a failure mode under load or under ambiguity.
Two rules hold regardless of modality: a number without the hardware that
produced it is not a result, and a subsystem that never fired must be named
rather than reported as a pass.

### In a full-duplex speech model

The interesting failures are **conversational**,
not mechanical. "Does it emit valid JSON" is table stakes. The questions that
matter:

- What does the agent do during the gap while a tool runs? Fill it, go silent,
  repeat itself, or hallucinate the result before it arrives?
- Does barge-in work *while a tool is in flight*, not just during speech?
- Does it distinguish backchannel agreement from redirection?
- Does it ask for a missing required argument, or invent one?
- Does it recover verbally from a tool error, or wedge?
- How does behaviour degrade as tool latency grows — 150 ms vs 2 s vs 8 s?

That last one is the curve. A single latency is a point, not a finding.

## Scenario design — speech harness only

From here to the end of the event-log section, this is about
`src/playground/` and the VoiceChat work. A text or ASR model is driven by its
own probe under `models/<slug>/probes/` and has none of this machinery.

A scenario is a Python file exporting `SPEC` — instructions plus a tool
registry. See `scenarios/nvidia_demo.py` (matches NVIDIA's own demo tools) and
`scenarios/support_desk.py`. The second is shaped like a support desk for the
same reason a tutorial is a to-do app: it gives tools an obvious reason to
exist and an obvious spread of latencies. Nothing measured through it is
domain-specific.

**Injected `latency` is the independent variable.** Spread tools across the
range a real integration lives on, so one run exercises the whole curve:

| Tool latency | Models |
|---|---|
| ~150 ms | cache hit |
| ~2 s | ordinary network round trip |
| ~8 s | cold backend — where behaviour is expected to break |

**Make handlers deterministic.** `generate_random_number` returns the midpoint,
not a random value. A harness that introduces variance the comparison cannot
account for cannot support run-to-run claims.

**Include a side-effecting, non-idempotent tool.** `transfer_to_queue` is the
case where cancelling on barge-in is the *dangerous* option, not the safe one.

## Reading the event log

`logs/*.jsonl` is the artifact. One JSON object per line, `t` in seconds from
session start.

```bash
# tool round trips
jq -c 'select(.kind|startswith("tool"))' logs/session.jsonl

# barge-in decisions with what was heard
jq -c 'select(.kind=="barge_in.decision")' logs/session.jsonl

# was anything pending when the caller spoke?  <- the honesty check
jq -c 'select(.kind=="caller.speech_started") | {t,pending}' logs/session.jsonl

# events you are silently dropping
jq -r 'select(.kind=="unhandled") | .type' logs/session.jsonl | sort | uniq -c

# transcript as a conversation
jq -r 'select(.kind=="agent.said" or .kind=="caller.said")
       | "\(.t)  \(.kind)  \(.text)"' logs/session.jsonl
```

### Latencies worth computing

The log carries the timestamps; nobody has computed these yet:

- `speech_stopped` → first `response.output_audio.delta` — **the turn-taking
  number** (claimed ~450 ms)
- `function_call_output` sent → speech resuming — the tool-gap cost
- `tool.call` → `caller.said` — how far ahead of ASR the model commits

## How a run looks clean while measuring nothing

Each of these has actually happened here.

### An entire subsystem never executed

Across every run from 2026-09-21 the barge-in policy **never fired**. Every `speech_started` across
four runs carried `pending=[]` — no tool was ever in flight when the caller
spoke. The run output looked perfect. The central question was untouched.

The cause was the **fixture**, not the code, and it was computable from the logs
that already existed:

```
tool.call @ 11.2449  ->  next caller.speech_started @ 20.2046   gap  8.96 s
tool.call @ 26.9193  ->  next caller.speech_started @ 38.7683   gap 11.85 s
```

Injected latency was 0.1 s and 1.5 s, so every tool finished long before the
caller spoke again and the two events could not overlap. Sweeping latency past
those gaps fires the policy reliably — measured threshold between 8 s (0/3 runs)
and 10 s (fires), with the recorded `elapsed` landing on 8.80 s and 8.96 s.

Two lessons, and the second is the one that generalises:

**Always check what did not run.** If a code path is inert, say so in the
result, in the same breath as the success. A passing report for a subsystem that
never executed is worse than a failure.

**An inert subsystem usually means the independent variable is too small.** The
instinct is to suspect the code; the arithmetic is cheaper. Before concluding a
behaviour does not happen, compute whether your fixture could have produced the
conditions for it at all.

### Client-caused silence attributed to the model

Awaiting a tool in the receive loop stops draining agent audio. The model keeps
talking; you record a multi-second silence and conclude it breaks under slow
tools. See `harness-internals`.

### Pacing drift attributed to the model

Per-iteration `sleep()` feeds a real-time model slower than real time. It
responds punctually; your numbers say it responds late.

### `--speed != 1.0`

Compresses the timeline. Every latency number from such a run is meaningless.
It exists to check plumbing.

### JIT warmup spikes

```
Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
```

Warmup does not cover every shape. **Discard the first exchange** of a timing
run, or run a throwaway session first.

### A fixture that cannot exercise the thing

`interruptions.wav` has 4 s of overlapping speech — but asks knowledge questions
requiring no tools, so it cannot test barge-in *during a tool call*. Check that
your fixture can actually produce the state you are measuring.

### Trusting a filename or a dataset's reputation

11 files named `.mix.wav` were genuinely stereo. 2 784 files from the benchmark
NVIDIA itself cites were mono stimulus and useless. Only per-file inspection
distinguished them.

## The bar for reporting a result

- [ ] Round trip proven — a value from *your* handler spoken back
- [ ] Injected latency measured against configured, and matching
- [ ] `--speed 1.0`
- [ ] First turn discarded or warmup controlled
- [ ] Fixture verified capable of producing the measured state
- [ ] **Inert subsystems named explicitly**
- [ ] Measured separated from assumed, in writing

## Stating what a measurement cannot see

Every gate has a blind spot; name it rather than implying coverage.

- Transcripts cannot see audio quality, prosody, or whether overlap sounded
  natural — only a human listening can.
- Event timings cannot distinguish model latency from network latency to a
  remote host.
- A single session cannot establish a distribution. Point measurements are
  points.
- Nothing here tests concurrency; all runs are single-session.

## Known-untested, as of the VoiceChat work

Carried here so it is not rediscovered: on-hold message behaviour, the
missing-argument claim, tool-error recovery, concurrency. (Barge-in and the
VRAM floor have since been measured; turn-taking latency was *unmeasurable*
until the client was taught to timestamp the agent becoming audible.) The current list lives in the note's open-questions
section: `models/nemotron-voicechat-11b/README.md`.
