---
name: evaluating
description: Use when designing a measurement for ANY model in model-playground — a probe, a scenario, a benchmark — or when analysing its output and deciding whether the result can be trusted. Covers choosing what is worth measuring rather than what is easy to measure, proving the round trip before believing a number, and the bar a result has to clear before it is reported.
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

## How a run looks clean while measuring nothing

Each of these has actually happened here.

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
- Nothing in the speech harness tests concurrency; all its runs are
  single-session.
