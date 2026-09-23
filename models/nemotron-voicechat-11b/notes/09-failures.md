# Failures worth recording

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


Each of these returned success while doing nothing, which is the failure mode
this repository exists to catch. They get a lot of space here because the same
shape keeps coming back.

## The central experiment never ran, and every run looked clean

Every session recorded on 2026-09-21 reported healthy: tools dispatched, results
delivered, transcripts sensible, audio captured, zero errors. On the strength
of that, the harness was described as working.

The harness was working; the experiment was not running. Every
`input_audio_buffer.speech_started` across four sessions arrived with
`pending=[]`, so `policy.decide()` had never executed once. That function is
the whole subject of the evaluation, and it was fully implemented and wired.

The cause was the fixture, not the model or the code, and the September log
shows it as soon as you look:

```
tool.call @ 11.2449  ->  next caller.speech_started @ 20.2046   gap  8.96 s
tool.call @ 26.9193  ->  next caller.speech_started @ 38.7683   gap 11.85 s
```

Injected latencies were 0.1 s and 1.5 s. Every tool completed long before the
caller spoke again, so the two events could never coincide. The fixture does
contain real barge-in (the caller starts talking 0.31 s after the agent begins
its reply), but it never overlapped a pending call.

This was expensive because nothing was broken. There was no error, no crash
and no odd number to chase. A subsystem that never executes leaves the same
evidence as one that executes perfectly, and the only way to catch it was to
ask the log something nobody had asked it yet: did this code path run at all?

That is now an invariant. A passing result for a code path that never executed
is the most dangerous output this harness can produce, so assert that the code
ran as well as what it returned.

## Three memory numbers, one conflated claim

Covered in [requirements](01-requirements.md). In short, the checkpoint is F32
on disk, the runtime is bf16, and the card requirement is a preallocation
fraction that ignores both. The earlier note merged these into one "F32 costs
you 80 GB" story, which made precision look like the lever when it is not.

Every measurement was right. The F32 finding was correct and was independently
re-verified here. The mistake was joining two correct measurements with an
assumed mechanism and then reporting the join with the same confidence as the
parts.

## The instrument could not see its own headline claim

NVIDIA claims ~450 ms turn-taking. Testing that needs the moment the agent
becomes audible, and the harness recorded the agent's audio without ever
timestamping it. `response.output_audio.delta` was consumed at
`client.py:167`, appended to the output buffer, and never logged.

The only agent-side timestamp was `response.output_audio_transcript.done`,
which lands a whole utterance later. Every turn-taking figure you can derive
from any log this harness produced before 2026-09-23 overstates latency by
seconds, so the vendor claim could not be tested either way.

The fix records `agent.audio.first_delta` on the first audio chunk after
`caller.speech_stopped`, and it is four lines. Nobody noticed the gap because
the logs did not look incomplete; the missing event was one nobody had tried
to use yet.

---

Previous: [Results: latency, headroom, VRAM](08-results-latency.md) | [Contents](../README.md#contents) | Next: [Traps, collected](10-traps.md)
