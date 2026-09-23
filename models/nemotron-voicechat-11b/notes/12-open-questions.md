# Open questions

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


Ordered by how much they matter. Questions answered since the first pass stay
in the list, struck through, with a pointer, because the gap between what was
expected and what was found is often the finding.

## Answered

| Was open | Answer | Where |
|---|---|---|
| ~~Does full-duplex tool calling hold up when a tool is slow and the caller interrupts?~~ | Fires above a measured latency threshold; the more interesting finding is what the model does while a tool is outstanding | [barge-in results](07-results-barge-in.md) |
| ~~Whether NIM exposes precision or `gpu_memory_utilization`~~ | `LLM_GPU_MEM_UTIL` / `TTS_GPU_MEM_UTIL`, env-overridable, defaults 0.45 / 0.4. These set the footprint; precision does not | [requirements](01-requirements.md) |
| ~~What is in the NIM EULA, exactly?~~ | Nothing. `/opt/nim/LICENSE` is 538 bytes of pointers to three externally-hosted agreements; the prescribed check does not answer the question | [licensing](02-licensing.md) |
| ~~What does turn-taking latency actually look like?~~ | Was unmeasurable: nothing timestamped the agent becoming audible, and the obvious fix measures the transport instead | [latency results](08-results-latency.md), [failures](09-failures.md) |

## Still open

**1. Does the smaller-card prediction hold?**
[requirements](01-requirements.md) shows the footprint comes from a configured
fraction, not from what the weights require. Weights need ~19 GB plus
~4 GB TTS, and the rest is elective KV. Lowering both `*_GPU_MEM_UTIL`
variables *should* fit a 40-48 GB card. One step has been tested: at
`0.30` / `0.20` it served full sessions in 44,716 MiB, and the cost appeared as
occasional 5-chunk replies ([latency results](08-results-latency.md)). The
floor is still unknown. The failure to watch for is less an OOM than silent KV
thrashing that shows up only as latency.

**2. Are on-hold messages spoken at all?**
`on_hold_message` reaches the model only as prompt text, with no server
machinery behind it. My prediction is that it will be inconsistent: sometimes
honoured because the field name describes itself, often ignored. Untested. If
on-hold filler matters, the likely fix is to state the behaviour in
`instructions` instead.

**3. Does "never guess a missing argument" survive duplex pressure?**
The model card states that tool arguments must be values the user spoke, and
that the model should ask for a missing one instead of inventing it. It
declined to invent a weather tool, which is adjacent but a different test. The
real probe is an utterance that names a tool and omits a required argument.
This is more urgent now: [barge-in results](07-results-barge-in.md) shows the
model inventing tool calls under latency, so the claim is under more pressure
than a quiet session suggests.

**4. Concurrency.**
`max_batch_size: 8`, one GPU instance, and 2.42× single-stream compute headroom
([latency results](08-results-latency.md)). Whether 8 concurrent sessions
actually hold, and how latency degrades on the way there, is unmeasured. The
2.42× figure means batching must be doing real work to reach 8. That is
arithmetic, not a measurement.

**5. `"type": "dict"` versus `"type": "object"`.**
The server's own warmup uses `dict`; the docs say `object`. Both appear to work,
since the value only ends up in prompt text. Not A/B tested. If tool-call
accuracy ever looks marginal, flip it.

**6. Can tools be changed mid-session?**
`session.update` sets tools, and `tools_not_set` implies they must exist before
a triggering turn. Whether a second `session.update` mid-conversation swaps
them is unknown.

**7. Does the model handle a tool *error* gracefully?**
If an error string comes back as `function_call_output`, does the model recover
verbally, retry, or wedge? The harness supports it; it has never been run.

**8. How hard is the Apache-2.0 serving path really?**
`nemo/collections/speechlm2/` is Apache 2.0 and contains the duplex
implementation. It is unknown whether it includes a streaming server or only
offline scripts, how much of NIM's value is TensorRT optimisation and how much
is plumbing, and what latency a self-hosted path would reach. Those answers
separate "a permissive path exists in principle" from "a permissive path is a
week of work." [licensing](02-licensing.md) raises the stakes, since
the NIM and HF channels carry different licence instruments for the same
weights.

---

Previous: [Probes and method](11-method.md) | [Contents](../README.md#contents)
