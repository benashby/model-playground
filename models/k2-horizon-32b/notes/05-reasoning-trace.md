# The reasoning trace

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

An earlier version of this note wrongly reported that the trace was
"generated, billed, and thrown away", and built a root-cause diagnosis on that
claim. The claim is corrected here instead of deleted, because how the mistake
happened is the most instructive thing in the investigation.

## What is actually true [MEASURED]

vLLM 0.30.0 returns the reasoning trace in `message.reasoning` (non-streaming)
and `delta.reasoning` (streaming). The trace is complete and free to read.

Measured across all three effort levels and both transports
(`probes/reasoning_fields.py`, raw output in `results/reasoning-fields.log`):

| effort | transport | `reasoning` | `reasoning_content` | content |
|---|---|---|---|---|
| low | non-streaming | 0 | 0 | 304 |
| low | streaming | 0 | 0 | 308 |
| medium | non-streaming | 194 | 0 | 295 |
| medium | streaming | 520 | 0 | 339 |
| high | non-streaming | **1205** | 0 | 179 |
| high | streaming | **666** | 0 | 390 |

```
message keys: ['annotations','audio','content','function_call','reasoning','refusal','role']
delta keys:   ['content','reasoning','role']
```

Two real findings survive from the original investigation:

- `usage.completion_tokens_details.reasoning_tokens` is stuck at 0 in every
  response, including those returning over a kilobyte of trace. The field is
  wrong, and it was the evidence that made the trace look unbilled-for and
  invisible.
- At `reasoning_effort="low"` there is usually no separate trace at all. In
  `results/k2-bench.log`, 8 of 9 low-effort runs returned an empty `reasoning`
  field with each way of passing the setting, while every medium and high run
  returned one. Low effort mostly skips the thinking block rather than
  shortening it, but not always.

## How the original conclusion happened

The probe read `reasoning_content`, the OpenAI/DeepSeek spelling and the one
every OpenAI-compatible client reaches for. vLLM 0.30.0 serves `reasoning`.

Nothing errored. The wrong key returned an empty string, which looks the same
as a server that withheld the field, and because that was plausible it
attracted an explanation. The chat template pre-seeds the opening think tag
into the prompt (a different tag for each effort level, visible in the
template's own code in `results/chat-template-excerpts.txt`). The model
therefore emits only a closing tag, and a parser looking for a matched pair
would find nothing.

That mechanism is coherent and built from real observations, but it explains
something that does not happen.

> **Open:** the template pre-seeding is real, but its role, if any, is
> unknown. `reasoning` is populated regardless, so either the parser handles the
> pre-seeded case or it has nothing to do with that field. It does not change
> what a caller should do, and it is listed with the other open questions in
> [method](07-method.md).

## What this cost, and the resulting rule

The claim stood because the probe was not committed. The prose said "empty in
every case", which nobody can check; the code says
`msg.get("reasoning_content")`, which anyone can check in three seconds.

The wrong observation also grew a mechanism, and the mechanism made the
observation harder to doubt.

Every note in this repository now follows the rule that *every number must
have committed code that produced it.* See `models/README.md`.

## Practical guidance

- Read `reasoning`, not `reasoning_content`. An OpenAI-compatible SDK that
  maps only the latter will silently show an empty trace.
- Do not use `reasoning_tokens` for budgeting, because it reports 0. Use
  `completion_tokens`, which does count the thinking.
- Budget `max_tokens` for the trace. How much longer it is than the answer
  depends on the question. On the train question with "give the number only",
  high effort produced 376 to 513 characters of trace for a 4-character answer
  (`results/k2-bench.log`). On an open question it was 666 to 1205 characters
  against 179 to 390 (`results/reasoning-fields.log`). Too small a limit
  truncates the output before the model reaches its conclusion.
- If you don't set `reasoning_effort`, this server uses `high`: the launch
  command sets it as the default, and the chat template falls back to it too.

---

Previous: [Results](04-results.md) | [Contents](../README.md#contents) | Next: [Failures worth recording](06-failures.md)
