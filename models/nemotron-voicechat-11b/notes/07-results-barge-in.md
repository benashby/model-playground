# Results: barge-in under slow tools

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


Hardware as in [requirements](01-requirements.md): one H100 80GB, NIM container, single session, `--speed 1.0`.
The fixture is NVIDIA's own `tool_call.wav` (84.5 s, caller on channel 0).
Probe: `probes/bargein_sweep.py`; analysis `probes/analyse_sweep.py`; raw logs
in `results/bargein-lat*.jsonl`.

## Barge-in under slow tools

Injected tool latency is the independent variable, swept across five values
with three runs each. 1.5 s is the original configuration, included as a
control that should not fire.

| injected latency | policy fired | tool calls | unrequested calls | barge-ins |
|---|---|---|---|---|
| 1.5 s | 0 / 3 | 2.0 | 0.0 | 6.0 |
| 8.0 s | 0 / 3 | 2.0 | 0.0 | 6.0 |
| 10.0 s | **3 / 3** | 3.7 | 2.0 | 3.7 |
| 12.0 s | **3 / 3** | 4.0 | 2.0 | 3.7 |
| 14.0 s | **3 / 3** | 4.0 | 2.0 | 3.0 |

The threshold is between 8 s and 10 s, and its location was predicted before
any of this ran. The September log gives the gap between a tool call and the
caller's next utterance as 8.96 s, and a tool must outlive that gap to still be
pending when the caller speaks. Measured `elapsed` at the moment of firing, over
all nine fires:

```
n=9   min 8.7959   mean 8.8197   max 8.9589   spread 0.163 s
```

That number comes from the fixture, which is why it barely moves across three
very different latency settings. Prediction and measurement agree to two
decimals.

## The disposition flip comes from the policy's arithmetic

| latency | disposition | why |
|---|---|---|
| 10 s | `complete_and_hold` ×3 | 8.82 ≥ 0.7 × 10 = 7.00 |
| 12 s | `complete_and_hold` ×3 | 8.82 ≥ 0.7 × 12 = 8.40 |
| 14 s | `cancel` ×3 | 8.82 < 0.7 × 14 = 9.80 |

`policy.py` holds a call that is ≥70% elapsed and cancels one that is not.
Because the fixture pins `elapsed` at ~8.82 s, raising the expected latency
lowers the completion ratio, and the rule flips at `8.82 / 0.7 = 12.60 s`.
That falls between the 12 s and 14 s rungs, which is where the flip was
observed.

The flip is the client's own heuristic behaving as written, so it says nothing
about the model. It is visible at all only because the policy now executes.

## `heard` is empty on every fire, so the backchannel branch is dead

All nine decisions carried `heard=''`.

`input_audio_buffer.speech_started` fires on the first detected token of caller
speech, before any transcription is available, so `user_transcript_so_far` is
empty at decision time. `policy.py`'s `BACKCHANNEL` set ("yeah", "mm hm",
"right", the distinction between agreement and redirection) can never match on
a recorded fixture, and the branch is unreachable.

The docstring anticipated this ("usually only a word or two, often an empty
string"), and the policy was written to depend on it regardless.
Distinguishing backchannel from redirection is a real requirement, but it
cannot be done at `speech_started`. It needs a later decision point, or a
policy that revises itself once transcription arrives.

## Slow tools degrade grounding

Nobody was looking for this. The model's own behaviour changes at the same
latency threshold, reproducibly, in three ways.

**It invents tool calls.** The fixture's caller never mentions news. Above the
threshold the model calls `get_news_headlines` twice per run, every run,
cycling `business` / `sports` / `entertainment`, for `France`:

| latency | unrequested calls |
|---|---|
| 1.5 s | none |
| 8.0 s | none |
| 10.0 s | `{'get_news_headlines': 6}` |
| 12.0 s | `{'get_news_headlines': 6}` |
| 14.0 s | `{'get_news_headlines': 6}` |

That is exactly 6 across 3 runs at every firing latency, with no drift.

**It fabricates argument values.** The caller says "twenty-five". Below the
threshold the model passes 25. Above it, 100:

| latency | `convert_currency` amounts |
|---|---|
| 1.5 s | `[25, 25, 25]` |
| 8.0 s | `[25, 25, 25]` |
| 10.0 s | `[100, 100]` |
| 12.0 s | `[100, 100, 100]` |
| 14.0 s | `[100, 100, 100]` |

> **This contradicts the model card**, which states: *"Tool-call arguments must
> be values the user spoke. If a required argument is missing, ask the user;
> never guess."* Under slow tools it guesses, consistently, and 100 is a round
> number that appears nowhere in the audio. The claim holds at 1.5 s and 8 s and
> fails at 10 s and above. It is conditional, and the condition is not stated
> anywhere.

**The conversation itself deforms.** Caller barge-ins drop from 6.0 per run to
3.7 and then 3.0. The model talks over more of the fixture, so fewer of the
caller's utterances register as interruptions of a listening agent.

This note does not establish the mechanism and does not guess at one. The
effect itself is threshold-shaped and reproducible: nothing at 8 s, everything
at 10 s, stable through 14 s, 3/3 every time.

---

Previous: [The harness and its fixtures](06-harness.md) | [Contents](../README.md#contents) | Next: [Results: latency, headroom, VRAM](08-results-latency.md)
