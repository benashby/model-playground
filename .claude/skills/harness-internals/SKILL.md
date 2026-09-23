---
name: harness-internals
description: VoiceChat/speech-specific — this skill is about the full-duplex speech harness only, and is irrelevant to the text and ASR models in models/. Use when changing src/playground/client.py, audio.py, live.py or tools.py — the DuplexSession send/receive loops, wall-clock pacing, tool dispatch, barge-in handling, the microphone client, or the tool registry. Covers why each design decision was forced (each naive alternative produces confident wrong numbers rather than visible breakage), the pw-record/pw-play subprocess approach and why not sounddevice, subprocess lifecycle and artifact-saving order, and the extension points for adding capabilities to the session loop.
---

# Harness internals

**Scope: the full-duplex speech harness in `src/playground/`, and nothing
else.** Only the VoiceChat work uses it; the other models in `models/` are
driven by their own probes and none of this applies to them.

Read `evaluating` for *why* the constraints exist. This file is *how* the code
implements them.

## The governing principle

> **The measurement apparatus must not distort the measurement.**

Every decision below had a simpler alternative that produces **confident wrong
numbers** rather than obvious breakage. That is the dangerous failure mode, and
the reason each is called out in a comment at its site.

## `DuplexSession` — two independent tasks

```python
receiver = asyncio.create_task(self._receive_loop())
sender   = asyncio.create_task(self._send_loop(source))
```

They run for the whole session and **never await each other**. That
independence *is* full duplex; a request/response loop would serialize the
property under test.

### `run()` vs `run_stream()`

`run_stream(source)` takes an async iterator of PCM chunks and is the real
entry point. `run(array)` is a thin wrapper that paces an array into one.

The asymmetry is deliberate: **a file needs pacing imposed, a microphone has it
intrinsically.** Both present the same interface, so tool dispatch, policy and
logging are shared rather than forked.

## Tool dispatch — `create_task`, never `await`

```python
task = asyncio.create_task(self._run_tool(tool, call))
self._tool_tasks.add(task)
task.add_done_callback(self._tool_tasks.discard)
```

Awaiting inline blocks `_receive_loop`, so the client stops draining
`response.output_audio.delta` for the tool's whole duration. **The model keeps
talking — you just don't hear it**, and it arrives as a burst when the socket
drains.

You would measure a multi-second silence your own client caused, and conclude
the model fails under slow tools.

The strong reference to `_tool_tasks` matters: asyncio only holds weak
references to tasks, so a fire-and-forget task can be garbage collected
mid-flight.

## Pacing — absolute monotonic deadlines

```python
deadline = start + (i + 1) * period
drift = deadline - time.monotonic()
if drift > 0:
    await asyncio.sleep(drift)
```

A per-iteration `sleep(0.08)` accumulates each send's cost (~1–3 ms here),
drifting the stream a second or more behind wall clock over a 60 s clip. You
would feed a real-time model *slower than real time* and call it late.

Verified: 1.0 s of audio → 13 chunks in 1.041 s. The 41 ms is the final partial
chunk zero-padded to 80 ms (12.5 rounds up), not drift.

**No pacing on the microphone.** `readexactly(3840)` already blocks for exactly
as long as 80 ms of audio takes to exist. Adding `paced_chunks` would
double-count.

## The linger window

The naive client closes when input audio ends. The agent is almost always
mid-utterance and in-flight tools still have to land — closing there makes the
model look like it failed to answer when it simply wasn't finished.

## Channel selection

```python
load_pcm16(path, channel=0)   # caller
```

`channel=None` downmixes, which for a two-party recording feeds the model
**both halves of its own conversation**. Channel 0 is the caller by convention;
verify with per-second RMS (whoever speaks first).

Resampling is a band-limited polyphase filter (Kaiser-windowed sinc; it
replaced linear interpolation on 2026-09-23 after the old one measurably
changed transcripts). Still pre-convert with `ffmpeg -ar 24000 -ac 1` when
fidelity itself is the variable, and never resample audio headed for an ASR
model: pass its native rate.

## The tool registry

```python
@registry.tool(
    description="...",
    parameters=obj(account_id=string("...")),
    on_hold="One moment.",
    latency=2.0,
)
def get_balance(account_id: str) -> dict: ...
```

- **`latency`** injects artificial delay — the evaluation's independent
  variable. Real tools span 50 ms to 10 s.
- **`obj()` marks every property required**, because testing "ask, never guess a
  missing argument" needs arguments actually marked required.
- **ASCII clamping** at `Tool.invoke`, since the server demands it and fails
  deep otherwise.
- **Malformed arguments do not raise** — `ToolCall.parse` captures them as
  `{"__unparsed__": raw}`. A model emitting bad JSON under duplex load is a
  *finding*, not a reason to kill the receive loop.
- Handlers may be sync or async; both are awaited correctly.
- Unknown tool names are **answered anyway** with an error payload. Leaving a
  `call_id` unanswered can wedge the turn, and "model invented a tool" is worth
  capturing rather than crashing on.

## Barge-in

On `speech_started` with pending calls, `_on_barge_in` asks `policy.decide()`
per call and applies the verdict. `decide()` must **not block** — it returns a
decision, it does not do work.

Accumulated caller transcript deltas feed `user_transcript_so_far`. These were
originally discarded, which made the policy's backchannel branch unreachable
dead code — 64 of 75 "unhandled" events in the first real run were exactly
those deltas.

`policy.py` is the one file that encodes a **product decision** rather than an
engineering constraint. Change it freely; the thresholds are guesses and the
structure is the argument.

## The live client

### Why subprocesses, not `sounddevice`

pip-installed `sounddevice`/`pyaudio` wheels expect a system `libportaudio` that
a pure-pip venv on NixOS has no reliable way to locate. `pw-record`/`pw-play`
are already on PATH and speak raw PCM on stdin/stdout.

### `--raw`, not `--container raw`

`pw-play` uses sndfile to read and will try to parse stdin as a container:

```
sndfile: failed to open audio file "-": Format not recognised.
```

`pw-record` accepts **both** spellings, so capture works while playback silently
fails — which reads as "the model isn't sending audio."

### stderr is inherited, never DEVNULL

That error above was originally discarded. Diagnosis cost far more than the bug.
Both processes are also checked for immediate exit after launch.

### Save before cleanup

```python
finally:
    session.save_agent_audio(...)   # FIRST
    session.log.save(...)
    for proc in (rec, play):        # then best-effort cleanup
        with contextlib.suppress(...):
            proc.terminate()
```

A group-wide SIGTERM reaps `pw-record` first, so `terminate()` raises
`ProcessLookupError` — which, when it ran first, escaped the `finally` and
destroyed the run's only artifact. **Ctrl-C does not reproduce this**; only
SIGTERM does.

Both SIGINT and SIGTERM are wired through `loop.add_signal_handler`.

### Headphones

Without them the mic captures the agent's voice, which a full-duplex model hears
as a barge-in and talks over, escalating until killed. Not subtle.

## Extension points

| Want | Do |
|---|---|
| New event handled | add a branch in `_dispatch`; it falls through to `kind:"unhandled"` with the type |
| Live playback / streaming sink | `on_audio` callback — must not block, it runs in the receive loop |
| New audio source | any `AsyncIterator[bytes]` of 3840-byte chunks → `run_stream` |
| New tool behaviour | `tools.py`; keep `Tool.invoke` the only place results are serialized |
| Different barge-in semantics | `policy.py` only — `client.py` applies verdicts, it does not make them |
| New backend | **not here** — see `adding-a-model` |

## What is NOT abstracted

`protocol.py` is imported directly as `proto` throughout `client.py`. There is
no adapter interface. That is correct for one backend and is the first thing to
change when a second arrives.
