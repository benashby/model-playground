# The harness and its fixtures

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


The harness lives at the repository root
(aliases `playground`, `voicechat`).

## The harness

### Layout

| File | Role |
|---|---|
| `protocol.py` | Realtime event constructors, `ToolCall` parsing, wire constants |
| `audio.py` | Load, resample, wall-clock-paced chunking, channel split |
| `tools.py` | `@registry.tool` decorator, JSON-Schema export, latency injection |
| `policy.py` | Barge-in disposition (the decision under study) |
| `client.py` | `DuplexSession`: concurrent sender + receiver |
| `live.py` | Microphone client via `pw-record`/`pw-play` |
| `scenarios/*.py` | Agent specs: instructions + tools, each exporting `SPEC` |

Everything NIM-specific is confined to `protocol.py`. The rest is
protocol-agnostic, which keeps a future move to a self-hosted Apache-2.0
serving layer cheap.

### The governing principle

> **The measurement apparatus must not distort the measurement.**

Every significant design decision below follows from that. Each one had a naive
alternative that produces confident wrong numbers instead of obvious breakage,
and a harness that fabricates a finding costs more than one that crashes.

### Decision 1: two independent tasks

A sender and a receiver run for the whole session and never await each other.

```python
receiver = asyncio.create_task(self._receive_loop())
sender   = asyncio.create_task(self._send_loop(source))
```

That independence is what full duplex means here. A request/response loop
would serialize the property under test and measure nothing.

### Decision 2: tools dispatch via `create_task`, never `await`

```python
task = asyncio.create_task(self._run_tool(tool, call))
```

Awaiting the tool inline would block `_receive_loop`, so the client would stop
draining `response.output_audio.delta` for the tool's entire duration. The
model keeps talking but you do not hear it, and the audio arrives as a burst
when the socket drains.

You would measure a 5-second "silence" that was purely a client artifact and
conclude the model fails under slow tools, when the failure was in your client.

### Decision 3: absolute monotonic deadlines instead of `sleep(0.08)`

```python
deadline = start + (i + 1) * period
drift = deadline - time.monotonic()
if drift > 0:
    await asyncio.sleep(drift)
```

A per-iteration `sleep(0.08)` accumulates the cost of each send, roughly 1 to
3 ms here. Over a 60-second clip that drifts the stream a second or more behind
real time. You would then be feeding a real-time model slower than real time,
and it would appear to respond late when your client had delivered late.

Verified: 1.0 s of audio became 13 chunks in 1.041 s. The 41 ms is the final
partial chunk zero-padded to 80 ms (12.5 chunks rounds up), not drift.

### Decision 4: no pacing on the microphone

A capture device already runs at wall clock. `readexactly(3840)` blocks for
exactly as long as 80 ms of audio takes to exist. Layering `paced_chunks` on top
would add a second 80 ms per chunk and double-count latency.

This asymmetry is why `run_stream()` takes an async iterator. A file needs
pacing imposed on it and a device has pacing built in, but both present the
same interface.

### Decision 5: a linger window

The naive client closes the socket when the input WAV ends. The agent is
almost always mid-utterance at that point, and an in-flight tool still has to
land. Closing there makes the model look like it failed to answer when it
simply had not finished.

### Decision 6: explicit channel selection

```python
load_pcm16(path, channel=0)   # caller
```

For a stereo two-party recording, downmixing feeds the model both halves of
its own conversation, which is never what you want. Channel 0 is the caller by
convention, verified by per-second RMS (ch0 speaks first, ch1 responds).

### The tool registry

```python
@registry.tool(
    description="Get the current outstanding balance for an account.",
    parameters=obj(account_id=string("Account ID from lookup_account")),
    on_hold="One moment while I check that balance.",
    latency=2.0,          # the independent variable
)
def get_balance(account_id: str) -> dict: ...
```

- `latency` injects an artificial delay. It is the evaluation's independent
  variable: real tools span 50 ms (cache hit) to 10 s (cold backend query), and
  the question is where conversational behaviour degrades.
- All properties are required by default. The model card claims "if a required
  argument is missing, ask the user; never guess." Testing that claim needs
  arguments that are actually marked required.
- Arguments are clamped to ASCII at the boundary, because the server demands it
  and fails deep inside instead of at the callsite.
- Malformed arguments do not raise. A model emitting bad JSON under duplex load
  is a finding, and no reason to kill the receive loop.

### The barge-in policy

`policy.py` is the one piece that is a design judgement rather than an
engineering constraint. There are three defensible options:

- CANCEL abandons the in-flight tool. It has the lowest latency, but you burned
  a query, and if the interruption was "yes, that one" you discarded the answer
  being confirmed.
- COMPLETE_AND_DELIVER lets it finish and delivers late. It never wastes work,
  but it may splice a stale result into a conversation that has moved on.
- COMPLETE_AND_HOLD finishes, caches, and delivers only if the result is still
  relevant. It is closest to what a human agent does and has the most state to
  manage.

Implemented, in priority order:

1. Side-effecting tools are never cancelled. You cannot un-fire a transfer.
   Cancelling discards the receipt, leaving the caller transferred while the
   agent believes it never happened, which is worse than a late result.
2. Backchannel is continuation. "yeah", "mm hm", "right" over the agent is
   agreement. Treating it as a topic change is the most common way duplex
   clients feel broken.
3. Past 70% of expected latency, hold. Pay the remaining milliseconds rather
   than barge a stale result into a turn that has moved on.
4. Otherwise cancel: the tool is idempotent, barely started, and the caller
   wants something else.

The thresholds are guesses. The ordering is the part with an argument behind it.

### Output

Every run writes:
- `logs/<name>-agent.wav`: the agent side, captured
- `logs/<name>.jsonl`: timestamped events, and the artifact that matters

The JSONL supports latency analysis: `tool.call` → `tool.result`
durations, `barge_in.decision` with `elapsed` and `heard`, and turn boundaries.

## Fixtures

A full-duplex harness needs recordings with the two speakers on separate
channels. A mono mixdown is useless, because you cannot feed one side to the
model without also feeding it the other side's speech.

### NVIDIA's shipped demo audio is the best material available

The HF weights repo contains three recordings the model card never mentions.
They are already at the 24 kHz wire rate and in stereo, one speaker per channel:

| File | Length | Overlapping speech | Purpose |
|---|---|---|---|
| `tool_call.wav` | 84.5 s | 1 s | tool calling end to end |
| `interruptions.wav` | 30.0 s | **4 s** | barge-in |
| `turn_taking.wav` | 41.1 s | 0 s | clean turn-taking baseline |

Channel roles were verified by per-second RMS energy. ch0 speaks first and ch1
responds, so ch0 is the caller and ch1 is NVIDIA's own reference agent output.

That second channel is free ground truth: whatever the harness captures in
`logs/agent.wav` should roughly match the shape of ch1.

```bash
scp $PLAYGROUND_HOST:/opt/models/NVIDIA-NemotronLabs-VoiceChat-11B/{tool_call,interruptions,turn_taking}.wav audio/
```

Finding these made the HF download worth it.

### HCRC Map Task Corpus, the bulk fixture

From the University of Edinburgh: 128 unscripted task-oriented dialogues, of
which we fetched 11 (419 MiB).

- 20 000 Hz, 2-channel, 16-bit PCM, one speaker per channel
- Free direct HTTP download, no signup
- Source: https://groups.inf.ed.ac.uk/maptask/signals/dialogues/

The best barge-in fixtures by overlapping speech (both channels above RMS 0.01
in the same second):

| File | Overlap | Total |
|---|---|---|
| `q4nc3.mix.wav` | **239 s** | 881 s |
| `q1nc1.mix.wav` | **233 s** | 1113 s |
| `q3nc2.mix.wav` | **194 s** | 538 s |

These carry far more real overlap than NVIDIA's fixtures. Map-task dialogue is
dense with backchannel and interruption, which is the stress case.

> Filename trap: `.mix.wav` reads as "mono mixdown", but every file was
> verified as 2-channel. No separate `.g.wav`/`.f.wav` variant is published;
> `.mix.wav` is the correct stereo file.

At 20 kHz they need resampling to 24 kHz. The harness will do it with linear
interpolation, but pre-convert for anything where audio quality is the
variable:

```bash
ffmpeg -i q4nc3.mix.wav -ar 24000 out.wav   # real polyphase filter
```

### Full-Duplex-Bench, downloaded and discarded

`Ssshangfu/Full-Duplex-Bench-Data` on Hugging Face. NVIDIA cites this benchmark
for turn-taking and barge-in evaluation, so it looked like the obvious fixture
set.

All 2 784 files were deleted. Every WAV sampled is a mono single-stream test
stimulus, a prompt to play at a model, rather than a two-channel human
conversation. That suits the benchmark's own scoring method and is useless for
a harness that feeds one side of a conversation and captures the other.

It is recorded here as a negative result: the obvious dataset was the wrong
shape, and only per-file channel inspection showed it.

### Montclair Map Task Corpus, not obtained

https://digitalcommons.montclair.edu/mmt_corpus/ has 48 dialogues, 2-channel,
with Praat TextGrid alignments. The download endpoint sits behind a Cloudflare
JS challenge, which was not bypassed.

### CANDOR, skipped

1 656 conversations, 850+ hours, per-participant tracks. It requires a data
request form. It is referenced in turn-taking research and worth pursuing if
this work continues.

### The verification that mattered

For every downloaded file, check sample rate, channel count, duration and
subtype, then delete anything mono. Then compute per-second RMS per channel and
count the seconds where both exceed 0.01, to rank barge-in value.

Without the channel check, 2 784 unusable files would have looked like a
successful corpus acquisition.

---

Previous: [The protocol](05-protocol.md) | [Contents](../README.md#contents) | Next: [Results: barge-in under slow tools](07-results-barge-in.md)
