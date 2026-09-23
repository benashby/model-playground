# model-playground

A personal project where I pick AI models apart to understand them.

The loop: find a model worth a look, get it actually running on hardware I can
name, learn what it really is, stress it until it shows its edges, then write it
up. Each write-up is a practical guide to *using* the model plus a deeper
account of *what it is and why it behaves that way*.

This is not a product or a library, and it is not meant to be cloned and run.
There is no support, stability promise, roadmap or install path. It's public
because several of these things were annoying to find out and the write-ups
might save someone an afternoon. Nothing here is connected to any job or
company; it's curiosity.

The write-ups are what matter. The code is just how I got there.

## The models so far

| Model | What it is | Why it was interesting |
|---|---|---|
| [**NemotronLabs VoiceChat 11B**](models/nemotron-voicechat-11b/) | Full-duplex speech-to-speech, 11B, NIM-served | The original subject. Tool calls under conversational pressure: what a model *does* while a tool is slow and you talk over it. |
| [**Parakeet Redux**](models/parakeet-redux/) | Ternary-quantised ASR, CPU-class | A 2-bit-ish model that is genuinely usable, and a study of how streaming transcripts *revise themselves*. |
| [**K2-Horizon-32B**](models/k2-horizon-32b/) | 32B dense reasoning LLM, vLLM | Deployment forensics: three separate ways a serving stack returns HTTP 200 while doing nothing you asked. |

Each directory holds a short overview (`README.md`) that links to the
write-up's articles in `notes/`, the probe code that produced every number
(`probes/`), and the raw output (`results/`). The standard they're all held
to is in [`models/README.md`](models/README.md).

Some of what's in them, so you know whether to open one:

- The VoiceChat realtime socket is 24 kHz in both directions. The model card
  says 16 kHz, but that figure is for the offline path.
- Its advertised per-tool `on_hold_message` does nothing special; the server
  serialises your whole tool dict into the system prompt as text.
- The checkpoint carries a `function_head` parallel to `lm_head`, so tool calls
  don't compete with speech for the output channel. They fire *before* the
  caller's transcript finalises, by 1.12 s and 2.73 s in the two observations
  there are. The single "1.1 s" this README used to quote was the first of them.
- It ships as F32 and runs as bf16, and neither is why it wants 80 GB: two
  environment variables preallocate 85% of the card regardless of model size.
- Parakeet Redux runs an hour of audio in about 80 s on a desktop CPU, with
  permissive weights and a licence-encumbered runtime.
- K2's reasoning trace looked like it was generated, billed and discarded. It
  is returned in full, under `reasoning`; the probe had read `reasoning_content`.

Each of those is stated properly, with hardware and with the probe that
produced it, in the note it belongs to.

## What's in here

| Path | |
|---|---|
| `models/<slug>/README.md` | Overview of the write-up: what the model is, what was found, and the contents |
| `models/<slug>/notes/` | The write-up itself, split into linked articles. This is the deliverable. |
| `models/<slug>/probes/` | The code behind every number in it, re-runnable |
| `models/<slug>/results/` | Raw probe output, committed as evidence |
| `src/playground/` | The speech harness, used by the VoiceChat work only |
| `scenarios/` | Agent specs for the harness: instructions + tools |
| `logs/` | Scratch run logs; anything worth keeping moves to `results/` |

The harness in `src/playground/` predates the rest and is specific to
full-duplex speech: `protocol.py` (realtime event constructors and wire
constants), `audio.py` (load / resample / wall-clock-paced chunking),
`tools.py` (tool registry with injectable latency), `client.py`
(`DuplexSession`, concurrent send + receive), `policy.py` (what to do with an
in-flight tool on barge-in), `live.py` (microphone client via
`pw-record`/`pw-play`). The other two models needed none of it and are driven
by their own probes.

Most of its non-obvious decisions exist because the naive alternative produced
confident wrong numbers instead of visible breakage. Two examples are pacing
with absolute deadlines instead of `sleep(0.08)`, and never awaiting a tool
inside the receive loop. Those are written up in
[`models/nemotron-voicechat-11b/`](models/nemotron-voicechat-11b/) and in
`.claude/skills/harness-internals/`.

The design choice I find most interesting is making tool *latency* the
independent variable. Real tools span 50 ms to 10 s (a cache hit, a network
round trip, a cold backend), and the question is where conversational
behaviour falls apart. The scenarios are a plausible-sounding support desk for
the same reason a tutorial is a to-do app: it gives the tools an obvious reason
to exist. Nothing being measured is domain-specific.

`policy.py` is the only file that's a judgement call. The rest is plumbing. When
the caller interrupts a running tool, do you cancel it, deliver it late, or
hold it? My answer: never cancel something side-effecting (you can't un-fire a
transfer, and cancelling only discards the receipt), treat backchannel
("yeah", "mm hm") as continuation, not redirection, hold anything past
70% done, and cancel the rest.

## Running it, for future me

There is no install guide. These are the commands I use, and they need a
VoiceChat NIM container on the other end.

```bash
uv sync
export PLAYGROUND_HOST=<server>          # or pass --host
.venv/bin/python -m playground scenarios/nvidia_demo.py --audio audio/tool_call.wav
.venv/bin/python -m playground.live scenarios/nvidia_demo.py     # headphones!
```

Runs write a timestamped JSONL event log plus the captured agent audio into
`logs/`. Wear headphones: the mic otherwise feeds the agent its own voice,
which a full-duplex model hears as barge-in.

## Test audio

NVIDIA ships three 24 kHz stereo recordings inside the weights repo, with the
caller on ch0 and their reference agent on ch1, and nobody mentions them:
`tool_call.wav` (84.5 s), `interruptions.wav` (30 s, 4 s of overlapping
speech), `turn_taking.wav` (41 s, none).

For more, the [HCRC Map Task Corpus](https://groups.inf.ed.ac.uk/maptask/signals/dialogues/)
is free, direct-download, and genuinely 2-channel despite `.mix.` in the
filenames. Full-Duplex-Bench looks ideal and isn't. It's all mono
single-stream stimulus, which is useless if you need two sides.

## License

MIT. Take anything useful. No weights or vendor code are vendored here; each
model carries its own terms, recorded in its note.
