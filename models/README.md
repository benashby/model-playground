# Model notes

One directory per model I have taken apart: the write-up, the probes that
produced its numbers, and their raw output.

The models have nothing in common except that I wanted to understand them. One
of them is driven by the speech harness in `src/playground/`; the others are
driven by their own probes and never touch it. They share the standard below
and no machinery.

These are research notes, and nothing here is a script to run or documentation
for a product. They record what was asked, what was measured, what it turned
out to mean, and which plausible-looking approaches were wrong. A finding is
only recorded together with the method that produced it.

## The standard every note is held to

The governing rule of the repository is that *the measurement apparatus must
not distort the measurement*. Its documentation corollary is that a reader must
always be able to tell what was measured from what was merely read.

So every note separates:

- **[CLAIM]**: what the vendor says. It is recorded because it is the
  hypothesis, and because the gap between claim and result is often the
  finding.
- **[MEASURED]**: what happened here, on stated hardware, with stated versions.
  It is never a number copied from a model card.
- **Open**: what is still unknown, named explicitly.

A number needs the hardware it came from. "43.7× real time" says little until
you know it came from a CPU with no AVX-512, against a vendor figure produced
on one that had it, so every results section states its exact silicon.

## Required sections

Every note covers these, because each one has at some point been the thing that
mattered. A note may leave a section out only if it says why the section does
not apply.

| Section | Contents |
|---|---|
| What it is / how it works | The mechanisms, explained well enough to predict the model's failure modes. |
| Licensing | Every artifact in the runtime path separately: weights, inference code, serving container, compiled engines. Link the actual licence text, not a summary of the marketing. |
| Architecture | As measured from the weights (safetensors headers), not as described. |
| Host & launch | What it ran on, and the configuration that mattered, including settings deliberately not inherited from a neighbouring model. |
| Protocol / interface | Where the server or template deviates from its own documentation. |
| Fixtures | What input was used, and whether it could exercise the intended behaviour at all. |
| Results | With the exact silicon, with caveats, and with an explicit list of what was not exercised. |
| Probes | The committed code that produced every number in the note, under `probes/`. A number with no probe is a rumour, including one you are confident about. |
| Results data | Raw probe output under `results/`, committed, so that a later run can correct an earlier claim instead of merely disagreeing with it. |
| Repeating it | The order of questions that made the findings appear, and what each step protects against. |
| Open questions | Each one stated explicitly. |

Probe output is public. `results/` is committed, so anything a probe prints
ships with the repository. Take the endpoint from `PLAYGROUND_HOST` and never
echo it, and describe hosts by their nature, never by hostname, address, cloud
project or account. The first probe written under this rule leaked the node's
IP into its own log on the first run, so redaction has to be built into the
probe instead of left to memory.

Notes are written for someone reading them in the GitHub web UI. Before a note
counts as finished, the humanizer strips AI writing habits from its prose, and
`check_rewrite.py` then fails the rewrite if it changed any code, link, tag or
number. A note much longer than ~400 lines is split into articles under
`notes/`, with the model's `README.md` as the hub; `nemotron-voicechat-11b/` is
the example. The procedure is in
`.claude/skills/writing-notes/SKILL.md`.

See `.claude/skills/adding-a-model/SKILL.md` for the onboarding process itself.

## Why the failures are written up at length

Several notes spend more words on a failure than on the working configuration.
I do that on purpose: it is the part of the repository most likely to be useful
elsewhere.

The failures that get this much space have one thing in common: the system
returned success while doing nothing. A chat template rendered a structurally
valid, empty prompt. A dependency pin silently did not apply. A container held
73 GB where the tool you use to stop it could not see it. None of these raised
an error, and all of them produced clean-looking numbers that measured nothing.

An evaluation instrument that fabricates a finding costs more than one that
crashes, so the notes catalogue the specific ways this kind of tool lies.

## The notes

| Note | Model | What it is |
|---|---|---|
| [`nemotron-voicechat-11b/`](nemotron-voicechat-11b/README.md) | NemotronLabs VoiceChat 11B | Full-duplex speech-to-speech. The one model the harness drives. |
| [`parakeet-redux/`](parakeet-redux/README.md) | Moondream Parakeet Redux (ternary ASR) | Transcription on a CPU, and a study of streaming-transcript stability. |
| [`k2-horizon-32b/`](k2-horizon-32b/README.md) | IFM K2-Horizon-32B | Text-only reasoning LLM. Documented for the deployment findings. |

## What each model is actually for

The short version, so you can skip to the right note.

| If you need… | Model | What to expect |
|---|---|---|
| Transcription, cheaply, at volume, with no GPU | Parakeet Redux | ~44× real time on a desktop CPU (an hour of audio in ~80 s), 178 MB, 25 languages, word timestamps free. Weak in noise. Runtime is proprietary. |
| Best-accuracy transcription, GPU available | Parakeet Ultra | Same architecture at full precision; ~5.3 % mean WER, near the top of the Open ASR Leaderboard. Not yet tested here. |
| Live transcription for an agent | Parakeet Redux, streaming | First preview about 4 s after audio starts, then updates every ~2 s. Settled words never change, but punctuation can be revised long after it first appears. Final text matches batch. |
| A conversation, end to end, in speech | VoiceChat 11B | The only full-duplex model here. Real barge-in. CUDA only; 73 GB at stock settings, and it also served in 44.7 GB with two memory settings lowered. |
| Long-context reasoning or agentic text work | K2-Horizon-32B | 65.6 tok/s decode, 131 k context served, tool calling works in three formats, Apache-2.0 with published training data. Stage-1 checkpoint, so weaker at agentic and coding work than mature peers. |
| Speaker labels / diarisation | none of these | No model here does it. |

Two things are easy to miss until you hit them. Parakeet's runtime is
licence-encumbered even though its weights are CC-BY-4.0
([licensing](parakeet-redux/notes/02-licensing.md)). And K2 returns its
reasoning trace in `reasoning`, which an OpenAI-style client that only reads
`reasoning_content` will show as empty
([the reasoning trace](k2-horizon-32b/notes/05-reasoning-trace.md)).

## How an investigation is structured

This is an order of questions that has kept turning out to matter, with the
reasoning for each. It is not meant to be followed mechanically.

1. Classify before coding: modality shape, serving stack, streaming or
   request/response, compute requirement, licensing. See
   `.claude/skills/adding-a-model/SKILL.md`. Getting these wrong is most of
   the cost of onboarding a model, and every one of them can be answered from
   documents.

2. Establish licensing across every artifact separately: weights, inference
   code, runtime, container, compiled engines. "The model is open" says nothing
   about a deployment, and the encumbered piece is routinely the one nobody
   thinks of as the model.

3. Prove the round trip before measuring anything. Get one honest output
   whose content could only have come from your input. A canary value helps
   here: a distinctive token you can search for in the rendered prompt or the
   response. One saved two separate investigations in this repo.

4. Only then measure, and state the hardware. A performance number is a
   property of a model and a machine together.

5. Name what did not execute. If a subsystem never fired (barge-in on a
   model with no interruption signal, a policy that was never consulted), say
   so in the result. A clean run with an inert subsystem reads as a pass.

## Reproducing a result

Each note's results section states its hardware, its software versions, and the
shape of what was done. That is enough to repeat it on different silicon, which
is usually the interesting question: does this number survive a machine
without AVX-512, or a GPU generation without FP4? In both cases the answer
turned out to be the finding.
