# Probes and method

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


## Probes

Every number in this note has code behind it in `probes/`, with its raw output
in `results/`. A finding without its method is a rumour. Until 2026-09-23 several of this
note's figures had no committed probe behind them, which is why the rule exists.

| Probe | Answers | Needs |
|---|---|---|
| `bargein_sweep.py` | Does the barge-in policy fire, at what tool latency, and what does the model do while a tool is outstanding? | the server |
| `analyse_sweep.py` | Summarises a sweep: firing threshold, dispositions, unrequested calls, argument fidelity | the sweep's logs |
| `inference_budget.py` | Per-chunk compute headroom, read from the server's own self-grading | container logs on stdin |
| `turn_taking.py` | `speech_stopped` to agent audible, versus the ~450 ms claim. `--run N TAG` drives its own sessions | the server |
| `vram_floor.sh` | Restarts the container with lower `LLM_GPU_MEM_UTIL` / `TTS_GPU_MEM_UTIL` and reports residency | shell on the node |

Probe output is committed, so probe output is public. No probe prints a
hostname; the endpoint comes from `PLAYGROUND_HOST` and is redacted in anything
written to `results/`. The first probe written under that rule leaked the node's
address into its own log on its first run, so the redaction is built into the
probes instead of left to memory.

## Repeating this investigation

The steps below are in order because each one protects against a specific way
the next can produce a confident wrong answer.

1. **Check what the weights actually are before believing any size claim.**
   Read the safetensors headers (dtype and parameter count) instead of the
   model card. Then check what the *serving* config declares, because here the
   two disagree (F32 on disk, bf16 resident) and the difference is a factor of
   two in VRAM.

2. **Find the memory knobs before sizing hardware.** `LLM_GPU_MEM_UTIL` and
   `TTS_GPU_MEM_UTIL` set almost the whole footprint, independently of the
   model. If you size from observed residency without finding them, the
   hardware requirement you get is really a default.

3. **Prove the prompt arrives.** Check that the input reached the model
   intact; a response coming back does not show that. Both other models in this
   repository returned fluent, plausible output from prompts that were empty or
   malformed.

4. **Run the fixture and assert that the code path under test executed.** A
   successful run does not show that. `pending=[]` on every interruption means
   the policy never fired, and a run where the subject never executes looks the
   same as one where it executed perfectly ([failures](09-failures.md)).

5. **Make the independent variable large enough to matter.** Injected tool
   latency must exceed the gap between a tool call and the caller's next
   utterance, or the two events cannot overlap and the experiment is inert by
   construction. Measure the gap from a previous log first.

6. **Check the instrument can observe the thing you are claiming.** Before
   quoting a turn-taking number, confirm that something timestamps the agent
   becoming audible. Until 2026-09-23 nothing did ([failures](09-failures.md)).

7. **Repeat, then report the distribution.** Single runs have been wrong by
   2.4× in this repository's own notes. Discard warmup runs, since the first
   turn after a cold start carries a JIT spike.

---

Previous: [Traps, collected](10-traps.md) | [Contents](../README.md#contents) | Next: [Open questions](12-open-questions.md)
