# Traps, collected

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).

Every bug and near-miss from this session. Several looked like "the model is
broken" and were not.

## Audio and playback

### `pw-play` rejects `--container raw` on stdin

Transcripts printed and agent audio arrived and saved correctly to WAV, but
nothing was audible. The player's error was:
```
sndfile: failed to open audio file "-": Format not recognised.
error: open failed: Input/output error
```
`pw-play` uses sndfile to *read*, and sndfile tries to parse stdin as a
container. The flag for headerless input is `--raw`, not `--container raw`.

An asymmetry hid it. `pw-record --container raw -` works fine, so capture
succeeded and playback failed with the same flag. That reads as "the model
isn't sending audio" when the real problem is "my player flag is wrong."

### `stderr=DEVNULL` turned a one-line error into a mystery

Both subprocesses were launched with stderr discarded, so the sndfile error
above was printed and thrown away. Diagnosing it cost far more than the bug.

The fix inherits stderr and checks both processes for immediate exit after
launch.

> This is the classic `2>/dev/null` trap: hiding a throw converts an error into
> apparent success.

### Echo feedback without headphones

The mic captures the agent's own voice and feeds it back as caller audio. The
model is full-duplex, so it hears itself mid-utterance, treats that as a
barge-in, and talks over itself in an escalating loop. It is unusable
immediately, not gradually. Headphones are mandatory.

## Process lifecycle

### Cleanup ordering destroyed the run's only artifact

```python
finally:
    rec.terminate()        # <- raises ProcessLookupError under group SIGTERM
    ...
    session.log.save(path) # <- never reached
```

A group-wide SIGTERM (`timeout`, a closed terminal, a supervisor) reaps
`pw-record` first. `terminate()` then raises `ProcessLookupError`, which escapes
the `finally` before the save lines run.

The fix is to save first and clean up after. Cleanup is best-effort; the event
log is not.

It nearly slipped through because Ctrl-C (SIGINT) does not reproduce it: the
subprocess survives long enough. Only SIGTERM exposes it.

### SIGINT handling alone is insufficient

`except KeyboardInterrupt` catches Ctrl-C but not SIGTERM. Both are now wired
through `loop.add_signal_handler`.

## SSH and shell

### `ssh -n` silently eats heredoc stdin

```bash
ssh -n host 'bash -s' <<'EOF'    # runs an EMPTY script
```
`-n` redirects stdin from `/dev/null`, so `bash -s` receives nothing. Exit code
0, with no output, looks exactly like success.

### Piping a secret *and* a heredoc into the same ssh

```bash
gopass show -o key | ssh host 'bash -s' <<'EOF'
read -r NGC_KEY     # gets the SCRIPT, not the key
```
There is only one stdin. Transfer the secret to a 0600 file first, then run the
script that reads it.

### `pgrep -f` matches its own command line

```bash
pgrep -f "hf download" && echo "downloading"
```
This kept reporting activity after the download finished, because the `bash -c`
wrapper containing that string is itself a process.

## Toolchain

### A dangling symlink reports "No such file or directory" about its *target*

```
~/.local/bin/hf -> ~/.local/share/uv/tools/huggingface-hub/bin/hf   [gone]
```
`ls` shows `hf`. `command -v hf` finds nothing. `~/.local/bin/hf --version`
says *"No such file or directory"*, which reads as "the file you typed does not
exist" when the file exists and its target does not.

`uv tool install` then refuses with `Executable already exists: hf` and needs
`--force`.

### `huggingface_hub` extras changed in v1.x

`huggingface_hub[cli,hf_transfer]` fails in 1.32.0 because both extras are
rejected. `hf_transfer` is now a separate package (`--with hf_transfer`), and
`HF_HUB_ENABLE_HF_TRANSFER` is deprecated in favour of `HF_XET_HIGH_PERFORMANCE`.

### The NVIDIA container toolkit metapackage is held/broken

`nvidia-container-toolkit` fails to install; `nvidia-container-toolkit-base` is
present and sufficient. I confirmed that with a GPU smoke test inside a
container instead of assuming it. The root cause is unresolved.

## Measurement

### Per-iteration `sleep()` drifts

`await asyncio.sleep(0.08)` at the bottom of a send loop adds the send cost on
every iteration, and over a long clip the stream drifts seconds behind wall
clock. Use absolute deadlines against a monotonic clock.

Otherwise you feed a real-time model slower than real time and conclude it
responds late.

### Awaiting a tool inside the receive loop fabricates silence

Blocking the receive loop stops draining agent audio for the tool's duration.
The model keeps talking and you do not hear it, so you measure a multi-second
silence that your own client caused.

### JIT compilation spikes the first turns

```
Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
```
Warmup does not cover every shape. Discard the first exchange of any timing run.

### `.mix.` filenames are not mono

11 HCRC Map Task files named `.mix.wav` are really 2-channel. Trusting the name
would have thrown away the best barge-in corpus available.

### The obvious dataset was the wrong shape

Full-Duplex-Bench, the benchmark NVIDIA cites, is entirely mono single-stream
stimulus. The 2 784 downloaded files were deleted; only inspecting the
channels of each file revealed it.

## Reasoning

### Enumerate every artifact in the runtime path separately

Saying the model is permissively licensed tells you nothing about your
deployment. Weights, inference code and serving container each carried
different terms here, and the encumbered one was the piece nobody thinks of as
"the model."

I asserted the HF checkpoint was a licensing escape hatch before checking
whether an open-source serving path existed. One does (Apache-2.0 `speechlm2`),
but I reached the right conclusion by accident, without the reasoning to back
it.

### "Pull down" is ambiguous

"pull down qwen" can mean *download* or *shut down*. The two readings lead to
opposite actions, and one of them stops a running inference server. Ask rather
than guess.

---

Previous: [Failures worth recording](09-failures.md) | [Contents](../README.md#contents) | Next: [Probes and method](11-method.md)
