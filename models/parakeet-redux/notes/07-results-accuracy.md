# Results: accuracy and streaming

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

Hardware and versions for every number here are in
[the throughput results](06-results-throughput.md#hardware-and-software).

## WER: scoring VoiceChat's own transcription [MEASURED]

This comparison is the main reason Parakeet is in the repository: VoiceChat
logs its caller transcript, and until now there was nothing to score it
against.

Parakeet Redux's batch transcript of the caller channel is the reference, and
VoiceChat's `caller.said` events from two September sessions, joined in order,
are the hypothesis. [`probes/wer_voicechat.py`](../probes/wer_voicechat.py)
produced [`results/wer_voicechat.log`](../results/wer_voicechat.log).

Both sides go through the same normaliser,
[`probes/textnorm.py`](../probes/textnorm.py): lowercase; hyphens become
spaces; punctuation is stripped except apostrophes; and runs of English number
words become digits ("fifty" to "50", "twenty-five" to "25", "one hundred and
five" to "105"). Parakeet writes numerals and VoiceChat spells them out, so
converting both to digits stops "1 and 50" against "one and fifty" from
counting as two errors. The module has a self-test covering 11 to 19,
hyphenated numbers, "and" inside a number, and runs of separate small numbers.

Correction: an earlier version of this note said number words were "spelled
out". The code did the opposite, mapping words to digits, and its map was
partial: it had no 11 to 19, and it turned "twenty-five" into "20 5". Neither
fixture contains a number the old map got wrong, so the published figures do
not change; the probe prints the old normaliser's result next to the new one
to show it.

| Fixture | Reference words | Errors | VoiceChat WER vs Parakeet | Fillers removed on both sides |
|---|---|---|---|---|
| `interruptions` | 28 | 0 | **0.0 %** | 0.0 % (28 words) |
| `tool_call` | 69 | 4 (2 substitutions, 2 deletions) | **5.8 %** | 4.5 % (3 errors, 67 words) |

The four `tool_call` disagreements, in order:

| Position | Parakeet | VoiceChat |
|---|---|---|
| 0 | "hello" | "o" |
| 11 | "okay" | (missing) |
| 41 | "what" | "about" |
| 50 | "uh" | (missing) |

Correction: an earlier version of this note said the errors "cluster at the
very start". One of the four is at the start. The others are spread through
the clip: a dropped "okay", "what the weather is like" heard as "about the
weather is like", and a dropped filler. The first one, Parakeet's "Hello!"
logged by VoiceChat as "o", fits a transcript that starts mid-word better than
a recognition failure, but that is a guess and is untested.

Parakeet is a reference here, not ground truth. The vendor reports 6.96 % WER
for Redux on its business-speech set and 6.55 % on the English leaderboard sets
[CLAIM], and where the two systems disagree either can be wrong. Correction: an
earlier version of this paragraph said Redux carries "~6 %" on business speech,
which rounded 6.96 % the wrong way. Read these numbers as "VoiceChat and an
independent ASR agree to within 5.8 % on these two clips", not as "VoiceChat
has 5.8 % WER", and from a small sample of 97 reference words in total.

## Streaming latency and stability [MEASURED]

Each fixture's caller channel was fed at wall-clock rate (`speed=1.0`)
through the harness's pacer into `atranscribe(..., stream=True)`, three
times, with a fresh Photon client each time.
[`probes/pk_stream.py`](../probes/pk_stream.py) wrote
[`results/pk_stream.log`](../results/pk_stream.log) and the raw snapshot logs
`results/asr-<fixture>-run<k>.jsonl`; every derived figure below comes from
[`probes/stream_analysis.py`](../probes/stream_analysis.py), in
[`results/stream_analysis.log`](../results/stream_analysis.log).

| Fixture | Audio | Snapshots | First snapshot after first audio | After client open | NE | UPWR | Final words |
|---|---|---|---|---|---|---|---|
| tool_call | 84.5 s | 42 | 4.00-4.01 s | 5.37-6.80 s | **1.087** | 0.0461 | 69 |
| interruptions | 30.0 s | 15 | 4.00-4.02 s | 5.21-5.30 s | **0.143** | 0.0191 | 28 |
| turn_taking | 41.1 s | 20 | 4.01 s | 5.26-5.30 s | **0.150** | 0.0359 | 20 |

Ranges are over n=3 runs per fixture. Snapshot counts, NE, UPWR and the text
of every snapshot were identical in all three runs.

The first preview arrives 4.00 to 4.02 s after the first audio chunk (median
4.01 s, n=9), which matches the vendor's "Live previews begin after four
seconds of audio" [CLAIM] almost exactly. After that, snapshots arrive every
2.00 s (median of 222 intervals, range 0.65 to 2.19 s), matching "scheduled
every two seconds after that" [CLAIM].

Correction: earlier versions of this note reported the first preview at
5.2-6.6 s and called the time above 4 s "pipeline latency" that "varies with
system load". Both parts were wrong. The earlier figures came from
`playground.asr.transcribe_streaming`, whose clock then started before it
opened the Photon client, so its `first_snapshot_s` included the client's load
time: 1.21 to 1.37 s on a warm start and 2.78 s on the first stream of the
session. Measured from the first audio chunk, the extra pipeline latency is
negligible. The function now starts its clock there too, and reports 4.00 to
4.02 s (`results/asr-fixes-check.log`).
The "earlier run of the same fixture gave 5.06 s" came from an early
exploratory log (`results/asrtest.log`) that had no committed probe; that file
is now deleted.

Streaming costs no accuracy on these clips. In all 9 runs the final streamed
transcript (`aresult()["text"]`) was byte-identical to the batch transcript,
with `timestamps="segment"` and with `"none"`. Correction: the previous
version stated this for all three fixtures, but the committed log had only the
heading of that section with nothing under it. The run that produced it had
ended before the section finished. Re-running that same probe unchanged
completed the section and printed `identical=True` for all three fixtures,
and the new probe repeats the check three times.

### The two stability metrics disagree

Both are published measures, and on `tool_call` they differ by more than 20×:

- Normalized erasure (NE) is 1.087. Roughly one token is taken back per word of
  final transcript, because the tail is being rewritten constantly.
- Unstable word ratio (UPWR) is 0.046. Only 4.6 % of all words ever displayed
  were not already their final value.

The logged snapshots show what the churn consists of. These are the first four
revisions in `tool_call` run 1, with times from the first audio chunk:

```
t=10.07  revised from word 0  (punct)
   was: Hello
   now: Hello. Can you generate a rand
t=12.14  revised from word 5  (frontier)
   was: rand
   now: random number between 1 and 50
t=14.14  revised from word 10  (punct)
   was: 50
   now: 50?
t=26.16  revised from word 25  (frontier)
   was: Par
   now: Paris. Can you convert that
```

Correction: the excerpt in an earlier version of this note had different
times (11.09, 13.13 and 15.14 s) and came from a log that was never
committed. The one above is printed by `stream_analysis.py` from a committed
log. (The logs behind the earlier table were also overwritten during this
revision, when the old probe was re-run to diagnose the empty section; their
NE, UPWR and snapshot counts were reproduced exactly.)

Across all 9 runs, the analysis classifies every rewritten word. There were 45:
27 were punctuation or case changes to the same word, 18 completed a word cut
off at the end of the previous snapshot ("rand" to "random", "Par" to
"Paris."), and none changed a settled word into a different word.

Correction: an earlier version said the unsafe region is "the last word or
two plus any trailing punctuation" and that "Everything earlier was stable in
every snapshot examined". That holds for the words and not for their
punctuation. In every `tool_call` run, the last snapshot, arriving about 1.7 s
after the audio ended, changed the very first word from "Hello." to "Hello!",
69 words back from the end, and "euros" (word 31 of 69) to "euros,". The other
revisions were all within 1 word of the end.

So the figure for an agent acting on words more than two back from the end of
each snapshot is:

| Fixture | Words shown (all snapshots but the last) | Equal to final, exactly | Equal ignoring punctuation and case |
|---|---|---|---|
| tool_call | 4647 | 96.06 % | 100.00 % |
| interruptions | 543 | 100.00 % | 100.00 % |
| turn_taking | 561 | 100.00 % | 100.00 % |
| all 9 runs | 5751 | 96.82 % | 100.00 % |

Correction: the earlier "~95-98 %" had no computation behind it. The measured
value is 96.82 % if punctuation counts and 100 % if it does not. For an agent
that acts on words, the second is the relevant number. For a UI, the late
"Hello." to "Hello!" change means even the start of the transcript can flicker
once, at the end.

Stability depends on the content more than on the model, so a stability number
means little without the audio it came from. NE ranges from 0.143 to 1.087
across three fixtures, a 7.6× spread. `tool_call` is long, conversational and
full of hesitation; `interruptions` is short, clipped questions.

Either metric alone would have given a materially wrong impression of the same
data. That is why the snapshots are logged raw with arrival times and the
metrics computed afterwards: capture is the expensive half and cannot be
repeated.

---

Previous: [Results: hardware and throughput](06-results-throughput.md) | [Contents](../README.md#contents) | Next: [Results: what the input audio does](08-results-input.md)
