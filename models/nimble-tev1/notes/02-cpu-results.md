# CPU results

> Part of the [Nimble 9B and Tev1](../README.md) investigation. See also [all model notes](../../README.md).

Everything here ran on one CPU host with no GPU: an Intel Xeon E5-2698 v4
(Broadwell, 2016) with AVX2 and no AVX-512, 12 usable CPUs in a container
whose `/proc/cpuinfo` reports the socket's 20 cores, and 16.0 GiB of memory.
The model is `tev1:0.8b` at Q8_0 (GGUF SHA-256 `fa9732e3…`) in every run,
because it is the only one of the three small enough to be quick on this
machine. These results are about the scoring machinery, which is the same for
all three models. Accuracy is not measured on this page.

Three llama-server builds appear:

| Name in the logs | What it is |
|---|---|
| `ollama-runner` | the llama-server Ollama 0.35.0 spawns itself: llama.cpp `b11232` plus Ollama's compat patch, reporting `b1-161755f29` |
| `ollama-0.35.0-bundled` | the same binary, started by hand with `-t 12` |
| `upstream-b11232` | llama.cpp `b11232` built clean by `probes/build_llama_cpp.sh cpu`, reporting `b1-6f767fe` |
| `distro-b9190` | a separately packaged older llama.cpp, `b9190-b64739e` |

## The port reproduces Ollama

[MEASURED] `probes/fidelity.py` sends the two ticket fixtures (3 and 8
questions) to Ollama's `/v1/systemone` and scores the same requests through
`systemone.py` on each build (`results/fidelity.log`).

| Scored through | Largest probability difference from Ollama | Same answer |
|---|---:|---|
| `ollama-runner`, the process Ollama itself uses | 1.30e-08 | 11 of 11 questions |
| `upstream-b11232`, clean upstream | 1.30e-08 | 11 of 11 |
| `distro-b9190` | 3.86e-02 | 11 of 11 |

Token counts agree exactly (727 and 3891 input tokens), as the prompt parity
check predicts. A difference of 1.30e-08 is the size float32 rounding would
produce.

Ollama's compat patch does not affect scoring, so a clean llama.cpp at the tag
Ollama pins is a faithful System One backend with no Ollama involved. The
llama.cpp version does affect it: `b9190` picked
every answer the same but moved probabilities by up to 3.86 points, on the
`human` question of the 8-question fixture (0.256934 against 0.295555), so a
probability threshold tuned on one build has to be checked again on another.

## A repeated prompt kills an older build

[MEASURED] `probes/identical_prompt.py` starts each build, sends one short
prompt twice with `cache_prompt`, and checks whether the process survived
(`results/identical-prompt.log`).

| Build | First request | Second request | Alive after |
|---|---|---|---|
| `distro-b9190` | 200 | no response | no |
| `upstream-b11232` | 200 | 200 | yes |
| `ollama-0.35.0-bundled` | 200 | 200 | yes |

`b9190` aborts with `failed to remove sequence 0 with p0=5, p1=-1` in
`common/common.cpp:1489`: on a fully cached prompt the server must re-evaluate
one token, tries to roll the recurrent state back by one position, and cannot.
A decision service that retries a failed request sends exactly this. `b11232`
handles it. Which change between the two fixed it is Open.

## Priming changes the probabilities

[MEASURED] `probes/priming.py` scores each fixture four ways, three times over,
on `ollama-0.35.0-bundled` started with `--cache-ram 0` (`results/priming.log`).
"Cold" means the slot was emptied first; "plain" means no primer.

| Fixture | Mode | Total, three repeats (ms) | Tokens evaluated per question | Largest change from cold plain |
|---|---|---|---|---:|
| 3 questions | cold plain | 4111, 4693, 3916 | 242, 242, 243 | 0 |
| 3 questions | cold primed | 1753, 1665, 2298 | 7, 11, 12, after a 235-token primer | 1.68e-02 |
| 3 questions | warm plain | 482, 474, 422 | 11, 11, 12 | 1.68e-02 |
| 8 questions | cold plain | 20942, 20834, 20889 | 486 or 487 each | 0 |
| 8 questions | cold primed | 4147, 6898, 3695 | 7 to 12, after a 479-token primer | 1.85e-02 |
| 8 questions | warm plain | 1173, 1244, 1216 | 11 or 12 | 1.85e-02 |

Without the primer every question re-reads its whole prompt, so a cold
8-question request costs about 21 s on this CPU and a primed one about 4 s.
Once the prefix is cached a question costs 141 to 161 ms here, primed or not,
in eleven of the twelve warm runs; the twelfth took 367 ms a question.

The primer also changes the probabilities. Every primed or warm run gives the
same probabilities as every other primed or warm run, and each differs from
cold plain by up to 1.85 points, identically on every repeat. The difference
is deterministic: a prefix evaluated in one pass and a prefix resumed from a
checkpoint give different recurrent states. The size is under two points
here; whether it is larger on a GPU, or for Nimble, is Open.

An earlier run of this probe reported warm numbers as cold. Emptying the slot
is not enough, because llama-server also keeps a host-memory prompt cache
(8192 MiB by default) and restored this fixture from it on every repeat after
the first. The probe now requires `--cache-ram 0` and marks a cold run that
evaluated fewer tokens than its prompt holds as `NOT COLD`.

## Unrelated questions move each other

[MEASURED] `probes/schema_coupling.py` scores the first question of the
8-question fixture, `team`, alone and then alongside the next k questions,
cold and without the primer (`results/schema-coupling.log`).

| Questions in the request | Prompt tokens for `team` | P(billing) | Change from alone |
|---:|---:|---:|---:|
| 1 | 142 | 0.982854 | 0 |
| 2 | 186 | 0.973065 | 1.33e-02 |
| 3 | 242 | 0.985265 | 3.76e-03 |
| 8 | 486 | 0.993033 | 1.02e-02 |

The questions are scored independently, as documented, but the Ollama format
puts every question's schema into every question's prompt, so the prompt for
`team` changes when `churn` is added. The `tev1` format sends each question
alone: its prompt for `team` is 146 tokens in every row and its probabilities
do not move at all. They are different probabilities, though (0.960481 for
`billing` and 0.039262 for `other`, against 0.982854 and 0.004144), and which
format is more accurate is Open.

## History moves a single question

[MEASURED] `probes/public_suite.py` runs Bespoke's own benchmark runner against
a local backend. On 8 BoolQ records it gave 7 of 8 correct through Ollama,
through the port in the Ollama format, and through the port in Tev1's format
(`results/public-suite-smoke.log`). That is a plumbing check, with far too
few records to say anything about accuracy.

It also caught one record that differed between Ollama and the port:
`boolq-02738b66ae7328d2` scored P(true) 0.840204 through Ollama and 0.859640
through the port, while the other seven agreed to six decimals. These are
single-question requests with no primer, so the only difference is what each
server had evaluated just before. With `--isolate` on a `--cache-ram 0`
server, which empties the slot before every record, two runs separated by an
unrelated request agreed exactly, and gave 0.859640 for that record: the
clean-slate value. Ollama's 0.840204 came from resuming a previous record's
checkpoint.

## The benchmark data rebuilds exactly

[MEASURED] `probes/fetch_public_data.py` fetches the upstream data and rebuilds
all 13 of Bespoke's public benchmark subsets with their converter. Every
subset's `source_sha256` and `dataset_sha256` match the manifests Bespoke
committed, 3,880 records in all (`results/public-data.log`). None of the data
is committed here.

## Not exercised

- Any GPU. Every number on this page is from the CPU described at the top.
- Nimble and Tev1 4B. The machinery is shared, but the size of the build,
  primer and coupling effects is measured for `tev1:0.8b` only.
- Accuracy and calibration, apart from the 8-record plumbing check.
- Ollama's own latency. Ollama appeared to run 20 threads on 12 usable CPUs
  here, which would make its timing on this host unrepresentative; no probe
  captured that, so it is recorded as an observation only.
- llama.cpp's open System One pull request, which adds a native tool.

---

Previous: [How it works](01-how-it-works.md) | [Contents](../README.md#contents)
