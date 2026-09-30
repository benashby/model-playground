# GPU results

> Part of the [Nimble 9B and Tev1](../README.md) investigation. See also [all model notes](../../README.md).

This page covers all three models on three GPUs and three CPUs. It measures
whether the port still reproduces Ollama, how far the other backends drift from
it, how accurate and how well calibrated the models are on Bespoke's 3,880
human-labelled decisions, whether each model's own prompt format helps, and
what a decision costs in time. It ran on 2026-09-30, the same day as
[the CPU results](02-cpu-results.md).

## Hardware and builds

| Name in the logs | Machine | GPU | Build |
|---|---|---|---|
| `port-cuda-h100` | a rented cloud VM: Xeon Platinum 8481C (AVX-512), 460 GiB, Ubuntu 24.04 | one H100 80GB HBM3 of two, driver 595.91.07 | llama.cpp `b11232` (`6f767fe`), GCC 13.3, CUDA 12.9.86, sm_90, in the `nvidia/cuda:12.9.1-devel-ubuntu24.04` container |
| `port-cpu-vm` | the same VM | none | the same tag, built for CPU in the same container |
| `port-cuda-3090` | a desktop: Ryzen 7 3700X (AVX2), 32 GB, native Windows 11 25H2 | RTX 3090 24 GB, WDDM, driver 617.14; it also drives the display | the same tag, MSVC 19.51, CUDA 13.4.59, sm_86 |
| `port-cpu-win` | the same desktop | none | the same tag, built for CPU with MSVC |
| `port-vulkan-rx9070` | the workstation: Core i9-12900KS, 62.7 GiB, Linux | RDNA4 Radeon 16 GB, RADV | the same tag, GCC 15.2, Vulkan |
| `port-cpu-ws` | the workstation | none | the same tag, built for CPU with GCC 15.2 |
| `ollama-runner` | the VM or the desktop | as above | the llama-server Ollama 0.35.0 spawns (`b11232` plus its compat patch) |

Every model is the Ollama GGUF at Q8_0, SHA-256 verified against Ollama's
manifest on each machine (`fa9732e3…`, `35f9281a…`, `bbf1d6fc…`). Ollama is the
0.35.0 release for Linux on the VM and for Windows on the desktop, each checked
against the digest GitHub publishes. Every port server ran with `-np 1
--cache-ram 0 -ngl 99`, the context Ollama uses (`-c 2050` for Tev1, `-c 8194`
for Nimble), and the batch size Ollama chose for that model, which the next
section explains. Each log's first line records its exact configuration.

## The port still reproduces Ollama on a GPU

[MEASURED] `probes/fidelity.py` scores the two ticket fixtures through Ollama's
`/v1/systemone` and through the port on each server
(`results/fidelity-<model>-h100.log`, `-rtx3090.log`).

| Model | GPU | Ollama's own runner through the port | Clean `b11232` CUDA build through the port |
|---|---|---:|---:|
| `tev1:0.8b` | H100 | 1.61e-08 | 1.61e-08 |
| `tev1` | H100 | 8.00e-09 | 8.00e-09 |
| `nimble` | H100 | 9.61e-09 | 9.61e-09 |
| `tev1:0.8b` | RTX 3090 | 1.20e-08 | 1.20e-08 |
| `tev1` | RTX 3090 | 7.80e-09 | 7.80e-09 |
| `nimble` | RTX 3090 | 6.06e-09 | 6.06e-09 |

The figures are the largest probability difference from Ollama over 11
questions; every answer matched. The CPU result carries over to CUDA on Linux
and on Windows: a clean build of the tag Ollama pins, with the port, gives
Ollama's numbers without Ollama.

## Batch size is a fifth source of drift

The first Nimble run on the H100 did not match. Through Ollama's runner the port
agreed to 9.61e-09, so the port was right, but the clean build differed by
6.12e-03 (`results/fidelity-nimble-h100-b512.log`). Ollama's runner command line
showed why: it started Nimble with `-b 1024 -ub 1024` and both Tev1 models with
`-b 512 -ub 512`. The clean build had used 512 for all three. Restarted at 1024,
it matched to 9.61e-09 (`results/fidelity-nimble-h100.log`).

The rule is in Ollama's scheduler (`server/sched.go`, `automaticGenerationBatch`
and `generationBatchForContext`, ollama/ollama `1abe35e`). With flash attention
on, the batch is 512 for a context up to 4,096 tokens, 1024 above that and 2048
above 32,768, stepped back down when the predicted VRAM leaves too little
headroom. Nimble's `num_ctx` is 8,194, so it gets 1024 on a card with room; on
both GPUs here it did. On a smaller or fuller card the same request would run
at a different batch size and give different probabilities. The micro-batch
size changes how prefill splits the prompt, so it changes the arithmetic.

This joins the four sources [the CPU results](02-cpu-results.md) found (build,
primer, other questions, cache history). Like them it is deterministic, and
like the history effect it depends on the machine rather than the request.

## Backends drift, and the drift can flip a close answer

[MEASURED] Against Ollama on CUDA, the same GGUF and the same tag on other
backends (`results/fidelity-<model>-h100.log`, `-rtx3090.log`, `-rx9070.log`;
the Vulkan and workstation CPU runs used the H100 VM's Ollama as their
reference, reached over an SSH tunnel):

| Model | CPU, Xeon (AVX-512) | CPU, Ryzen (MSVC) | CPU, i9 (GCC) | Vulkan, RX 9070 |
|---|---:|---:|---:|---:|
| `tev1:0.8b` | 3.85e-02, 1 answer changed | 1.81e-02 | 2.69e-02 | 2.68e-02, 1 answer changed |
| `tev1` | 9.87e-03 | 1.66e-02 | 1.01e-02 | 4.24e-03 |
| `nimble` | 9.54e-03 | 7.45e-03 | 1.58e-02 | 8.04e-03 |

The changed answer on the Xeon is the `urgency` score question of the 3-question
fixture, where Ollama's own distribution is nearly tied (0.398996 for level 2,
0.392589 for level 1) and the CPU build tips it to level 1.

Ollama itself moves between GPUs. The same Ollama 0.35.0, model and request on
the H100 and on the RTX 3090 differ by up to 2.51e-02 for `tev1:0.8b`, 1.10e-02
for `tev1` and 9.89e-03 for `nimble`, with no top answer changing on the
fixtures (`results/ollama-across-gpus.log`, from `probes/ollama_across_logs.py`). A
probability threshold tuned on one GPU does not carry to another GPU, let alone
to a CPU.

## The primer and the other questions, on a GPU

[MEASURED] `probes/priming.py` and `probes/schema_coupling.py` on the two CUDA
GPUs (`results/priming-<model>-<gpu>.log`, `results/schema-coupling-<model>-<gpu>.log`),
with the CPU half's figures for comparison:

| Model | GPU | Primer moves probabilities by | Other questions move `team` by |
|---|---|---:|---:|
| `tev1:0.8b` | CPU half (Xeon) | 1.85e-02 | 1.33e-02 |
| `tev1:0.8b` | RTX 3090 | 2.43e-02 | 1.70e-02 |
| `tev1:0.8b` | H100 | 1.89e-02 | 1.39e-02 |
| `tev1` | RTX 3090 | 1.66e-02 | 5.62e-03 |
| `tev1` | H100 | 8.65e-03 | 6.68e-03 |
| `nimble` | RTX 3090 | 4.17e-03 | 1.45e-02 |
| `nimble` | H100 | 6.36e-03 | 1.41e-02 |

Both effects are present on every GPU and for every model, and they are of the
same order as the backend drift. The Tev1 format, which sends each question
alone, again showed no coupling at all (0 on both GPUs, for both Tev1 models).
The Nimble format keeps Ollama's one-document structure, and couples Nimble's
questions as much as Ollama's does (1.75e-02 on both GPUs).

## The repeated-prompt crash

[MEASURED] `probes/identical_prompt.py` with the Windows CUDA and CPU builds
(`results/identical-prompt-windows.log`): both answered the identical fully
cached prompt twice and stayed up, as `b11232` did on Linux.

## Accuracy on the public suite

[MEASURED] `probes/public_suite.py` ran Bespoke's runner over all 13 subsets,
3,880 records, for each model and format, through the port on one H100 with
`--isolate` (`results/public-suite/<run>.json` holds every subset's summary,
manifest and host record; the table is `results/public-suite/accuracy-h100.md`,
generated by `probes/suite_table.py`). No request failed: every run has zero
error rows, including Tev1 on `helpsteer2`, whose longest prompt was the
context risk the plan named.

| Subset | Nimble, Ollama format | Nimble, own format | Tev1 4B, Ollama format | Tev1 4B, own format | Tev1 0.8B, Ollama format | Tev1 0.8B, own format |
|---|---:|---:|---:|---:|---:|---:|
| vitaminc-dev | 78.6% (471/599) | 79.3% (475/599) | 74.3% (445/599) | 74.5% (446/599) | 68.6% (411/599) | 67.6% (405/599) |
| massive-en-US | 84.3% (295/350) | 86.6% (303/350) | 85.7% (300/350) | 85.4% (299/350) | 79.1% (277/350) | 78.0% (273/350) |
| massive-de-DE | 83.1% (291/350) | 83.7% (293/350) | 83.1% (291/350) | 82.0% (287/350) | 70.6% (247/350) | 72.6% (254/350) |
| boolq | 86.3% (259/300) | 85.0% (255/300) | 85.0% (255/300) | 83.7% (251/300) | 78.7% (236/300) | 78.7% (236/300) |
| squad2 | 74.2% (222/299) | 75.3% (225/299) | 76.3% (228/299) | 78.3% (234/299) | 70.6% (211/299) | 72.6% (217/299) |
| paws | 74.0% (185/250) | 74.0% (185/250) | 82.4% (206/250) | 83.2% (208/250) | 64.8% (162/250) | 71.6% (179/250) |
| multinli | 90.0% (269/299) | 90.6% (271/299) | 92.0% (275/299) | 91.3% (273/299) | 75.3% (225/299) | 79.6% (238/299) |
| civil_comments | 78.0% (234/300) | 81.3% (244/300) | 73.3% (220/300) | 78.0% (234/300) | 76.7% (230/300) | 83.3% (250/300) |
| aegis2 | 82.4% (206/250) | 81.2% (203/250) | 77.2% (193/250) | 80.0% (200/250) | 56.4% (141/250) | 68.4% (171/250) |
| helpsteer2 | 34.1% (85/249) | 34.9% (87/249) | 35.7% (89/249) | 36.1% (90/249) | 33.3% (83/249) | 35.7% (89/249) |
| summeval-relevance | 48.8% (117/240) | 47.9% (115/240) | 50.4% (121/240) | 47.5% (114/240) | 13.8% (33/240) | 13.8% (33/240) |
| summeval-consistency | 82.6% (119/144) | 82.6% (119/144) | 79.9% (115/144) | 81.2% (117/144) | 84.0% (121/144) | 84.0% (121/144) |
| pubmedqa | 78.0% (195/250) | 76.0% (190/250) | 74.4% (186/250) | 71.6% (179/250) | 60.0% (150/250) | 59.6% (149/250) |
| **macro mean** | 75.0% | 75.3% | 74.6% | 74.8% | 64.0% | 66.6% |
| micro mean | 76.0% | 76.4% | 75.4% | 75.6% | 65.1% | 67.4% |
| error rows | 0 | 0 | 0 | 0 | 0 | 0 |

The macro mean is the unweighted mean of the 13 subsets, the average Ollama
and Bespoke both quote, and Bespoke's own `summarize_public_suite` gives the same
figures (`results/public-suite/summarize-public-suite.md`).

Against the published numbers, in the Ollama format:

| Model | Here | Ollama's blog [CLAIM] | Bespoke's table [CLAIM] |
|---|---:|---:|---:|
| Nimble | 75.0% | 75.7% | 74.8% (BF16 transformers, temperature 1.0) |
| Tev1 4B | 74.6% | 73.3% | not in the table |
| Tev1 0.8B | 64.0% | 63.5% | not in the table |

Every model lands within 1.3 points of what Ollama published, Nimble 0.7 below
and both Tev1 models above (by 1.3 and 0.5). Bespoke's table, from BF16
weights and possibly another revision, is 0.2 points under the Q8_0 GGUF here.

Tev1 0.8B answers level 4 on every one of the 240 `summeval-relevance` records,
in both formats, which is where its 13.8% comes from: the human targets spread
over levels 1 to 4.

### Through Ollama itself

[MEASURED] `boolq` scored through Ollama's own `/v1/systemone` on the VM,
against the isolated port run (`results/rows-diff.log`, from
`probes/rows_diff.py`): 268 of 300 records agree to 1e-6, the other 32 differ by
up to 0.0383 in P(true), and no prediction changes, 86.3% both ways. Those 32
are the cache-history effect: Ollama cannot be isolated, so it resumes each
record from whatever the previous one left.

### Across GPUs

[MEASURED] The Ollama-format runs repeated on the RTX 3090
(`results/public-suite/accuracy-backends.md`, `results/rows-diff.log`,
`results/public-suite/compare-<model>-backend-table.md`):

| Model | H100 macro | RTX 3090 macro | Records whose answer differs | Only H100 right / only 3090 right | McNemar p |
|---|---:|---:|---:|---:|---:|
| Nimble | 75.0% | 75.0% | 22 of 3,880 (0.57%) | 8 / 8 | 1 |
| Tev1 4B | 74.6% | 74.7% | 33 (0.85%) | 11 / 12 | 1 |
| Tev1 0.8B | 64.0% | 64.0% | 30 (0.77%) | 10 / 13 | 0.678 |

The drift changes individual answers, a little under one record in a hundred,
and leaves accuracy where it was: the flips cancel. The median per-record
difference is 8.60e-04 for Nimble and 6.67e-03 for Tev1 0.8B, with a largest of
0.147.

## Each model's own prompt format

[MEASURED] Bespoke's `compare_public` pairs each model's Ollama-format run with
its own-format run on the same records (`results/public-suite/compare-<model>-format-h100-table.md`,
from `probes/compare_table.py`; the full reports are alongside). Pooled over
all 3,880 records:

| Model | Only Ollama format right | Only own format right | Exact McNemar p | Subsets significant at p < 0.05 |
|---|---:|---:|---:|---|
| Nimble | 54 | 71 | 0.152 | `civil_comments` (own format better, p 0.0309) |
| Tev1 4B | 81 | 89 | 0.591 | `civil_comments` (own format better, p 0.00258) |
| Tev1 0.8B | 167 | 255 | 2.14e-05 | `aegis2` (p 0.000287), `civil_comments` (p 0.000535), `paws` (p 0.0331), all own format better |

Sending Tev1 0.8B the structure it was trained on is worth 2.6 macro points,
and the pooled test is far from chance. For Nimble and Tev1 4B the format makes
no difference that survives the test, except on `civil_comments`, where both
do better in their own format. With 13 subsets per model, one marked subset in
twenty is expected by chance, so the per-subset marks for the two larger models
are weaker evidence than Tev1 0.8B's pooled result.

The Tev1 format's handling of `noul` and of missing descriptions is this
repository's choice ([how it works](01-how-it-works.md#three-prompt-formats)),
and it holds for these numbers. The Nimble format refused no records.

## Calibration, and one temperature

[MEASURED] `probes/temperature.py` fits one temperature per run on the records
of half the families, chosen by a hash of the family ID so that paired records
are never split, and evaluates on the other half; then it swaps the halves
(`results/temperature/<run>.log`).

| Run | Fitted T, each half | ECE at T = 1 | ECE at the fitted T |
|---|---|---|---|
| Nimble, Ollama format, H100 | 1.767 / 1.744 | 0.1169 / 0.1234 | 0.0285 / 0.0337 |
| Nimble, own format, H100 | 1.755 / 1.728 | 0.1094 / 0.1200 | 0.0179 / 0.0257 |
| Tev1 4B, Ollama format, H100 | 1.574 / 1.534 | 0.0797 / 0.0793 | 0.0151 / 0.0206 |
| Tev1 4B, own format, H100 | 1.630 / 1.624 | 0.0836 / 0.0886 | 0.0186 / 0.0134 |
| Tev1 0.8B, Ollama format, H100 | 1.781 / 1.667 | 0.1152 / 0.1433 | 0.0191 / 0.0294 |
| Tev1 0.8B, own format, H100 | 1.957 / 1.839 | 0.0956 / 0.1292 | 0.0440 / 0.0348 |

All three models are overconfident as shipped, and one temperature between 1.53
and 1.96 removes most of it: ECE falls to 0.013 to 0.044 on held-out families,
and NLL and Brier improve with it (each log has them per subset). The fit is
stable, within about 0.1 between halves, and the RTX 3090 runs give the same
temperatures (1.767 / 1.745 for Nimble). Ollama applies no temperature. Bespoke
fitted T = 2.179 for an earlier Nimble revision [CLAIM]; the checkpoint Ollama
ships wants about 1.75 on this suite. A caller who wants calibrated
probabilities can apply `p^(1/T)`, renormalised, to what Ollama returns.

## What a decision costs

[MEASURED] `probes/latency_sweep.py` varies the state length and the number of
questions, and times three ways to answer each: cold without the primer, cold
with it (a new state, which is what a streaming transcript produces), and warm.
Five repeats, the first discarded, medians shown; the probe runs on the same
machine as the server. Each cell is the wall time the caller waits, and in
brackets llama-server's own prompt time (`results/latency-sweep-<model>-<gpu>.log`).
The RTX 3090 was otherwise idle for these runs.

| Model | State tokens | Questions | Mode | Prompt tokens per question | RTX 3090 (ms) | H100 (ms) |
|---|---:|---:|---|---:|---:|---:|
| `tev1:0.8b` | 64 | 1 | warm | 189 | 20.9 (13.8) | 15.1 (9.9) |
| `tev1:0.8b` | 1536 | 1 | cold | 1737 | 137.5 (113.3) | 78.7 (67.3) |
| `tev1:0.8b` | 1024 | 8 | cold primed | 1539 | 465.2 (342.4) | 398.0 (311.4) |
| `tev1:0.8b` | 64 | 32 | warm | 1724 | 1364.2 (816.6) | 1358.4 (987.5) |
| `tev1` | 1536 | 1 | cold | 1737 | 319.4 (305.4) | 172.1 (160.9) |
| `tev1` | 1024 | 8 | cold primed | 1539 | 796.7 (685.9) | 785.6 (700.0) |
| `tev1` | 64 | 32 | warm | 1724 | 2134.1 (1672.5) | 2769.3 (2380.2) |
| `nimble` | 64 | 1 | warm | 227 | 43.3 (36.1) | 49.1 (42.4) |
| `nimble` | 4096 | 1 | cold | 4467 | 1067.3 (1039.9) | 498.6 (478.3) |
| `nimble` | 1024 | 8 | cold | 1577 | 2453.8 (2321.2) | 1368.9 (1278.3) |
| `nimble` | 1024 | 8 | cold primed | 1577 | 1022.4 (884.8) | 853.4 (761.7) |
| `nimble` | 64 | 32 | cold | 1762 | 9545.5 (8975.6) | 5357.5 (4970.5) |
| `nimble` | 64 | 32 | cold primed | 1762 | 2958.4 (2399.5) | 2834.6 (2450.1) |
| `nimble` | 64 | 32 | warm | 1762 | 2558.5 (2041.8) | 2657.4 (2277.8) |

A single question costs what its prefill costs, and the H100 is 1.7 to 2.1
times as fast at that: Nimble reads 4,467 tokens in 1,067 ms on the RTX 3090 and 499 ms
on the H100. A warm single question costs 15 to 50 ms depending on the model.

Many questions hit a floor that a faster GPU does not lower. With the primer or
warm, 32 questions cost about 42 to 44 ms each for `tev1:0.8b`, 67 to 91 ms for
`tev1` and 80 to 92 ms for `nimble`, on either GPU, and Tev1 4B is slower warm on
the H100 than on the RTX 3090. The server's own time accounts for most of it,
which leaves little for HTTP. It fits llama-server restoring the recurrent-state
checkpoint before each question, which is host-side work. That explanation is
inferred from the timings; no probe here measured the restore itself.

The primer matters for time once prefill is expensive. On the CPU it cut a cold
8-question request from 21 s to 4 s; on a GPU it makes little difference for
the small Tev1 but cuts a cold 32-question Nimble request from 9.5 s to 3.0 s on
the RTX 3090. For the streaming case, a new 1,024-token state with 8 questions
costs Nimble about 1.0 s on the RTX 3090 and 0.85 s on the H100, and Tev1 0.8B
about 0.47 s and 0.40 s.

## Repeating it

1. Build `b11232` for each backend with `probes/build_llama_cpp.sh` (Linux;
   `CUDA_ARCH=90` for an H100, and its header shows the CUDA container used on
   a host with no toolkit) or `probes/build_llama_cpp.cmd` (native Windows).
2. Fetch the GGUFs with `probes/registry.py --download` and check each digest.
   The registry serves one connection at about 3 to 6 MB/s here, so a
   parallel ranged download checked against the manifest's SHA-256 is much
   faster for the 9.5 GB Nimble file.
3. Serve each model with `-c 2050` (Tev1) or `-c 8194` (Nimble), `-np 1 -ngl 99
   --cache-ram 0`, and the batch size Ollama would choose on that card: read it
   from Ollama's runner command line, or apply the rule above (1024 for Nimble
   when VRAM allows, 512 for Tev1). Check `/props` reports the model you just
   started before measuring anything.
4. Run the probes as each log's first line records, then the suite with
   `probes/public_suite.py ... --isolate` per subset, and the analysis with
   `suite_table.py`, `compare_table.py` (over Bespoke's `compare_public`
   output), `rows_diff.py` and `temperature.py`.

## Not exercised

- Quantisation (experiment G in the plan): only the Q8_0 GGUFs Ollama ships were
  run.
- llama.cpp's open System One pull request.
- Timing Ollama itself; every latency is the port's, which reproduces Ollama's
  requests.
- Tev1 0.8B and Tev1 4B on the RTX 3090 in their own format; the format
  comparison ran on the H100 only.
- Which llama.cpp change moved the CPU half's `b9190` numbers, and whether the
  per-question floor is the checkpoint restore.

---

Previous: [CPU results](02-cpu-results.md) | [Contents](../README.md#contents)
