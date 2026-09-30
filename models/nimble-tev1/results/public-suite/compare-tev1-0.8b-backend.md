# tev1-0.8b on vitaminc-dev: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 411/599 | 68.6% | 64.8%–72.2% | 42/171 | 0.136 | 0.859 | 0.463 | 0 |
| rtx3090 | 414/599 | 69.1% | 65.3%–72.7% | 43/171 | 0.130 | 0.863 | 0.465 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| vitaminc-real | 153/259 | 154/259 |
| vitaminc-synthetic | 258/340 | 260/340 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.2% of records. Both correct 410, neither 184, only h100 1, only rtx3090 4. Exact McNemar two-sided p = 0.3750.


# tev1-0.8b on massive-en-US: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 277/350 | 79.1% | 74.6%–83.1% | 277/350 | 0.087 | 0.917 | 0.327 | 0 |
| rtx3090 | 275/350 | 78.6% | 74.0%–82.5% | 275/350 | 0.094 | 0.923 | 0.328 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-en-US | 277/350 | 275/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.4% of records. Both correct 275, neither 73, only h100 2, only rtx3090 0. Exact McNemar two-sided p = 0.5000.


# tev1-0.8b on massive-de-DE: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 247/350 | 70.6% | 65.6%–75.1% | 247/350 | 0.109 | 1.211 | 0.419 | 0 |
| rtx3090 | 247/350 | 70.6% | 65.6%–75.1% | 247/350 | 0.110 | 1.217 | 0.422 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-de-DE | 247/350 | 247/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.4% of records. Both correct 247, neither 103, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on boolq: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 236/300 | 78.7% | 73.7%–82.9% | 213/274 | 0.077 | 0.527 | 0.333 | 0 |
| rtx3090 | 234/300 | 78.0% | 73.0%–82.3% | 211/274 | 0.066 | 0.525 | 0.331 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| boolq | 236/300 | 234/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.7% of records. Both correct 233, neither 63, only h100 3, only rtx3090 1. Exact McNemar two-sided p = 0.6250.


# tev1-0.8b on squad2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 211/299 | 70.6% | 65.2%–75.4% | 1/31 | 0.074 | 0.590 | 0.396 | 0 |
| rtx3090 | 212/299 | 70.9% | 65.5%–75.8% | 0/31 | 0.079 | 0.589 | 0.396 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| squad2 | 211/299 | 212/299 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.0% of records. Both correct 210, neither 86, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on paws: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 162/250 | 64.8% | 58.7%–70.5% | 162/250 | 0.172 | 0.780 | 0.517 | 0 |
| rtx3090 | 162/250 | 64.8% | 58.7%–70.5% | 162/250 | 0.168 | 0.774 | 0.515 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| paws | 162/250 | 162/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 162, neither 88, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on multinli: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 225/299 | 75.3% | 70.1%–79.8% | 44/103 | 0.043 | 0.584 | 0.340 | 0 |
| rtx3090 | 225/299 | 75.3% | 70.1%–79.8% | 44/103 | 0.023 | 0.581 | 0.340 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| multinli-fiction | 42/60 | 43/60 |
| multinli-government | 50/61 | 50/61 |
| multinli-slate | 45/60 | 44/60 |
| multinli-telephone | 43/60 | 43/60 |
| multinli-travel | 45/58 | 45/58 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.3% of records. Both correct 224, neither 73, only h100 1, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on civil_comments: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 230/300 | 76.7% | 71.6%–81.1% | 230/300 | 0.078 | 0.538 | 0.345 | 0 |
| rtx3090 | 231/300 | 77.0% | 71.9%–81.4% | 231/300 | 0.079 | 0.537 | 0.345 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| civil_comments | 230/300 | 231/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.0% of records. Both correct 229, neither 68, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on aegis2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 141/250 | 56.4% | 50.2%–62.4% | 141/250 | 0.158 | 0.783 | 0.556 | 0 |
| rtx3090 | 140/250 | 56.0% | 49.8%–62.0% | 140/250 | 0.161 | 0.779 | 0.554 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| aegis2 | 141/250 | 140/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.6% of records. Both correct 140, neither 109, only h100 1, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on helpsteer2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 83/249 | 33.3% | 27.8%–39.4% | 16/125 | 0.339 | 2.116 | 0.920 | 0 |
| rtx3090 | 84/249 | 33.7% | 28.1%–39.8% | 16/125 | 0.326 | 2.110 | 0.917 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| helpsteer2 | 83/249 | 84/249 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.4% of records. Both correct 83, neither 165, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on summeval-relevance: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 33/240 | 13.8% | 10.0%–18.7% | 0/15 | 0.650 | 2.375 | 1.282 | 0 |
| rtx3090 | 33/240 | 13.8% | 10.0%–18.7% | 0/15 | 0.651 | 2.372 | 1.283 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-relevance | 33/240 | 33/240 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 33, neither 207, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on summeval-consistency: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 121/144 | 84.0% | 77.2%–89.1% | 1/9 | 0.100 | 0.709 | 0.294 | 0 |
| rtx3090 | 121/144 | 84.0% | 77.2%–89.1% | 1/9 | 0.097 | 0.711 | 0.294 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-consistency | 121/144 | 121/144 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 121, neither 23, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on pubmedqa: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 150/250 | 60.0% | 53.8%–65.9% | 150/250 | 0.097 | 0.921 | 0.525 | 0 |
| rtx3090 | 152/250 | 60.8% | 54.6%–66.6% | 152/250 | 0.091 | 0.923 | 0.527 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| pubmedqa | 150/250 | 152/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.4% of records. Both correct 150, neither 98, only h100 0, only rtx3090 2. Exact McNemar two-sided p = 0.5000.
