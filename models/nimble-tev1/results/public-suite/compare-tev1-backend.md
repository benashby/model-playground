# tev1 on vitaminc-dev: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 445/599 | 74.3% | 70.6%–77.6% | 78/171 | 0.144 | 0.776 | 0.387 | 0 |
| rtx3090 | 443/599 | 74.0% | 70.3%–77.3% | 77/171 | 0.148 | 0.782 | 0.389 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| vitaminc-real | 190/259 | 189/259 |
| vitaminc-synthetic | 255/340 | 254/340 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.5% of records. Both correct 443, neither 154, only h100 2, only rtx3090 0. Exact McNemar two-sided p = 0.5000.


# tev1 on massive-en-US: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 300/350 | 85.7% | 81.7%–89.0% | 300/350 | 0.062 | 0.548 | 0.224 | 0 |
| rtx3090 | 300/350 | 85.7% | 81.7%–89.0% | 300/350 | 0.058 | 0.548 | 0.224 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-en-US | 300/350 | 300/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.7% of records. Both correct 300, neither 50, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1 on massive-de-DE: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 291/350 | 83.1% | 78.9%–86.7% | 291/350 | 0.050 | 0.622 | 0.272 | 0 |
| rtx3090 | 290/350 | 82.9% | 78.6%–86.4% | 290/350 | 0.061 | 0.625 | 0.274 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-de-DE | 291/350 | 290/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.1% of records. Both correct 290, neither 59, only h100 1, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1 on boolq: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 255/300 | 85.0% | 80.5%–88.6% | 230/274 | 0.069 | 0.355 | 0.215 | 0 |
| rtx3090 | 256/300 | 85.3% | 80.9%–88.9% | 231/274 | 0.073 | 0.352 | 0.214 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| boolq | 255/300 | 256/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.7% of records. Both correct 255, neither 44, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1 on squad2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 228/299 | 76.3% | 71.1%–80.7% | 6/31 | 0.121 | 0.548 | 0.338 | 0 |
| rtx3090 | 229/299 | 76.6% | 71.5%–81.0% | 6/31 | 0.116 | 0.548 | 0.337 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| squad2 | 228/299 | 229/299 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.7% of records. Both correct 228, neither 70, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1 on paws: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 206/250 | 82.4% | 77.2%–86.6% | 206/250 | 0.086 | 0.463 | 0.268 | 0 |
| rtx3090 | 205/250 | 82.0% | 76.8%–86.3% | 205/250 | 0.091 | 0.466 | 0.270 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| paws | 206/250 | 205/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.6% of records. Both correct 205, neither 44, only h100 1, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1 on multinli: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 275/299 | 92.0% | 88.3%–94.5% | 81/103 | 0.022 | 0.248 | 0.128 | 0 |
| rtx3090 | 275/299 | 92.0% | 88.3%–94.5% | 81/103 | 0.025 | 0.247 | 0.127 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| multinli-fiction | 52/60 | 53/60 |
| multinli-government | 57/61 | 57/61 |
| multinli-slate | 57/60 | 57/60 |
| multinli-telephone | 56/60 | 56/60 |
| multinli-travel | 53/58 | 52/58 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.3% of records. Both correct 274, neither 23, only h100 1, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1 on civil_comments: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 220/300 | 73.3% | 68.1%–78.0% | 220/300 | 0.065 | 0.527 | 0.350 | 0 |
| rtx3090 | 221/300 | 73.7% | 68.4%–78.3% | 221/300 | 0.077 | 0.529 | 0.352 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| civil_comments | 220/300 | 221/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.0% of records. Both correct 219, neither 78, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# tev1 on aegis2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 193/250 | 77.2% | 71.6%–82.0% | 193/250 | 0.079 | 0.504 | 0.318 | 0 |
| rtx3090 | 194/250 | 77.6% | 72.0%–82.3% | 194/250 | 0.089 | 0.502 | 0.317 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| aegis2 | 193/250 | 194/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.6% of records. Both correct 193, neither 56, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# tev1 on helpsteer2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 89/249 | 35.7% | 30.0%–41.9% | 16/125 | 0.238 | 1.715 | 0.819 | 0 |
| rtx3090 | 88/249 | 35.3% | 29.7%–41.5% | 14/125 | 0.241 | 1.717 | 0.820 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| helpsteer2 | 89/249 | 88/249 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 97.6% of records. Both correct 86, neither 158, only h100 3, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# tev1 on summeval-relevance: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 121/240 | 50.4% | 44.1%–56.7% | 0/15 | 0.095 | 1.139 | 0.631 | 0 |
| rtx3090 | 122/240 | 50.8% | 44.5%–57.1% | 0/15 | 0.098 | 1.142 | 0.632 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-relevance | 121/240 | 122/240 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 97.9% of records. Both correct 120, neither 117, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# tev1 on summeval-consistency: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 115/144 | 79.9% | 72.6%–85.6% | 0/9 | 0.166 | 0.745 | 0.316 | 0 |
| rtx3090 | 115/144 | 79.9% | 72.6%–85.6% | 0/9 | 0.161 | 0.749 | 0.317 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-consistency | 115/144 | 115/144 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.6% of records. Both correct 115, neither 29, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# tev1 on pubmedqa: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 186/250 | 74.4% | 68.6%–79.4% | 186/250 | 0.103 | 0.766 | 0.381 | 0 |
| rtx3090 | 187/250 | 74.8% | 69.1%–79.8% | 187/250 | 0.094 | 0.765 | 0.381 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| pubmedqa | 186/250 | 187/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.4% of records. Both correct 185, neither 62, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.
