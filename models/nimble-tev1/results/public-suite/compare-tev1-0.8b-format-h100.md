# tev1-0.8b on vitaminc-dev: Ollama format vs native format, H100

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 411/599 | 68.6% | 64.8%–72.2% | 42/171 | 0.136 | 0.859 | 0.463 | 0 |
| native | 405/599 | 67.6% | 63.8%–71.2% | 49/171 | 0.132 | 0.864 | 0.450 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| vitaminc-real | 153/259 | 162/259 |
| vitaminc-synthetic | 258/340 | 243/340 |

## Paired correctness

**ollama vs native**: predictions agree on 86.0% of records. Both correct 380, neither 163, only ollama 31, only native 25. Exact McNemar two-sided p = 0.5044.


# tev1-0.8b on massive-en-US: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 277/350 | 79.1% | 74.6%–83.1% | 277/350 | 0.087 | 0.917 | 0.327 | 0 |
| native | 273/350 | 78.0% | 73.4%–82.0% | 273/350 | 0.131 | 0.952 | 0.328 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-en-US | 277/350 | 273/350 |

## Paired correctness

**ollama vs native**: predictions agree on 88.3% of records. Both correct 260, neither 60, only ollama 17, only native 13. Exact McNemar two-sided p = 0.5847.


# tev1-0.8b on massive-de-DE: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 247/350 | 70.6% | 65.6%–75.1% | 247/350 | 0.109 | 1.211 | 0.419 | 0 |
| native | 254/350 | 72.6% | 67.7%–77.0% | 254/350 | 0.165 | 1.329 | 0.430 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-de-DE | 247/350 | 254/350 |

## Paired correctness

**ollama vs native**: predictions agree on 85.7% of records. Both correct 235, neither 84, only ollama 12, only native 19. Exact McNemar two-sided p = 0.2810.


# tev1-0.8b on boolq: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 236/300 | 78.7% | 73.7%–82.9% | 213/274 | 0.077 | 0.527 | 0.333 | 0 |
| native | 236/300 | 78.7% | 73.7%–82.9% | 213/274 | 0.090 | 0.527 | 0.330 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| boolq | 236/300 | 236/300 |

## Paired correctness

**ollama vs native**: predictions agree on 94.7% of records. Both correct 228, neither 56, only ollama 8, only native 8. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on squad2: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 211/299 | 70.6% | 65.2%–75.4% | 1/31 | 0.074 | 0.590 | 0.396 | 0 |
| native | 217/299 | 72.6% | 67.3%–77.3% | 3/31 | 0.044 | 0.555 | 0.371 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| squad2 | 211/299 | 217/299 |

## Paired correctness

**ollama vs native**: predictions agree on 82.6% of records. Both correct 188, neither 59, only ollama 23, only native 29. Exact McNemar two-sided p = 0.4885.


# tev1-0.8b on paws: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 162/250 | 64.8% | 58.7%–70.5% | 162/250 | 0.172 | 0.780 | 0.517 | 0 |
| native | 179/250 | 71.6% | 65.7%–76.8% | 179/250 | 0.029 | 0.525 | 0.353 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| paws | 162/250 | 179/250 |

## Paired correctness

**ollama vs native**: predictions agree on 77.2% of records. Both correct 142, neither 51, only ollama 20, only native 37. Exact McNemar two-sided p = 0.0331.


# tev1-0.8b on multinli: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 225/299 | 75.3% | 70.1%–79.8% | 44/103 | 0.043 | 0.584 | 0.340 | 0 |
| native | 238/299 | 79.6% | 74.7%–83.8% | 52/103 | 0.052 | 0.522 | 0.297 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| multinli-fiction | 42/60 | 50/60 |
| multinli-government | 50/61 | 50/61 |
| multinli-slate | 45/60 | 46/60 |
| multinli-telephone | 43/60 | 42/60 |
| multinli-travel | 45/58 | 50/58 |

## Paired correctness

**ollama vs native**: predictions agree on 87.0% of records. Both correct 212, neither 48, only ollama 13, only native 26. Exact McNemar two-sided p = 0.0533.


# tev1-0.8b on civil_comments: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 230/300 | 76.7% | 71.6%–81.1% | 230/300 | 0.078 | 0.538 | 0.345 | 0 |
| native | 250/300 | 83.3% | 78.7%–87.1% | 250/300 | 0.062 | 0.390 | 0.242 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| civil_comments | 230/300 | 250/300 |

## Paired correctness

**ollama vs native**: predictions agree on 89.3% of records. Both correct 224, neither 44, only ollama 6, only native 26. Exact McNemar two-sided p = 0.0005.


# tev1-0.8b on aegis2: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 141/250 | 56.4% | 50.2%–62.4% | 141/250 | 0.158 | 0.783 | 0.556 | 0 |
| native | 171/250 | 68.4% | 62.4%–73.8% | 171/250 | 0.059 | 0.618 | 0.426 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| aegis2 | 141/250 | 171/250 |

## Paired correctness

**ollama vs native**: predictions agree on 73.6% of records. Both correct 123, neither 61, only ollama 18, only native 48. Exact McNemar two-sided p = 0.0003.


# tev1-0.8b on helpsteer2: Ollama format vs native format, H100

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 83/249 | 33.3% | 27.8%–39.4% | 16/125 | 0.339 | 2.116 | 0.920 | 0 |
| native | 89/249 | 35.7% | 30.0%–41.9% | 17/125 | 0.310 | 1.903 | 0.887 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| helpsteer2 | 83/249 | 89/249 |

## Paired correctness

**ollama vs native**: predictions agree on 82.7% of records. Both correct 74, neither 151, only ollama 9, only native 15. Exact McNemar two-sided p = 0.3075.


# tev1-0.8b on summeval-relevance: Ollama format vs native format, H100

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 33/240 | 13.8% | 10.0%–18.7% | 0/15 | 0.650 | 2.375 | 1.282 | 0 |
| native | 33/240 | 13.8% | 10.0%–18.7% | 0/15 | 0.728 | 2.801 | 1.434 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-relevance | 33/240 | 33/240 |

## Paired correctness

**ollama vs native**: predictions agree on 100.0% of records. Both correct 33, neither 207, only ollama 0, only native 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on summeval-consistency: Ollama format vs native format, H100

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 121/144 | 84.0% | 77.2%–89.1% | 1/9 | 0.100 | 0.709 | 0.294 | 0 |
| native | 121/144 | 84.0% | 77.2%–89.1% | 1/9 | 0.114 | 0.696 | 0.297 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-consistency | 121/144 | 121/144 |

## Paired correctness

**ollama vs native**: predictions agree on 100.0% of records. Both correct 121, neither 23, only ollama 0, only native 0. Exact McNemar two-sided p = 1.0000.


# tev1-0.8b on pubmedqa: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 150/250 | 60.0% | 53.8%–65.9% | 150/250 | 0.097 | 0.921 | 0.525 | 0 |
| native | 149/250 | 59.6% | 53.4%–65.5% | 149/250 | 0.078 | 0.927 | 0.533 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| pubmedqa | 150/250 | 149/250 |

## Paired correctness

**ollama vs native**: predictions agree on 88.0% of records. Both correct 140, neither 91, only ollama 10, only native 9. Exact McNemar two-sided p = 1.0000.
