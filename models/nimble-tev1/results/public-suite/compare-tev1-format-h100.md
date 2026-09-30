# tev1 on vitaminc-dev: Ollama format vs native format, H100

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 445/599 | 74.3% | 70.6%–77.6% | 78/171 | 0.144 | 0.776 | 0.387 | 0 |
| native | 446/599 | 74.5% | 70.8%–77.8% | 83/171 | 0.141 | 0.822 | 0.403 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| vitaminc-real | 190/259 | 193/259 |
| vitaminc-synthetic | 255/340 | 253/340 |

## Paired correctness

**ollama vs native**: predictions agree on 95.8% of records. Both correct 434, neither 142, only ollama 11, only native 12. Exact McNemar two-sided p = 1.0000.


# tev1 on massive-en-US: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 300/350 | 85.7% | 81.7%–89.0% | 300/350 | 0.062 | 0.548 | 0.224 | 0 |
| native | 299/350 | 85.4% | 81.3%–88.7% | 299/350 | 0.076 | 0.584 | 0.228 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-en-US | 300/350 | 299/350 |

## Paired correctness

**ollama vs native**: predictions agree on 97.4% of records. Both correct 296, neither 47, only ollama 4, only native 3. Exact McNemar two-sided p = 1.0000.


# tev1 on massive-de-DE: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 291/350 | 83.1% | 78.9%–86.7% | 291/350 | 0.050 | 0.622 | 0.272 | 0 |
| native | 287/350 | 82.0% | 77.6%–85.7% | 287/350 | 0.098 | 0.688 | 0.282 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-de-DE | 291/350 | 287/350 |

## Paired correctness

**ollama vs native**: predictions agree on 95.7% of records. Both correct 283, neither 55, only ollama 8, only native 4. Exact McNemar two-sided p = 0.3877.


# tev1 on boolq: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 255/300 | 85.0% | 80.5%–88.6% | 230/274 | 0.069 | 0.355 | 0.215 | 0 |
| native | 251/300 | 83.7% | 79.1%–87.4% | 226/274 | 0.068 | 0.329 | 0.202 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| boolq | 255/300 | 251/300 |

## Paired correctness

**ollama vs native**: predictions agree on 96.7% of records. Both correct 248, neither 42, only ollama 7, only native 3. Exact McNemar two-sided p = 0.3438.


# tev1 on squad2: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 228/299 | 76.3% | 71.1%–80.7% | 6/31 | 0.121 | 0.548 | 0.338 | 0 |
| native | 234/299 | 78.3% | 73.2%–82.6% | 6/31 | 0.100 | 0.521 | 0.328 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| squad2 | 228/299 | 234/299 |

## Paired correctness

**ollama vs native**: predictions agree on 94.6% of records. Both correct 223, neither 60, only ollama 5, only native 11. Exact McNemar two-sided p = 0.2101.


# tev1 on paws: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 206/250 | 82.4% | 77.2%–86.6% | 206/250 | 0.086 | 0.463 | 0.268 | 0 |
| native | 208/250 | 83.2% | 78.1%–87.3% | 208/250 | 0.056 | 0.390 | 0.239 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| paws | 206/250 | 208/250 |

## Paired correctness

**ollama vs native**: predictions agree on 98.4% of records. Both correct 205, neither 41, only ollama 1, only native 3. Exact McNemar two-sided p = 0.6250.


# tev1 on multinli: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 275/299 | 92.0% | 88.3%–94.5% | 81/103 | 0.022 | 0.248 | 0.128 | 0 |
| native | 273/299 | 91.3% | 87.6%–94.0% | 78/103 | 0.023 | 0.253 | 0.137 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| multinli-fiction | 52/60 | 52/60 |
| multinli-government | 57/61 | 57/61 |
| multinli-slate | 57/60 | 56/60 |
| multinli-telephone | 56/60 | 57/60 |
| multinli-travel | 53/58 | 51/58 |

## Paired correctness

**ollama vs native**: predictions agree on 97.3% of records. Both correct 270, neither 21, only ollama 5, only native 3. Exact McNemar two-sided p = 0.7266.


# tev1 on civil_comments: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 220/300 | 73.3% | 68.1%–78.0% | 220/300 | 0.065 | 0.527 | 0.350 | 0 |
| native | 234/300 | 78.0% | 73.0%–82.3% | 234/300 | 0.050 | 0.454 | 0.300 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| civil_comments | 220/300 | 234/300 |

## Paired correctness

**ollama vs native**: predictions agree on 93.3% of records. Both correct 217, neither 63, only ollama 3, only native 17. Exact McNemar two-sided p = 0.0026.


# tev1 on aegis2: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 193/250 | 77.2% | 71.6%–82.0% | 193/250 | 0.079 | 0.504 | 0.318 | 0 |
| native | 200/250 | 80.0% | 74.6%–84.5% | 200/250 | 0.082 | 0.486 | 0.308 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| aegis2 | 193/250 | 200/250 |

## Paired correctness

**ollama vs native**: predictions agree on 94.0% of records. Both correct 189, neither 46, only ollama 4, only native 11. Exact McNemar two-sided p = 0.1185.


# tev1 on helpsteer2: Ollama format vs native format, H100

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 89/249 | 35.7% | 30.0%–41.9% | 16/125 | 0.238 | 1.715 | 0.819 | 0 |
| native | 90/249 | 36.1% | 30.4%–42.3% | 18/125 | 0.268 | 1.810 | 0.852 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| helpsteer2 | 89/249 | 90/249 |

## Paired correctness

**ollama vs native**: predictions agree on 83.1% of records. Both correct 77, neither 147, only ollama 12, only native 13. Exact McNemar two-sided p = 1.0000.


# tev1 on summeval-relevance: Ollama format vs native format, H100

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 121/240 | 50.4% | 44.1%–56.7% | 0/15 | 0.095 | 1.139 | 0.631 | 0 |
| native | 114/240 | 47.5% | 41.3%–53.8% | 0/15 | 0.091 | 1.149 | 0.638 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-relevance | 121/240 | 114/240 |

## Paired correctness

**ollama vs native**: predictions agree on 89.2% of records. Both correct 110, neither 115, only ollama 11, only native 4. Exact McNemar two-sided p = 0.1185.


# tev1 on summeval-consistency: Ollama format vs native format, H100

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 115/144 | 79.9% | 72.6%–85.6% | 0/9 | 0.166 | 0.745 | 0.316 | 0 |
| native | 117/144 | 81.2% | 74.1%–86.8% | 0/9 | 0.113 | 0.589 | 0.262 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-consistency | 115/144 | 117/144 |

## Paired correctness

**ollama vs native**: predictions agree on 96.5% of records. Both correct 115, neither 27, only ollama 0, only native 2. Exact McNemar two-sided p = 0.5000.


# tev1 on pubmedqa: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 186/250 | 74.4% | 68.6%–79.4% | 186/250 | 0.103 | 0.766 | 0.381 | 0 |
| native | 179/250 | 71.6% | 65.7%–76.8% | 179/250 | 0.132 | 0.832 | 0.432 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| pubmedqa | 186/250 | 179/250 |

## Paired correctness

**ollama vs native**: predictions agree on 93.6% of records. Both correct 176, neither 61, only ollama 10, only native 3. Exact McNemar two-sided p = 0.0923.
