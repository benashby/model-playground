# nimble on vitaminc-dev: Ollama format vs native format, H100

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 471/599 | 78.6% | 75.2%–81.7% | 92/171 | 0.135 | 0.702 | 0.334 | 0 |
| native | 475/599 | 79.3% | 75.9%–82.4% | 91/171 | 0.128 | 0.673 | 0.330 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| vitaminc-real | 199/259 | 205/259 |
| vitaminc-synthetic | 272/340 | 270/340 |

## Paired correctness

**ollama vs native**: predictions agree on 97.0% of records. Both correct 466, neither 119, only ollama 5, only native 9. Exact McNemar two-sided p = 0.4240.


# nimble on massive-en-US: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 295/350 | 84.3% | 80.1%–87.7% | 295/350 | 0.079 | 0.506 | 0.230 | 0 |
| native | 303/350 | 86.6% | 82.6%–89.7% | 303/350 | 0.066 | 0.487 | 0.210 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-en-US | 295/350 | 303/350 |

## Paired correctness

**ollama vs native**: predictions agree on 95.1% of records. Both correct 292, neither 44, only ollama 3, only native 11. Exact McNemar two-sided p = 0.0574.


# nimble on massive-de-DE: Ollama format vs native format, H100

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 291/350 | 83.1% | 78.9%–86.7% | 291/350 | 0.084 | 0.588 | 0.259 | 0 |
| native | 293/350 | 83.7% | 79.5%–87.2% | 293/350 | 0.088 | 0.568 | 0.249 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| massive-de-DE | 291/350 | 293/350 |

## Paired correctness

**ollama vs native**: predictions agree on 95.4% of records. Both correct 286, neither 52, only ollama 5, only native 7. Exact McNemar two-sided p = 0.7744.


# nimble on boolq: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 259/300 | 86.3% | 82.0%–89.8% | 235/274 | 0.082 | 0.359 | 0.206 | 0 |
| native | 255/300 | 85.0% | 80.5%–88.6% | 231/274 | 0.082 | 0.357 | 0.211 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| boolq | 259/300 | 255/300 |

## Paired correctness

**ollama vs native**: predictions agree on 98.0% of records. Both correct 254, neither 40, only ollama 5, only native 1. Exact McNemar two-sided p = 0.2188.


# nimble on squad2: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 222/299 | 74.2% | 69.0%–78.9% | 5/31 | 0.180 | 0.764 | 0.415 | 0 |
| native | 225/299 | 75.3% | 70.1%–79.8% | 6/31 | 0.163 | 0.700 | 0.385 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| squad2 | 222/299 | 225/299 |

## Paired correctness

**ollama vs native**: predictions agree on 98.3% of records. Both correct 221, neither 73, only ollama 1, only native 4. Exact McNemar two-sided p = 0.3750.


# nimble on paws: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 185/250 | 74.0% | 68.2%–79.0% | 185/250 | 0.189 | 0.928 | 0.425 | 0 |
| native | 185/250 | 74.0% | 68.2%–79.0% | 185/250 | 0.195 | 0.923 | 0.443 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| paws | 185/250 | 185/250 |

## Paired correctness

**ollama vs native**: predictions agree on 96.8% of records. Both correct 181, neither 61, only ollama 4, only native 4. Exact McNemar two-sided p = 1.0000.


# nimble on multinli: Ollama format vs native format, H100

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 269/299 | 90.0% | 86.0%–92.9% | 76/103 | 0.039 | 0.259 | 0.148 | 0 |
| native | 271/299 | 90.6% | 86.8%–93.4% | 78/103 | 0.031 | 0.251 | 0.144 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| multinli-fiction | 54/60 | 54/60 |
| multinli-government | 54/61 | 55/61 |
| multinli-slate | 51/60 | 52/60 |
| multinli-telephone | 57/60 | 57/60 |
| multinli-travel | 53/58 | 53/58 |

## Paired correctness

**ollama vs native**: predictions agree on 99.0% of records. Both correct 269, neither 28, only ollama 0, only native 2. Exact McNemar two-sided p = 0.5000.


# nimble on civil_comments: Ollama format vs native format, H100

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 234/300 | 78.0% | 73.0%–82.3% | 234/300 | 0.087 | 0.500 | 0.317 | 0 |
| native | 244/300 | 81.3% | 76.5%–85.3% | 244/300 | 0.067 | 0.448 | 0.283 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| civil_comments | 234/300 | 244/300 |

## Paired correctness

**ollama vs native**: predictions agree on 94.0% of records. Both correct 230, neither 52, only ollama 4, only native 14. Exact McNemar two-sided p = 0.0309.


# nimble on aegis2: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 206/250 | 82.4% | 77.2%–86.6% | 206/250 | 0.122 | 0.600 | 0.293 | 0 |
| native | 203/250 | 81.2% | 75.9%–85.6% | 203/250 | 0.116 | 0.595 | 0.299 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| aegis2 | 206/250 | 203/250 |

## Paired correctness

**ollama vs native**: predictions agree on 95.6% of records. Both correct 199, neither 40, only ollama 7, only native 4. Exact McNemar two-sided p = 0.5488.


# nimble on helpsteer2: Ollama format vs native format, H100

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 85/249 | 34.1% | 28.5%–40.2% | 14/125 | 0.347 | 1.797 | 0.858 | 0 |
| native | 87/249 | 34.9% | 29.3%–41.0% | 17/125 | 0.343 | 1.871 | 0.885 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| helpsteer2 | 85/249 | 87/249 |

## Paired correctness

**ollama vs native**: predictions agree on 88.4% of records. Both correct 76, neither 153, only ollama 9, only native 11. Exact McNemar two-sided p = 0.8238.


# nimble on summeval-relevance: Ollama format vs native format, H100

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 117/240 | 48.8% | 42.5%–55.0% | 0/15 | 0.175 | 1.241 | 0.685 | 0 |
| native | 115/240 | 47.9% | 41.7%–54.2% | 0/15 | 0.150 | 1.193 | 0.662 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-relevance | 117/240 | 115/240 |

## Paired correctness

**ollama vs native**: predictions agree on 95.0% of records. Both correct 112, neither 120, only ollama 5, only native 3. Exact McNemar two-sided p = 0.7266.


# nimble on summeval-consistency: Ollama format vs native format, H100

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 119/144 | 82.6% | 75.6%–88.0% | 0/9 | 0.131 | 0.591 | 0.246 | 0 |
| native | 119/144 | 82.6% | 75.6%–88.0% | 0/9 | 0.126 | 0.575 | 0.239 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| summeval-consistency | 119/144 | 119/144 |

## Paired correctness

**ollama vs native**: predictions agree on 99.3% of records. Both correct 119, neither 25, only ollama 0, only native 0. Exact McNemar two-sided p = 1.0000.


# nimble on pubmedqa: Ollama format vs native format, H100

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama | 195/250 | 78.0% | 72.5%–82.7% | 195/250 | 0.144 | 0.773 | 0.358 | 0 |
| native | 190/250 | 76.0% | 70.3%–80.9% | 190/250 | 0.144 | 0.770 | 0.363 | 0 |

## Per domain

| Domain | ollama | native |
|---|---:|---:|
| pubmedqa | 195/250 | 190/250 |

## Paired correctness

**ollama vs native**: predictions agree on 96.8% of records. Both correct 189, neither 54, only ollama 6, only native 1. Exact McNemar two-sided p = 0.1250.
