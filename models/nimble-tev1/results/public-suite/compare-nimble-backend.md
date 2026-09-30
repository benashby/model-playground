# nimble on vitaminc-dev: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 599 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 471/599 | 78.6% | 75.2%–81.7% | 92/171 | 0.135 | 0.702 | 0.334 | 0 |
| rtx3090 | 472/599 | 78.8% | 75.3%–81.9% | 93/171 | 0.136 | 0.703 | 0.334 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| vitaminc-real | 199/259 | 201/259 |
| vitaminc-synthetic | 272/340 | 271/340 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.3% of records. Both correct 470, neither 126, only h100 1, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# nimble on massive-en-US: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 295/350 | 84.3% | 80.1%–87.7% | 295/350 | 0.079 | 0.506 | 0.230 | 0 |
| rtx3090 | 295/350 | 84.3% | 80.1%–87.7% | 295/350 | 0.079 | 0.505 | 0.229 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-en-US | 295/350 | 295/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.1% of records. Both correct 294, neither 54, only h100 1, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# nimble on massive-de-DE: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 350 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 291/350 | 83.1% | 78.9%–86.7% | 291/350 | 0.084 | 0.588 | 0.259 | 0 |
| rtx3090 | 291/350 | 83.1% | 78.9%–86.7% | 291/350 | 0.086 | 0.591 | 0.259 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| massive-de-DE | 291/350 | 291/350 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.1% of records. Both correct 291, neither 59, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on boolq: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 259/300 | 86.3% | 82.0%–89.8% | 235/274 | 0.082 | 0.359 | 0.206 | 0 |
| rtx3090 | 258/300 | 86.0% | 81.6%–89.5% | 234/274 | 0.079 | 0.359 | 0.206 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| boolq | 259/300 | 258/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.7% of records. Both correct 258, neither 41, only h100 1, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on squad2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 222/299 | 74.2% | 69.0%–78.9% | 5/31 | 0.180 | 0.764 | 0.415 | 0 |
| rtx3090 | 222/299 | 74.2% | 69.0%–78.9% | 5/31 | 0.179 | 0.761 | 0.414 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| squad2 | 222/299 | 222/299 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 222, neither 77, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on paws: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 185/250 | 74.0% | 68.2%–79.0% | 185/250 | 0.189 | 0.928 | 0.425 | 0 |
| rtx3090 | 185/250 | 74.0% | 68.2%–79.0% | 185/250 | 0.189 | 0.929 | 0.425 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| paws | 185/250 | 185/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 185, neither 65, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on multinli: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 299 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 269/299 | 90.0% | 86.0%–92.9% | 76/103 | 0.039 | 0.259 | 0.148 | 0 |
| rtx3090 | 269/299 | 90.0% | 86.0%–92.9% | 76/103 | 0.045 | 0.259 | 0.147 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| multinli-fiction | 54/60 | 54/60 |
| multinli-government | 54/61 | 54/61 |
| multinli-slate | 51/60 | 51/60 |
| multinli-telephone | 57/60 | 57/60 |
| multinli-travel | 53/58 | 53/58 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 269, neither 30, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on civil_comments: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 300 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 234/300 | 78.0% | 73.0%–82.3% | 234/300 | 0.087 | 0.500 | 0.317 | 0 |
| rtx3090 | 234/300 | 78.0% | 73.0%–82.3% | 234/300 | 0.087 | 0.501 | 0.319 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| civil_comments | 234/300 | 234/300 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.7% of records. Both correct 232, neither 64, only h100 2, only rtx3090 2. Exact McNemar two-sided p = 1.0000.


# nimble on aegis2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 206/250 | 82.4% | 77.2%–86.6% | 206/250 | 0.122 | 0.600 | 0.293 | 0 |
| rtx3090 | 206/250 | 82.4% | 77.2%–86.6% | 206/250 | 0.115 | 0.598 | 0.292 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| aegis2 | 206/250 | 206/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 100.0% of records. Both correct 206, neither 44, only h100 0, only rtx3090 0. Exact McNemar two-sided p = 1.0000.


# nimble on helpsteer2: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 249 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 85/249 | 34.1% | 28.5%–40.2% | 14/125 | 0.347 | 1.797 | 0.858 | 0 |
| rtx3090 | 86/249 | 34.5% | 28.9%–40.6% | 14/125 | 0.343 | 1.798 | 0.858 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| helpsteer2 | 85/249 | 86/249 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.6% of records. Both correct 85, neither 163, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# nimble on summeval-relevance: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 240 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 117/240 | 48.8% | 42.5%–55.0% | 0/15 | 0.175 | 1.241 | 0.685 | 0 |
| rtx3090 | 117/240 | 48.8% | 42.5%–55.0% | 0/15 | 0.174 | 1.241 | 0.686 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-relevance | 117/240 | 117/240 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.2% of records. Both correct 116, neither 122, only h100 1, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# nimble on summeval-consistency: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 144 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 119/144 | 82.6% | 75.6%–88.0% | 0/9 | 0.131 | 0.591 | 0.246 | 0 |
| rtx3090 | 120/144 | 83.3% | 76.4%–88.5% | 0/9 | 0.125 | 0.590 | 0.246 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| summeval-consistency | 119/144 | 120/144 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 98.6% of records. Both correct 119, neither 24, only h100 0, only rtx3090 1. Exact McNemar two-sided p = 1.0000.


# nimble on pubmedqa: H100 (Linux, CUDA 12.9) vs RTX 3090 (Windows, CUDA 13.4), Ollama format

All runs answered the same 250 records, joined by ID with identical reference labels. Labels come from the public dataset's human annotation; the subset was selected by a seeded, label-blind rule that keeps contrastive families together.

| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h100 | 195/250 | 78.0% | 72.5%–82.7% | 195/250 | 0.144 | 0.773 | 0.358 | 0 |
| rtx3090 | 193/250 | 77.2% | 71.6%–82.0% | 193/250 | 0.141 | 0.777 | 0.359 | 0 |

## Per domain

| Domain | h100 | rtx3090 |
|---|---:|---:|
| pubmedqa | 195/250 | 193/250 |

## Paired correctness

**h100 vs rtx3090**: predictions agree on 99.2% of records. Both correct 193, neither 55, only h100 2, only rtx3090 0. Exact McNemar two-sided p = 0.5000.
