"""Shared text normaliser and word-level edit distance for the Parakeet probes.

Not a probe on its own. `wer_voicechat.py`, `resample.py`, `compare_models.py`
and the other probes import it so that every WER-style number in the note is
scored the same way. Running it directly runs its self-test:

    uv run python models/parakeet-redux/probes/textnorm.py

The normalisation, applied identically to BOTH sides of every comparison:

1. Lowercase.
2. Hyphens and slashes become spaces ("twenty-five" -> "twenty five").
3. Thousands separators inside digit groups are dropped ("1,000" -> "1000");
   a decimal point between digits is kept ("3.5").
4. Every other character except a-z, 0-9, apostrophe and "." between digits
   becomes a space.
5. Runs of English cardinal number words are converted to one digit string:
   "one" -> "1", "eleven" -> "11", "twenty five" -> "25",
   "one hundred and five" -> "105", "two thousand twenty" -> "2020".
   "and" is absorbed only directly after "hundred" or "thousand" and only when
   another number word follows, so "between one and fifty" stays
   "between 1 and 50". Two small numbers in a row are not summed:
   "one two three" -> "1 2 3".

Known limits, stated rather than hidden: ordinals ("first"), "a hundred",
"oh" for zero, and "one" used as a pronoun are all left as written or
converted literally, on both sides alike. Filler words ("um", "uh") are kept
by default; `strip_fillers=True` removes them.

This replaces the normaliser that used to live in `pk_exp.py`, which mapped
only 0-10 and a few tens, turned "twenty-five" into "20 5", and could not
produce 11-19 at all.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "ah", "eh"}


def _is_num(w: str) -> bool:
    return w in UNITS or w in TENS or w in SCALES


def _numbers_to_digits(words: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if not _is_num(w) or w in SCALES:
            out.append(w)
            i += 1
            continue
        total, current, last = 0, 0, None  # last: "unit" | "tens" | "scale"
        while i < len(words):
            w = words[i]
            if w == "and" and last == "scale" and i + 1 < len(words) and (
                words[i + 1] in UNITS or words[i + 1] in TENS
            ):
                i += 1
                continue
            if w in UNITS:
                v = UNITS[w]
                # a unit may follow a tens word (twenty five) only if < 10,
                # or start the sub-hundred slot when that slot is empty
                if last == "tens" and v < 10:
                    current += v
                elif current % 100 == 0 and last in (None, "scale"):
                    current += v
                else:
                    break
                last = "unit"
            elif w in TENS:
                if current % 100 == 0 and last in (None, "scale"):
                    current += TENS[w]
                else:
                    break
                last = "tens"
            elif w in SCALES:
                s = SCALES[w]
                if s == 100:
                    current = (current or 1) * 100
                else:
                    total += (current or 1) * s
                    current = 0
                last = "scale"
            else:
                break
            i += 1
        out.append(str(total + current))
    return out


def normalise(text: str, *, strip_fillers: bool = False) -> list[str]:
    s = text.lower()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x00", s)
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"[^a-z0-9'\x00 ]", " ", s)
    s = s.replace("\x00", ".")
    words = [w.strip("'") for w in s.split()]
    words = [w for w in words if w]
    words = _numbers_to_digits(words)
    if strip_fillers:
        words = [w for w in words if w not in FILLERS]
    return words


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment: list of (op, ref_word, hyp_word), op in = S D I."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def wer(ref_text: str, hyp_text: str, *, strip_fillers: bool = False) -> dict:
    """Word error rate of hyp against ref, with the S/D/I breakdown."""
    r = normalise(ref_text, strip_fillers=strip_fillers)
    h = normalise(hyp_text, strip_fillers=strip_fillers)
    ops = align(r, h)
    s = sum(o == "S" for o, _, _ in ops)
    dl = sum(o == "D" for o, _, _ in ops)
    ins = sum(o == "I" for o, _, _ in ops)
    return {
        "wer": (s + dl + ins) / len(r) if r else float("nan"),
        "errors": s + dl + ins,
        "ref_words": len(r),
        "S": s, "D": dl, "I": ins,
        "ops": ops,
    }


def errors_only(ops) -> list[str]:
    """Human-readable list of the non-matching alignment steps."""
    out = []
    for k, (o, r, h) in enumerate(ops):
        if o == "S":
            out.append(f"@{k} S {r!r}->{h!r}")
        elif o == "D":
            out.append(f"@{k} D {r!r}")
        elif o == "I":
            out.append(f"@{k} I {h!r}")
    return out


SELF_TEST = [
    ("between one and fifty", "between 1 and 50"),
    ("Between 1 and 50?", "between 1 and 50"),
    ("twenty-five", "25"),
    ("eleven twelve nineteen", "11 12 19"),
    ("one hundred and five", "105"),
    ("two thousand twenty", "2020"),
    ("one two three", "1 2 3"),
    ("ninety nine", "99"),
    ("fifty five hundred", "5500"),
    ("It's 1,000 dollars, or 3.5%", "it's 1000 dollars or 3.5"),
    ("Um, can you", "um can you"),
]


def self_test() -> None:
    for src, want in SELF_TEST:
        got = " ".join(normalise(src))
        status = "ok" if got == want else "FAIL"
        print(f"  {status:4s} {src!r:36s} -> {got!r}  (want {want!r})")
        assert got == want, (src, got, want)
    assert normalise("Um, can you", strip_fillers=True) == ["can", "you"]
    print("  ok   strip_fillers removes 'um'")


if __name__ == "__main__":
    self_test()
