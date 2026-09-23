#!/usr/bin/env python3
"""Does a cheaper configuration still give the same answers?

Written to compare FP8 weights and an FP8 KV cache against the BF16 baseline.
Speed is measured by k2_bench.py; this measures what speed might cost:

  answers   24 short questions with one exact answer each (arithmetic, units,
            facts, simple logic), three runs each at the vendor's sampling
            (temperature 1.0, top_p 0.95) and medium effort. A sanity check
            for gross damage, not a benchmark: it cannot see small quality
            differences.
  tools     the weather-tool request, at low and at high effort, with the
            default `xml` format. Tool-call reliability was the most fragile
            behaviour found on the baseline.
  needle    a random six-digit code planted at three depths in a long filler
            prompt, then asked for. Whether the model can use context that far
            back had not been tested at all, and it is exactly what a
            lower-precision KV cache could damage.

    PLAYGROUND_HOST=<host> NEEDLE_TOKENS=480000 \
        python models/k2-horizon-32b/probes/k2_quality.py [answers] [tools] [needle]
"""
from __future__ import annotations

import os
import re
import secrets
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from k2_bench import WEATHER, filler, nonce, stream_chat  # noqa: E402

QA = [
    # Each answer is a string, or a tuple of accepted forms ("6" or "six").
    ("What is 17 multiplied by 23?", "391"),
    ("What is 1234 plus 5678?", "6912"),
    ("What is 144 divided by 12?", ("12", "twelve")),
    ("What is 2 to the power of 10?", "1024"),
    ("What is 15% of 240?", "36"),
    ("What is the square root of 169?", ("13", "thirteen")),
    ("How many minutes are in 3.5 hours?", "210"),
    ("How many seconds are in one day?", "86400"),
    ("A train leaves at 14:05 and arrives at 17:40. How many minutes is the journey?", "215"),
    ("How many centimetres are in 2.5 metres?", "250"),
    ("What is 9 degrees Celsius above freezing, in Kelvin, rounded to the nearest whole number?", "282"),
    ("How many days are in a leap year?", "366"),
    ("What is the capital of Australia?", "canberra"),
    ("What is the capital of Canada?", "ottawa"),
    ("What is the chemical symbol for gold?", "au"),
    ("Which planet is closest to the Sun?", "mercury"),
    ("How many sides does a hexagon have?", ("6", "six")),
    ("What is the largest ocean on Earth?", "pacific"),
    ("In which year did the first human land on the Moon?", "1969"),
    ("How many bits are in a byte?", ("8", "eight")),
    ("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.", "yes"),
    ("Alice is taller than Bob, and Bob is taller than Carol. Who is the shortest?", "carol"),
    ("What is the next number in the sequence 2, 4, 8, 16?", ("32", "thirty two")),
    ("What is 7 factorial?", "5040"),
]
RUNS = int(os.environ.get("QA_RUNS", "3"))


def norm(text: str) -> str:
    # Thousands separators are removed; every other non-alphanumeric becomes a
    # space. The model writes a narrow no-break space (U+202F) between a number
    # and its unit, "250\u202fcm"; an earlier version deleted it, which turned
    # that into "250cm" and scored 13 correct answers as wrong.
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def answers_section() -> None:
    print(f"### ANSWERS: {len(QA)} questions x {RUNS} runs, effort medium, temperature 1.0, top_p 0.95")
    right, wrong = 0, []
    tokens = []
    for q, a in QA:
        for _ in range(RUNS):
            r = stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] {q} Answer briefly."}],
                             "reasoning_effort": "medium", "temperature": 1.0, "top_p": 0.95,
                             "max_tokens": 3000})
            tokens.append(r["usage"].get("completion_tokens", 0))
            accepted = a if isinstance(a, tuple) else (a,)
            if any(re.search(rf"\b{re.escape(x)}\b", norm(r["content"])) for x in accepted):
                right += 1
            else:
                wrong.append((q[:50], a, r["content"].strip()[:60]))
    n = len(QA) * RUNS
    print(f"  correct {right}/{n} ({100 * right / n:.1f}%)   completion tokens median {st.median(tokens)}")
    for w in wrong:
        print(f"      wrong: {w}")


def tools_section() -> None:
    n = int(os.environ.get("TOOL_REPEATS", "10"))
    print(f"### TOOLS: weather request, default xml format, n={n} per effort")
    for effort in ("low", "high"):
        ok = 0
        for _ in range(n):
            r = stream_chat({"messages": [{"role": "user", "content": f"[{nonce()}] What's the weather in Oslo, in celsius?"}],
                             "tools": [WEATHER], "tool_choice": "auto", "reasoning_effort": effort,
                             "max_tokens": 4000})
            ok += bool(r["tool_calls"]) and r["tool_calls"][0]["name"] == "get_weather" \
                and "oslo" in r["tool_calls"][0]["arguments"].lower()
        print(f"  effort {effort:<5} structured and correct {ok}/{n}")


def haystack(total: int) -> list[str]:
    """Words for the needle prompt, about one token each.

    NEEDLE_TEXT=<path to a plain-text book> uses natural prose, repeated if the
    book is too short. Without it the filler is random draws from ten words.
    That synthetic filler was the first haystack tried, and at 475k tokens the
    model degenerated into word salad; natural text separates "cannot use long
    context" from "cannot cope with ten words repeated half a million times".
    The runs here used Project Gutenberg's War and Peace (ebook 2600, public
    domain), which is about 1.3 tokens per word, so it is cut by token budget.
    """
    path = os.environ.get("NEEDLE_TEXT")
    if not path:
        return filler(total).split()
    words = open(path, encoding="utf-8", errors="replace").read().split()
    budget = int(total / 1.3)
    reps = budget // len(words) + 1
    return (words * reps)[:budget]


def needle_section() -> None:
    lengths = [int(x) for x in os.environ.get("NEEDLE_TOKENS", "480000").split(",")]
    depths = [float(x) for x in os.environ.get("NEEDLE_DEPTHS", "0.1,0.5,0.9").split(",")]
    kind = "natural text" if os.environ.get("NEEDLE_TEXT") else "synthetic ten-word filler"
    print(f"### NEEDLE: a six-digit code in {kind}, lengths {lengths}, depths {depths}, effort low")
    for total in lengths:
        for depth in depths:
            code = f"{secrets.randbelow(900000) + 100000}"
            words = haystack(total)
            at = int(len(words) * depth)
            words[at:at] = f"The access code for the vault is {code}. Remember it.".split()
            prompt = (f"[{nonce()}] " + " ".join(words) +
                      "\n\nWhat is the access code for the vault? Reply with the six digits only.")
            r = stream_chat({"messages": [{"role": "user", "content": prompt}],
                             "reasoning_effort": "low", "max_tokens": 2000})
            got = re.findall(r"\d{6}", r["content"] + " " + r["reasoning"])
            # Record why a reply is empty, not just that it is: a first run at
            # max_tokens=200 came back empty three times with nothing to say why.
            print(f"  depth {depth:.0%}  prompt_tokens {r['usage'].get('prompt_tokens', 0):>7,}  "
                  f"found: {code in got}  in content: {code in r['content']}  "
                  f"finish={r['finish']}  completion_tokens={r['usage'].get('completion_tokens')}  "
                  f"time to first token {r['t_first_any'] or 0:.1f}s")
            print(f"      reply {r['content'].strip()[:60]!r}  reasoning tail {r['reasoning'].strip()[-120:]!r}")


if __name__ == "__main__":
    print("host=$PLAYGROUND_HOST model=k2-horizon-32b  (address redacted by design)\n")
    sections = {"answers": answers_section, "tools": tools_section, "needle": needle_section}
    for name in (sys.argv[1:] or list(sections)):
        sections[name]()
        print()
