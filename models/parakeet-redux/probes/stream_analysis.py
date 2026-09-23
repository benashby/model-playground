"""Derive every streaming-stability figure in the note from the committed snapshot logs.

Reads models/parakeet-redux/results/asr-<fixture>-run<k>.jsonl (written by
pk_stream.py). No model is run. For each log it computes:

  - snapshot count, first-snapshot latency since first audio, and the
    interval between consecutive snapshots (min / median / max);
  - normalized erasure and unstable-word ratio via `playground.asr.stability`;
  - every REVISION: a pair of consecutive snapshots where the earlier one is
    not a word-prefix of the later one. Each rewritten word is classified:
        punct   : same word once punctuation and case are ignored
                  ("Hello" -> "Hello.", "50" -> "50?")
        frontier: the last word of the earlier snapshot, completed or extended
                  in the later one ("rand" -> "random")
        other   : anything else, i.e. a real change of a settled word;
    and the depth of the revision (how many words back from the end of the
    earlier snapshot the first changed word was). Words after the first
    change that are unchanged are not counted as rewritten;
  - "settled accuracy": over every non-final snapshot, the fraction of words
    at least K words back from that snapshot's end that equal the final
    transcript's word at the same position, for K = 0, 1, 2, 3, both exactly
    and ignoring punctuation and case. K=2 answers "an agent acting only on
    words more than two back from the frontier would have been right X% of
    the time".

It also prints the first revisions of tool_call run 1 in the format the note
quotes.

Run from the repo root:

    uv run python models/parakeet-redux/probes/stream_analysis.py \
        > models/parakeet-redux/results/stream_analysis.log
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from playground.asr import stability  # noqa: E402

RESULTS = Path("models/parakeet-redux/results")


def bare(w: str) -> str:
    return re.sub(r"[^\w']", "", w).lower()


def load(path: Path):
    ev = [json.loads(ln) for ln in path.read_text().splitlines()]
    snaps = [(e["since_audio"], e["text"]) for e in ev if e["kind"] == "asr.snapshot"]
    final = [e for e in ev if e["kind"] == "asr.final"][0]["text"]
    return snaps, final


def revisions(snaps):
    out = []
    seqs = [t.split() for _, t in snaps]
    for idx, (prev, nxt) in enumerate(zip(seqs, seqs[1:])):
        k = 0
        while k < len(prev) and k < len(nxt) and prev[k] == nxt[k]:
            k += 1
        if k == len(prev):
            continue  # prev is a prefix of nxt: pure append, no revision
        kinds = []
        for i in range(k, len(prev)):
            a = prev[i]
            b = nxt[i] if i < len(nxt) else None
            if b is not None and a == b:
                continue  # unchanged word after the first change: not a rewrite
            if b is not None and bare(a) == bare(b):
                kinds.append("punct")
            elif i == len(prev) - 1 and b is not None and bare(b).startswith(bare(a)):
                kinds.append("frontier")
            else:
                kinds.append("other")
        out.append({
            "t": snaps[idx + 1][0], "from_word": k, "depth": len(prev) - k,
            "changed": [(i, prev[i], nxt[i] if i < len(nxt) else None)
                        for i in range(k, len(prev)) if i >= len(nxt) or prev[i] != nxt[i]],
            "was": " ".join(prev[k:]), "now": " ".join(nxt[k:len(prev) + 6]),
            "kinds": kinds,
        })
    return out


def settled(snaps, final_text, K):
    final = final_text.split()
    tot = exact = loose = 0
    for _, t in snaps[:-1]:
        s = t.split()
        for i in range(max(0, len(s) - K)):
            tot += 1
            if i < len(final):
                exact += s[i] == final[i]
                loose += bare(s[i]) == bare(final[i])
    return tot, exact, loose


logs = sorted(RESULTS.glob("asr-*-run*.jsonl"))
print(f"{len(logs)} snapshot logs")
agg_kinds: Counter = Counter()
agg_depth: Counter = Counter()
agg_settled = {K: [0, 0, 0] for K in (0, 1, 2, 3)}
per_fixture_settled: dict[str, dict[int, list[int]]] = {}
firsts, intervals_all = [], []
for p in logs:
    stem, run = re.match(r"asr-(.+)-run(\d+)\.jsonl", p.name).groups()
    snaps, final = load(p)
    st = stability(snaps)
    iv = [b[0] - a[0] for a, b in zip(snaps, snaps[1:])]
    intervals_all += iv
    firsts.append(snaps[0][0])
    revs = revisions(snaps)
    kinds = Counter(k for r in revs for k in r["kinds"])
    agg_kinds.update(kinds)
    agg_depth.update(r["depth"] for r in revs)
    print(f"\n=== {p.name}")
    print(f"  snapshots={st['snapshots']} first_since_audio={snaps[0][0]:.2f}s "
          f"interval min/med/max {min(iv):.2f}/{statistics.median(iv):.2f}/{max(iv):.2f}s "
          f"NE={st['normalized_erasure']} UPWR={st['unstable_word_ratio']} "
          f"final_words={st['final_words']} final==last_snapshot={final == snaps[-1][1]}")
    print(f"  revisions={len(revs)} rewritten-word kinds={dict(kinds)} "
          f"depths={sorted(r['depth'] for r in revs)}")
    for K in (0, 1, 2, 3):
        tot, ex, lo = settled(snaps, final, K)
        for d in (agg_settled[K], per_fixture_settled.setdefault(stem, {}).setdefault(K, [0, 0, 0])):
            d[0] += tot
            d[1] += ex
            d[2] += lo
        print(f"  words >= {K} back from frontier: {tot:5d} shown, "
              f"{ex / tot * 100:6.2f}% exact final, {lo / tot * 100:6.2f}% ignoring punct/case")
    for r in revs:
        if r["depth"] > 2:
            print(f"  DEEP revision t={r['t']:.2f} from word {r['from_word']} depth {r['depth']}:"
                  f" changed words {r['changed']}")
    others = [r for r in revs if "other" in r["kinds"]]
    for r in others:
        print(f"  OTHER t={r['t']:.2f} from word {r['from_word']} depth {r['depth']}: "
              f"was: {r['was']!r} now: {r['now']!r}")

print("\n=== pooled over all logs")
print(f"  first snapshot since audio: {min(firsts):.2f}-{max(firsts):.2f}s, "
      f"median {statistics.median(firsts):.2f}s, n={len(firsts)}")
print(f"  snapshot interval: min {min(intervals_all):.2f} median {statistics.median(intervals_all):.2f}"
      f" max {max(intervals_all):.2f}s, n={len(intervals_all)}")
print(f"  rewritten-word kinds: {dict(agg_kinds)}")
print(f"  revision depth (words back from the end of the earlier snapshot): {dict(sorted(agg_depth.items()))}")
for K, (tot, ex, lo) in agg_settled.items():
    print(f"  words >= {K} back: {tot} shown, {ex / tot * 100:.2f}% exact, {lo / tot * 100:.2f}% ignoring punct/case")
print("  per fixture, words >= 2 back:")
for stem, d in per_fixture_settled.items():
    tot, ex, lo = d[2]
    print(f"    {stem:14s} {tot:5d} shown, {ex / tot * 100:.2f}% exact, {lo / tot * 100:.2f}% ignoring punct/case")

print("\n=== excerpt: first revisions of tool_call run 1")
snaps, _ = load(RESULTS / "asr-tool_call-run1.jsonl")
for r in revisions(snaps)[:4]:
    print(f"t={r['t']:.2f}  revised from word {r['from_word']}  ({', '.join(r['kinds'])})")
    print(f"   was: {r['was']}")
    print(f"   now: {r['now']}")
