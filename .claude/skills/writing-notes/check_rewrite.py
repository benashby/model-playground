#!/usr/bin/env python3
"""Did a prose rewrite change anything that isn't prose?

Snapshot the files first, rewrite them, then run:

    python .claude/skills/writing-notes/check_rewrite.py <snapshot_dir>

<snapshot_dir> mirrors repo-relative paths (e.g. snapshot/models/x/README.md).
For each file it compares fenced code, inline code, link targets, the
[MEASURED]/[CLAIM] tags and every number in the prose, and counts dashes left
in prose. Hard failures (changed code, links or tags, or ANY added number) exit
non-zero. Lost numbers are warnings to review by hand: a number may move
between sentences, but it must not disappear.
"""
import re, sys, pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
BEFORE = pathlib.Path(sys.argv[1])

def split(text):
    fences, prose, cur, inf = [], [], [], False
    for l in text.splitlines():
        if l.lstrip().startswith("```"):
            if inf: fences.append("\n".join(cur)); cur = []
            inf = not inf; continue
        (cur if inf else prose).append(l)
    return fences, "\n".join(prose)

NUM = re.compile(r'(?<![\w.])[-+±~]?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|×|x\b|ms\b|s\b|GB|GiB|MiB|B\b|kHz))?')
def facts(prose):
    inline = Counter(re.findall(r'`[^`]+`', prose))
    bare = re.sub(r'`[^`]+`', ' ', prose)
    links = Counter(re.findall(r'\]\(([^)]+)\)', bare))
    nums = Counter(n.replace(" ", "") for n in NUM.findall(re.sub(r'\]\([^)]+\)', ' ', bare)))
    tags = Counter(re.findall(r'\[(MEASURED|CLAIM)\]', bare))
    dashes = len(re.findall(r'[—–]| -- ', bare))
    return inline, links, nums, tags, dashes

bad = 0
for b in sorted(BEFORE.rglob("*.md")):
    rel = b.relative_to(BEFORE); a = ROOT / rel
    fb, pb = split(b.read_text()); fa, pa = split(a.read_text())
    ib, lb, nb, tb, db = facts(pb); ia, la, na, ta, da = facts(pa)
    probs = []
    if fb != fa: probs.append(f"FENCED CODE CHANGED ({len(fb)} -> {len(fa)} blocks)")
    if ib != ia: probs.append(f"inline code: -{dict(ib-ia)} +{dict(ia-ib)}")
    if lb != la: probs.append(f"links: -{dict(lb-la)} +{dict(la-lb)}")
    if tb != ta: probs.append(f"[MEASURED]/[CLAIM] tags: {dict(tb)} -> {dict(ta)}")
    lost, added = nb - na, na - nb
    if lost or added: probs.append(f"numbers lost {dict(lost)} added {dict(added)}")
    hard = any(p.startswith(("FENCED","inline","links","[MEAS")) for p in probs) or added
    print(f"{'FAIL' if hard else ('warn' if probs else ' ok ')}  {rel}  dashes {db}->{da}  lines {len(b.read_text().splitlines())}->{len(a.read_text().splitlines())}")
    for p in probs: print("        ", p)
    bad |= bool(hard) or da > 0
sys.exit(1 if bad else 0)
