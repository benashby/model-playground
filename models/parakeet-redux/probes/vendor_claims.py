"""Fetch the vendor pages the note quotes, and print each quoted claim in context.

Every [CLAIM] figure in the Parakeet note is checked here against the live
source rather than copied from memory. For each source the probe prints the
revision or date it saw, then every line (or sentence, for HTML pages) that
matches the patterns for the claims the note cites. If a claim disappears from
a page, its section prints "NOT FOUND", which is the signal to revisit the note.

Sources (network):
  - HF model cards (raw README.md) and self-reported eval YAML for
    moondream/parakeet-redux and moondream/parakeet-ultra, and the model card
    of nvidia/parakeet-tdt-0.6b-v3;
  - https://docs.moondream.ai/transcription
  - the release post, https://moondream.ai/blog/introducing-parakeet-redux-and-ultra
  - "Photon is now free", https://moondream.ai/blog/photon-1-3-0-update
  - the Photon 2.0 launch post, https://moondream.ai/blog/photon-2-launch
  - https://moondream.ai/photon (FAQ) and https://moondream.ai/terms

Run from the repo root:

    python models/parakeet-redux/probes/vendor_claims.py \
        > models/parakeet-redux/results/vendor_claims.log
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import date

UA = {"User-Agent": "Mozilla/5.0 (research notebook probe)"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def page_text(url: str) -> str:
    t = fetch(url)
    t = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", " ", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def grep_lines(text: str, patterns: list[str]) -> None:
    lines = text.splitlines()
    for pat in patterns:
        hits = [ln.strip() for ln in lines if re.search(pat, ln, re.I)]
        print(f"  /{pat}/: " + ("NOT FOUND" if not hits else ""))
        for h in hits[:6]:
            print(f"    | {h[:220]}")


def grep_sentences(text: str, patterns: list[str], width: int = 170) -> None:
    for pat in patterns:
        hits = [m for m in re.finditer(pat, text, re.I)]
        print(f"  /{pat}/: " + ("NOT FOUND" if not hits else ""))
        for m in hits[:3]:
            a, b = max(0, m.start() - width), min(len(text), m.end() + width)
            print(f"    | ...{text[a:b].strip()}...")


print(f"fetched {date.today().isoformat()}")

for repo, pats in [
    ("moondream/parakeet-redux", [
        r"113", r"Open ASR Leaderboard, 7 English", r"FLEURS, 25", r"Business speech",
        r"Background noise, 9", r"TED-LIUM long-form", r"^\| Weights", r"EPYC", r"AVX-512",
        r"sherpa-onnx", r"onnx-asr", r"parakeet\.cpp", r"substitutes similar", r"Dropped or invented",
        r"segmenter cuts", r"at most 30 seconds", r"license",
    ]),
    ("moondream/parakeet-ultra", [
        r"Open ASR Leaderboard, 7 English", r"FLEURS, 25", r"Business speech",
        r"Background noise, 9", r"TED-LIUM long-form", r"B200", r"license",
    ]),
    ("nvidia/parakeet-tdt-0.6b-v3", [
        r"up to 24 minutes", r"license", r"RTFx",
    ]),
]:
    info = json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))
    print(f"\n=== HF card {repo} (revision {info.get('sha', '?')[:12]}, lastModified {info.get('lastModified')})")
    grep_lines(fetch(f"https://huggingface.co/{repo}/raw/main/README.md"), pats)
    if repo.startswith("moondream/"):
        y = fetch(f"https://huggingface.co/{repo}/raw/main/.eval_results/open_asr_leaderboard.yaml")
        print(f"  eval YAML header and mean_wer:")
        for ln in y.splitlines():
            if ln.startswith("#"):
                print(f"    | {ln}")
        m = re.search(r"task_id: mean_wer\s+value: ([\d.]+)", y)
        print(f"    | mean_wer value: {m.group(1) if m else 'NOT FOUND'}")

for url, pats in [
    ("https://docs.moondream.ai/transcription", [
        r"on supported NVIDIA GPUs", r"Multichannel files are downmixed[^.]*\.",
        r"Parakeet detects language implicitly[^.]*\.", r"Use an asynchronous PCM iterator[^.]*\.",
        r"limited to 24 hours", r"\"none\" , \"segment\" , or \"word\"",
        r"Word entries include[^.]*\.", r"language : the detected[^;]*;",
    ]),
    ("https://moondream.ai/blog/introducing-parakeet-redux-and-ultra", [
        r"(January|February|March|April|May|June|July|August|September|October|November|December) \d+, 2026",
        r"One user has already moved[^.]*\.", r"Live previews begin[^;]*;",
        r"pass in the whole recording[^.]*\.", r"memory bandwidth[^.]*\.",
        r"minute",
    ]),
    ("https://moondream.ai/blog/photon-1-3-0-update", [
        r"(January|February|March|April|May|June|July|August|September|October|November|December) \d+, 2026",
        r"running Moondream locally with Photon is totally free[^.]*\.", r"No API key required[^.]*\.",
    ]),
    ("https://moondream.ai/blog/photon-2-launch", [
        r"(January|February|March|April|May|June|July|August|September|October|November|December) \d+, 2026",
        r"The Photon inference engine is Apache 2\.0[^.]*\.[^.]*\.",
    ]),
    ("https://moondream.ai/photon", [r"Is it free\?[^.]*\.[^.]*\."]),
    ("https://moondream.ai/terms", [r"access to and use of the Moondream Cloud Service[^.]*\."]),
]:
    print(f"\n=== {url}")
    try:
        grep_sentences(page_text(url), pats)
    except Exception as e:  # noqa: BLE001
        print(f"  fetch failed: {type(e).__name__}: {e}")
