"""Licence tags and licence wording on the Hugging Face repos behind each model.

    python models/nimble-tev1/probes/licences.py > models/nimble-tev1/results/licences.log

Compare with the licence layers Ollama ships (results/registry.log). A tag is
the uploader's declaration; the card's own words are printed where they say
more than the tag.
"""
import json
import urllib.request

REPOS = ["bespokelabs/Bespoke-Nimble-9B", "bespokelabs/Bespoke-Nimble-9B-v2", "bespokelabs/Nimble-V3",
         "togethercomputer/Tev1-4B-experimental", "togethercomputer/Tev1-0.8B-experimental",
         "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"]

for repo in REPOS:
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60))
    tags = [t for t in d.get("tags", []) if t.startswith("license:")] or ["no licence tag"]
    print(f"{repo:42} {d.get('sha', '')[:10]}  {', '.join(tags)}  (modified {d.get('lastModified')})")
    card = urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/README.md", timeout=60).read().decode()
    lines = card.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() in ("## license", "## licence"):
            body = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            print(f"    card, License section: {body[:240]}")
