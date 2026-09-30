"""Fetch the upstream data behind Bespoke's public benchmark and rebuild all 13 subsets.

    uv run --no-project --with pyarrow --with transformers --with jinja2 \
        python models/nimble-tev1/probes/fetch_public_data.py --nimble-src $NIMBLE_SRC

Writes into $NIMBLE_SRC/data/public/, which that repository gitignores, never
into this one: the records are third-party text under their own licences.

Each raw file is written the way Bespoke describes (docs/PUBLIC_BENCHMARKS.md):
the Hugging Face parquet export as JSON Lines with the original field names;
MASSIVE's tarball and the VitaminC and MultiNLI zips from their original
hosts. Each subset is then built with the exact converter command from that
document, and its source_sha256 and dataset_sha256 are compared with the
manifest Bespoke committed. A subset that does not match is reported, not
used silently: a different payload means a different benchmark.

Four subsets need the Qwen3.5-9B tokenizer for the length filter. Its files
are fetched at the revision Nimble was trained from.
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HF = "https://huggingface.co"
QWEN = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")

# raw file -> how to obtain it
PARQUET = {
    "boolq/validation.jsonl": ("google/boolq", "default", "validation"),
    "squad_v2/validation.jsonl": ("rajpurkar/squad_v2", "squad_v2", "validation"),
    "paws/test.jsonl": ("google-research-datasets/paws", "labeled_final", "test"),
    "civil_comments/test.jsonl": ("google/civil_comments", "default", "test"),
    "aegis2/test.jsonl": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "test"),
    "helpsteer2/validation.jsonl": ("nvidia/HelpSteer2", "default", "validation"),
    "summeval/test.jsonl": ("mteb/summeval", "default", "test"),
    "pubmedqa/train.jsonl": ("qiaojin/PubMedQA", "pqa_labeled", "train"),
}
VITAMINC = "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip"
MASSIVE = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MULTINLI = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

# (subset dir, dataset module, raw source, extra args); commands from PUBLIC_BENCHMARKS.md
BUILDS = [
    ("vitaminc-dev", "vitaminc", "vitaminc/dev.jsonl", ["--limit", "600"]),
    ("massive-en-US", "massive", "massive/amazon-massive-dataset-1.1.tar.gz", ["--subset", "en-US", "--limit", "350"]),
    ("massive-de-DE", "massive", "massive/amazon-massive-dataset-1.1.tar.gz",
     ["--subset", "de-DE", "--ids-from", "data/public/massive-en-US/manifest.json"]),
    ("boolq", "boolq", "boolq/validation.jsonl", ["--limit", "300"]),
    ("squad2", "squad2", "squad_v2/validation.jsonl", ["--limit", "300"]),
    ("paws", "paws", "paws/test.jsonl", ["--limit", "250"]),
    ("multinli", "multinli", "multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl", ["--limit", "300"]),
    ("civil_comments", "civil_comments", "civil_comments/test.jsonl", ["--limit", "300"]),
    ("aegis2", "aegis2", "aegis2/test.jsonl", ["--limit", "250", "TOKENIZER"]),
    ("helpsteer2", "helpsteer2", "helpsteer2/validation.jsonl", ["--limit", "250", "TOKENIZER"]),
    ("summeval-relevance", "summeval", "summeval/test.jsonl", ["--subset", "relevance", "--limit", "250", "TOKENIZER"]),
    ("summeval-consistency", "summeval", "summeval/test.jsonl", ["--subset", "consistency", "--limit", "150", "TOKENIZER"]),
    ("pubmedqa", "pubmedqa", "pubmedqa/train.jsonl", ["--limit", "250"]),
]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "probe"}), timeout=600) as r:
        return r.read()


def parquet_to_jsonl(repo, config, split, out):
    import pyarrow.parquet as pq
    urls = json.loads(fetch(f"{HF}/api/datasets/{repo}/parquet/{config}/{split}"))
    with open(out, "w", encoding="utf-8") as f:
        for url in urls:
            for row in pq.read_table(io.BytesIO(fetch(url))).to_pylist():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_files(raw):
    for rel, (repo, config, split) in PARQUET.items():
        out = raw / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"  fetching {repo} {config}/{split}", file=sys.stderr)
            parquet_to_jsonl(repo, config, split, out)
    if not (raw / "vitaminc/dev.jsonl").exists():
        (raw / "vitaminc").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(VITAMINC))) as z:
            name = next(n for n in z.namelist() if n.endswith("dev.jsonl"))
            (raw / "vitaminc/dev.jsonl").write_bytes(z.read(name))
    if not (raw / "massive/amazon-massive-dataset-1.1.tar.gz").exists():
        (raw / "massive").mkdir(parents=True, exist_ok=True)
        (raw / "massive/amazon-massive-dataset-1.1.tar.gz").write_bytes(fetch(MASSIVE))
    target = raw / "multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(MULTINLI))) as z:
            target.write_bytes(z.read("multinli_1.0/multinli_1.0_dev_matched.jsonl"))


def tokenizer_dir(root):
    out = root / "tokenizer-qwen3.5-9b"
    if (out / "tokenizer.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    repo, rev = QWEN
    info = json.loads(fetch(f"{HF}/api/models/{repo}/revision/{rev}"))
    for s in info["siblings"]:
        name = s["rfilename"]
        if name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "chat_template")) \
                or name in ("config.json", "generation_config.json"):
            (out / name).write_bytes(fetch(f"{HF}/{repo}/resolve/{rev}/{name}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimble-src", required=True, type=Path)
    ap.add_argument("--only", help="comma-separated subset names")
    a = ap.parse_args()
    src = a.nimble_src.resolve()
    raw = src / "data/public/raw"
    raw_files(raw)
    tok = None
    bad = 0
    for subset, module, source, extra in BUILDS:
        if a.only and subset not in a.only.split(","):
            continue
        if "TOKENIZER" in extra:
            tok = tok or tokenizer_dir(src / "data/public")
            extra = [x for x in extra if x != "TOKENIZER"] + ["--tokenizer", str(tok)]
        out = src / "data/public" / subset
        if not (out / "manifest.json").exists():
            if out.exists():
                shutil.rmtree(out)
            cmd = [sys.executable, "-m", "nimble.datasets.public_benchmarks", "--dataset", module,
                   "--source", str(raw / source), "--output-dir", f"data/public/{subset}", *extra]
            done = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
            if done.returncode:
                print(f"{subset}: converter failed\n{done.stderr[-2000:]}")
                bad += 1
                continue
        mine = json.loads((out / "manifest.json").read_text())
        ref = json.loads((src / f"docs/assets/public-benchmarks/subsets/{subset}-manifest.json").read_text())
        verdict = {k: mine.get(k) == ref.get(k) for k in ("source_sha256", "dataset_sha256", "count")}
        bad += not all(verdict.values())
        print(f"{subset:21} count {mine.get('count'):4}  source {'ok' if verdict['source_sha256'] else 'DIFFERS'}"
              f"  dataset {'ok' if verdict['dataset_sha256'] else 'DIFFERS'}  "
              f"{hashlib.sha256((out / 'all.jsonl').read_bytes()).hexdigest()[:12]}")
    print("all subsets match Bespoke's manifests" if not bad else f"{bad} subset(s) differ or failed")
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
