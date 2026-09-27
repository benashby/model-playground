"""Download one evaluation corpus from its pinned upstream revision.

Run by the DVC stages in dvc.yaml, not by hand:

    uv run --extra corpora dvc repro            # fetch both from upstream
    uv run --extra corpora dvc pull             # or: from a configured mirror

Each corpus is pinned to an immutable upstream revision, and dvc.lock records
the md5 of every file this script produces. A later run that downloads
different bytes shows up as a change to dvc.lock, so the audio a published
number was measured on can always be checked.

The output directory is created from scratch and holds only the corpus
files, the upstream licence and README (attribution is a condition of both
licences), and nothing generated. Probes write their own output elsewhere.

    python audio/corpora/fetch.py harper_valley audio/corpora/harper_valley
    python audio/corpora/fetch.py apptek audio/corpora/apptek
"""

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HARPER_VALLEY = {
    # Gridspace-Stanford HarperValleyBank, CC-BY-4.0. Audio is committed
    # directly in the git repository (not LFS), so the commit tarball has it.
    "repo": "cricketclub/gridspace-stanford-harper-valley",
    "commit": "0bd721e877c4a85d8c13ff837e68661ea6200a98",
    "keep": ("LICENSE", "README.md", "data/"),
}

APPTEK = {
    # AppTek Call-Center Dialogues, CC-BY-SA-4.0. Only the accents the WER
    # probe uses; score.py and word_mappings.py are the corpus's own scorer.
    "repo": "apptek-com/apptek_callcenter_dialogues",
    "revision": "b98967d9946f7f59f58d08624a2a00fe98fe0219",
    "patterns": ["README.md", "score.py", "word_mappings.py",
                 "test/en-US_General/*", "test/en-IN/*", "test/en-GB_SCT/*"],
}


def fetch_harper_valley(out: Path) -> None:
    url = f"https://codeload.github.com/{HARPER_VALLEY['repo']}/tar.gz/{HARPER_VALLEY['commit']}"
    print(f"downloading {url}", flush=True)
    # Stream the tarball ("r|gz"): it is gigabytes, and nothing needs random access.
    with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not m.isfile() or not rel.startswith(HARPER_VALLEY["keep"]):
                continue
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def fetch_apptek(out: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"downloading {APPTEK['repo']}@{APPTEK['revision']}", flush=True)
    snapshot_download(APPTEK["repo"], repo_type="dataset", revision=APPTEK["revision"],
                      allow_patterns=APPTEK["patterns"], local_dir=out, max_workers=8)
    # huggingface_hub keeps download metadata in <local_dir>/.cache; it is not
    # corpus content and would make the recorded output differ between runs.
    shutil.rmtree(out / ".cache", ignore_errors=True)


def main() -> int:
    name, out = sys.argv[1], Path(sys.argv[2])
    fetchers = {"harper_valley": fetch_harper_valley, "apptek": fetch_apptek}
    if name not in fetchers:
        raise SystemExit(f"unknown corpus {name!r}; one of {sorted(fetchers)}")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    fetchers[name](out)
    n = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{name}: {n} files, {size / 1e9:.2f} GB in {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
