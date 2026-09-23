"""Inventory the licence of every artifact in Parakeet Redux's runtime path.

For each installed distribution that `pip install moondream` pulls in for
Photon (moondream, kestrel, kestrel-native, kestrel-kernels and its four
bundles) it prints:

  - the version, the `License` / `License-Expression` metadata fields and any
    `License ::` classifier, as declared in the installed METADATA;
  - every licence-like file shipped in the dist-info (LICENSE*, NOTICE*,
    COPYING*, licenses/, sboms/), with its size in bytes and sha256;
  - any "License" section in the package README (the long description);
  - the installed size of the import package, excluding __pycache__;
  - the size of the wheel on PyPI for this platform (network: pypi.org JSON).

It then prints the clauses the note quotes from the kestrel-kernels LICENSE,
verbatim, by searching for them in the installed file, so the quotation can be
checked against the shipped text.

Also fetches the licence field from the HuggingFace model API for the three
checkpoints (network: huggingface.co).

Run from the repo root:

    uv run --extra asr python models/parakeet-redux/probes/licensing.py \
        > models/parakeet-redux/results/licensing.log

Prints package-relative paths only, never a path under /home.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as im
import json
import re
import urllib.request
from pathlib import Path

DISTS = [
    ("moondream", "moondream"),
    ("kestrel", "kestrel"),
    ("kestrel-native", "kestrel_native"),
    ("kestrel-kernels", "kestrel_kernels"),
    ("kestrel-kernels-bundle-a", "kestrel_kernels_bundle_a"),
    ("kestrel-kernels-bundle-b", "kestrel_kernels_bundle_b"),
    ("kestrel-kernels-bundle-c", "kestrel_kernels_bundle_c"),
    ("kestrel-kernels-bundle-d", "kestrel_kernels_bundle_d"),
]
LICENCE_RE = re.compile(r"(licen[cs]e|notice|copying|sbom)", re.I)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def pkg_size(root: Path) -> int:
    return sum(
        p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )


for dist_name, pkg in DISTS:
    dist = im.distribution(dist_name)
    meta = dist.metadata
    print(f"=== {dist_name} {dist.version}")
    print(f"  License field        : {(meta.get('License') or '<absent>').splitlines()[0]}")
    print(f"  License-Expression   : {meta.get('License-Expression') or '<absent>'}")
    classifiers = [c for c in meta.get_all("Classifier") or [] if "License" in c]
    print(f"  License classifiers  : {classifiers or '<none>'}")
    files = [f for f in (dist.files or []) if ".dist-info" in str(f) and LICENCE_RE.search(str(f))]
    if not files:
        print("  licence files        : <none in dist-info>")
    for f in files:
        p = Path(dist.locate_file(f))
        b = p.read_bytes()
        print(f"  licence file         : {f}  {len(b)} bytes  sha256 {hashlib.sha256(b).hexdigest()[:16]}...")
        if p.suffix == ".json" and "sbom" in str(f).lower():
            sbom = json.loads(b)
            comp = sbom.get("metadata", {}).get("component", {})
            print(f"    sbom top component licences: {comp.get('licenses') or '<none declared>'}")
    body = meta.get_payload() or ""
    m = re.search(r"^#+\s*License\s*$(.+?)(?=^#+\s|\Z)", body, re.M | re.S)
    if m:
        print("  README 'License' section:")
        for line in m.group(1).strip().splitlines():
            print(f"    | {line}")
    lic_line = [ln for ln in body.splitlines() if ln.strip().startswith("**License:**")]
    for ln in lic_line:
        print(f"  README licence line  : {ln.strip()}")
    root = Path(dist.locate_file(pkg))
    print(f"  installed package    : {pkg}/ {pkg_size(root)} bytes (excluding __pycache__)")
    try:
        pj = get_json(f"https://pypi.org/pypi/{dist_name}/{dist.version}/json")
        wheels = [
            u for u in pj["urls"]
            if u["filename"].endswith(".whl")
            and ("none-any" in u["filename"] or "x86_64" in u["filename"])
            and ("cp312" in u["filename"] or "py3-none" in u["filename"])
        ]
        for u in wheels:
            print(f"  PyPI wheel           : {u['filename']}  {u['size']} bytes")
    except Exception as e:  # noqa: BLE001
        print(f"  PyPI lookup failed   : {type(e).__name__}: {e}")

print("\n=== clauses quoted in the note, located in the installed kestrel-kernels LICENSE")
kk = im.distribution("kestrel-kernels")
lic = [f for f in kk.files if str(f).endswith("licenses/LICENSE")][0]
text = Path(kk.locate_file(lic)).read_text()
flat = re.sub(r"\s+", " ", text)
QUOTES = [
    "It is licensed, not sold, and is made available only under the terms of a "
    "separate written agreement between M87 Labs, Inc. and the licensee (\"the "
    "Agreement\"). If you have not entered into such an Agreement, you have no "
    "license to use this software.",
    "reverse engineer, decompile, disassemble, deobfuscate, decrypt, or otherwise "
    "attempt to derive, reconstruct, or recover the source code",
    "extract, unpack, or separate the kernel binaries or their metadata from the "
    "packaged collection for any purpose other than the software's normal execution;",
    "copy, modify, translate, create derivative works from, publish, sublicense, "
    "sell, rent, lease, or otherwise distribute the software or any part of it, "
    "except as expressly permitted by the Agreement.",
    "Subject to the Agreement, you are granted a limited, non-exclusive, "
    "non-transferable license to install and run this software",
]
for q in QUOTES:
    print(f"  found verbatim={q in flat}: \"{q}\"")
print(f"  numbered sections: {re.findall(r'^\s*(\d+)\. ([A-Z ;,]+)\.', text, re.M)}")

print("\n=== HuggingFace model licences (api/models)")
for repo in ("moondream/parakeet-redux", "moondream/parakeet-ultra", "nvidia/parakeet-tdt-0.6b-v3"):
    try:
        info = get_json(f"https://huggingface.co/api/models/{repo}")
        lic_tag = [t for t in info.get("tags", []) if t.startswith("license:")]
        print(f"  {repo:30s} {lic_tag}  revision {info.get('sha', '?')[:12]}")
    except Exception as e:  # noqa: BLE001
        print(f"  {repo:30s} lookup failed: {type(e).__name__}: {e}")
