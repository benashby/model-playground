"""What the three Parakeet checkpoints contain, read from their own files.

Tests the licensing note's claim that Redux's ternary weights can only be used
by Photon's kernels. For each checkpoint in the local Hugging Face cache
(moondream/parakeet-redux, moondream/parakeet-ultra, nvidia/parakeet-tdt-0.6b-v3),
it reads the safetensors header (never the weights' contents beyond one
tensor) and reports tensor counts by dtype and the declared architecture.
Then, for Redux:

  - prints the packing specification in its ternary.json;
  - checks every packed tensor against it: each `qweight` must have
    ceil(in / elements_per_byte) bytes per row, and a matching `scales`
    tensor with one scale per group of `group_size` inputs;
  - unpacks one layer following the documented rule and reports the shape and
    the set of values, to show the rule is complete enough to reconstruct a
    dense weight matrix without Photon.

It does not run the unpacked weights in any runtime. Reading the published
files is not reverse engineering the kernels: nothing from `kestrel-kernels`
is opened.

    uv run --extra asr python models/parakeet-redux/probes/weights_format.py \\
        > models/parakeet-redux/results/weights_format.log
"""

import collections
import json
import math
import struct
from pathlib import Path

import numpy as np

HUB = Path.home() / ".cache" / "huggingface" / "hub"
REPOS = ["moondream/parakeet-redux", "moondream/parakeet-ultra", "nvidia/parakeet-tdt-0.6b-v3"]


def snapshot(repo: str) -> Path:
    snaps = sorted((HUB / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())
    return snaps[-1]


def header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def main():
    heads = {}
    for repo in REPOS:
        d = snapshot(repo)
        h, _ = header(d / "model.safetensors")
        heads[repo] = h
        cfg = json.load(open(d / "config.json", encoding="utf-8"))
        dt = collections.Counter(v["dtype"] for v in h.values())
        print(f"=== {repo} (snapshot {d.name[:12]})")
        print(f"  architectures {cfg.get('architectures')}, model_type {cfg.get('model_type')}")
        print(f"  {len(h)} tensors: " + ", ".join(f"{k} {v}" for k, v in sorted(dt.items())))

    red, ult, nv = (heads[r] for r in REPOS)
    print("\n=== tensor inventories")
    print(f"  ultra vs nvidia: {len(set(ult) & set(nv))} shared names, "
          f"{len(set(ult) - set(nv))} only in ultra ({sorted(set(ult) - set(nv))[:3]}...), "
          f"{len(set(nv) - set(ult))} only in nvidia; shared shapes all equal: "
          f"{all(ult[k]['shape'] == nv[k]['shape'] for k in set(ult) & set(nv))}")
    packed = sorted(k for k in red if k.endswith(".qweight"))
    print(f"  redux: {len(packed)} packed (.qweight) tensors, {len(set(nv) - set(red))} nvidia tensor names absent")

    d = snapshot("moondream/parakeet-redux")
    spec = json.load(open(d / "ternary.json", encoding="utf-8"))
    print("\n=== redux ternary.json")
    for k in ("format", "packing", "weight_rule", "quant"):
        print(f"  {k}: {json.dumps(spec.get(k))}")
    per_byte = spec["packing"]["elements_per_byte"]
    group = spec["quant"]["group_size"]

    # Every packed tensor must be consistent with the spec, using the dense
    # shape of the same weight in NVIDIA's checkpoint.
    ok = bad = 0
    for q in packed:
        dense = nv.get(q.replace(".qweight", ".weight"))
        sc = red.get(q.replace(".qweight", ".scales"))
        rows, cols = dense["shape"][0], math.prod(dense["shape"][1:])
        good = (red[q]["shape"] == [rows, math.ceil(cols / per_byte)] and sc is not None
                and sc["shape"] == [rows, math.ceil(cols / group)])
        ok, bad = ok + good, bad + (not good)
    print(f"  packed tensors consistent with the spec and the dense shapes: {ok} of {ok + bad}")

    # Unpack the first packed layer by the documented rule.
    q = packed[0]
    h, off = header(d / "model.safetensors")
    def read(name):
        m = h[name]
        a, b = m["data_offsets"]
        with open(d / "model.safetensors", "rb") as f:
            f.seek(off + a)
            raw = f.read(b - a)
        dt = {"U8": np.uint8, "F16": np.float16, "F32": np.float32}[m["dtype"]]
        return np.frombuffer(raw, dtype=dt).reshape(m["shape"])
    qw, sc = read(q), read(q.replace(".qweight", ".scales")).astype(np.float32)
    rows, cols = nv[q.replace(".qweight", ".weight")]["shape"][0], math.prod(nv[q.replace(".qweight", ".weight")]["shape"][1:])
    digits = np.stack([(qw // (3 ** i)) % 3 for i in range(per_byte)], axis=-1).reshape(rows, -1)[:, :cols]
    w = sc[:, np.arange(cols) // group] * (digits.astype(np.float32) - 1)
    print(f"\n=== unpacked {q}")
    print(f"  dense shape {list(w.shape)} (NVIDIA's {nv[q.replace('.qweight', '.weight')]['shape']}), "
          f"codes used {sorted(np.unique(digits).tolist())}, "
          f"share of zero weights {float((digits == 1).mean()):.3f}")


if __name__ == "__main__":
    main()
