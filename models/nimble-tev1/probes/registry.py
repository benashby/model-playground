"""What Ollama actually ships for each decision model, read from its registry.

    python models/nimble-tev1/probes/registry.py > models/nimble-tev1/results/registry.log
    python models/nimble-tev1/probes/registry.py --download DIR      # also fetch the GGUFs

For every tag: the manifest layers, the config blob, the small text layers
(system prompt, parameters, licences) and an inventory of the GGUF itself:
header metadata and per-tensor types. The GGUF is streamed only as far as the
end of its tensor table, so the inventory costs a few MB, not the whole file.

--download writes <DIR>/<name>-<tag>.gguf and refuses to keep a file whose
SHA-256 does not match its manifest digest. Standard library only.
"""
import argparse
import collections
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

REGISTRY = "https://registry.ollama.ai/v2/library"
TAGS = [("nimble", "latest"), ("tev1", "latest"), ("tev1", "0.8b")]
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}


def get(url, accept=None):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    return urllib.request.urlopen(req, timeout=120)


class Stream:
    """Sequential little-endian reader over an HTTP body."""

    def __init__(self, resp):
        self.r, self.n = resp, 0

    def read(self, k):
        b = self.r.read(k)
        while len(b) < k:
            more = self.r.read(k - len(b))
            if not more:
                raise EOFError
            b += more
        self.n += k
        return b

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.read(self.u("Q")).decode("utf-8", "replace")

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.u(scalar[t])
        if t == 7:
            return bool(self.u("B"))
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def gguf_inventory(resp):
    s = Stream(resp)
    if s.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = s.u("I"), s.u("Q"), s.u("Q")
    kv = {}
    for _ in range(n_kv):
        key = s.string()
        kv[key] = s.value(s.u("I"))
    types, params = collections.Counter(), 0
    per_type_params = collections.Counter()
    for _ in range(n_tensors):
        name = s.string()
        dims = [s.u("Q") for _ in range(s.u("I"))]
        t = GGML_TYPES.get(s.u("I"), "?")
        s.u("Q")
        n = 1
        for d in dims:
            n *= d
        types[t] += 1
        per_type_params[t] += n
        params += n
    resp.close()
    return version, n_tensors, kv, types, per_type_params, params, s.n


def summarise(kv):
    """Scalars worth reading; arrays and the chat template summarised, not dumped."""
    out = {}
    for k, v in kv.items():
        if isinstance(v, list):
            out[k] = f"<array of {len(v)}>"
        elif k == "tokenizer.chat_template":
            out[k] = (f"<{len(v)} chars, sha256 {hashlib.sha256(v.encode()).hexdigest()[:16]}, "
                      f"enable_thinking={'enable_thinking' in v}>")
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=Path)
    a = ap.parse_args()
    for name, tag in TAGS:
        m = json.load(get(f"{REGISTRY}/{name}/manifests/{tag}", ACCEPT))
        print(f"=== {name}:{tag}")
        config = json.load(get(f"{REGISTRY}/{name}/blobs/{m['config']['digest']}"))
        print("config:", json.dumps(config))
        model_layer = None
        for layer in m["layers"]:
            kind = layer["mediaType"].rsplit(".", 1)[-1]
            print(f"layer {kind:8} {layer['size']:>12} {layer['digest']}")
            if kind == "model":
                model_layer = layer
            elif layer["size"] < 4096:
                text = get(f"{REGISTRY}/{name}/blobs/{layer['digest']}").read().decode()
                print(f"  {kind} text: {text!r}")
            else:
                text = get(f"{REGISTRY}/{name}/blobs/{layer['digest']}").read().decode()
                print(f"  {kind} text, first line: {text.strip().splitlines()[0]!r}")
        version, n_t, kv, types, per_type, params, header_bytes = gguf_inventory(
            get(f"{REGISTRY}/{name}/blobs/{model_layer['digest']}"))
        print(f"gguf: version {version}, {n_t} tensors, {params:,} parameters, "
              f"header+tensor table {header_bytes:,} bytes")
        for t, count in types.most_common():
            print(f"  {t:5} {count:4} tensors  {per_type[t]:>14,} params")
        for k, v in summarise(kv).items():
            print(f"  {k} = {v}")
        if a.download:
            a.download.mkdir(parents=True, exist_ok=True)
            path = a.download / f"{name}-{tag}.gguf"
            digest, h = model_layer["digest"].split(":", 1)[1], hashlib.sha256()
            with get(f"{REGISTRY}/{name}/blobs/{model_layer['digest']}") as r, open(path, "wb") as f:
                while chunk := r.read(1 << 22):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != digest:
                path.unlink()
                sys.exit(f"{path}: SHA-256 mismatch, deleted")
            print(f"downloaded {path.name}, SHA-256 verified")
        print()


if __name__ == "__main__":
    main()
