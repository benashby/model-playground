#!/usr/bin/env python3
"""What the checkpoint on disk actually is: config values, parameter count,
dtypes, shard count and size, all read from the files rather than the card.

Stdlib only, because it runs wherever the weights are (here, the GPU node):

    ssh <gpu-node> 'python3 - /opt/models/K2-Horizon-32B' < models/k2-horizon-32b/probes/weights_inventory.py

Reads each safetensors header (an 8-byte length, then JSON) without loading
any tensor data.
"""
import json, os, struct, sys
from collections import Counter

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
keys = ["architectures", "model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "max_position_embeddings", "rope_theta",
        "torch_dtype", "tie_word_embeddings"]
print("config.json")
for k in keys:
    if k in cfg:
        print(f"  {k} = {cfg[k]}")

shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
params, dtypes, nbytes, tensors = 0, Counter(), 0, 0
for f in shards:
    p = os.path.join(d, f)
    nbytes += os.path.getsize(p)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensors += 1
        dtypes[meta["dtype"]] += 1
        size = 1
        for dim in meta["shape"]:
            size *= dim
        params += size
print("safetensors")
print(f"  shards  = {len(shards)}")
print(f"  tensors = {tensors}")
print(f"  dtypes  = {dict(dtypes)}")
print(f"  params  = {params:,} ({params / 1e9:.2f} B)")
print(f"  on disk = {nbytes / 2**30:.2f} GiB ({nbytes:,} bytes)")
print(f"  bytes/param = {nbytes / params:.3f}")
tpl = os.path.join(d, "chat_template.jinja")
print(f"chat_template.jinja present: {os.path.exists(tpl)}")
