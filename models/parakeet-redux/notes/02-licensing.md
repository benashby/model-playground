# Licensing

> Part of the [Parakeet Redux](../README.md) investigation. See also [all model notes](../../README.md).

The weights have one licence. The runtime that executes them is spread over
eight Python distributions, and only one of those states a licence at all.

## Every artifact in the runtime path [MEASURED]

`uv sync --extra asr` (which depends on `moondream>=2.4.0`) installed the
versions below. The licence columns come from each distribution's installed
`METADATA` and `dist-info`, and the wheel sizes from PyPI, all printed by
[`probes/licensing.py`](../probes/licensing.py) into
[`results/licensing.log`](../results/licensing.log).

| Artifact | Version | Declared licence | Licence file shipped | Wheel (x86_64, cp312) |
|---|---|---|---|---|
| Weights, `moondream/parakeet-redux` | HF revision `2bf128600aac` | CC-BY-4.0 (HF `license:` tag) | n/a | 177,774,490-byte `model.safetensors` |
| `moondream` (client) | 2.4.1 | none: no `License` field, no classifier | none | 110,592 bytes |
| `kestrel` (engine orchestration, pure Python) | 0.8.1 | none in metadata; its README's "License" section says only that local inference is free and needs no API key | none | 550,004 bytes |
| `kestrel-native` (Rust extension) | 0.1.8 | none; its CycloneDX SBOM declares no licence for the component itself | SBOM only | 2,331,778 bytes |
| `kestrel-kernels` | 0.7.1 | `Kestrel Kernels — Proprietary Software License`, classifier `License :: Other/Proprietary License` | `licenses/LICENSE`, 4,392 bytes | 4,195,874 or 7,291,184 bytes (two glibc tags) |
| `kestrel-kernels-bundle-{a,b,c,d}` | 0.7.1 | none | none | about 81 to 86 MB each |

So the earlier version of this table was wrong in two ways. It said
`pip install moondream` "pulls in four licences" and then gave a licence for
only two rows. The measured answer is one open licence on the weights, one
proprietary licence on `kestrel-kernels`, and no declared licence on the other
seven distributions. The "~110 kB" given for the client holds: the wheel is
110,592 bytes (158,230 bytes installed, excluding `__pycache__`).

The kernel bundles are covered by the `kestrel-kernels` LICENSE by its own
wording, which names "every artifact it installs (... the packed CUDA kernel
collection `bundles.kstlc` and its contents)", and the `kestrel-kernels`
README adds: "These kernels are provided for use with Kestrel only. Other use
is not permitted."

## What the kernel licence says [MEASURED]

The shipped `LICENSE` has an unnumbered preamble and six numbered sections
(1 GRANT, 2 PROHIBITED CONDUCT, 3 RESERVATION OF RIGHTS, 4 NOTE ON
CIRCUMVENTION LAW, 5 NO WARRANTY; LIMITATION OF LIABILITY, 6 TERMINATION).
The preamble reads, verbatim:

> It is licensed, not sold, and is made available only under the terms of a
> separate written agreement between M87 Labs, Inc. and the licensee ("the
> Agreement"). If you have not entered into such an Agreement, you have no
> license to use this software.

Correction: an earlier version of this note called that passage "Clause 1"
and set its last sentence in bold. It is the preamble, and the original has no
emphasis. Section 1 is the grant, which is itself conditional: "Subject to the
Agreement, you are granted a limited, non-exclusive, non-transferable license
to install and run this software".

Section 2 forbids, among other things, attempts to "reverse engineer,
decompile, disassemble, deobfuscate, decrypt, or otherwise attempt to derive,
reconstruct, or recover the source code", to "extract, unpack, or separate the
kernel binaries or their metadata from the packaged collection for any purpose
other than the software's normal execution;", and to "copy, modify, translate,
create derivative works from, publish, sublicense, sell, rent, lease, or
otherwise distribute the software or any part of it, except as expressly
permitted by the Agreement." The probe checks each of these quotations against
the installed file and prints whether it was found verbatim; all were.

## What the vendor says in public [CLAIM]

Checked on 2026-09-23 by [`probes/vendor_claims.py`](../probes/vendor_claims.py)
([`results/vendor_claims.log`](../results/vendor_claims.log)):

- The post "Photon is now free" (June 8, 2026, Photon 1.3.0): "Starting with
  version 1.3.0, running Moondream locally with Photon is totally free. No API
  key required, just download and start calling it."
- The Photon page's FAQ: "Is it free? Yes. pip install moondream and run any
  supported model on your own hardware at no cost."
- The Photon 2.0 launch post (August 3, 2026): "The Photon inference engine is
  Apache 2.0, and the megakernels are free to run. Our compiler itself is
  proprietary."
- The published Terms of Service govern "access to and use of the Moondream
  Cloud Service", the hosted API. They say nothing about local use.

The installed packages do not match the public statements. Nothing in
`kestrel` 0.8.1 or `kestrel-native` 0.1.8 declares Apache 2.0, or any other
licence, and the one licence that is shipped tells a user with no written
agreement that they have no licence to use the kernels. No page I found offers
such an agreement for local use.

The likeliest explanation is an unreconciled document: a proprietary template
written for enterprise deals, shipped inside every artifact and never updated
when a free tier launched. That is my reading; the vendor has not said so. The
clauses with real force are the ones written for this package: no
reverse engineering, no unpacking the packed kernel containers, no
redistribution. This repository respects them without qualification, and none
of the probes here inspects the kernels; `licensing.py` reads only package
metadata and the licence text.

## The weights

All three checkpoints carry the same weights licence. The HF API returns
`license:cc-by-4.0` for `moondream/parakeet-redux`, `moondream/parakeet-ultra`
and `nvidia/parakeet-tdt-0.6b-v3`. NVIDIA's card states: "GOVERNING TERMS: Use
of this model is governed by the CC-BY-4.0 license." [CLAIM]

## The practical consequence

- Redux is runtime-locked. What makes it worth using is the ternary encoder,
  and only Photon's proprietary kernels multiply against that packed
  representation.
  There is no ONNX or `sherpa-onnx` path to the 178 MB weights, so without
  Photon, Redux cannot run at all.
- Ultra may not be. It ships full-precision safetensors with the same
  architecture. Open: whether NeMo or an ONNX runtime will load it. If one
  does, the arm the vendor reports as most accurate survives without a
  proprietary runtime.
- The original is unencumbered today: CC-BY-4.0 weights that run under NeMo
  or ONNX runtimes. The vendor's own card measured those runtimes on its AMD
  EPYC 9575F at 45× (parakeet.cpp, q8_0), 42× (sherpa-onnx, int8) and 28×
  (onnx-asr, int8) [CLAIM]. Correction: an earlier version of this note gave
  "28-42× real time" for the original with no source. Those figures are the
  vendor's, from the Redux model card, on an AVX-512 server CPU, and none of
  them was measured here. What was measured here is the original checkpoint
  running inside Photon on this CPU; see
  [the throughput results](06-results-throughput.md).

This confirmed twice a rule the repo's onboarding skill already states:
enumerate every artifact in the runtime path separately. The open weights said
nothing about the runtime that executes them, and the runtime's public
description disagreed with its own package metadata.

---

Previous: [How it works](01-how-it-works.md) | [Contents](../README.md#contents) | Next: [Setup](03-setup.md)
