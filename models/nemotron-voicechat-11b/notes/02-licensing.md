# Licensing

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


## Three licenses

VoiceChat is covered by three licenses, one for each of three separately
distributed artifacts.

| Component | License | Commercial use |
|---|---|---|
| Model weights (Hugging Face) | OpenMDW-1.1 | Yes, unrestricted |
| Inference code (NeMo Speech repo) | Apache 2.0 | Yes, unrestricted |
| NIM serving container (`nvcr.io`) | NVIDIA terms | Dev/test free; production needs NVIDIA AI Enterprise |

## OpenMDW-1.1 on the weights

[Full text](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).
This is the Linux Foundation's "MIT for model artifacts." It is notably better
than the NVIDIA Open Model License or NSCLv1 that most NVIDIA models ship under.

- Commercial use: permitted ("deal in the Model Materials without restriction")
- Fine-tune, modify, redistribute weights: yes
- Restrictions on generated output: explicitly none. The license states it
  "does not impose any restrictions or obligations with respect to any outputs"
- Acceptable-use / field-of-use clause: none (unlike Llama, Gemma, NVIDIA OML)
- Conditions: retain the license text and origin notices on redistribution
- Patent grant: yes, with defensive termination. Sue over the model and you
  lose your grant

OpenMDW exists because MIT and Apache were written for *source code* and are
ambiguous about weights, training data, and eval sets. OpenMDW defines "Model
Materials" to cover all of it, then grants copyright, patent, database, and
trade secret rights. The last two matter in the EU, where trained weights may
attract *sui generis* database protection that a code license never
contemplated.

The output disclaimer is the commercially significant clause. Llama and most
NVIDIA OML models attach conditions to what you *generate*. OpenMDW does not,
so synthesized speech from this model carries no downstream encumbrance.

## Apache 2.0 on the inference code

The `NVIDIA-NeMo/Speech` repo is Apache 2.0. That includes
`nemo/collections/speechlm2/` and `duplex_stt_model.py`, the actual streaming
duplex implementation.

That settles the question. NVIDIA open-sourced the duplex implementation, so
the streaming capability is not proprietary to the container. The container is
a packaged, TensorRT-optimized convenience wrapper around code you are free to
use and ship.

## The NIM container

`nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest` carries NVIDIA's own terms.
NVIDIA's published positioning is:

- Hosted API endpoints: free, rate-limited, prototyping
- Downloadable containers: free for development and testing
- Production: requires NVIDIA AI Enterprise, from ~$4,500/GPU/year

NVIDIA defines "production" as serving real end users or business transactions.
Evaluation on your own hardware is explicitly the free tier.

> **Caveat on this section.** The $4,500 figure and the dev/test-vs-production
> boundary come from NVIDIA's published marketing and third-party summaries. I
> did not read the EULA text shipped inside the image for them. Before anything
> ships, read the actual license in the container. I'm confident of the
> direction, but none of it is quotable to procurement.

## The mistake I made

I initially told Ben the HF checkpoint was his "licensing escape hatch": keep
the permissive weights and you're safe. That was wrong, for an instructive
reason.

The weights were never the encumbered part; the *server* is. At the time I
believed the only realtime server was the NIM container, so swapping which
weights you feed it changes nothing about the terms you operate under. Reading
`generate-model-repo.md` confirmed the container is in the loop twice: it
converts the checkpoint and it serves it.

What I'd missed was the Apache 2.0 code. With that, a permissive realtime path
does exist. It means implementing the serving layer from `speechlm2` instead
of using NVIDIA's prebuilt one.

The general lesson: when reasoning about ML licensing, enumerate every artifact
in the runtime path separately. "The model is permissively licensed" says
nothing about your deployment. Weights, inference code, serving container, and
any optimizer/compiler output can each carry different terms.

## The practical conclusion

There are two phases:

- Evaluation (now): the NIM container. It is explicitly licensed for dev/test,
  costs nothing, and runs in minutes. Findings are not a derivative work.
- Production (if it earns it): Apache-2.0 `speechlm2` plus OpenMDW weights,
  with serving built in-house. It is fully unencumbered and has no per-GPU fee.

Phase 2 stays cheap only if everything NIM-specific is confined to one module.
In our harness that is `protocol.py`; `audio.py`, `tools.py`, `policy.py`, and
the scenarios are all protocol-agnostic.

## The container's own LICENSE file [MEASURED]

The previous version of this note flagged the production boundary as unverified
and prescribed the fix: *"Read `/LICENSE` in the container before anything
ships."*

That was done, and it does not resolve the question, because there is no
`/LICENSE` at the container root. The licence lives at `/opt/nim/LICENSE`. It
is 538 bytes and contains no terms at all, only pointers to three separately
hosted agreements:

| Referenced agreement | Governs |
|---|---|
| [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) | the software |
| [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) | AI product usage |
| [AI Foundation Models Community License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/) | the model |

Grepping it for `production`, `AI Enterprise`, `evaluation`, `non-commercial`,
`internal use` and `deploy` returns nothing. The image does not state the
dev/test-versus-production boundary.

This has two consequences:

1. The prescribed verification step does not work. Reading the container tells
   you which agreements apply, not what they say. Anyone who needs the actual
   boundary has to read three web-hosted documents whose contents can change
   independently of the image.
2. The same weights carry two different licence claims depending on where you
   got them. Via Hugging Face they are OpenMDW-1.1. Via NIM, the container's
   own LICENSE says the model is governed by the *AI Foundation Models
   Community License Agreement*. That is a materially different instrument, so
   the choice of distribution channel has licensing consequences that neither
   channel advertises.

> **Still open:** the ~$4,500/GPU/year AI Enterprise figure is vendor
> positioning, not licence text. Nothing measured here confirms or refutes it.

---

Previous: [Requirements and runtime](01-requirements.md) | [Contents](../README.md#contents) | Next: [Architecture and distributions](03-architecture.md)
