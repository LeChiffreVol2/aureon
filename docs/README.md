# AUREON documentation

**Generative Foundation Models for Earth Observation — from Sentinel-2 input to synthetic S1GRD.**

[Repository](../README.md) · [Download v0.1.0](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0) · [Get help](../packaging/SUPPORT.md)

## New to AUREON?

1. **Choose a workflow.** Review the [supported models](MODEL-SUPPORT.md): start with TerraMind S2L2A → synthetic S1GRD generation, or choose KAO-TerraMind B segmentation.
2. **Install and prepare.** Follow [installation and first result](../packaging/START-HERE.md) for your dependency profile, pinned assets and Sentinel-2 input.
3. **Run and inspect.** Execute your first explicit job, then inspect the COG, mask and provenance together.

Linux x86_64 / WSL2 and Python 3.12 are the documented platform. Keep scene and
generation in separate environments. Installation and metadata inspection do
not load weights or run inference.

## Find your guide

| I want to… | Read this |
| --- | --- |
| Try the input pipeline without weights or imagery | [CPU-only preparation walkthrough](CPU-WALKTHROUGH.md) |
| Understand the Engine and its interfaces | [Architecture and result lifecycle](ARCHITECTURE.md) |
| Install reproducibly or use a local dependency cache | [Locked dependency profiles](../packaging/LOCKED-INSTALL.md) |
| Prepare my own Sentinel-2 imagery | [Bands, radiometry, grids and masks](../packaging/INPUTS.md) |
| Find the exact checkpoint and tokenizer | [Pinned model assets](../packaging/ASSETS.md) |
| Call inference from Python | [Public API and resident sessions](../packaging/PUBLIC-API.md) |
| Use the command line | [First-result commands](../packaging/START-HERE.md) |
| Upload, submit and download through local HTTP | [Authenticated local service](../packaging/ASYNC-SERVICE.md) |
| Check whether a model or profile is qualified | [Support matrix](MODEL-SUPPORT.md) |
| Diagnose a failed install or job | [Errors and support](../packaging/SUPPORT.md) |

## Release and project reference

| Reference | What it contains |
| --- | --- |
| [v0.1.0 release notes](releases/v0.1.0.md) | Exact artifact checksums, interfaces and validation scope |
| [Contributing](../CONTRIBUTING.md) | Repository map, local checks and pull-request guidance |
| [Rights statement](RIGHTS.md) | Approved constants/class-name scope and archive wording |
| [Third-party notices](../packaging/review/THIRD-PARTY-NOTICES.md) | Source attribution, licenses and upstream discrepancy |
| [Security policy](../SECURITY.md) | Supported boundary and private vulnerability reporting |

These guides cover the local library and local service. The public alpha does not
include a hosted inference endpoint or a production SLA.
