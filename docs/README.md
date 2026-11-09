# AUREON documentation

**Current distribution: Best v0.1.0 (`e09e710c`).** Use the same package for
Python, CLI and local HTTP. Historical release assets and comparison results
retain their own hashes; they are not additional installation choices.

[Repository](../README.md) · [Download](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0) · [Get help](../packaging/SUPPORT.md)

## New to AUREON?

1. [Install the current package](../packaging/CANDIDATE-QUICKSTART.md), or use the
   [verified first-use installer](FIRST-USE.md). Linux x86_64 / WSL2 and Python 3.12
   are the qualified installation platform.
2. [Prepare assets and your Sentinel-2 input](../packaging/START-HERE.md).
   Choose TerraMind generation or KAO-TerraMind B segmentation; check their
   dependency and qualification scope in [model support](MODEL-SUPPORT.md).
3. Inspect the result's COG, validity and provenance. Use [whole scenes and
   resident jobs](../packaging/BEST-WORKFLOWS.md) beyond a native tile.

Installation does not download weights or execute a model. The CPU walkthrough
checks preprocessing only. PyPI and hosted inference remain closed.

## Find your guide

The guides describe interfaces in the selected wheel. Available interfaces and
successful GPU qualification are different; each model/profile keeps its recorded
evidence scope.

| I want to… | Read this |
| --- | --- |
| Run a whole scene beyond a single model tile | [Scene planning, masks, bounded memory and COG delivery](../packaging/PRODUCT-WORKFLOWS.md) |
| Keep a model resident across scene jobs | [Resident scene and feature workflows](../packaging/RESIDENT-WORKFLOWS.md) |
| Read local or STAC/HTTP-range inputs | [Bounded raster windows](../packaging/STAC-WINDOWS.md) |
| Configure queues, GPU lanes or replicas | [Concurrency](../packaging/CONCURRENT-GENERATION.md), [scaling runtime](../packaging/SCALABLE-RUNTIME.md) |
| Install the selected Best runtime | [Installation guide](../packaging/CANDIDATE-QUICKSTART.md) |
| Try the input pipeline without weights or imagery | [CPU-only preparation walkthrough](CPU-WALKTHROUGH.md) |
| Understand the Engine and its interfaces | [Architecture and result lifecycle](ARCHITECTURE.md) |
| Install reproducibly or use a local dependency cache | [Locked dependency profiles](../packaging/LOCKED-INSTALL.md) |
| Prepare my own Sentinel-2 imagery | [Bands, radiometry, grids and masks](../packaging/INPUTS.md) |
| Find the exact checkpoint and tokenizer | [Pinned model assets](../packaging/ASSETS.md) |
| Call inference from Python | [Public API and resident sessions](../packaging/PUBLIC-API.md) |
| Integrate TorchGeo/TerraTorch and arrays | [GenerationEngine, bridges and spatial helpers](DEVELOPER-INTEGRATION.md) |
| Plan memory or contribute a serving adapter | [Resource policies, compatibility and conformance](../packaging/SCALABLE-POLICY.md) |
| Use the command line | [First-result commands](../packaging/START-HERE.md) |
| Upload, submit and download through local HTTP | [Authenticated local service](../packaging/ASYNC-SERVICE.md) |
| Check whether a model or profile is qualified | [Support matrix](MODEL-SUPPORT.md) |
| Diagnose a failed install or job | [Errors and support](../packaging/SUPPORT.md) |

Measure [efficiency, correctness and usability](EVALUATION.md), or contribute
[first-use feedback](FIRST-USE.md).

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

Local scalability interfaces: [runtime and replica SDK](../packaging/SCALABLE-RUNTIME.md), [resource policies](../packaging/SCALABLE-POLICY.md), [generation execution and concurrency](../packaging/CONCURRENT-GENERATION.md). Physical scaling still requires qualification.
