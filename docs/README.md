# AUREON documentation

**Runtime selection:** use the [selected best wheel](../packaging/CANDIDATE-QUICKSTART.md), `0fa5ef2b...`, v0.1.0. Later default/core builds and working-source extensions described below are unpromoted development, not the recommended execution artifact. Their recorded checks remain historical and are not transferred to the selected wheel.

**Generative Foundation Models for Earth Observation — from Sentinel-2 input to synthetic S1GRD.**

[Repository](../README.md) · [Download v0.1.0](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0) · [Get help](../packaging/SUPPORT.md)

## New to AUREON?

1. **Choose a workflow.** Review the [supported models](MODEL-SUPPORT.md): start with TerraMind S2L2A → synthetic S1GRD generation, or choose KAO-TerraMind B segmentation.
2. **Choose your artifact and install.** The [published alpha quickstart](../packaging/START-HERE.md) uses the existing GitHub download. For the selected best local wheel, use the [selected-runtime quickstart](../packaging/CANDIDATE-QUICKSTART.md). Their APIs, hashes and evidence scopes are distinct.
3. **Run and inspect.** Execute your first explicit job, then inspect the COG, mask and provenance together.

Linux x86_64 / WSL2 and Python 3.12 are the documented platform. Keep scene and
generation in separate environments. Installation and metadata inspection do
not load weights or run inference.

## Find your guide

Whole-scene, bridge and scaling guides describe retained development source or
their explicitly identified artifacts. They are not an assertion that those
interfaces are included in the selected historical wheel. Choose the install
guide first; use only interfaces available in that artifact.

| I want to… | Read this |
| --- | --- |
| Run a whole scene beyond a single model tile | [Scene planning, masks, bounded memory and COG delivery](../packaging/PRODUCT-WORKFLOWS.md) |
| Keep a model resident across scene jobs | [Resident scene and feature workflows](../packaging/RESIDENT-WORKFLOWS.md) |
| Read local or STAC/HTTP-range inputs | [Bounded raster windows](../packaging/STAC-WINDOWS.md) |
| Configure queues, GPU lanes or replicas | [Concurrency](../packaging/CONCURRENT-GENERATION.md), [scaling runtime](../packaging/SCALABLE-RUNTIME.md) |
| Install the exact selected best local runtime | [Candidate quickstart](../packaging/CANDIDATE-QUICKSTART.md) |
| Try the input pipeline without weights or imagery | [CPU-only preparation walkthrough](CPU-WALKTHROUGH.md) |
| Understand the Engine and its interfaces | [Architecture and result lifecycle](ARCHITECTURE.md) |
| Install reproducibly or use a local dependency cache | [Locked dependency profiles](../packaging/LOCKED-INSTALL.md) |
| Prepare my own Sentinel-2 imagery | [Bands, radiometry, grids and masks](../packaging/INPUTS.md) |
| Find the exact checkpoint and tokenizer | [Pinned model assets](../packaging/ASSETS.md) |
| Call inference from Python | [Public API and resident sessions](../packaging/PUBLIC-API.md) |
| Integrate the unreleased source API | [GenerationEngine, bridges and spatial helpers](DEVELOPER-INTEGRATION.md) |
| Plan memory or contribute a serving adapter | [Resource policies, compatibility and conformance](../packaging/SCALABLE-POLICY.md) |
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

Local scalability interfaces: [runtime and replica SDK](../packaging/SCALABLE-RUNTIME.md), [resource policies](../packaging/SCALABLE-POLICY.md), [generation execution and concurrency](../packaging/CONCURRENT-GENERATION.md). These are unpublished source additions.
