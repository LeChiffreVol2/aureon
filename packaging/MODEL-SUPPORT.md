# Models and supported scope

**Runtime selection:** one local v0.1.0 distribution, identified by its exact delivery receipt ([installation guide](CANDIDATE-QUICKSTART.md)). Compatible TerraMind base file, array and HTTP scene routes share the preserved graph/reuse primitives. Version alone does not identify artifact bytes or qualify every route. Historical cohorts below keep their original artifact scope; current acceptance and timing belong to the receipt shipped with the selected delivery.

## Retained development candidate acceptance

**Unpublished v0.1.0, accepted on 2026-11-01.** Generation wheel
`ca996c1063b39ebc8a803273778f8dfdadfe79f221118321163c21880f96710b` passed six installed scene jobs against the preceding wheel, producing
eight COGs with exact highest-FP32 output comparisons on RTX 3070 / WSL,
Torch 2.10.0 and TerraTorch 1.2.13; TF32 and autocast were disabled.

| Workload | Scene size | Tiles per wheel | Accepted scope |
| --- | --- | ---: | --- |
| KAO-TerraMind B | 512 × 512 | 9 | `fp32-full-reference-v1`, `scene-legacy`; exact logits against fresh eager references, class/validity COGs and provenance |
| TerraMind 1.0 base, S2L2A → synthetic S1GRD | 448 × 320 | 4 | Exact raw output, tokens and post-request RNG against saved native references; reconstruction, COG and provenance |
| TerraMind 1.0 base, S2L2A → synthetic S1GRD | 449 × 289 | 6 | Same checks, including non-multiple edges |

The comparison covered 20 complete TerraMind generations across both wheels;
six jobs is not a model-call count. All six model owners closed with CUDA
allocated/reserved bytes at 0/0. This is finite-job cleanup evidence, not a
sustained-residency or long-run leak test. B's guarded-v3 profile was not newly
qualified. Other families, profiles, HTTP workflows and physical multi-GPU
operation retain their separate artifact-specific qualification.

Core wheel `04905165fca56bd064b804bc6cfedf17435e23e7dc386e4dad18b40d62d7dafe` passed installed CPU checks and contains the same 201 runtime
payloads, but was not separately run on a GPU. Neither profile establishes fresh
dependency provisioning. Exact parity preserves the existing TerraMind tile
seams; it does not establish seamless imagery or scientific realism. Recorded
scene times are acceptance observations, not a new speedup qualification.

## TerraMind family extension (local v0.1.0 candidate)

| TerraMind 1.0 size | Generation | Backbone | TiM |
| --- | --- | --- | --- |
| tiny | `terramind-v1-tiny` | `terramind-v1-tiny-backbone` | `terramind-v1-tiny-tim` |
| small | `terramind-v1-small` | `terramind-v1-small-backbone` | `terramind-v1-small-tim` |
| base | `terramind-v1-base` | `terramind-v1-base-backbone` | `terramind-v1-base-tim` |
| large | `terramind-v1-large` | `terramind-v1-large-backbone` | `terramind-v1-large-tim` |

Generation retains the all-valid S2L2A224 → synthetic S1GRD SDK/CLI workflow. The selected HTTP worker executes TerraMind base `generation` and `generation_scene` only; other generation sizes and the `features` executor are rejected before loading. Backbone and TiM expose resident feature extraction through `TerraMindEncoder` and the `encode` CLI; those outputs need a downstream task head and are not synthetic image-generation results. Feature transport schemas remain testable but are not executable HTTP support. See [resident workflows](RESIDENT-WORKFLOWS.md) and [full contracts and examples](TERRAMIND-FAMILIES.md). Earlier artifact-specific cohorts passed all twelve size/mode SDK/CLI routes with exact FP32 parity on RTX 3070; one earlier tiny-generation HTTP flow also passed. That result does not enable tiny-generation HTTP in the currently selected worker. The linked family guide records the exact artifact, input, TiM target, profiles and limits. Prior evidence stays artifact-bound. These additions are not in the original published GitHub wheel.

| Model | Environment / input | Result | Alpha boundary |
| --- | --- | --- | --- |
| KAO-TerraMind B | scene; prepared S2L2A up to4096 per side | class/validity COGs, provenance | Full-reference first-user route; guarded resident mode requires exact qualification |
| TerraMind 1.0 base | generation; all-valid S2L2A224 | synthetic continuous S1GRD VV/VH COG, raw output, provenance | 8token/10diffusion highestFP32, resident exact encoder reuse and guarded decoder graph |
| KAO-TerraMind A/C | scene-legacy; prepared S2L2A | segmentation | Implemented full-reference routes; model acceptance remains on earlier artifacts |

`aureon-engine models` lists interfaces; `assets` lists pinned official sources;
`profiles` and `check --environment NAME` inspect metadata without loading models.
The [first-result guide](START-HERE.md) covers preparation/configuration. All
models require separately obtained assets. No checkpoint download happens during
installation, import, metadata inspection or model factory construction.

The original published alpha preserves its measured numerical/reuse/graph
implementation. That original wheel passed one guarded-v3 B512 and one TerraMind224
real HTTP job on RTX3070/WSL, retaining FP32 reference parity, verified artifacts,
replay/once-only usage and CUDA0/0 close. See the [public support matrix](../docs/MODEL-SUPPORT.md)
and [release notes](../docs/releases/v0.1.0.md). CPU tests alone do not qualify
GPU execution, and this functional cohort does not create a new benchmark or
new-hardware/provider claim. Earlier cohorts retain their separate scopes.

`fp32-full-reference-v1` compares B to an eager reference per job. Guarded B
`fp32-stream-guarded-v3` uses a separately verified qualification and reports
per-job numerical comparison `not_evaluated`; it does not rerun eager reference.
The guarded receipt must match hardware, dependencies, checkpoint and numerical
sources. A/C cannot borrow the B guarded route.

TerraMind reuse is exact and eligibility checked. `denoiser=graph` retains guards
and fallback; `denoiser=native` / `reuse=off` remain supported choices. A completed
job receipt and a successful close receipt are separate facts. Generated S1GRD
is synthetic; parity is not proof of geophysical accuracy. The scene sizes qualified above do not imply arbitrary-shape, seamless-image,
training, hosted SLA or new-hardware coverage.
