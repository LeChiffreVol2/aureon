# Models and supported scope

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

This release preserves the previously measured numerical/reuse/graph
implementation. The final wheel passed one guarded-v3 B512 and one TerraMind224
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
is synthetic; parity is not proof of geophysical accuracy. No arbitrary shapes,
large-scene generation, training, hosted SLA or new-hardware coverage is claimed.
