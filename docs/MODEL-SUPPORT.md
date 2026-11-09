# Supported models and qualification

**Runtime selection:** one selected public v0.1.0 distribution, identified by its exact delivery receipt ([installation guide](../packaging/CANDIDATE-QUICKSTART.md)). Compatible TerraMind base file, array and HTTP scene routes share the preserved graph/reuse primitives. Version alone does not identify artifact bytes or qualify every route. Historical cohorts below keep their original artifact scope; current acceptance and timing belong to the receipt shipped with the selected delivery.

## Current Best release (2026-11-02)

The current GitHub download is wheel
`e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64`.
Its retained acceptance covers TerraMind base native 224 and 448 × 320
four-tile scenes, plus local HTTP and private Cloud scene paths: 26 complete
generation calls with exact FP32 output/token/RNG parity and seven resident
owners closing with CUDA allocation/reservation at zero. The private Cloud
path is test evidence, not an available hosted API. The later 14-call diagnostic
retained parity and does not establish an additional speedup. Publication ran
no new inference or benchmark. See [release notes](releases/v0.1.0.md).

KAO-TerraMind and other family qualifications below retain their original
artifact identities; publishing this wheel does not relabel those earlier
results as a fresh Best-wheel acceptance. The historical performance table
also keeps its original measured wheel and workload scope.

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

## Available interfaces and separate qualification

TerraMind 1.0 tiny/small/base/large have explicit Generation, Backbone and TiM
SDK/CLI interfaces in the selected v0.1.0 package. The selected HTTP worker executes
TerraMind base `generation` and `generation_scene`; it rejects other generation
sizes and the `features` executor before model loading. Feature transport schemas
remain available for integration tests, not executable HTTP support. Use the
Backbone/TiM SDK/CLI for feature outputs. Large-scene generation and separate
SDK process dispatch are described in [resident workflows](../packaging/RESIDENT-WORKFLOWS.md). The original published wheel below
is unchanged. See [the 12-mode support matrix](../packaging/MODEL-SUPPORT.md)
and [asset/input/output contracts](../packaging/TERRAMIND-FAMILIES.md).
Earlier artifact-specific cohorts passed all twelve size/mode SDK/CLI routes on
RTX 3070 with exact FP32 parity; one earlier tiny-generation HTTP flow also passed on its recorded artifact.
See the family guide for the exact artifact and restricted input/recipe scope.
No new comparative performance or general production qualification is claimed.

The selected package provides hardware-independent resource planning,
bounded independent graph lanes and [replica/state/delivery contracts](../packaging/SCALABLE-RUNTIME.md).
CPU simulations and loopback workers validate control flow, not GPU scaling.
The unpublished tensor-batch route is retired; recommended/reference generation remains available.
Generic adapter registration does not qualify new modalities or precision profiles.

## Original published release qualification

| Model | Environment | Implemented workflow | Recorded qualification |
| --- | --- | --- | --- |
| TerraMind 1.0 base | `generation` | All-valid S2L2A224 → synthetic S1GRD; 8 token / 10 diffusion steps, highest FP32, eligible exact encoder reuse and decoder graph | Final v0.1.0 wheel: seed17 HTTP job, exact raw/token/RNG/COG parity, replay and clean shutdown |
| KAO-TerraMind B | `scene` | Prepared S2L2A segmentation; full-reference or separately qualified guarded resident serving | Final v0.1.0 wheel: guarded-v3 B512 HTTP job, retained FP32 reference, exact class/mask/grid/provenance, replay and clean shutdown |
| KAO-TerraMind A/C | `scene-legacy` | Full-reference segmentation interfaces | Earlier-artifact model evidence; no new A/C final-wheel qualification |

The original published wheel SHA256 is
`a11b67e859ce0e2e89277bf5e35e1b8d917bfa0aa577617bf9e66d8860c2c1bf`.
The functional cohort used one job per leading model on RTX 3070 8 GiB / WSL.
Repeat downloads and idempotent replay triggered no additional inference or usage
row. Both owned workers closed with CUDA allocated/reserved at 0/0.

B supports bounded prepared model grids up to4096 pixels per side, but the final
functional cohort above was512. Earlier larger-scene evidence remains separate.
In this original published artifact TerraMind generation accepts one224 tile. Current-source large-scene generation is described above; it does not retroactively extend that wheel.
Native denoising and reuse-off remain explicit alternatives; guards/fallback are
retained. The independent first-user B route is full-reference and does not
borrow someone else's guarded qualification.

These results demonstrate functional fidelity within a declared configuration.
They do not establish accuracy on new scenes, sustained-load behavior, absence of
long-run leaks, arbitrary GPU/driver support, provider performance or hosted SLA.
Exact parity is not physical validation of synthetic SAR.

See [first result](../packaging/START-HERE.md), [runtime details](../packaging/MODEL-SUPPORT.md)
and [release checksums](releases/v0.1.0.md). No new benchmark was run for publication.

## Historical performance reference

These retained results belong to the original measured Best wheel `0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97`.
The current integrated package is `e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64`;
it retains its separate functional acceptance, not these competitor percentages.
The original alpha is preserved separately; the current published download is the integrated Best wheel. The 2026-10-27 comparison used TerraMind
1.0 base, all-valid S2L2A 224 × 224 → synthetic S1GRD, highest FP32, RTX 3070,
8 token / 10 diffusion steps, and local IPC batch-one execution with FIFO depth
three. This was not an HTTP benchmark. All 28 requests retained exact reference
parity; the small cohort provides descriptive results, not production guarantees.

| Full-delivery metric | Independent baseline | AUREON reference wheel | Observed change |
| --- | ---: | ---: | ---: |
| Pooled burst throughput | 1.8667 jobs/s | 2.3498 jobs/s | +25.88% |
| Sequential median | 617.43 ms | 446.40 ms | −27.70% |
| Burst completion median | 1102.20 ms | 852.64 ms | −22.64% |
| Warm THB per 1,000 jobs at 1 THB/hour | 0.148807 | 0.118213 | −20.56% |

The baseline is TerraTorch + native vLLM with an independent TerraMind adapter,
corrected conventional I/O and disclosed batch-one/resource overrides. It is not
an unmodified default end-to-end product. The cost row normalizes measured warm
execution time at 1 THB/hour; it is not provider pricing or startup-inclusive cost.
Memory is mixed: peak CUDA allocation is 2,113,873,408 → 2,112,630,272 bytes, while
warm reserved CUDA memory is 2,264,924,160 → 2,313,158,656 bytes (+46 MiB).
The current integrated wheel has no new competitor performance qualification.
Its separate four-warm-call-per-arm check observed 2.464 → 2.434 serial jobs/s;
that is neither this queued-throughput workload nor an established causal loss.
A later diagnostic changed timing direction and did not establish non-regression.
Earlier evidence and measurements remain attached to their original artifacts.
