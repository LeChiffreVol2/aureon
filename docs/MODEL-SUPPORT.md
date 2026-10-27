# Supported models and qualification

| Model | Environment | Implemented workflow | Recorded qualification |
| --- | --- | --- | --- |
| TerraMind 1.0 base | `generation` | All-valid S2L2A224 → synthetic S1GRD; 8 token / 10 diffusion steps, highest FP32, eligible exact encoder reuse and decoder graph | Final v0.1.0 wheel: seed17 HTTP job, exact raw/token/RNG/COG parity, replay and clean shutdown |
| KAO-TerraMind B | `scene` | Prepared S2L2A segmentation; full-reference or separately qualified guarded resident serving | Final v0.1.0 wheel: guarded-v3 B512 HTTP job, retained FP32 reference, exact class/mask/grid/provenance, replay and clean shutdown |
| KAO-TerraMind A/C | `scene-legacy` | Full-reference segmentation interfaces | Earlier-artifact model evidence; no new A/C final-wheel qualification |

The final wheel SHA256 is
`a11b67e859ce0e2e89277bf5e35e1b8d917bfa0aa577617bf9e66d8860c2c1bf`.
The functional cohort used one job per leading model on RTX 3070 8 GiB / WSL.
Repeat downloads and idempotent replay triggered no additional inference or usage
row. Both owned workers closed with CUDA allocated/reserved at 0/0.

B supports bounded prepared model grids up to4096 pixels per side, but the final
functional cohort above was512. Earlier larger-scene evidence remains separate.
TerraMind generation is224 only; B streaming does not imply large-scene generation.
Native denoising and reuse-off remain explicit alternatives; guards/fallback are
retained. The independent first-user B route is full-reference and does not
borrow someone else's guarded qualification.

These results demonstrate functional fidelity within a declared configuration.
They do not establish accuracy on new scenes, sustained-load behavior, absence of
long-run leaks, arbitrary GPU/driver support, provider performance or hosted SLA.
Exact parity is not physical validation of synthetic SAR.

See [first result](../packaging/START-HERE.md), [runtime details](../packaging/MODEL-SUPPORT.md)
and [release checksums](releases/v0.1.0.md). No new benchmark was run for publication.
