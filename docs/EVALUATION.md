# Evaluating AUREON

A useful inference library must deliver correct results, save resources and let
developers finish their work. These are separate measurements. A fast kernel,
many tests or a short installation command cannot establish all three.

The current download is **v0.1.0, wheel `e09e710c…`**. Its FP32 acceptance and
the historical competitor comparison have different artifact identities.
See [support and measured results](MODEL-SUPPORT.md) and the machine-readable
[quality record](../release/quality-v0.1.0.json). No formal certification is claimed.

## What to measure

| Dimension | Measurements | Evidence needed for a claim |
| --- | --- | --- |
| Execution correctness | Tokens, raw outputs, RNG/state isolation; shape, dtype, grid, units, masks, saved COG and provenance | Same checkpoint, preprocessing, recipe, seed, stack and declared precision; verify delivered artifacts, not only model tensors |
| Scientific usefulness | Modality-specific physical checks, seams, invalid/cloud coverage, paired real observations where available and downstream task performance | Independent acquisitions/regions/dates, reference data and uncertainty; synthetic S1GRD is not measured SAR |
| Efficiency | Cold readiness, warm model latency, end-to-end latency, completed jobs/s, valid pixels/s, peak RAM and CUDA allocated/reserved bytes | Explicit timing boundaries, load pattern, resource budget, failures and matched baseline |
| Serving quality | Queue wait, completion latency, goodput within a stated deadline, rejection/error rate, recovery and cancellation | Offered load and completed/failed work, sustained tests, mixed scene sizes and fault cases |
| Usability | Unassisted completion, time to first verified result, active user time, errors, assistance and satisfaction | Representative GeoAI developers and newcomers performing the same tasks |
| Distribution | Clean-install success, payload identity, dependency consistency, working commands/examples and actionable diagnostics | Installed wheel outside the source tree, supported OS/Python matrix and reviewed public package boundary |
| Reuse and portability | TorchGeo/TerraTorch integration effort, stable public contracts, hardware capability and numerical qualification | Reusable examples; record which adapters/GPUs were actually tested; simulation is not physical scaling |

## Fair performance comparisons

Maintain two complementary comparisons. A **controlled execution** comparison
uses the same I/O contract and tuned native resident PyTorch/TerraTorch baseline
to isolate runtime contributions. A **product workflow** comparison measures each
recommended installation end to end, including preparation and durable delivery.
Do not attribute custom adapter or receiver inefficiency to upstream defaults.
Record adapter code and configuration; the TerraTorch + vLLM comparator must not
import AUREON model code. Include native resident TerraTorch to expose costs
introduced by a serving integration. Defaults and tuned profiles need explicit labels.

Record artifact/checkpoint/input hashes, normalization, masks/grid, output
modality, sampling steps, precision/TF32/autocast, versions, device/driver,
concurrency, power/clock/thermal conditions and graph/reuse eligibility.
Synchronize asynchronous CUDA work at timing boundaries. Account for graph
capture and warmup separately and publish their break-even job count.

Alternate or randomize execution order in independent run blocks. Acquisitions
and run blocks are the sampling units; tiles and seeds within them are not
independent replicates. Keep all valid observations and predefined exclusions.
Report paired effect sizes and uncertainty at the block/acquisition level;
cluster bootstrap intervals are appropriate only with enough independent units.
A tiny pilot should stay descriptive. A confidence interval crossing no change
does not prove equality; non-inferiority needs a practical margin set before
looking at results. Choose margins and sample size from user requirements and
pilot variability, not from the percentage that happens to win.

Use `completed_jobs / total_elapsed_seconds` for pooled throughput, not the
mean of reciprocals of job latency. Measure serial latency and concurrent
throughput separately. Report p50 and tail latency with the sample count; a
handful of jobs cannot characterize production p99. Include failed/rejected
work when reporting reliability and cost. Account for upload, queue, read,
preparation, inference, reconstruction, export and client verification explicitly.
Do not sum overlapping stage timers as wall time or multiply speedups from
different cohorts. Ablate reuse, graph and I/O on the same workload.

At the development normalization of **1 THB/hour**, warm cost per 1,000 jobs is
`1000 / (3600 * completed_jobs_per_second)` THB. Cold start, idle allocation,
failed attempts, storage and transfer must be added for a service cost estimate.
This normalization is not a cloud provider quote.

## Precision and quality

Keep exact highest-FP32 parity for the unchanged qualified profile. Broader
GPU/precision support needs a declared numerical contract: maximum absolute
and relative error, distributions of errors, token/RNG effects and task quality.
Do not silently loosen existing acceptance or claim FP32 implies bitwise equality
across devices or batching algorithms. PyTorch documents that mathematically
equivalent batched operations can differ numerically.

Evaluate FP16/BF16/TF32 against FP32 with the same model/data/seed and a quality
budget agreed for the intended task. Exact parity and useful scientific output
are distinct requirements. Use paired real observations and downstream metrics
where appropriate; image similarity alone cannot establish physical fidelity.

## Developer first-use study

Use [the task and feedback form](FIRST-USE.md). Evaluate both an EO practitioner
and a developer new to EO; report their experience separately. Have each install,
prepare their input, obtain a result, inspect geospatial metadata and recover a
failed/repeated request. Time setup, downloads and model execution separately.
For comparing two guides or interfaces, counterbalance order or use separate
groups to limit learning effects. Keep incomplete attempts and assistance visible.

Team-operated install tests are engineering evidence. External user usability
and independent research reproduction remain unmeasured until those people run
the workflow. No telemetry or contact upload is enabled by the library.

## Sources and interpretation

- [NIST usability testing](https://www.nist.gov/programs-projects/usability-testing)
  supports task success, time, errors and satisfaction with representative users.
- [MLPerf inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
  distinguish workload scenarios and accuracy/performance conditions. AUREON's
  EO evaluation adapts these principles; it is not an MLPerf submission.
- [ACM SIGIR artifact policy](https://sigir.org/general-information/acm-sigir-artifact-badging/)
  distinguishes reusable artifacts from results reproduced by another team.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
  explains precision and batched-operation differences.
- [Python packaging publication guidance](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
  recommends Trusted Publishing and separating build from publication.

These sources inform the measurement design. They do not certify this library
or supply a universal speedup threshold for a useful product or research paper.
