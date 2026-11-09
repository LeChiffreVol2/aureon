# Queued generation and bounded parallelism

## Execution profile and scene size

Use `execution_profile="recommended"` for eligible resident encoder reuse and
decoder graphs, or `"reference"` for reference execution. A 224x224 tensor is a
TerraMind generation tile, not the scene-size limit. Prepared rasters use bounded
tiling/reconstruction through [scene workflows](PRODUCT-WORKFLOWS.md); repeated
scenes may borrow a resident engine through [resident workflows](RESIDENT-WORKFLOWS.md).
Admission uses declared queue, memory and storage budgets. Independent lanes and
replicas do not imply fused neural batching or a speedup on every GPU.

The unpublished `NeuralBatchConfig`, `NeuralBatchSession` and `neural_batch=`
experiment are retired. Omit them from old examples; immutable older wheels keep
their original contents. Existing queue/lane/replica interfaces remain available.

**Best v0.1.0 interfaces.** This checkout contains `GenerationEngine.submit`, ordered `generate_many`,
explicit upstream adapters, raster-scene helpers and a placement planner. The
package version remains **0.1.0**. A reviewed source build needs its own wheel
and source-archive SHA256 values. The original public GitHub release and earlier
private candidates remain separate artifacts. This document does not announce
a new public release.

## Submission and ownership

By default, one resident model owner executes **one request at a time**. Multiple callers can
submit work through a bounded queue. `max_pending`, `max_results` and input/output
byte budgets bound admission and retained outputs; a busy queue rejects admission.
`generate_many` pulls lazily within its window and yields in input order. Consume
each handle once, or discard it to release the retained result credit. Use a
stream context manager when breaking iteration early.

`GenerationJob` carries each input's `GenerationRecipe`, execution profile,
job ID and attempt ID. Request seeds and sampling state remain separate. Queued
work can be cancelled; an active model call finishes before its result is
discarded. Draining and cancellation do not preempt CUDA execution. See the
[developer guide](../docs/DEVELOPER-INTEGRATION.md) for runnable API patterns.

Concurrent submission alone does not select parallel GPU execution. The default
remains `parallel="off"`, with a maximum neural batch size of one.

## Explicit two-lane generation

Set `parallel="two-lane"` on `GenerationEngine` to collect up to two compatible
FIFO requests for independent B1 decoder graphs. `pair_wait_ms=5.0` bounds the
collection window; the accepted range is 0–1000 milliseconds. Submit two jobs or
use an ordered stream with at least two input/result credits:

```python
with GenerationEngine(
    model_file=model_file, tokenizer_file=tokenizer_file,
    work=fresh_work_directory, execute_model=True,
    parallel="two-lane", pair_wait_ms=5.0,
) as engine:
    first = engine.submit(first_input, seed=17)
    second = engine.submit(second_input, seed=101)
    first_result = first.result()
    second_result = second.result()
```

Use validated `GenerationInput` objects and the pinned assets from the developer
guide. Each result owns its own CPU arrays. Each lane owns a stream, graph pool,
input/output buffers and scheduler; model weights are shared and immutable.
Stochastic token preparation stays request-local. Neural batch size remains
one; this is not the fused B2 candidate.

The imported implementation was previously qualified on **RTX 3070 8 GiB, Linux/WSL**, with the
pinned TerraMind base/S1GRD tokenizer, all-valid S2L2A224, eight token steps,
ten DDIM steps, temperature1.0/top-p0.8/top-k0, highest FP32. Cold seeds17/101
and warm seeds101/17 preserve exact raw outputs, token payloads and request RNG;
both lane pools close and CUDA allocation/reservation returns to zero. Current
admission uses capability and resource checks, with B1 fallback for ineligible
groups. That evidence belongs to the original private artifacts; it does not qualify
a new build of this checkout. The recorded seed pairs are the tested sample,
not exhaustive scene/seed coverage.

Timeout singletons, incompatible neighbors, reference-profile requests and
insufficient pair credits use the existing B1 entry point. Only an explicit
pre-execution admission rejection permits serial fallback. An execution failure
never retries the pair. Queued cancellation removes that request; running discard
drops only its eventual result and allows its neighbor to finish. Failed lane
cleanup terminates the session, including after cancellation.

`engine.capabilities()` describes the source qualification and
`result["parallel_selection"]` records the selected path. Source qualification
and installed-artifact acceptance are separate. Stream event intervals include
waits and do not prove overlapping kernels or throughput improvement. No new
performance comparison or physical multi-GPU scaling measurement is claimed.

## Workflow bridges and spatial scope

The TorchGeo adapter requires explicit bands, units, normalization, validity,
grid and source identity. It does not infer validity from a segmentation label
or silently replace dataset transforms. The TerraTorch bridge accepts only the
documented TerraMind 1.0 base configuration for TerraTorch 1.2.13. It does not
accept arbitrary factories, mutate upstream model objects or download weights.

The raster helper admits bounded, all-valid 224-pixel windows, derives explicit
per-tile seeds and exports georeferenced FP32 COG output with provenance. CPU
contracts cover scheduling, blending and storage bounds. Real-model evidence
covers **one 224 x 224 tile only**; multi-tile seams, geophysical accuracy and
large-scene generative quality remain unqualified. Generated S1GRD is synthetic.

## Planner and service scope

The placement planner compares explicit compatible worker snapshots and capacity
estimates. It is a pure planner: it does not reserve a GPU, acquire a lease or
dispatch work. The service separately implements cooperative single-host device
ownership and explicit device binding. Each worker still owns a single model
execution at a time. Planner snapshots alone are not atomic scheduling or a
qualified multi-GPU service. No Ray or LLM router is part of this path.

Use the [replica service guide](REPLICA-SERVICE.md) for explicit fresh-state v2
workers and their resource ownership. This path keeps B1 numerical execution;
the library pair collector is not an automatic service batching policy.
Use the [async service guide](ASYNC-SERVICE.md) for authenticated local
upload/submit/status/download. Hosted operation, distributed storage and
multi-GPU execution require their own implementation and qualification.

## Evidence boundary

The imported implementation has functional evidence on earlier, separately
identified private artifacts, including exact FP32 output and cleanup checks.
Those results are not GPU acceptance of this checkout or a future source build.
Current CPU contract checks exercise API behavior and deterministic doubles;
they do not establish a speedup, fresh dependency resolution or broad hardware
support. Published GitHub archives remain unchanged.

## Configurable parallelism

The generic API accepts `parallelism=1`, a positive integer N, or `"auto"`.
N is the requested maximum number of B1 decoder lanes in one resident model;
it is separate from neural batch size, queued requests, retained results and
replica count. The default is1. The older `parallel="two-lane"` spelling is
retained as an alias for2; choose one spelling, not both.

For prepared `GenerationInput` data and explicit local model assets:

```python
from aureon.api import GenerationEngine, GenerationJob, GenerationRecipe

with GenerationEngine(
    model_file="/path/to/TerraMind_v1_base.pt",
    tokenizer_file="/path/to/TerraMind_Tokenizer_S1GRD.pt",
    work="/path/to/existing-parent/fresh-session",
    execute_model=True,
    parallelism=3,
    allow_unqualified_parallelism=True,
    parallel_fallback="error",
) as engine:
    print(engine.capabilities()["parallel"])
    jobs = [GenerationJob(data, recipe=GenerationRecipe(seed=seed))
            for seed in (17, 101, 17)]
    with engine.generate_many(jobs) as results:
        for result in results:
            print(result["parallel_selection"])
            # Export explicitly to a unique COG path when the input has a grid.
```

The example performs real inference when run. The fresh work directory must have
an existing parent; no model download is performed. A prepared, all-valid S2L2A224
input and the pinned TerraMind base/S1GRD assets remain required. Arbitrary shapes,
models and output modalities are not enabled by increasing parallelism.

`parallel_wait_ms` bounds collection delay (default5ms, range0–1000ms). Omitted
pending/result count credits derive from the requested lanes within the workspace
budget, while preserving the default two queue/result slots for serial use.
Explicit `max_pending`, `max_results`, `max_input_bytes` and `max_output_bytes`
override those defaults. Partial groups and singleton work need not wait for N
requests indefinitely. The core grows graph pools only to the admitted group
size, shares immutable weights and reuses already initialized lanes.

Memory options are `parallel_resident_bytes` (default3GiB) and
`parallel_workspace_bytes` (default2GiB). Planning derives lane reservations from
the [workload memory estimate](SCALABLE-POLICY.md#workload-and-device-planning),
with a default 512MiB free-device floor. These are planning assumptions, not
measured universal bounds or an allocator hard limit. Raising a budget does not
create memory or qualify the GPU. Queued snapshots
and delivered results use separate host byte/count credits. Caller-retained
arrays after delivery are owned by the caller.

Each result and its provenance report `requested`, `planned`, `admitted`, `actual`,
`group_actual`, qualification and fallback reason. A large N is never evidence
that N lanes actually ran. `parallel_fallback="serial"` permits reported B1
fallback before execution; `"error"` rejects failed group admission. It does not
force every partial FIFO group to contain N requests. A failed or partly executed
group is never retried silently as singleton work.

`auto` uses queue/memory credits and native capability admission, independently
of historical evidence counts and GPU names. The compatibility flag
`allow_unqualified_parallelism` is retained but is no longer a hardware whitelist
override. Current source receipts say `executable_not_qualified`; they preserve
historical evidence without transferring it to this installed artifact or device.
No GPU parity or speed advantage is inferred for untested profiles. Highest FP32
remains the numerical policy.

For several GPUs on one host, use the separate replica service and explicit
operator limits described in `REPLICA-SERVICE.md`. Each HTTP replica still executes
one B1 request at a time. Library N-lane collection is not automatic HTTP batching.
Multi-host scale requires an external transactional authority and storage adapter;
the current SQLite service is confined to one Linux host/UID/PID namespace.
