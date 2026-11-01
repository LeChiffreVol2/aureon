# Borrowed models, temporal features and explicit adapters

These additive Python interfaces belong to the unpublished v0.1.0 candidate.
They do not change the native TerraMind generation kernels, preprocessing or
checkpoint identities. CPU compatibility is tested with real tensors and the
installed TerraTorch 1.2.13 implementation. No pretrained, speed or new GPU
qualification follows from those tests.

## Reuse a caller-owned model

```python
from aureon.api import BorrowedModelBridge, DeviceTensorPolicy

# model is an existing FP32 nn.Module, already in eval mode on CPU.
with BorrowedModelBridge(model,
        forward=lambda module, values: module(values['image']),
        output=lambda result: {'features': result},
        policy=DeviceTensorPolicy('cpu')) as bridge:
    first = bridge.predict({'image': image}, seed=17)
    second = bridge.predict({'image': next_image}, seed=18)
```

The exact model instance is used for every call. The bridge never calls its
`eval`, `to`, `half`, `close`, or `deepcopy`. The caller must exclude concurrent
access and supply trusted, read-only forward/output adapters. This API accepts
registered finite dense tensors and bounded plain metadata; custom mutable
attributes, unregistered tensors, hooks and inference tensors are rejected.
Floating state and inputs/outputs must be FP32 on the declared device.

In the default audit mode, a baseline copy of registered parameters, buffers and existing gradients is
retained once, bounded by `DeviceTensorPolicy(max_state_bytes=...)`. Before and
after each request, the bridge checks module metadata, tensor identities,
versions and exact bytes. This requires state-sized memory and scans on every
request; it is **not a measured acceleration claim**. Inputs and outputs are
detached owned copies on the same device. No CPU staging is inserted for CUDA
tensors. Returned tensors are ordinary tensors, including under inference mode.

For a validated immutable adapter, an explicit
[`BorrowedValidationPolicy` serving contract](SCALABLE-POLICY.md#borrowed-model-audit-and-serving)
avoids weight snapshots and per-request value scans while retaining metadata,
identity/version and input/output checks. Serving requires exclusive ownership
and a caller-declared contract ID; unsafe tensor-storage mutation is not detected.

Model mutation or a failure after execution begins makes the bridge terminal.
Mutated caller state is not rolled back. `close()` releases bridge references;
the caller remains responsible for the model. Invalid input/seed, global RNG
contention and cancellation before execution remain retryable. Cancellation
after successful synchronized execution discards the result. Running Python
forward code is not forcibly interrupted; cancellation is cooperative.

Python, NumPy and the selected Torch RNG states, FP32 policy, default dtype/device,
gradient mode and the documented deterministic/cuDNN flags are restored on exit.
Other threads must cooperate with the process-wide RNG/policy lock. This is not
a sandbox for arbitrary Python side effects, external globals, background work,
or transient writes hidden through unsafe storage access.

## Explicit device and stream contract

CPU is the tested default. CUDA requires all of the following:

- `DeviceTensorPolicy('cuda:N', allow_unqualified_cuda=True)` with an explicit index.
- A CUDA context already initialized and owned by the caller.
- A caller-supplied `torch.cuda.Stream` on that exact device.
- Model state and all inputs already on that device in the required dtype.

The execution stream waits on the caller's current stream, then synchronizes on
both success and failure before control returns. Producers on other streams
must be explicitly ordered by the caller first. Adapters must not launch
untracked foreign-stream or background work. Keep input/model storage alive
through the call. Closing the bridge does not close the caller's stream/context.
CUDA execution, multi-device behavior and performance remain **unqualified**;
CPU tests check admission without initializing or probing GPUs. No graph capture,
implicit device movement, reduced precision or automatic batching is provided.

## Temporal and geographic inputs

`TemporalSpec` declares ordered band names, units and unique chronological UTC
timestamps. `TemporalInput(values, mask, spec)` copies CPU FP32 **B,C,T,H,W**
values and checks an explicit boolean **B,T,H,W** mask. It requires all-valid,
finite samples; no gap filling or normalization is inferred. Its input budget is
256 MiB. Callers prepare model-specific normalization before constructing it.

`TemporalBridge(encoder, spec=spec, pooling='mean')` wraps the same caller encoder
with the **actual upstream** `terratorch.models.utils.TemporalWrapper`. The
encoder must already be eval/FP32/CPU and return a nonempty list of features.
`mean`, `max`, `keep`, `concat` and `diff` delegate to that upstream implementation.
Optional positive `subset_lengths` must partition all frames; subsets are averaged
first. `diff` requires two frames or two subsets and preserves upstream order
(first minus second). Result timestamps label input frames, not a guessed mapping
between output channels and dates. Feature layout follows upstream pooling.

Set `CUDA_VISIBLE_DEVICES=''` and `LIGHTLY_DID_VERSION_CHECK=True` before upstream
import; use the existing generation dependency profile. The CPU example also
disables upstream online checks. No weights or data are downloaded.

`GeographicConditioning` separately validates explicit latitude/longitude degrees
and encodes either `degrees_lat_lon` or `unit_sphere_xyz`. Image CRS describes a
grid; it is not automatically model conditioning. The temporal wrapper consumes
neither those coordinates nor timestamp embeddings. A model-specific adapter
must explicitly consume any such conditioning and record its semantics.

## Register an explicit adapter

`AdapterRegistry[Options, Adapter]` is instance-local. Register an `AdapterSpec`
with an explicit trusted factory, then call `plan(id, request)` before `create`.
The spec declares input/output kinds, CPU/CUDA, FP32, maximum batch size, graph
support and exact base dependency versions. Installed local build suffixes are
recorded separately; no modules are auto-discovered, imported by string, or
downloaded to satisfy a pin. Unknown adapters raise `KeyError`; unsupported
requests return admission reasons and are rejected before the factory executes.

Only an explicitly requested `graph_fallback='eager'` permits graph-to-eager
fallback. No hidden batch splitting, precision/device substitution or input
dropping occurs. Capability declarations are not runtime qualification; trusted
factories must honor the admitted request. Returned adapters belong to the caller.
Unregistering affects future admissions and never closes an existing adapter.

Run [the complete CPU example](../examples/generation/borrowed_temporal.py) from
an installed package. Source and installed checks use `packaging/ci_terratorch.py`;
the `upstream-cpu` CI job installs the existing hash-locked generation profile
and exercises real random-weight upstream code with network/CUDA initialization
blocked. Preparing that job does not attest a hosted CI run.

For imagery-to-scene delivery, see [bounded STAC windows](STAC-WINDOWS.md).
