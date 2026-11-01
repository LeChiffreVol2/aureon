# Resource policies and adapter serving

AUREON separates executable capabilities, dependency compatibility and recorded
qualification. A device name is not an admission rule. CPU simulations in this
increment validate decisions and ownership, not GPU throughput or scientific quality.

## Workload and device planning

```python
from aureon.api import DeviceCapability, ParallelPolicy, WorkloadMemory

policy = ParallelPolicy("auto", workspace_bytes=8 * 1024**3,
                        workload=WorkloadMemory(batch_size=1))
limits = dict(max_pending=8, max_results=8,
              max_input_bytes=8 * (12 * 4 + 1) * 224 * 224,
              max_output_bytes=8 * (2 * 4 + 1) * 224 * 224)
# Caller-supplied metadata for planning; this does not probe or reserve a GPU.
device = DeviceCapability("cuda", total_bytes=24 * 1024**3,
                          free_bytes=20 * 1024**3,
                          compute_capability=(8, 6), graph=True)
plan = policy.plan(limits, None, capability=device)
print(plan["planned"], plan["fallback_reason"])
```

`WorkloadMemory` declares input/output shapes and dtypes, batch size, model-state
bytes, latent/token shapes, activation multiplier, retained graph copies and
pinned copies. Weights are charged once; request tensors scale with batch size.
Graph pools are charged independently for each retained lane/shape bucket.
Multipliers are explicit estimates, not measured peak-memory formulas.
Use measured peaks from your workload to choose conservative values.

`ParallelPolicy.plan` bounds concurrency by queue credits, workspace, pinned RAM,
output buffers and optional current free-device memory. Planning happens before
loading a model: it subtracts the resident budget
and reserve from the supplied free-memory snapshot. Runtime admission instead
uses observed residency after model load. Call `bound_queue_limits` before
creating a queue so completed outputs stay bounded until consumed or discarded.
Insufficient resources for even one estimated lane raise `MemoryError`; choose the explicit reference
path or revise the budget. An incompatible backend/topology explains a serial
reference selection. Unsupported shapes are not silently padded or resized.

The native `open_lane_session` path uses this estimate for retained graph pools,
new lane admission and free-memory reserve, with observed pre-lane residency and
a post-execution reservation check. It still implements only the existing
TerraMind base S2L2A224 → synthetic S1GRD FP32 B1 graph lanes. Generic estimator
support for other shapes/dtypes/batches does not implement those native routes.
Native recipe eligibility is checked separately before preparation. A rejected
group can use the existing explicit serial fallback; a failed running group is
never retried automatically. Reference execution remains available.

Auto admission no longer compares marketing GPU names or historical lane counts.
The old `allow_unqualified` flag remains accepted for source compatibility;
capability admission is independent of historical qualification. Every new lane
receipt says `executable_not_qualified` and preserves the old evidence as a
historical reference, without transferring it to another artifact or hardware.

These budgets are accounting and admission controls, **not an allocator hard
cap**. CUDA fragmentation, other processes and provider allocations can exceed
estimates. An observed envelope violation terminates the group without retry.
Close the owning session to release retained graph pools. Output ownership lasts
until consumption/discard, beyond device completion.

For adapters and schedulers, `MemoryBudget` provides atomic device/pinned/output
credits. Reserve model/cache credits once, retain graph credits until eviction,
release request workspace after synchronized completion, and release output
credits only after consumption or discard. A budget refuses close with live
leases and refuses double release. It does not itself allocate/free memory.

## Borrowed model audit and serving

The default remains `BorrowedValidationPolicy(mode="audit")`: one state snapshot
and exact state checks before and after each request. Existing callers retain this
behavior. The module is never copied, moved, cast or put into eval mode by AUREON.

For an adapter whose read-only behavior you have validated, explicitly select:

```python
from aureon.api import BorrowedModelBridge, BorrowedValidationPolicy

# model, forward and output are supplied by the caller, already in eval mode.
validation = BorrowedValidationPolicy(
    mode="serving", exclusive_ownership=True, contract_id="my-adapter-v1")
# bridge = BorrowedModelBridge(model, forward=forward, output=output,
#                              validation=validation)
```

The contract ID is the caller's assertion, not a certification issued by AUREON.
The caller excludes all other use and mutation until close. Serving scans finite
state once at admission and retains identities/version counters, without a
weight-value snapshot or per-request weight-value scans. Module metadata,
registered state identity/version, device/dtype and input/output values remain
checked. Input/output storage is still detached and owned.

Serving cannot detect unsafe `.data` or storage writes that bypass version
counters. Neither mode sandboxes arbitrary Python side effects. Do not use serving
to test an untrusted adapter. Changing a model requires closing the bridge and
constructing a new one; no hot replacement is implicit. Mutation or a started
forward failure remains terminal, with no rollback of caller state. Close only
drops bridge references. `DeviceTensorPolicy.max_state_bytes` still bounds the
admitted registered state in either mode; audit additionally stores that snapshot.

## Compatibility and installation

| Contract | Current scope | What it establishes |
|---|---|---|
| Default package install recipe | Mandatory pins in `pyproject.toml` | Supported native resolver inputs; unchanged |
| Explicit core build profile | No mandatory runtime dependencies; exact backend extras | Integrate with an existing CPU Torch environment without installing a native stack |
| Generation reproducibility | [Exact hash lock](lock-generation-linux-py312-cu128.txt) | Repeatable dependency artifacts, with separate model/hardware receipts |
| Borrowed CPU bridge | Python 3.12, NumPy 2.2.6; local CPU fixtures on Linux/Torch 2.10.0 and Windows/Torch 2.11.0 | Tested API behavior on those pairs; not every platform/build combination or a GPU guarantee |
| TerraMind native generation | Exact [generation profile](requirements-generation-linux-cu128.txt) | Dependency admission; runtime/model qualification is separate |
| COP-GEN | [Explicit torch210/torch260 profiles](../aureon/copgen_native/PROFILE.md) | Separate exact backend constraints, including runtime CUDA build |

`inspect_backend_compatibility("borrowed-cpu")` reads installed distribution
metadata without importing Torch, probing CUDA or installing packages. Its
`dependency_compatible` result is separate from `execution_qualified`, which stays
false. Optional version snapshots support deployment planning but are caller
assertions, not worker attestations. Local build suffixes are retained in the
result; compatible CPU API versions do not qualify those builds for GPU execution.

No dependency ranges were widened. The default wheel retains its native pins.
An **explicit core build profile** offers the same package/import name with no
mandatory runtime dependencies and separate exact backend extras. It is an
alternative build of version 0.1.0, not a second package to co-install. From a
reviewed source checkout, ordinary pip dependency resolution can use:

```bash
# Existing Torch 2.11 CPU integration; pip resolves dependencies normally.
python -m pip install ".[borrowed-torch211]" --config-settings aureon-profile=core

# Or select the tested Torch 2.10 CPU bridge.
python -m pip install ".[borrowed-torch210]" --config-settings aureon-profile=core
```

The core extras are `borrowed-torch210`, `borrowed-torch211`, `geo`, `generation`,
`copgen` and `scene-legacy`. The last three retain their exact native dependencies;
selecting `generation` does not permit Torch 2.11. Without extras, core supports
dependency-light planning/imports only; actual adapters need their declared
runtime dependencies. No automatic downloads or backend selection are added.
PEP 517 metadata preparation and wheel construction use the same selected profile.
The staged sdist records it explicitly so rebuilding does not revert to default.

Core is currently a local developer build option, not a separately published
release. Use separate output directories and record profile plus artifact hash:
core/default share the package name, version and wheel filename. Do not put both
in one index or directory. If replacing an already installed AUREON at the same
version, remove that AUREON distribution first; retain the environment's Torch
and other dependencies, then install the selected profile normally. Native GPU
acceptance does not transfer to core or to Torch 2.11.

Maintainers can stage/build it with `packaging/build_review.py --profile core
--output <fresh-directory>` in the pinned build environment. The default command
still builds the original native dependency profile. Metadata-only resolver
validation does not establish an installed core wheel; that check belongs to the
final merged build.

Use the [locked install guide](LOCKED-INSTALL.md) for ordinary native installation and
keep conflicting native stacks separate. Hosted CI coverage is configured in the
workflow, not claimed as a completed remote run.

## Contributing an adapter

Declare an instance-local `AdapterSpec`, exact tested dependency versions,
input/output semantics, supported devices, precision and batch limit. Registration
does not confer qualification. Keep downloads, model creation and CUDA setup out
of imports. Return independent output storage and expose explicit `close`.

`check_adapter_conformance(factory, inputs, expected)` runs two caller-supplied
CPU fixtures through a fresh disposable adapter, verifies exact fixture output,
input preservation and independent output storage, and closes the owner even on
failure. It mutates the first returned output to exercise isolation. This small
fixture check does not replace model-specific mask/band/grid, cancellation,
failure and checkpoint-identity tests. It executes no benchmark.

Run in the existing CPU test environment:

```bash
python -I -B packaging/ci_scalability.py --scratch /tmp/aureon-policy-new --output /tmp/aureon-policy.json
python -I -B packaging/check_implementation.py --scratch /tmp/aureon-types-new --output /tmp/aureon-types.json
```

Use fresh output paths. The runner blocks network and CUDA initialization, checks
runtime import locations, and can take `--runtime` for an installed candidate.
The typed CI matrix runs the same fixture checks for both supported Torch CPU
versions. A model-specific GPU acceptance remains separate work.
