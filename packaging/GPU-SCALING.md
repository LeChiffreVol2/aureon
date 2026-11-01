# GPU execution and scaling

The SDK owns model execution. HTTP is an optional client transport over the same
engine; use the SDK first to measure execution without upload/export overhead.
All changes described here are local v0.1.0 source until an artifact-bound release
record explicitly includes them. The older published wheel is unchanged.

## Inspect and select physical GPUs

```sh
aureon-engine devices
```

This explicit driver query reports GPU UUID, compute capability and current free
memory. It imports no Torch, creates no CUDA context and executes no model.
It reports the physical devices visible to `nvidia-smi`, not CUDA process ordinal
numbers or MIG instances. An unavailable driver/query is an error, not zero memory.

```python
from aureon.api import probe_nvidia_gpus, cuda_device_slots

inventory = probe_nvidia_gpus()
# Operator-selected full GPU UUIDs from the inventory, not CUDA ordinals.
selected_uuids = ["GPU-00000000-0000-0000-0000-000000000001"]  # replace
slots = cuda_device_slots(
    inventory,
    allowed_uuids=selected_uuids,
    memory_bytes=4 * 1024**3,   # explicit per-GPU pool envelope
    reserve_bytes=512 * 1024**2,
    max_residents=1,
)
```

Pass `slots` into `ModelPool` and declare the same slot names in each route's
`devices` tuple. Memory figures above illustrate configuration; they are not
estimates for arbitrary checkpoints. The inventory snapshot expires after five
seconds by default and cannot reserve memory. Live worker attestation/admission
and measured route budgets still apply. Device choices explicitly select UUIDs;
they do not implicitly inherit `CUDA_VISIBLE_DEVICES` ordinal semantics.

## Three distinct forms of parallelism

1. **Within one resident model:** `GenerationEngine(parallelism=N)` executes
   compatible batch-one decoder lanes sharing immutable weights. Each lane owns
   its stream, graph buffers and request RNG. Use `generate_many()` or queued
   `submit()` so the collector actually receives concurrent work. A Python loop
   calling `generate()` sequentially is not a concurrent workload.
2. **Several resident models on one GPU:** configure one physical `DeviceSlot`
   with `max_residents=N`. ModelPool admits distinct routes only while combined
   resident/transient/output budgets fit. It does not duplicate one identical
   route N times on that device; generation lanes serve that case.
3. **Several physical GPUs:** provide distinct UUID-bound slots. ModelPool can
   place replicas of an exact route on different devices. Request replication
   does not require splitting every forward across GPUs. Tensor/model parallelism
   and cross-host orchestration are separate, unimplemented execution features.

Do not represent one real GPU as several production slots: alias detection is
intentional. A second slot does not create additional VRAM. CPU test fixtures
may simulate separate identities without claiming physical GPU execution.

## Choose N from workload measurements

`parallelism` accepts any positive integer. The current implementation estimates
each lane from tensor shape/dtype, activations, latent/token state and graph pools,
plus separately declared resident memory and headroom. See the
[resource policy guide](SCALABLE-POLICY.md) for the explicit estimate and lifecycle
budgets. These are admission estimates, not allocator limits.
Increasing queue depth, CUDA streams, resident processes and true neural batch
size are different operations. Current generation lanes retain neural batch one.

Use a finite experiment with the same checkpoint/input/recipe/seed, collect cold
load/capture separately, then measure completed warm requests and actual admitted
lane counts. Compare elapsed wall time, queue latency, allocated/reserved VRAM,
host memory and exact reference outputs. Consume results promptly. Stop a failed
parity or cleanup cohort; preserve it rather than silently retrying it.

Choose the fastest measured feasible configuration, not the largest N that fits.
More lanes can contend for memory bandwidth/SM resources and lose throughput.
The `auto` policy uses resource credits and executable capabilities without GPU
name matching. It does not infer performance or qualification from VRAM/SM count.
Historical evidence remains artifact-bound; inspect reported fallbacks, effective
N and `executable_not_qualified` status for this source revision.

## Simulation on a one-GPU workstation

From a source checkout, with existing test dependencies:

```sh
python -m unittest discover -s tests -p test_pool_scaling.py -v
```

The tests use the real ModelPool scheduler, IPC and spawned process lifecycle,
substituting only device attestation and model calls. They exercise four logical
GPU slots, eight overlapping resident processes, 96 jobs with finite credits,
failed-worker isolation, quarantine and verified closure. The `node` labels are
logical names; these are not networked machines. A separate 64-device placement
case tests stale snapshots, wrong execution identities and insufficient memory.

Simulation proves these control-path properties. It cannot establish CUDA
overlap, physical scaling, PCIe/NVLink bandwidth, network latency, NCCL behavior
or distributed durability. Private provider transport tests remain outside this
library. Deploying a multi-host coordinator requires actual transport, fenced
durable leases and independent validation; placement alone does not dispatch.

## GPU architecture and performance work

Inspect a CPU/CUDA trace before choosing an intervention. On Ampere/SM86, useful
questions include whether the decoder has launch gaps, whether H2D/D2H copies
overlap compute, whether kernels saturate bandwidth or compute, and whether graph
buffer residency limits concurrency. CUDA Graph replay addresses launch cost;
it does not by itself fuse kernels or reduce arithmetic.

Use pinned staging and stream/events only with explicit buffer lifetime ownership.
Use library kernels/SDPA or compilation before introducing a custom Triton/CUDA
kernel, and validate any arithmetic/order change against the reference. A GPU's
TF32/BF16 support is an eligibility fact, not model quality qualification. Keep
precision experiments explicit, preserve FP32 reference, and report numerical
error plus downstream quality independently from speed.

References: [NVIDIA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
[Ampere tuning](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[PyTorch 2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html).
