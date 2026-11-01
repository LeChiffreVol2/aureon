# Explicit Backbone compute precision

AUREON v0.1.0 keeps `highest_fp32` as the default. The Python
`TerraMindEncoder` interface also offers three **experimental, unqualified**
compute policies for Backbone only. TiM and generation keep their existing
FP32 execution; these options do not change their samplers or diffusion stages.

```python
from aureon.api import TerraMindEncoder

with TerraMindEncoder(
    "terramind-v1-base-backbone",
    model_file="/models/TerraMind_v1_base.pt",
    modalities=["S2L2A"],
    device="cuda:0",
    execute_model=True,
    precision="amp_bf16",
    allow_experimental_precision=True,
) as encoder:
    # User-owned, correctly preprocessed FP32 full-band input; no input conversion
    # or normalization is inferred from the selected compute precision.
    result = encoder.predict({"S2L2A": image}, seed=17)
    features = result["features"]       # Owned FP32 feature tensors
    policy = result["provenance"]["compute_precision"]
```

| Policy | Forward behavior | Supported device scope |
| --- | --- | --- |
| `highest_fp32` | Existing highest-FP32 policy; TF32/autocast disabled | Existing CPU/CUDA paths |
| `tf32` | FP32 tensors; permits TF32 in eligible CUDA matmul/CuDNN kernels | NVIDIA CUDA, compute capability 8.0+ |
| `amp_bf16` | CUDA autocast targets BF16 for eligible operators; other operators may remain FP32 | NVIDIA CUDA, compute capability 8.0+, native BF16 query succeeds |
| `amp_fp16` | CUDA autocast targets FP16 for eligible operators; other operators may remain FP32 | NVIDIA CUDA, compute capability 7.0+ |

The FP16 architecture boundary is AUREON's conservative supported scope, not a
claim that older hardware cannot perform FP16 arithmetic. Construction validates
the explicit opt-in, mode and device without loading a model or probing a GPU.
At `predict`, preflight checks the selected device, available NVIDIA CUDA build,
capability and, for BF16, `is_bf16_supported(including_emulation=False)`. ROCm,
CPU, emulated BF16 and unsupported devices are rejected for experimental modes.
TF32 is also rejected when `NVIDIA_TF32_OVERRIDE=0` disables it in the environment.
There is no automatic fallback to FP32. This check does not prove a checkpoint
fits memory, that every kernel is supported, or that its results are acceptable.

Weights are still strictly loaded as FP32 from the same pinned local checkpoints.
Continuous public inputs remain FP32; categorical LULC inputs remain int64.
Input copies and resident weights are not converted to half precision. Only the
explicit Backbone forward runs inside the requested compute context, including
any upstream input-tokenizer operations used by that configured Backbone.
Experimental policies select the resolved CUDA device before constructing
autocast and keep it selected through the forward, then restore the caller's
device. AMP disables its autocast weight cache and FP16/BF16 reduced-precision
matmul reductions plus reduced-precision math-SDP reduction. Split-K heuristics
remain permitted, matching Torch's normal backend-compatible setting; disabling
them would require cuBLASLt on Torch 2.10. AUREON does not switch the BLAS backend.
FP16 accumulation is disabled when that Torch setting is available. These
settings constrain eligible operations; they do not establish numerical parity.
Its output must be FP32 or the requested AMP dtype, finite, and in the native
feature layout. Invalid dtypes, NaN and Inf fail the call and invalidate the
resident instance. Valid low-precision outputs are copied to owned FP32 tensors;
this conversion does not recover precision lost during computation.

The request holds AUREON's RNG capsule across the policy scope. Default dtype,
matmul precision, CuDNN TF32/benchmark flags, the selected CUDA device and
thread-local autocast dtype/enabled/cache state are restored on success and
failure. AMP also restores both reduction and split-k flags exactly, math-SDP
reduction and optional FP16 accumulation. The unchanged highest-FP32 policy and
TF32 policy retain caller reduction/accumulation settings and cache policy.
The scope uses the existing matmul precision and TF32 APIs; it does not mix them
with Torch's newer per-backend `fp32_precision` controls.
Existing isolated cold imports and strict FP32
loading keep their own unchanged policy scopes. Overlapping AUREON RNG scopes
fail immediately. Applications must not concurrently change process-global
precision flags outside this coordination boundary.

`configuration()` records the requested precision and experimental opt-in.
Returned provenance separately records the requested/applied policy, active
autocast dtype/cache state, TF32 permissions, AMP reduction/accumulation settings,
observed native output dtypes, and public FP32 output dtype. An unavailable
optional FP16 accumulation setting is recorded as `None`.
TF32 permission is not proof that every kernel used TF32; the AMP
target is not a per-operator dtype trace. New policies never inherit the FP32
artifact's numerical or performance qualification.

The compute scope is factored for later, deliberately scoped execution stages.
It is not wired into generation, TiM, ModelPool, HTTP or CLI precision selection.
Do not wrap a full generation sampler in autocast based on these Backbone checks.

Validation here consists of synthetic CPU policy/runtime tests, unchanged encoder
cold-import/lifecycle/file-flow regressions, strict implementation typing and CPU
execution of the policy tests with existing Torch 2.10/2.11. CUDA discovery is
mocked and CUDA autocast thread-local state is exercised without GPU tensors or
kernels. No reduced-precision real-model parity, quality, memory, latency or
throughput result is claimed. Real GPU validation requires separate explicit work.
