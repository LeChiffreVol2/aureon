# Explicit model and preprocessing bridges

`ModelBridge` accepts an evaluation-mode CPU FP32 PyTorch/TerraTorch module and
explicit forward/output callables. See the [offline example](../examples/generation/model_bridge.py).
Imports do not load Torch or acquire assets. The bridge currently runs CPU only.

The borrowed constructor snapshots the module using standard `deepcopy` semantics.
Exclude caller mutation during construction. Each forward uses another private
copy, including buffers and ordinary Python module attributes. Inputs and returned
tensors are independent copies. Closing drops bridge references; it never calls
the caller model's close method. Copy hooks and adapters are trusted developer
code and must not mutate external objects or acquire resources. Modules with
external resources or nonstandard copy behavior require a dedicated adapter.
This compatibility path is intentionally separate from graph or automatic batch
eligibility; neither optimization is enabled. Model mode/device/dtype are checked,
not silently changed on the caller object. Request RNG and FP32 policy are scoped.
Overlapping predict/close calls fail immediately. Cancellation is cooperative
before forward and before returning results; active CPU work is not interrupted.
Forward/output failures invalidate the bridge; cancellation leaves it usable.
Invalid inputs/seeds and a busy global RNG capsule are rejected before execution
and leave a ready bridge retryable. The caller chooses when to retry; the library
does not wait or retry implicitly. Models and checkpoints must contain finite,
dense CPU tensors with FP32 floating-point weights. Complex and quantized models
are outside this bridge's declared contract. `metadata()` returns independent
JSON-compatible data, including the capability record.

This snapshot bridge copies the model at construction and again for each
request. It is a compatibility/isolation path for CPU workflows, not the
resident accelerated TerraMind engine. Use `GenerationEngine` or
`TerraMindEncoder` for the supported resident workflows. The separate
[BorrowedModelBridge](BORROWED-ADAPTERS.md) reuses a caller model under explicit
audit/serving ownership; accepting a module does not grant it accelerated or
device-native qualification. Use the [existing-environment installation](CANDIDATE-QUICKSTART.md)
to avoid replacing an existing training environment with native-generation pins.

For local checkpoints, use `ModelBridge.from_checkpoint(config, path,
expected_sha256=..., factory=..., forward=..., output=...)`. Config is JSON-compatible
data supplied to a trusted explicit factory, never a dynamically imported class
or executable YAML. The factory must construct locally with pretrained loading
disabled. The checkpoint must be a raw state dictionary readable with
`weights_only=True`. Keys, tensor shapes and dtypes must exactly match; values
are checked again after strict loading. Metadata includes file SHA256 and
per-tensor hashes. The generic ModelBridge reads raw dictionaries. For full-task
Lightning containers and a fixed supported native task schema use the
[TerraTorch task adapter](TERRATORCH-TASK.md); for explicit trusted upstream
constructors and segmentation/regression/features contracts use
[UpstreamTaskBridge](UPSTREAM-TASK.md). Neither resolves arbitrary class paths.

`PreprocessingRecipe` requires declared ordered bands and units. It reorders
exactly the declared band set, scales/offsets, then optionally applies declared
per-band mean/std. It performs no resizing, masking, augmentation or unit guessing.
It rejects a `PreparedTensor` passed back for transformation and rejects a second
normalization of data labeled standardized. Metadata cannot detect a caller who
strips the wrapper and lies about units. CPU B,C,H,W FP32 is the current input.
Dimensions must be positive. A no-op recipe preserves tensor bytes, including
signed zero. Results are ordinary owned tensors even when called inside
`torch.inference_mode()`, and remain usable by downstream code after that scope.
Normalization statistics always use the input's declared CPU device; the helper
does not change the caller's default device. Standardized data cannot be relabelled
as raw reflectance through this API.

The `generation_s2l2a` contract keeps all twelve pretrained bands and delivers
raw reflectance x10000; the native generation route owns normalization.
`encoder_s2l2a` requires the full bands and explicitly standardized output for
Backbone/TiM, using caller-supplied statistics or already-standardized input.
These recipes compose with existing APIs; they do not modify native numerical
paths or grant arbitrary-band support to Generation/TiM.

CPU fixtures establish direct callable/checkpoint parity and ownership contracts.
Random-weight TerraTorch task construction and strict checkpoint round trips have
CPU evidence in the task adapters above. Pretrained output parity, CUDA tensors,
additional generation modalities, source/sink reconstruction, coverage policies,
broader task-specific postprocessing and temporal/coordinate adapters remain pending.
Coordinates will require a model conditioning contract separate from COG CRS.
