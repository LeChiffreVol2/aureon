# Trusted upstream task constructors

`UpstreamTaskBridge` accepts an explicit typed Python `TaskConstructor`, JSON-data
`model_args`, a preprocessing recipe, and a declared task result contract. It
extends the [fixed task profile](TERRATORCH-TASK.md) without guessing classes or
claiming every TerraTorch training configuration is supported.

```python
from torch import nn
from aureon.api import UpstreamTaskBridge, PreprocessingRecipe

def construct(args: dict[str, object]) -> nn.Module:
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.models.smp_model_factory import SMPModelWrapper
    return SemanticSegmentationTask(
        model_args={"num_classes": 3},
        model=SMPModelWrapper(nn.Conv2d(2, 3, 1)),
        ignore_index=-1, plot_on_val=False,
    )

recipe = PreprocessingRecipe(('red', 'nir'), 'reflectance', 'reflectance')
with UpstreamTaskBridge(construct, {}, kind='segmentation', channels=3,
                        preprocessing=recipe) as bridge:
    result = bridge.predict(image, bands=('red', 'nir'), units='reflectance')
    logits, labels = result.values, result.labels
```

Start the process with `CUDA_VISIBLE_DEVICES` empty and
`LIGHTLY_DID_VERSION_CHECK=True` before Torch/upstream imports. Construction is
scoped to CPU FP32 and restores caller RNG/precision settings. The constructor
must create a **new** local module and must not acquire assets, reuse/mutate a
caller-owned model, or access external resources. Supplied Python code is trusted,
not sandboxed. Config rejects class-path/executable directives and enabled
pretrained/weights fields, but cannot prove what arbitrary Python code does.
The constructor name in metadata is descriptive, not a code attestation.

Optional `checkpoint`, `expected_sha256` and `checkpoint_format` load either a
Lightning container (default) or an explicitly selected raw state dictionary.
The entire constructed task's key set, including prefixes, must match. Shape,
dtype, finite values, strict loading and post-load equality are checked. Metadata
records file/per-tensor identities. Hyperparameters and optimizer fields in a
Lightning file never choose the constructor or change supplied configuration.

| kind | Required native result | Returned TaskPrediction |
| --- | --- | --- |
| segmentation | FP32 B,K,H,W; K equals declared channels | `.values` logits and `.labels` int64 argmax |
| regression | FP32 B,C,H,W; B,H,W also allowed for one channel | `.values` unchanged; `.labels` is None |
| features | FP32 B,tokens,D; D equals declared channels | `.values` token features; `.labels` is None |

`output_container='model_output'` reads upstream `.output` and rejects auxiliary
heads; `output_container='tensor'` requires a tensor directly. Features do not
claim a raster grid. Dense outputs must preserve **both batch and spatial
dimensions** of their request. There is no automatic resizing, interpolation,
softmax, unit conversion or calibration. Geometry, auxiliary-head, nonfinite or
dtype violations occur inside execution and invalidate the private bridge. The
fixed TerraTorchSegmentation path applies the same batch/spatial rejection.

The explicit recipe is part of predict, including band order/units and exactly-once
normalization rules. Tensor inputs/outputs and model snapshots follow
[ModelBridge ownership](MODEL-BRIDGE.md). Invalid seed/input or shared RNG
contention remains retryable; concurrent execution and close fail immediately.
Closing releases private snapshots. This is a CPU isolation path, not a live
borrowed model or zero-copy accelerated inference.

Tests execute actual randomly initialized TerraTorch segmentation and pixelwise
regression tasks with a custom Conv2d head, compare direct upstream outputs, and
round-trip full-task Lightning checkpoints. A raw checkpoint/token-feature path
uses a small PyTorch module. These are separate from the fixed Unet/resnet18
fixture and from pretrained scientific/GPU acceptance, which remains pending.
