# Local TerraTorch segmentation tasks

This adapter constructs an actual TerraTorch 1.2.13 `SemanticSegmentationTask`
using `SMPModelFactory`, SMP `Unet`, and the built-in `resnet18` encoder. The
supported schema fixes encoder depth 3 and decoder channels 16/8/4. It accepts
1–32 ordered named input channels and 2–256 classes. Pretrained loading is always
disabled. Only the tested three-channel/two-class random-weight fixture has CPU
execution evidence; other admitted sizes still need their own workload checks.

Use `TerraTorchTaskConfig(bands=(...), num_classes=...)`. Its `to_dict()` produces
the explicit JSON schema; `from_dict()` rejects extra keys, class paths, other
architectures and pretrained settings. There is no executable YAML or dynamic
class-name resolution. Existing arbitrary training configs need deliberate
conversion to this supported schema; unsupported architectures are not guessed.

Launch a CPU-only process with `CUDA_VISIBLE_DEVICES` set to the empty string
**before importing Torch**. Also set `LIGHTLY_DID_VERSION_CHECK=True` before
upstream imports to disable Lightly's optional background version-check request.
The adapter checks these settings explicitly and does not change them. Upstream optional import discovery initialized CUDA
in an exploratory visible-GPU process; the API therefore rejects visible-GPU
process configuration instead of changing visibility globally. Tests also block
network connections and CUDA initialization. Importing `aureon.api` remains cold
and does not import Torch/TerraTorch.

```python
from aureon.api import TerraTorchTaskConfig, TerraTorchSegmentation, PreprocessingRecipe

config = TerraTorchTaskConfig(('red', 'green', 'blue'), 2)
recipe = PreprocessingRecipe(config.bands, 'reflectance', 'reflectance')
with TerraTorchSegmentation.from_checkpoint(
    config, 'local-task.ckpt', expected_sha256=checkpoint_sha256,
    preprocessing=recipe,
) as engine:
    result = engine.predict(image, bands=config.bands, units='reflectance')
    logits, labels = result['logits'], result['labels']
```

The local Lightning file must contain `state_dict` for the **entire task**.
For a raw full-task state dictionary, explicitly pass `checkpoint_format='state_dict'`;
format is never inferred. All keys, including `model.` prefixes and persistent task tensors, must match.
Keys are never stripped or ignored; tensor shape, dtype, finiteness and exact
post-load values are checked. File hash is checked before/after weights-only
reading. Metadata reports every tensor hash. Other Lightning fields, including
`hyper_parameters`, are data only and never configure or execute a model.
Unsafe pickle fallback and remote asset resolution are unavailable.

The required preprocessing recipe is applied exactly once in the prediction
workflow. It explicitly handles band reordering, units, scaling and supplied
normalization statistics. No training augmentation is reused. Inputs are CPU
FP32 B,C,H,W; spatial dimensions must be positive multiples of eight. Outputs
are owned FP32 B,K,H,W logits and int64 B,H,W argmax labels. No softmax,
calibration, georeferencing or scientific quality is implied. Auxiliary outputs
are rejected rather than silently discarded.

This uses the existing isolated model snapshot bridge, including its request
copies, RNG/precision restoration, retryable validation/busy rejection and
terminal execution errors. It is not live borrowed zero-copy execution or an
accelerated resident path. See the [offline example](../examples/generation/terratorch_task.py)
and [bridge lifecycle contract](MODEL-BRIDGE.md). The example creates and removes
its own random-weight test checkpoint; it downloads nothing.

`packaging/ci_terratorch.py --scratch NEW_DIR --output NEW_JSON` exercises actual
upstream CPU components with network/CUDA initialization blocked. `--runtime`
can select an installed package prefix. This suite is separate from the existing
mock-only tensor suite. Torch 2.10/2.11 results, installed-package acceptance and
pretrained acceptance must be distinguished; the latter remains pending.
