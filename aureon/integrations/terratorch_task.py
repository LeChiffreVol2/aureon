"""Bounded TerraTorch 1.2.13 semantic task/Lightning checkpoint interoperability."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from .model_bridge import ModelBridge, _validate_model, _validate_seed
from .preprocessing import PreprocessingRecipe
from ..terramind_native.encoder_loader import fp32_policy
from ..terramind_native.rng import rng_capsule

if TYPE_CHECKING:
    from torch import Tensor, nn


@dataclass(frozen=True)
class TerraTorchTaskConfig:
    """Fixed offline SMP Unet/resnet18 depth-3 segmentation schema."""
    bands: tuple[str, ...]
    num_classes: int

    def __post_init__(self) -> None:
        if (not isinstance(self.bands, tuple) or not 1 <= len(self.bands) <= 32
                or any(not isinstance(b, str) or not b.strip() for b in self.bands)
                or len(set(self.bands)) != len(self.bands)):
            raise ValueError('Declare 1..32 unique ordered band names')
        if type(self.num_classes) is not int or not 2 <= self.num_classes <= 256:
            raise ValueError('num_classes must be 2..256')

    def to_dict(self) -> dict[str, object]:
        return {'schema': 'aureon.terratorch.segmentation.v1', 'bands': list(self.bands),
                'num_classes': self.num_classes, 'task': 'SemanticSegmentationTask',
                'factory': 'SMPModelFactory', 'architecture': 'Unet', 'backbone': 'resnet18',
                'encoder_depth': 3, 'decoder_channels': [16, 8, 4], 'pretrained': False}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TerraTorchTaskConfig:
        bands = value.get('bands')
        if not isinstance(bands, list) or any(not isinstance(b, str) for b in bands):
            raise ValueError('bands must be a JSON string list')
        config = cls(tuple(cast(list[str], bands)), cast(int, value.get('num_classes')))
        if dict(value) != config.to_dict() or value.get('pretrained') is not False:
            raise ValueError('Unsupported task config; use the exact declared schema, no class paths or pretrained assets')
        return config


def build_terratorch_task(config: TerraTorchTaskConfig, *, seed: int = 17) -> nn.Module:
    """Construct real upstream components with random weights in a CPU-only process.

    Launch with CUDA_VISIBLE_DEVICES empty before importing Torch. TerraTorch's
    optional import discovery can otherwise initialize a visible GPU. This API
    refuses to change process visibility or install global CUDA monkeypatches.
    """
    import torch
    _validate_seed(seed)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or cast(Callable[[], bool], torch.cuda.is_initialized)():
        raise ValueError('Launch a CPU-only process with CUDA_VISIBLE_DEVICES empty before Torch import')
    if os.environ.get('LIGHTLY_DID_VERSION_CHECK') != 'True':
        raise ValueError('Set LIGHTLY_DID_VERSION_CHECK=True before upstream import to disable its background network check')
    if importlib.metadata.version('terratorch') != '1.2.13':
        raise ValueError('This task adapter supports TerraTorch 1.2.13 only')
    with rng_capsule(seed), fp32_policy(), torch.device('cpu'):
        constructor = cast("Callable[..., nn.Module]", getattr(importlib.import_module('terratorch.tasks'), 'SemanticSegmentationTask'))
        task = constructor(model_factory='SMPModelFactory', model_args={
            'backbone': 'resnet18', 'model': 'Unet', 'bands': list(range(len(config.bands))),
            'in_channels': len(config.bands), 'num_classes': config.num_classes,
            'pretrained': False, 'smp_encoder_depth': 3, 'smp_decoder_channels': [16, 8, 4],
        }, ignore_index=-1, plot_on_val=False).eval()
        _validate_model(task)
    return task


class SegmentationPrediction(TypedDict):
    logits: Tensor
    labels: Tensor


class TerraTorchSegmentation:
    """Owned snapshot task with explicit preprocessing; no training transforms."""
    def __init__(self, task: nn.Module, config: TerraTorchTaskConfig,
                 preprocessing: PreprocessingRecipe, *, seed: int = 17) -> None:
        if preprocessing.bands != config.bands or preprocessing.workflow != 'custom':
            raise ValueError('Task preprocessing must declare matching bands and custom workflow')
        self.config, self.preprocessing = config, preprocessing
        self._bridge = ModelBridge(task, forward=self._forward, output=self._output, seed=seed)
        self._checkpoint: dict[str, object] = {}
        self._ownership = 'borrowed_snapshot'

    def _forward(self, model: nn.Module, inputs: Mapping[str, Tensor], /) -> object:
        import torch
        image = inputs['image']
        value = model(image)
        tensor = getattr(value, 'output', None)
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                or tensor.shape[0] != image.shape[0] or tensor.shape[-2:] != image.shape[-2:]
                or getattr(value, 'auxiliary_heads', None)):
            raise ValueError('Task output must preserve request batch and spatial grid, without auxiliary outputs')
        return value

    def _output(self, value: object, /) -> Mapping[str, Tensor]:
        import torch
        tensor = getattr(value, 'output', None)
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                or tensor.shape[1] != self.config.num_classes or getattr(value, 'auxiliary_heads', None)):
            raise ValueError('Expected upstream ModelOutput with class logits and no auxiliary heads')
        return {'logits': tensor}

    @classmethod
    def from_checkpoint(cls, config: TerraTorchTaskConfig, checkpoint: str | Path, *,
                        expected_sha256: str, preprocessing: PreprocessingRecipe,
                        seed: int = 17, checkpoint_format: Literal['lightning', 'state_dict'] = 'lightning') -> TerraTorchSegmentation:
        import torch
        _validate_seed(seed)
        if checkpoint_format not in ('lightning', 'state_dict'):
            raise ValueError('Declare lightning or state_dict checkpoint format')
        path = Path(checkpoint)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Checkpoint must be a regular local Lightning file')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected_sha256:
                raise ValueError('Checkpoint SHA256 mismatch')
            stream.seek(0)
            payload = torch.load(stream, map_location='cpu', weights_only=True)
            stream.seek(0)
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Checkpoint changed during read')
        if not isinstance(payload, dict):
            raise ValueError('Expected a checkpoint dictionary')
        state = payload.get('state_dict') if checkpoint_format == 'lightning' else payload
        if not isinstance(state, dict):
            raise ValueError('Expected Lightning container with explicit state_dict')
        # hyper_parameters/optimizer/trainer fields are never executed or used to
        # guess architecture. Every task tensor, including model. prefixes, stays.
        task = build_terratorch_task(config, seed=seed)
        expected = task.state_dict()
        if set(state) != set(expected):
            raise ValueError('Strict Lightning task key mismatch; no prefix stripping')
        tensors: dict[str, object] = {}
        for name, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Checkpoint tensor mismatch: ' + str(name))
            tensors[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        with rng_capsule(seed), fp32_policy():
            task.load_state_dict(state, strict=True)
            loaded = task.state_dict()
            if any(not torch.equal(loaded[k], v) for k, v in state.items()):
                raise ValueError('Loaded task tensors differ from checkpoint')
        result = cls(task, config, preprocessing, seed=seed)
        result._ownership = 'engine_owned_snapshot'
        result._checkpoint = {'sha256': digest, 'strict': True, 'tensor_identity': tensors,
                              'format': checkpoint_format, 'hyper_parameters_used': False}
        return result

    def predict(self, image: Tensor, *, bands: tuple[str, ...], units: str,
                seed: int = 17, cancel: threading.Event | None = None) -> SegmentationPrediction:
        import torch
        _validate_seed(seed)
        prepared = self.preprocessing.apply(image, bands=bands, units=units)
        if any(n < 8 or n % 8 for n in prepared.tensor.shape[-2:]):
            raise ValueError('This encoder requires spatial dimensions divisible by eight')
        result = self._bridge.predict({'image': prepared.tensor}, seed=seed, cancel=cancel)
        with torch.inference_mode(False), torch.no_grad():
            labels = result['logits'].argmax(dim=1)
        return {'logits': result['logits'], 'labels': labels}

    def metadata(self) -> dict[str, object]:
        from copy import deepcopy
        return {**self._bridge.metadata(), 'ownership': self._ownership, 'task_config': self.config.to_dict(),
                'checkpoint': deepcopy(self._checkpoint), 'preprocessing': repr(self.preprocessing),
                'qualification': 'upstream_random_weight_cpu_only'}

    def close(self) -> None:
        self._bridge.close()

    def __enter__(self) -> TerraTorchSegmentation:
        self._bridge.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
