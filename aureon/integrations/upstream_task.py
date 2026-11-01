"""Explicit trusted upstream constructors and CPU task output contracts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Literal, Protocol, cast

from .model_bridge import ModelBridge, _validate_model, _validate_seed
from .preprocessing import PreprocessingRecipe
from ..terramind_native.encoder_loader import fp32_policy
from ..terramind_native.rng import rng_capsule

if TYPE_CHECKING:
    from torch import Tensor, nn


class TaskConstructor(Protocol):
    def __call__(self, model_args: dict[str, object], /) -> nn.Module: ...


TaskKind = Literal['segmentation', 'regression', 'features']


@dataclass(frozen=True)
class TaskPrediction:
    kind: TaskKind
    values: Tensor
    labels: Tensor | None


def _config(value: Mapping[str, object]) -> dict[str, object]:
    def inspect(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError('Config keys must be strings')
                if key in ('class_path', '_target_', 'pretrained_path', 'checkpoint_path'):
                    raise ValueError('Executable/implicit asset config is unsupported')
                if ('pretrained' in key or key in ('weights', 'encoder_weights')) and child is not False and child is not None:
                    raise ValueError('Pretrained acquisition must be disabled explicitly')
                inspect(child)
        elif isinstance(item, list):
            for child in item: inspect(child)
    inspect(dict(value))
    normalized = cast(dict[str, object], json.loads(json.dumps(dict(value), allow_nan=False)))
    inspect(normalized)
    return normalized


class UpstreamTaskBridge:
    """CPU snapshot bridge for a caller-selected trusted upstream constructor.

    Constructor identity is a supplied code object, never resolved from config.
    It must construct a NEW local CPU model with no pretrained acquisition or
    external side effects. Config validation cannot sandbox trusted Python code.
    """
    def __init__(self, constructor: TaskConstructor, model_args: Mapping[str, object], *,
                 kind: TaskKind, preprocessing: PreprocessingRecipe, channels: int,
                 output_container: Literal['model_output', 'tensor'] = 'model_output',
                 checkpoint: str | Path | None = None, expected_sha256: str | None = None,
                 checkpoint_format: Literal['lightning', 'state_dict'] = 'lightning', seed: int = 17) -> None:
        import torch
        _validate_seed(seed)
        if not callable(constructor) or kind not in ('segmentation', 'regression', 'features'):
            raise ValueError('Provide a trusted constructor and supported task kind')
        if type(channels) is not int or channels < (2 if kind == 'segmentation' else 1):
            raise ValueError('Declare task output channels/classes/feature dimension')
        if output_container not in ('model_output', 'tensor') or checkpoint_format not in ('lightning', 'state_dict'):
            raise ValueError('Declare supported output/checkpoint containers')
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('LIGHTLY_DID_VERSION_CHECK') != 'True'
                or cast(Callable[[], bool], torch.cuda.is_initialized)()):
            raise ValueError('Start CPU-only/offline process before upstream imports: CUDA_VISIBLE_DEVICES empty, LIGHTLY_DID_VERSION_CHECK=True')
        args = _config(model_args)
        if (checkpoint is None) != (expected_sha256 is None):
            raise ValueError('Checkpoint and expected SHA256 must be supplied together')
        state: dict[str, Tensor] | None = None
        identity: dict[str, object] = {}
        if checkpoint is not None:
            path = Path(checkpoint)
            if path.is_symlink() or not path.is_file():
                raise ValueError('Checkpoint must be a regular local file')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != expected_sha256: raise ValueError('Checkpoint SHA256 mismatch')
                stream.seek(0)
                payload = torch.load(stream, map_location='cpu', weights_only=True)
                stream.seek(0)
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('Checkpoint changed during read')
            if not isinstance(payload, dict): raise ValueError('Expected checkpoint dictionary')
            extracted = payload.get('state_dict') if checkpoint_format == 'lightning' else payload
            if not isinstance(extracted, dict): raise ValueError('Missing explicit state_dict')
            state = extracted
            identity = {'sha256': digest, 'format': checkpoint_format, 'strict': True,
                        'hyper_parameters_used': False}
        with rng_capsule(seed), fp32_policy(), torch.device('cpu'):
            model = constructor(deepcopy(args))
            if not isinstance(model, torch.nn.Module): raise TypeError('Constructor must return nn.Module')
            model.eval()
            _validate_model(model)
            if state is not None:
                expected = model.state_dict()
                if set(state) != set(expected): raise ValueError('Strict task keys mismatch; prefixes are preserved')
                hashes: dict[str, object] = {}
                for key, tensor in state.items():
                    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                            or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype
                            or not bool(torch.isfinite(tensor).all())):
                        raise ValueError('Checkpoint shape/dtype/value mismatch: ' + str(key))
                    hashes[key] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'sha256': hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
                model.load_state_dict(state, strict=True)
                loaded = model.state_dict()
                if any(not torch.equal(loaded[k], v) for k, v in state.items()):
                    raise ValueError('Post-load tensor identity mismatch')
                identity['tensor_identity'] = hashes
        self._constructor_name = str(getattr(constructor, '__module__', '')) + '.' + str(getattr(constructor, '__qualname__', type(constructor).__name__))
        self.kind, self.channels, self.preprocessing = kind, channels, preprocessing
        self._container, self._args, self._identity = output_container, args, identity
        self._bridge = ModelBridge(model, forward=self._forward, output=self._output, seed=seed)

    @property
    def state(self) -> Literal['ready', 'failed', 'closed']:
        return self._bridge.state

    def _forward(self, model: nn.Module, inputs: Mapping[str, Tensor], /) -> object:
        import torch
        image = inputs['image']
        native = model(image)
        if self._container == 'model_output':
            if getattr(native, 'auxiliary_heads', None):
                raise ValueError('Auxiliary outputs require a dedicated contract')
            value = getattr(native, 'output', None)
        else:
            value = native
        if not isinstance(value, torch.Tensor) or value.shape[0:1] != image.shape[0:1]:
            raise ValueError('Task output must preserve request batch')
        if self.kind == 'features':
            if value.ndim != 3 or value.shape[1] < 1 or value.shape[2] != self.channels:
                raise ValueError('Features require B,tokens,declared_dimension')
        else:
            valid = value.ndim == 4 and value.shape[1] == self.channels
            if self.kind == 'regression' and self.channels == 1 and value.ndim == 3: valid = True
            if not valid or value.shape[-2:] != image.shape[-2:]:
                raise ValueError('Dense task output must preserve request spatial grid and declared channels')
        return value

    def _output(self, value: object, /) -> Mapping[str, Tensor]:
        return {'values': cast('Tensor', value)}

    def predict(self, image: Tensor, *, bands: tuple[str, ...], units: str,
                seed: int = 17, cancel: threading.Event | None = None) -> TaskPrediction:
        import torch
        _validate_seed(seed)
        prepared = self.preprocessing.apply(image, bands=bands, units=units)
        value = self._bridge.predict({'image': prepared.tensor}, seed=seed, cancel=cancel)['values']
        with torch.inference_mode(False), torch.no_grad():
            labels = value.argmax(1) if self.kind == 'segmentation' else None
        return TaskPrediction(self.kind, value, labels)

    def metadata(self) -> dict[str, object]:
        return {**self._bridge.metadata(), 'ownership': 'engine_owned_snapshot', 'kind': self.kind, 'channels': self.channels,
                'model_args': deepcopy(self._args), 'checkpoint': deepcopy(self._identity),
                'output_container': self._container, 'preprocessing': repr(self.preprocessing),
                'constructor': self._constructor_name, 'constructor_identity_scope': 'descriptive_name_not_code_attestation',
                'qualification': 'caller_constructor_cpu_contract_only'}

    def close(self) -> None: self._bridge.close()

    def __enter__(self) -> UpstreamTaskBridge:
        self._bridge.__enter__()
        return self

    def __exit__(self, *args: object) -> None: self.close()
