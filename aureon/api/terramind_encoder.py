"""Resident TerraMind Backbone/TiM features, with explicit local asset ownership."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import importlib
from os import PathLike
from pathlib import Path
import re
import sys
import threading
import traceback as traceback_tools
from types import TracebackType
from typing import TYPE_CHECKING, Literal, TypedDict

from ..terramind_native.encoder_loader import (
    CHANNELS, FeatureLayer, LoadedEncoder, MergeMethod, Metadata, fp32_policy,
    load_encoder, required_tokenizers, validate_options,
)
from ..terramind_native.rng import rng_capsule
from .v1 import ExecutionRequired
from .precision import (ComputePrecision, allowed_feature_dtypes, compute_precision_scope,
                        precision_preflight, validate_precision)

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class TerraMindFeatures(TypedDict):
    features: list[FeatureLayer]
    provenance: Metadata


_ENCODER_IMPORTS = (
    'terratorch.models.backbones.terramind.model.terramind_register',
    'terratorch.models.backbones.terramind.tokenizer.tokenizer_register',
)
_IMPORT_LOCK = threading.Lock()


def _encoder_import_target(devices: tuple[int, ...], errors: list[BaseException]) -> None:
    """No model/request/self arguments or closure may enter this thread's stack."""
    try:
        with rng_capsule(0, devices=devices), fp32_policy():
            for name in _ENCODER_IMPORTS:
                importlib.import_module(name)
    except BaseException as error:
        errors.append(error)


def _prepare_encoder_modules(devices: tuple[int, ...]) -> None:
    """Keep retained optional-import tracebacks outside model/request ancestry.

    TerraTorch can retain a caught optional-import exception in a module global.
    Its traceback/f_back frames would otherwise retain the active encoder,
    caller and outputs. An isolated thread imports before model construction,
    preserving upstream diagnostics and the loader's RNG/FP32 policy. Warm
    modules need no thread. This creates no model and loads no checkpoint.
    """
    with _IMPORT_LOCK:
        if all((module := sys.modules.get(name)) is not None and
               not getattr(getattr(module,'__spec__',None),'_initializing',False)
               for name in _ENCODER_IMPORTS):
            return
        errors: list[BaseException] = []
        worker = threading.Thread(target=_encoder_import_target, args=(devices,errors),
                                  name='aureon-encoder-import', daemon=True)
        worker.start()
        worker.join()
        if errors:
            raise errors.pop()


def _own_features(value: object, *, merge_method: MergeMethod,
                  precision: ComputePrecision = 'highest_fp32',
                  native_dtypes: set[str] | None = None) -> list[FeatureLayer]:
    import torch
    if not isinstance(value, list) or not value:
        raise ValueError('Upstream encoder did not return feature layers')
    def own(tensor: object) -> Tensor:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in allowed_feature_dtypes(precision) or tensor.ndim != 3:
            raise ValueError('Expected native FP32 or policy-selected (B, tokens, channels) feature tensors')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite native features')
        if native_dtypes is not None:
            native_dtypes.add(str(tensor.dtype))
        if tensor.dtype != torch.float32:
            return tensor.detach().to(dtype=torch.float32, copy=True)
        return tensor.detach().clone()
    result: list[FeatureLayer] = []
    for layer in value:
        if merge_method == 'dict':
            if not isinstance(layer, dict) or not layer or any(not isinstance(k, str) for k in layer):
                raise ValueError('Expected native modality feature dictionaries')
            result.append({k: own(v) for k, v in layer.items()})
        else:
            result.append(own(layer))
    return result


class TerraMindEncoder:
    """Lazy, engine-owned Backbone or TiM model; outputs are features, not predictions.

    Every configured input must be supplied as a full-band 224x224 batch. Inputs
    are copied before upstream use. This facade adds no normalization, band
    selection, masking, resizing, georeferencing or prediction head. TiM callers
    must supply original full-band inputs, never precomputed tokens/features.
    """

    def __init__(self, model_id: str, *, model_file: str | PathLike[str],
                 modalities: Sequence[str], tokenizer_files: Mapping[str, str | PathLike[str]] | None = None,
                 tim_modalities: Sequence[str] | None = None, device: str = 'cpu',
                 execute_model: bool = False, merge_method: MergeMethod = 'mean',
                 tim_decoding_steps: int = 1, tim_temperature: float = 1.0,
                 tim_top_p: float = 0.8, tim_top_k: int = 0,
                 tim_chained_generations: bool = True,
                 precision: ComputePrecision = 'highest_fp32',
                 allow_experimental_precision: bool = False) -> None:
        if execute_model is not True:
            raise ExecutionRequired('Pass execute_model=True explicitly; no model was loaded')
        self._spec, self._options = validate_options(model_id, modalities, tim_modalities,
            merge_method, tim_decoding_steps, tim_temperature, tim_top_p, tim_top_k, tim_chained_generations)
        if not isinstance(device, str) or re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device) is None:
            raise ValueError('device must be cpu, cuda or cuda:N')
        validate_precision(precision, mode=self._spec.mode, device=device,
                           allow_experimental_precision=allow_experimental_precision)
        self._precision = precision
        self._allow_experimental_precision = allow_experimental_precision
        supplied = dict(tokenizer_files or {})
        needed = required_tokenizers(self._options)
        if set(supplied) != set(needed):
            raise ValueError('tokenizer_files must exactly contain: ' + ', '.join(needed))
        self._tokenizer_files = {k: Path(v).absolute() for k, v in supplied.items()}
        self._model_file = Path(model_file).absolute()
        self._device_name = device
        self._loaded: LoadedEncoder | None = None
        self._state: Literal['unloaded', 'ready', 'failed', 'closed'] = 'unloaded'
        self._lock = threading.Lock()

    @property
    def state(self) -> Literal['unloaded', 'ready', 'failed', 'closed']:
        return self._state

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    def configuration(self) -> Metadata:
        """Detached explicit construction identity; no model/file/CUDA probing.

        Paths identify operator configuration, while load-time pinned hashes
        attest actual weights. Changing this returned dictionary cannot change
        the encoder's options or ownership.
        """
        return {'model_id': self.model_id, 'model_file': str(self._model_file.absolute()),
                'tokenizer_files': {k: str(v.absolute()) for k, v in self._tokenizer_files.items()},
                'device': self._device_name, 'options': asdict(self._options),
                'precision': self._precision, 'weights_verified': self._loaded is not None,
                'allow_experimental_precision': self._allow_experimental_precision}

    def _inputs(self, inputs: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
        import torch
        if not isinstance(inputs, Mapping) or set(inputs) != set(self._options.modalities):
            raise ValueError('Supply every configured modality exactly once, with no extra inputs')
        batch: int | None = None
        for name in self._options.modalities:
            tensor = inputs[name]
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.device.type == 'meta':
                raise ValueError('Inputs must be real dense torch tensors')
            shape = tuple(tensor.shape)
            expected_tail = (224, 224) if name == 'LULC' else (CHANNELS[name], 224, 224)
            if len(shape) != len(expected_tail) + 1 or shape[0] < 1 or shape[1:] != expected_tail:
                raise ValueError('Expected full-band 224x224 batch for ' + name)
            if batch is not None and batch != shape[0]:
                raise ValueError('Input batch sizes must match')
            batch = shape[0]
            if name == 'LULC':
                if tensor.dtype != torch.int64 or bool((tensor < 0).any()) or bool((tensor > 9).any()):
                    raise ValueError('LULC must contain int64 class indices 0..9 with shape (B, H, W)')
            elif tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Continuous inputs must be finite FP32 tensors: ' + name)
        return {k: inputs[k].detach().to(device=device, copy=True).contiguous() for k in self._options.modalities}

    def predict(self, inputs: Mapping[str, Tensor], *, seed: int = 17) -> TerraMindFeatures:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('TerraMindEncoder is busy')
        try:
            if self._state in ('failed', 'closed'):
                raise RuntimeError('TerraMindEncoder is ' + self._state)
            import torch
            device = torch.device(self._device_name)
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise ValueError('Requested CUDA device is unavailable')
                device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
            precision_preflight(self._precision, device)
            owned_inputs = self._inputs(inputs, device)
            devices = (device.index,) if device.type == 'cuda' and device.index is not None else ()
            model: torch.nn.Module | None = None
            modes: list[tuple[torch.nn.Module, bool]] = []
            native: object = None
            module: torch.nn.Module | None = None
            try:
                if self._loaded is None:
                    _prepare_encoder_modules(devices)
                    self._loaded = load_encoder(self._spec, self._options, self._model_file,
                                                self._tokenizer_files, device)
                    self._state = 'ready'
                model = self._loaded.model
                modes = [(module, module.training) for module in model.modules()]
                try:
                    with rng_capsule(seed, devices=devices) as rng, compute_precision_scope(self._precision, device) as compute:
                        model.eval()
                        native = model(owned_inputs)
                finally:
                    for module, training in modes:
                        module.training = training
                native_dtypes: set[str] = set()
                features = _own_features(native, merge_method=self._options.merge_method,
                                         precision=self._precision, native_dtypes=native_dtypes)
                compute['observed_native_output_dtypes'] = sorted(native_dtypes)
                provenance = deepcopy(self._loaded.provenance)
                provenance.update(seed=seed, rng=deepcopy(rng), output_kind='backbone_features',
                    feature_layout='B,tokens,channels', input_preprocessing='none; caller supplies full-band upstream inputs',
                    merge_method=self._options.merge_method, input_modalities=list(self._options.modalities),
                    tim_modalities=list(self._options.tim_modalities), ownership='engine-owned model; request-owned input/output copies',
                    qualification='implementation_and_synthetic_tests_only')
                provenance['compute_precision'] = compute
                if self._precision != 'highest_fp32':
                    provenance['precision'] = self._precision
                    provenance['qualification'] = 'unqualified_experimental_precision'
                return {'features': features, 'provenance': provenance}
            except BaseException as error:
                self._loaded = None
                self._state = 'failed'
                traceback_tools.clear_frames(error.__traceback__)
                raise
            finally:
                owned_inputs.clear()
                modes.clear()
                model = module = None
                native = None
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close a busy TerraMindEncoder')
        try:
            self._loaded = None
            self._state = 'closed'
        finally:
            self._lock.release()

    def __enter__(self) -> TerraMindEncoder:
        if self._state in ('failed', 'closed'):
            raise RuntimeError('TerraMindEncoder is ' + self._state)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()
