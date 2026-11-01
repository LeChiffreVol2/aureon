"""Offline TerraMind encoder construction with complete active tensor binding."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import traceback
from typing import TYPE_CHECKING, Literal, TypeAlias, cast
from unittest.mock import patch

from .families import AssetPin, TerraMindSpec, TOKENIZER_ASSETS, get_terramind_spec
from .rng import rng_capsule

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

MergeMethod: TypeAlias = Literal['mean', 'max', 'concat', 'dict'] | None
FeatureLayer: TypeAlias = 'Tensor | dict[str, Tensor]'
Metadata: TypeAlias = dict[str, object]
CHANNELS = {'S2L2A': 12, 'S2L1C': 13, 'S2RGB': 3, 'S1GRD': 2,
            'S1RTC': 2, 'DEM': 1, 'LULC': 1, 'NDVI': 1}
TOKEN_KEYS = {'S2L2A': 'tok_sen2l2a@224', 'S1GRD': 'tok_sen1grd@224',
              'S1RTC': 'tok_sen1rtc@224', 'DEM': 'tok_dem@224',
              'LULC': 'tok_lulc@224', 'NDVI': 'tok_ndvi@224'}


@dataclass(frozen=True)
class EncoderOptions:
    modalities: tuple[str, ...]
    tim_modalities: tuple[str, ...]
    merge_method: MergeMethod
    tim_decoding_steps: int
    tim_temperature: float
    tim_top_p: float
    tim_top_k: int
    tim_chained_generations: bool


@dataclass
class LoadedEncoder:
    model: nn.Module
    provenance: Metadata


def validate_options(model_id: str, modalities: Sequence[str], tim_modalities: Sequence[str] | None,
                     merge_method: MergeMethod, tim_decoding_steps: int, tim_temperature: float,
                     tim_top_p: float, tim_top_k: int, tim_chained_generations: bool) -> tuple[TerraMindSpec, EncoderOptions]:
    spec = get_terramind_spec(model_id)
    if spec.mode not in ('backbone', 'tim'):
        raise ValueError('TerraMindEncoder requires a backbone or tim model ID')
    def names(value: Sequence[str], allowed: Mapping[str, object]) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not value or any(not isinstance(v, str) or v not in allowed for v in value):
            raise ValueError('Provide a nonempty sequence of canonical supported modalities')
        if len(set(value)) != len(value):
            raise ValueError('Duplicate modalities are not allowed')
        return tuple(value)
    inputs = names(modalities, CHANNELS)
    targets = names(('LULC',) if tim_modalities is None else tim_modalities, TOKEN_KEYS) if spec.mode == 'tim' else ()
    if spec.mode == 'backbone' and tim_modalities is not None:
        raise ValueError('Backbone does not accept tim_modalities')
    if any(m in inputs and m not in ('LULC', 'NDVI') for m in targets):
        raise ValueError('TiM input/target overlap requires a tokenized input modality (LULC or NDVI)')
    if merge_method not in ('mean', 'max', 'concat', 'dict', None):
        raise ValueError('Unsupported feature merge method')
    if type(tim_decoding_steps) is not int or not 1 <= tim_decoding_steps <= 196:
        raise ValueError('tim_decoding_steps must be an integer in [1, 196]')
    if isinstance(tim_temperature, bool) or not math.isfinite(tim_temperature) or tim_temperature <= 0:
        raise ValueError('tim_temperature must be finite and positive')
    if isinstance(tim_top_p, bool) or not math.isfinite(tim_top_p) or not 0 <= tim_top_p <= 1:
        raise ValueError('tim_top_p must be finite in [0, 1]')
    if type(tim_top_k) is not int or tim_top_k < 0 or type(tim_chained_generations) is not bool:
        raise ValueError('Invalid TiM top-k or chained-generation option')
    return spec, EncoderOptions(inputs, targets, merge_method, tim_decoding_steps,
                                tim_temperature, tim_top_p, tim_top_k, tim_chained_generations)


def required_tokenizers(options: EncoderOptions) -> tuple[str, ...]:
    return tuple(sorted(set(options.tim_modalities) | (set(options.modalities) & {'LULC', 'NDVI'})))


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_asset(path: Path, pin: AssetPin) -> None:
    if not path.is_file() or path.stat().st_size != pin.bytes or file_digest(path) != pin.sha256:
        raise ValueError('Local asset does not match pinned bytes/SHA256: ' + pin.filename)


@contextmanager
def fp32_policy() -> Iterator[None]:
    """Caller must hold the native RNG capsule, also serializing global policy changes."""
    import torch
    dtype = torch.get_default_dtype()
    precision = torch.get_float32_matmul_precision()
    tf32, benchmark = torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = benchmark


def _checkpoint(path: Path) -> dict[str, Tensor]:
    import torch
    value: object = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(value, dict) or not value or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in value.items()):
        raise ValueError('Expected a nonempty flat tensor checkpoint')
    return cast('dict[str, Tensor]', value)


def bind_encoder_state(model: nn.Module, checkpoint: Mapping[str, Tensor],
                       tokenizer_states: Mapping[str, Mapping[str, Tensor]]) -> Metadata:
    """Bind all active state, duplicating enc/dec keys only for TiM's sampler namespace."""
    import torch
    expected = model.state_dict()
    selected: dict[str, Tensor] = {}
    used: set[str] = set()
    used_tokens: dict[str, set[str]] = {k: set() for k in tokenizer_states}
    for key, target in expected.items():
        if key.startswith('tokenizer.'):
            _, name, source_key = key.split('.', 2)
            source = tokenizer_states.get(name, {})
            if name in used_tokens:
                used_tokens[name].add(source_key)
        else:
            source_key = key.removeprefix('sampler.model.')
            source = checkpoint
            used.add(source_key)
        if source_key not in source:
            raise ValueError('Missing active tensor: ' + key)
        value = source[source_key]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Active tensor shape/dtype mismatch: ' + key)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite active tensor: ' + key)
        selected[key] = value
    for name, state in tokenizer_states.items():
        if set(state) != used_tokens[name]:
            raise ValueError('Unexpected or unused tokenizer tensors: ' + name)
    unused = sorted(set(checkpoint) - used)
    active_embedding_groups = {tuple(k.split('.')[:2]) for k in used if k.startswith(('encoder_embeddings.', 'decoder_embeddings.'))}
    has_decoder = any(k.startswith('sampler.model.') for k in expected)
    for key in unused:
        inactive_embedding = key.startswith(('encoder_embeddings.', 'decoder_embeddings.')) and tuple(key.split('.')[:2]) not in active_embedding_groups
        inactive_decoder = not has_decoder and (key.startswith(('decoder.', 'decoder_norm.', 'decoder_proj_context.')) or key == 'mask_token')
        if not inactive_embedding and not inactive_decoder:
            raise ValueError('Unexpected active checkpoint tensor: ' + key)
    loaded = model.load_state_dict(selected, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Strict loading did not bind all active tensors')
    hashes: dict[str, str] = {}
    for key, value in model.state_dict().items():
        if not torch.equal(value, selected[key]):
            raise ValueError('Loaded tensor value mismatch (including aliases): ' + key)
        hashes[key] = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {'all_active_tensors_matched': True, 'active_tensor_count': len(hashes),
            'active_tensor_sha256': hashes, 'unused_inactive_checkpoint_keys': unused,
            'random_fallback_tensors': 0}


def load_encoder(spec: TerraMindSpec, options: EncoderOptions, model_file: Path,
                 tokenizer_files: Mapping[str, Path], device: torch.device) -> LoadedEncoder:
    """Construct upstream modules with pretrained=False, then require exact local state."""
    import torch
    if set(tokenizer_files) != set(required_tokenizers(options)):
        raise ValueError('tokenizer_files must exactly cover the required input/TiM tokenizers')
    verify_asset(model_file, spec.checkpoint)
    for name, path in tokenizer_files.items():
        verify_asset(path, TOKENIZER_ASSETS[name])
    if importlib.metadata.version('terratorch') != '1.2.13':
        raise ValueError('Encoder source contract requires TerraTorch 1.2.13')
    # A cold TerraTorch import changes global matmul precision through clay_v15.
    # Capture before importing, including import failures, then enter the normal
    # construction policy below with the caller's original state restored.
    devices = (device.index if device.index is not None else torch.cuda.current_device(),) if device.type == 'cuda' else ()
    with rng_capsule(0, devices=devices), fp32_policy():
        registry = importlib.import_module('terratorch.models.backbones.terramind.model.terramind_register')
        tokenizers = importlib.import_module('terratorch.models.backbones.terramind.tokenizer.tokenizer_register')
    constructor = cast('Callable[..., nn.Module]', getattr(registry, spec.upstream_name))
    kwargs: dict[str, object] = {'pretrained': False, 'modalities': list(options.modalities),
                                 'img_size': 224, 'merge_method': 'mean' if options.merge_method == 'concat' else options.merge_method}
    if spec.mode == 'tim':
        kwargs.update(tim_modalities=list(options.tim_modalities), tim_decoding_steps=options.tim_decoding_steps,
                      tim_temps=options.tim_temperature, tim_top_p=options.tim_top_p,
                      tim_top_k=options.tim_top_k, tim_chained_generations=options.tim_chained_generations)
    model: nn.Module | None = None
    state: dict[str, Tensor] = {}
    token_states: dict[str, dict[str, Tensor]] = {}
    try:
        with rng_capsule(0), fp32_policy(), ExitStack() as stack:
            for target in (registry, tokenizers):
                stack.enter_context(patch.object(target, 'hf_hub_download', side_effect=RuntimeError('Implicit downloads forbidden')))
            with torch.device('cpu'):
                model = constructor(**kwargs)
            state = _checkpoint(model_file)
            token_states = {TOKEN_KEYS[k]: _checkpoint(path) for k, path in tokenizer_files.items()}
            bound = bind_encoder_state(model, state, token_states)
            # All configured modalities are present. Do not invent the upstream
            # concat placeholder parameter, which is only used for missing inputs.
            if options.merge_method == 'concat':
                setattr(model, 'merge_method', 'concat')
            model.eval().requires_grad_(False)
            if any(t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
                raise ValueError('Encoder/tokenizer state must be entirely FP32')
            verify_asset(model_file, spec.checkpoint)
            for name, path in tokenizer_files.items():
                verify_asset(path, TOKENIZER_ASSETS[name])
            model.to(device=device)
        return LoadedEncoder(model, {'model_id': spec.model_id, 'mode': spec.mode, 'size': spec.size,
            'checkpoint': asdict(spec.checkpoint), 'tokenizers': {k: asdict(TOKENIZER_ASSETS[k]) for k in tokenizer_files},
            'tensor_binding': bound, 'upstream_constructor': spec.upstream_name, 'terratorch': '1.2.13',
            'options': asdict(options), 'device': str(device), 'precision': 'float32/highest',
            'pretrained_downloads': False, 'new_real_weight_qualification': False})
    except BaseException as error:
        # Preserve traceback locations while releasing completed loader frames' tensors.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        model = None
        state.clear()
        token_states.clear()
