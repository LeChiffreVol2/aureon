"""Opt-in, unqualified batch-two denoising for pinned TerraMind S1GRD224.

Serial upstream token/quant/noise preparation stops immediately BEFORE its first
decoder forward. Only deterministic decoder calls are batched. Each DDIM state
and scheduler step remains request-owned. No accepted B1 implementation changes.

Upstream integration references: TerraTorch 1.2.13 TerraMindGeneration, DiVAE and
PipelineCond (IBM / EPFL / Apple, Apache-2.0); retain the project's third-party
notices. The installed, hash-verified implementations are called, not vendored.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast, NoReturn
from collections.abc import Callable, Iterator, Sequence, Mapping
from types import ModuleType
from ._types import TerraMindModel, Metadata, InputSnapshot, FloatArray, PreparedState, StageRunner, PrepareRunner
if TYPE_CHECKING:
    from torch import Tensor
    from .session import _NativeRuntime
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import random
import types

from .common import canonical, digest, read_json, validate_recipe
from .session import _decision

TARGET = 'tok_sen1grd@224'
PROFILE = 'experimental_s1grd224_fp32_ddim_batch2_eager_v1'
IMAGE_BYTES = 2 * 224 * 224 * 4
COND_BYTES = 5 * 14 * 14 * 4
# Explicit tensor reservation, excluding model/activation/kernel workspace.
WORKSPACE_BYTES = 8 * IMAGE_BYTES + 4 * COND_BYTES
_MISSING = object()


class BatchIneligible(ValueError):
    """Rejected before token preparation; the caller may explicitly select B1."""


class _PreparedBoundary(BaseException):
    """Internal suspension at the first native decoder call, never a result."""


def compatible_recipes(recipes: Sequence[Mapping[str, Any]]) -> list[Metadata]:
    if not isinstance(recipes, (tuple, list)) or len(recipes) != 2:
        raise BatchIneligible('Exactly two explicitly supplied recipes are required')
    return compatible_group_recipes(recipes)


def compatible_group_recipes(recipes: Sequence[Mapping[str, Any]]) -> list[Metadata]:
    if not isinstance(recipes,(tuple,list)) or not recipes:
        raise BatchIneligible('Supply a finite nonempty recipe group')
    recipes = [validate_recipe(r) for r in recipes]
    first = {k: v for k, v in recipes[0].items() if k != 'seed'}
    if any({k: v for k, v in r.items() if k != 'seed'} != first for r in recipes):
        raise BatchIneligible('Sampling recipes must match except for request-local seeds')
    if first['diffusion_steps'] != 10 or first['token_steps'] not in (1, 8):
        raise BatchIneligible('Batch-two candidate supports one/eight token steps and ten DDIM steps')
    return cast("list[Metadata]", recipes)


def scheduler_identity(scheduler: Any) -> Metadata:
    module = importlib.import_module(
        'terratorch.models.backbones.terramind.tokenizer.scheduling.scheduling_ddim')
    if type(scheduler) is not module.DDIMScheduler:
        raise BatchIneligible('Only the pinned deterministic DDIMScheduler is admitted; DDPM is unsupported')
    if any(name in vars(scheduler) for name in ('step', 'set_timesteps')):
        raise BatchIneligible('Instance scheduler overrides are unsupported')
    if inspect.signature(scheduler.step).parameters['eta'].default != 0.0:
        raise BatchIneligible('Native DDIM eta default changed')
    config = dict(scheduler.config)
    canonical(config)
    return {'class': type(scheduler).__module__ + '.' + type(scheduler).__name__,
            'config': config, 'config_sha256': digest(config), 'eta': 0.0,
            'generator': None, 'variance_noise': None, 'timesteps_mode': 'trailing'}


def eligibility(runtime: _NativeRuntime, recipes: Sequence[Mapping[str, Any]], *, max_workspace_bytes: int, min_free_cuda_bytes: int, graph: bool=False) -> Metadata:
    recipes = compatible_recipes(recipes)
    if graph is not False:
        raise BatchIneligible('Batch-two graph buffers are not implemented; the B1 graph is never reused')
    return native_group_eligibility(runtime,recipes,max_workspace_bytes=max_workspace_bytes,
                                    min_free_cuda_bytes=min_free_cuda_bytes)


def native_group_eligibility(runtime: _NativeRuntime,recipes: Sequence[Mapping[str, Any]],*,max_workspace_bytes: int,min_free_cuda_bytes: int) -> Metadata:
    recipes=compatible_group_recipes(recipes)
    if type(max_workspace_bytes) is not int or max_workspace_bytes < WORKSPACE_BYTES:
        raise BatchIneligible('Insufficient explicit denoiser tensor workspace reservation')
    if type(min_free_cuda_bytes) is not int or min_free_cuda_bytes < 0:
        raise BatchIneligible('min_free_cuda_bytes must be a nonnegative integer')
    if any(runtime.config['dependencies'].get(k) != v for k,v in
           {'terratorch':'1.2.13','torch':'2.10.0','diffusers':'0.39.0'}.items()):
        raise BatchIneligible('Batch adapter requires the inspected generation dependency versions')
    # load_native already verifies every file against this manifest. Require the
    # reviewed manifest as well, rather than any caller-supplied set of 37 files.
    if read_json(runtime.paths['source_manifest'])['files'] != \
            read_json(Path(__file__).with_name('upstream-source.json'))['files']:
        raise BatchIneligible('Batch adapter requires the packaged pinned upstream source inventory')
    from .denoiser_binding import eligibility as decoder_eligibility
    core = runtime.numerical_core
    check_decoder = core.graph_eligibility if core is not None else decoder_eligibility
    ok, reason = check_decoder(cast("TerraMindModel", runtime.model), recipes[0], runtime.device)
    if not ok: raise BatchIneligible(reason)
    model = cast("TerraMindModel", runtime.model)
    if any(module.training for module in model.modules()):
        raise BatchIneligible('All model modules must remain in eval mode')
    tokenizer = model.tokenizer[TARGET]
    if type(tokenizer).__name__ != 'DiVAE' or type(tokenizer).__module__ != \
            'terratorch.models.backbones.terramind.tokenizer.vqvae':
        raise BatchIneligible('Expected pinned DiVAE tokenizer')
    if any(name in vars(tokenizer) for name in ('decode_tokens', 'decode_quant', '_get_pipeline')):
        raise BatchIneligible('Tokenizer method overrides are unsupported')
    if 'forward' in vars(tokenizer.decoder):
        raise BatchIneligible('Decoder already patched; B1 graph/custom hooks cannot compose')
    if (model.temps != recipes[0]['temperature'] or model.top_p != recipes[0]['top_p']
            or model.top_k != recipes[0]['top_k']):
        raise BatchIneligible('Loaded sampler settings do not match request recipes')
    scheduler = scheduler_identity(tokenizer.noise_scheduler)
    free, total = cast("ModuleType", runtime.torch).cuda.mem_get_info()
    if free < min_free_cuda_bytes + WORKSPACE_BYTES:
        raise BatchIneligible('Insufficient free CUDA memory for the requested admission reserve')
    return {'profile_id': PROFILE, 'scheduler': scheduler, 'batch_size': len(recipes),
            'image_shape': [len(recipes), 2, 224, 224], 'conditioning_shape': [len(recipes), 5, 14, 14],
            'workspace_reservation_bytes': WORKSPACE_BYTES, 'max_workspace_bytes': max_workspace_bytes,
            'free_cuda_bytes_at_admission': free, 'total_cuda_bytes': total,
            'activation_memory_bound': False, 'graph': False,
            'qualification': 'unqualified_candidate_requires_real_model_parity_and_resource_checks'}


@contextmanager
def _unchanged_rng(t: ModuleType, devices: tuple[int, ...], *, capsule: ModuleType | None=None) -> Iterator[Metadata]:
    """No reseeding: reject any random draw during the deterministic batch stage."""
    import numpy as np
    if capsule is None:
        from . import rng as default_capsule
        capsule = default_capsule
    if not capsule._LOCK.acquire(blocking=False):
        raise RuntimeError('Native RNG capsule already in use')
    before = None
    receipt: Metadata = {}
    try:
        before = capsule.snapshot(t, np, devices)
        receipt['before_sha256'] = capsule.state_digest(before)
        yield receipt
        receipt['after_sha256'] = capsule.state_digest(capsule.snapshot(t, np, devices))
        receipt['unchanged'] = receipt['before_sha256'] == receipt['after_sha256']
        if not receipt['unchanged']:
            raise RuntimeError('Decoder/scheduler consumed RNG; batch stage is not deterministic')
    finally:
        try:
            if before is not None:
                random.setstate(before['python']); np.random.set_state(before['numpy'])
                t.set_rng_state(before['torch'])
                for device, value in zip(devices, before['cuda']): t.cuda.set_rng_state(value, device)
        finally:
            capsule._LOCK.release()


def prepare_request(runtime: _NativeRuntime, image: FloatArray, recipe: Metadata, *, reuse: str, cancel: Callable[[], None]) -> PreparedState:
    """Run original forward through its actual CPU-noise draw, then suspend."""
    from .compat import schedule_compatibility
    from .memory_output import array_identity, trace_memory
    from .reuse import exact_conditioning_reuse
    from .rng import rng_capsule
    core = runtime.numerical_core
    schedule = core.schedule if core is not None else schedule_compatibility
    rng_context = core.rng if core is not None else rng_capsule
    reuse_context = core.reuse if core is not None else exact_conditioning_reuse
    decision_function = core.decision if core is not None else _decision
    t, model = cast("ModuleType", runtime.torch), cast("TerraMindModel", runtime.model)
    tokenizer = model.tokenizer[TARGET]
    decoder = tokenizer.decoder
    original_scheduler, original_steps = tokenizer.noise_scheduler, model.decoding_steps
    old_forward = vars(decoder).get('forward', _MISSING)
    patched = False
    state: PreparedState = {}
    decision = decision_function(model, recipe, reuse)
    stats: Metadata = {'requested': False, 'cache_released': True, 'cache_hits': 0}
    tensor = None

    def boundary(this: Any, x: Tensor, timestep: Tensor, cond: Tensor, cond_mask: Tensor | None=None, **kwargs: Any) -> NoReturn:
        if state or cond_mask is not None or kwargs != {'orig_res': None}:
            raise RuntimeError('Unexpected native decoder boundary')
        if (tuple(x.shape) != (1, 2, 224, 224) or tuple(cond.shape) != (1, 5, 14, 14)
                or x.dtype != t.float32 or cond.dtype != t.float32 or x.device != cond.device
                or timestep.ndim != 0 or timestep.dtype != t.int64):
            raise RuntimeError('Unexpected native batch-one decoder signature')
        scheduler = tokenizer.noise_scheduler
        if scheduler is original_scheduler or len(scheduler.timesteps) != recipe['diffusion_steps']:
            raise RuntimeError('Request scheduler was not isolated/configured')
        if not t.equal(timestep, scheduler.timesteps[0]):
            raise RuntimeError('Capture must be before the first decoder step')
        cast(Metadata, state).update(image=x.detach().clone(), conditioning=cond.detach().clone(),
                     scheduler=scheduler, timestep=timestep.clone(),
                     initial_noise=array_identity(x.detach().cpu().numpy()),
                     conditioning_identity=array_identity(cond.detach().cpu().numpy()))
        raise _PreparedBoundary()

    try:
        cancel()
        tokenizer.noise_scheduler = deepcopy(original_scheduler)
        model.decoding_steps = recipe['token_steps']
        decoder.forward = types.MethodType(boundary, decoder)
        patched = True
        tensor = t.from_numpy(image.copy()).unsqueeze(0).to(runtime.device)
        devices = (0,) if runtime.device == 'cuda:0' else ()
        with t.no_grad(), rng_context(recipe['seed'], devices=devices) as rng, \
                schedule(model, token_steps=recipe['token_steps']) as compatibility, \
                trace_memory(model) as trace:
            context = reuse_context(model, enabled=True) if decision['enabled'] else nullcontext(stats)
            with context as stats:
                try:
                    model({'S2L2A': tensor}, standardize=True,
                          timesteps=recipe['diffusion_steps'], verbose=False)
                except _PreparedBoundary:
                    pass
                else:
                    raise RuntimeError('Upstream returned without reaching the admitted decoder boundary')
        if not (state and rng['restored'] and compatibility['method_restored']
                and trace['methods_restored'] and stats['cache_released']):
            raise RuntimeError('Request preparation did not restore native state')
        cast(Metadata, state).update(rng=rng, trace=trace, compatibility=compatibility,
                     reuse=stats, reuse_decision=decision)
        return state
    finally:
        tokenizer.noise_scheduler = original_scheduler
        model.decoding_steps = original_steps
        if patched:
            if old_forward is _MISSING: delattr(decoder, 'forward')
            else: decoder.forward = old_forward
        tensor = None


def denoise_pair(t: ModuleType, decoder: Callable[..., Tensor], states: list[PreparedState], *, cancel: Callable[[], None]) -> Metadata:
    """One actual batch-two decoder invocation per step, two owned scheduler steps."""
    if len(states) != 2 or states[0]['scheduler'] is states[1]['scheduler']:
        raise ValueError('Two independent scheduler states are required')
    steps = states[0]['scheduler'].timesteps
    if not t.equal(steps, states[1]['scheduler'].timesteps) or len(steps) != 10:
        raise ValueError('Identical ten-step DDIM schedules are required')
    device = states[0]['image'].device
    for state in states:
        if (state['image'].shape != (1, 2, 224, 224) or state['conditioning'].shape != (1, 5, 14, 14)
                or state['image'].dtype != t.float32 or state['conditioning'].dtype != t.float32
                or state['image'].device != device or state['conditioning'].device != device):
            raise ValueError('Prepared batch shape/dtype/device differs')
    images = t.empty((2, 2, 224, 224), dtype=t.float32, device=device)
    conditioning = t.cat([s['conditioning'] for s in states], dim=0)
    prediction: Tensor | None
    batch_sizes = []
    for timestep in steps:
        cancel()
        for i, state in enumerate(states): images[i:i+1].copy_(state['image'])
        prediction = decoder(images, timestep, conditioning, orig_res=None)
        if prediction.shape != images.shape or prediction.dtype != t.float32 or prediction.device != device:
            raise RuntimeError('Batched decoder output violates the admitted signature')
        batch_sizes.append(int(images.shape[0]))
        with t.amp.autocast('cuda', enabled=False):
            for i, state in enumerate(states):
                # Original native API/default eta and generator=None; no shared scheduler.
                state['image'] = state['scheduler'].step(prediction[i:i+1].float(), timestep,
                    state['image'], generator=None).prev_sample
        prediction = None
    return {'decoder_calls': len(batch_sizes), 'observed_decoder_batch_sizes': batch_sizes,
            'scheduler_steps_per_request': len(steps), 'graph': False}


def run_pair(runtime: _NativeRuntime, snapshots: list[InputSnapshot], recipes: list[Metadata], identifiers: list[tuple[str, str]], *, reuse: str, cancel: Callable[[], None],
             session_identity: Metadata, max_workspace_bytes: int, min_free_cuda_bytes: int, admission: Metadata | None=None, stage_runner: StageRunner | None=None) -> list[Metadata]:
    if admission is None:
        admission = eligibility(runtime, recipes, max_workspace_bytes=max_workspace_bytes,
                                min_free_cuda_bytes=min_free_cuda_bytes)
    return run_prepared_group(runtime,snapshots,recipes,identifiers,reuse=reuse,cancel=cancel,
        session_identity=session_identity,admission=admission,stage_runner=stage_runner)


def run_prepared_group(runtime: _NativeRuntime,snapshots: list[InputSnapshot],recipes: list[Metadata],identifiers: list[tuple[str, str]],*,reuse: str,cancel: Callable[[], None],session_identity: Metadata,
                       admission: Metadata,stage_runner: StageRunner | None,prepare_runner: PrepareRunner | None=None) -> list[Metadata]:
    """Request-local native preparation and ordered delivery for an admitted group."""
    import numpy as np
    from .memory_output import array_identity
    if not snapshots or not len(snapshots)==len(recipes)==len(identifiers):
        raise ValueError('Prepared group inputs, recipes and identifiers must have equal nonzero lengths')
    t, model = cast("ModuleType", runtime.torch), cast("TerraMindModel", runtime.model)
    if runtime._versions() != runtime.versions: raise RuntimeError('Resident model parameters changed')
    runtime._policy()
    states: list[PreparedState] = []
    try:
        with t.no_grad():
            runtime._assert_policy()
            devices = (0,) if runtime.device == 'cuda:0' else ()
            if prepare_runner is None:
                for snap, recipe in zip(snapshots, recipes):
                    states.append(prepare_request(runtime, snap[0], recipe, reuse=reuse, cancel=cancel))
                with _unchanged_rng(t, devices, capsule=runtime.numerical_core.rng_module if runtime.numerical_core is not None else None) as deterministic_rng:
                    stage = (denoise_pair(t, model.tokenizer[TARGET].decoder, states, cancel=cancel)
                             if stage_runner is None else stage_runner(states, cancel=cancel))
            else:
                def prepare(snap: InputSnapshot, recipe: Metadata) -> Callable[[], PreparedState]:
                    return lambda: prepare_request(runtime, snap[0], recipe, reuse=reuse, cancel=cancel)
                states, stage, deterministic_rng = prepare_runner(
                    [prepare(snap, recipe) for snap,recipe in zip(snapshots,recipes)],cancel=cancel)
            with _unchanged_rng(t, devices, capsule=runtime.numerical_core.rng_module if runtime.numerical_core is not None else None) as delivery_rng:
                arrays = []
                value: Tensor | None
                for state in states:
                    value = state['image']
                    key = model.mod_name_mapping['S1GRD']
                    if key in model.pretraining_mean:
                        value = value * model.pretraining_std[key] + model.pretraining_mean[key]
                    arrays.append(value.detach().cpu().numpy()[0].copy())
                    value = None
            if deterministic_rng['after_sha256'] != delivery_rng['before_sha256']:
                raise RuntimeError('Preparation/stage RNG boundary changed before delivery')
            runtime._assert_policy()
        if runtime._versions() != runtime.versions: raise RuntimeError('Resident model parameters changed')
        results = []
        for state, snap, recipe, ids, array in zip(states, snapshots, recipes, identifiers, arrays):
            if array.shape != (2, 224, 224) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise RuntimeError('Expected finite request-owned FP32 output')
            _, mask, metadata, identity = snap
            shared_core = ({'numerical_core':runtime.numerical_core.receipt(route='independent_b1_graph_lanes')}
                           if runtime.numerical_core is not None else {})
            provenance = {**shared_core, 'schema': 'aureon.experimental_denoiser_batch_job.v1',
                'session': session_identity, 'job_id': ids[0], 'attempt_id': ids[1], 'recipe': recipe,
                'model': runtime.config['model'], 'tokenizer': runtime.config['tokenizer'],
                'source_manifest': runtime.config['source_manifest'], 'dependencies': runtime.config['dependencies'],
                'loaded_identity': runtime.loaded_identity, 'input_identity': identity, 'input': metadata,
                'synthetic': True, 'sensor_observation': False, 'scientific_quality': 'not_evaluated',
                'numerical_fidelity': 'unqualified_batch_candidate', 'array': array_identity(array),
                'valid_mask': array_identity(mask), 'delivery': 'memory',
                'output': {'modality': 'S1GRD', 'channels': ['VV','VH'], 'units': 'dB'},
                'batch_admission': admission, 'batch_stage': stage, 'deterministic_stage_rng': deterministic_rng,
                'rng': state['rng'], 'trace': state['trace'], 'initial_noise': state['initial_noise'],
                'conditioning': state['conditioning_identity'], 'compatibility': state['compatibility'],
                'reuse': state['reuse'], 'reuse_decision': state['reuse_decision'],
                'precision': {'parameters': 'float32', 'matmul': 'highest', 'tf32': False, 'autocast': False}}
            canonical(provenance)
            results.append({**shared_core, 'status': 'completed', 'job_id': ids[0], 'attempt_id': ids[1],
                            'array': array, 'valid_mask': mask.copy(), 'provenance': provenance})
        return results
    finally:
        states.clear()
