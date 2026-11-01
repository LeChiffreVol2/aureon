"""Bounded exact-model dispatch with one spawned process per resident session.

No models, Torch or CUDA are imported by this module. Registration is a value;
execution requires an explicit pool. Capacities reserve application payloads and
declared runtime workspaces, not an operating-system RSS/VRAM enforcement limit.
The caller must provision and qualify those budgets for the selected hardware.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, TimeoutError
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import pickle
import queue as thread_queue
import re
import sys
import threading
from time import monotonic
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict, cast
import uuid
import weakref

from .contracts import ExecutionProfile, GenerationResult, OUTPUT_BYTES, positive, validate_timeout
from .generation_bridges import GenerationRecipe
from .lifecycle import close_preserving_error
from .memory_input import GenerationInput
from .routing import ExecutionKey
from .pool_placement import WorkerCapabilities,WorkerOffer,rank_worker_offers
from .v1 import ExecutionRequired
from ..terramind_native.encoder_loader import CHANNELS, MergeMethod, required_tokenizers, validate_options
from ..terramind_native.families import GENERATION_MODELS, PINNED_S1GRD, TOKENIZER_ASSETS, get_terramind_spec

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor
    from .generation_engine import GenerationEngine
    from .terramind_encoder import TerraMindEncoder

_META_LIMIT = 1024**2
_COPY_FACTOR = 8
_EXIT_DRAIN_SECONDS = 5.0
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
_State = Literal['queued', 'running', 'completed', 'failed', 'cancelled', 'consumed', 'discarded']
FloatArray: TypeAlias = 'NDArray[np.float32]'
FeatureArray: TypeAlias = 'FloatArray | dict[str, FloatArray]'


class PoolError(RuntimeError):
    """Base class for bounded pool admission, execution and ownership errors."""


class PoolBusy(PoolError):
    """Admission credits are exhausted; consume/discard results before retrying."""


class PoolClosed(PoolError):
    """Pool admission has closed."""


class PoolExecutionError(PoolError):
    """One exact route failed; it is disabled for this pool instance."""


class PoolCleanupError(PoolError):
    """A worker did not provide a verified close; its slot is quarantined."""


class PooledFeatures(TypedDict):
    features: list[FeatureArray]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PoolResult:
    """Owned CPU outputs. Features are NumPy arrays, not borrowed GPU tensors."""
    route: str
    route_identity: str
    execution_key: ExecutionKey
    device_slot: str
    worker_generation: str
    seed: int
    kind: Literal['generation', 'features']
    value: GenerationResult | PooledFeatures


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(field+' must be 1..128 ASCII letters/digits/._-')


def _seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    return value


def _path(value: str | Path) -> str:
    if not isinstance(value, (str, Path)) or not str(value) or len(str(value)) > 16384 or '://' in str(value):
        raise ValueError('Supply a bounded explicit local asset path')
    return str(Path(value).resolve())


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DeviceSlot:
    """One explicit physical CUDA selector, or CPU when selector is None.

    CUDA workers receive this single CUDA_VISIBLE_DEVICES selector before Torch
    import. Distinct slots must not alias one physical GPU; worker UUIDs are
    checked when available. No ambient CUDA device numbering is inferred.
    """
    name: str
    memory_bytes: int
    cuda_visible_device: str | None = None
    max_residents: int = 1
    capabilities: WorkerCapabilities = WorkerCapabilities()
    relative_capacity: float = 1.0

    def __post_init__(self) -> None:
        _name(self.name, 'Device slot name')
        if type(self.memory_bytes) is not int or self.memory_bytes < 0:
            raise ValueError('memory_bytes must be nonnegative integer bytes')
        positive(self.max_residents, 'max_residents')
        WorkerOffer(self.name,self.capabilities,self.memory_bytes,self.relative_capacity)
        if self.cuda_visible_device is None:
            if self.memory_bytes != 0:
                raise ValueError('CPU slots use the host budget and memory_bytes=0')
        elif (not isinstance(self.cuda_visible_device, str) or
              re.fullmatch(r'[A-Za-z0-9_:/.-]{1,256}', self.cuda_visible_device) is None or self.memory_bytes == 0):
            raise ValueError('Declare one CUDA selector and a positive device budget')


@dataclass(frozen=True)
class PoolResources:
    """Explicit per-resident and per-active-request workspace reservations.

    Device output capacity is added to transient_device_bytes automatically.
    Host payload credits reserve eight copies of input/output plus bounded IPC
    metadata. Declared host workspaces include interpreter/checkpoint/runtime
    overhead; external allocations and caller-owned consumed results are excluded.
    """
    resident_device_bytes: int
    resident_host_bytes: int
    transient_device_bytes: int
    transient_host_bytes: int
    max_input_bytes: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(name+' must be nonnegative integer bytes')
        for name in ('resident_host_bytes', 'max_input_bytes', 'max_output_bytes'):
            positive(getattr(self, name), name)


@dataclass(frozen=True)
class GenerationRoute:
    name: str
    model_id: str
    model_file: str | Path
    tokenizer_file: str | Path
    devices: tuple[str, ...]
    resources: PoolResources
    recipe: GenerationRecipe = GenerationRecipe()
    execution_profile: ExecutionProfile = 'recommended'
    required_capabilities: tuple[str,...] = ()

    def __post_init__(self) -> None:
        _route_fields(self)
        if self.model_id not in GENERATION_MODELS or not isinstance(self.recipe, GenerationRecipe):
            raise ValueError('GenerationRoute requires a supported generation model and recipe')
        if self.execution_profile not in ('recommended', 'reference'):
            raise ValueError('Choose recommended or reference execution_profile')
        if self.resources.max_output_bytes < OUTPUT_BYTES:
            raise ValueError('Generation output credit must cover the fixed FP32 output')
        self.recipe.native()
        object.__setattr__(self, 'model_file', _path(self.model_file))
        object.__setattr__(self, 'tokenizer_file', _path(self.tokenizer_file))


@dataclass(frozen=True)
class EncoderRoute:
    name: str
    model_id: str
    model_file: str | Path
    modalities: tuple[str, ...]
    devices: tuple[str, ...]
    resources: PoolResources
    tokenizer_files: tuple[tuple[str, str | Path], ...] = ()
    tim_modalities: tuple[str, ...] | None = None
    merge_method: MergeMethod = 'mean'
    tim_decoding_steps: int = 1
    tim_temperature: float = 1.0
    tim_top_p: float = 0.8
    tim_top_k: int = 0
    tim_chained_generations: bool = True
    required_capabilities: tuple[str,...] = ()

    def __post_init__(self) -> None:
        _route_fields(self)
        if not isinstance(self.modalities, tuple) or (self.tim_modalities is not None and not isinstance(self.tim_modalities, tuple)):
            raise TypeError('Encoder modality declarations must be immutable tuples')
        _, options = validate_options(self.model_id, self.modalities, self.tim_modalities,
            self.merge_method, self.tim_decoding_steps, self.tim_temperature, self.tim_top_p,
            self.tim_top_k, self.tim_chained_generations)
        if (not isinstance(self.tokenizer_files, tuple) or any(not isinstance(item, tuple) or len(item) != 2 for item in self.tokenizer_files)
                or len(dict(self.tokenizer_files)) != len(self.tokenizer_files)
                or set(dict(self.tokenizer_files)) != set(required_tokenizers(options))):
            raise ValueError('tokenizer_files must exactly declare the required immutable name/path pairs')
        object.__setattr__(self, 'model_file', _path(self.model_file))
        object.__setattr__(self, 'tokenizer_files', tuple(sorted((name, _path(path)) for name, path in self.tokenizer_files)))
        object.__setattr__(self, 'tim_modalities', options.tim_modalities or None)


ModelRoute: TypeAlias = GenerationRoute | EncoderRoute


def _route_fields(route: ModelRoute) -> None:
    _name(route.name, 'Route name')
    if not isinstance(route.resources, PoolResources):
        raise TypeError('resources requires PoolResources')
    if not isinstance(route.devices, tuple) or not route.devices or len(set(route.devices)) != len(route.devices):
        raise ValueError('Declare a nonempty immutable tuple of unique device slots')
    for name in route.devices:
        _name(name, 'Device slot')
    WorkerCapabilities(tags=route.required_capabilities)


def _route_identity(route: ModelRoute) -> str:
    value = asdict(route)
    value['checkpoint'] = asdict(get_terramind_spec(route.model_id).checkpoint)
    value['tokenizers'] = ({'S1GRD': asdict(PINNED_S1GRD)} if isinstance(route, GenerationRoute)
                           else {name: asdict(TOKENIZER_ASSETS[name]) for name, _ in route.tokenizer_files})
    return _digest(value)


def _execution_key(route: ModelRoute, seed: int) -> ExecutionKey:
    spec = get_terramind_spec(route.model_id)
    recipe: object
    if isinstance(route, GenerationRoute):
        recipe = {'sampling': replace(route.recipe, seed=seed).native(),
                  'execution_profile': route.execution_profile}
    else:
        recipe = {'seed': seed, 'modalities': route.modalities, 'tim_modalities': route.tim_modalities,
                  'merge_method': route.merge_method, 'steps': route.tim_decoding_steps,
                  'temperature': route.tim_temperature, 'top_p': route.tim_top_p,
                  'top_k': route.tim_top_k, 'chained': route.tim_chained_generations}
    return ExecutionKey(route.model_id, spec.checkpoint.sha256, _digest(recipe),
        'generation' if isinstance(route, GenerationRoute) else 'features',
        'S2L2A' if isinstance(route, GenerationRoute) else '+'.join(route.modalities),
        'S1GRD' if isinstance(route, GenerationRoute) else 'latent_features', 'highest_fp32')


class EncoderInput:
    """Immutable owned CPU pixel snapshots, with explicit upstream preprocessing.

    This performs no standardization. Supply native FP32 continuous batches and
    int64 LULC indices; shape/bands follow TerraMindEncoder. CUDA tensors are not
    transferred implicitly. Constructor bytes remain caller-owned until admitted.
    """
    __slots__ = ('_items', '_bytes', '_identity')

    def __init__(self, inputs: Mapping[str, object], *, max_bytes: int = 64*1024**2) -> None:
        import numpy as np
        positive(max_bytes, 'max_bytes')
        if not isinstance(inputs, Mapping) or not inputs or any(name not in CHANNELS for name in inputs):
            raise ValueError('Use canonical modality names and explicit CPU pixel batches')
        items: list[tuple[str, tuple[int, ...], str, bytes]] = []
        total = 0
        batch = None
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                array = value
            else:
                torch = sys.modules.get('torch')
                if torch is None or not isinstance(value, torch.Tensor):
                    raise TypeError('EncoderInput accepts NumPy arrays or already imported CPU Torch tensors')
                tensor = cast('Tensor', value)
                if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
                    raise ValueError('Explicit dense CPU tensors required before pool admission')
                array = tensor.detach().numpy()
            tail = (224, 224) if name == 'LULC' else (CHANNELS[name], 224, 224)
            dtype = np.dtype('int64' if name == 'LULC' else 'float32')
            if (isinstance(array, np.ma.MaskedArray) or array.dtype != dtype or not array.dtype.isnative
                    or array.ndim != len(tail)+1 or array.shape[0] < 1 or array.shape[1:] != tail):
                raise ValueError('Expected native full-band224 batch with explicit FP32/int64 dtype')
            if batch is not None and batch != array.shape[0]:
                raise ValueError('Input batch sizes must match')
            batch = array.shape[0]
            total += array.nbytes
            if total > max_bytes:
                raise ValueError('EncoderInput exceeds max_bytes')
            data = array.tobytes(order='C')
            owned = np.frombuffer(data, dtype=dtype)
            if (name == 'LULC' and ((owned < 0).any() or (owned > 9).any())) or not np.isfinite(owned).all():
                raise ValueError('Expected finite continuous pixels or LULC indices0..9')
            items.append((name, tuple(array.shape), dtype.str, data))
        self._items = tuple(items)
        self._bytes = total
        self._identity = _digest([(name, shape, dtype, hashlib.sha256(data).hexdigest()) for name, shape, dtype, data in items])

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self._items)

    @property
    def identity(self) -> str:
        return self._identity


PoolInput: TypeAlias = GenerationInput | EncoderInput
_Wire: TypeAlias = dict[str, Any]


def _input_wire(value: PoolInput) -> _Wire:
    if isinstance(value, EncoderInput):
        return {'kind': 'features', 'items': value._items, 'identity': value.identity}
    image, mask, metadata, identity = value.snapshot()
    return {'kind': 'generation', 'image': image.tobytes(), 'mask': mask.tobytes(),
            'metadata': metadata, 'identity': identity}


def _send(connection: Connection, message: _Wire, limit: int) -> None:
    payload = pickle.dumps(message, protocol=5)
    if len(payload) > limit:
        raise PoolError('IPC serialization exceeds its admitted bound')
    connection.send_bytes(payload)


def _receive(connection: Connection, limit: int) -> _Wire:
    value = pickle.loads(connection.recv_bytes(maxlength=limit))
    if not isinstance(value, dict):
        raise PoolError('Invalid owned-worker IPC message')
    return value


def _cleanup_receipt(value: object) -> dict[str, Any]:
    """Bound/detach worker diagnostics without letting malformed data pass close."""
    try:
        if not isinstance(value,dict):
            raise TypeError('Cleanup acknowledgement must be a dictionary')
        encoded=json.dumps(value,allow_nan=False)
        if len(encoded.encode('utf-8'))>_META_LIMIT//2:
            raise ValueError('Cleanup acknowledgement exceeds its metadata budget')
        return cast(dict[str,Any],json.loads(encoded))
    except Exception as error:
        return {'passed':False,'invalid_receipt':True,'receipt_type':type(value).__name__[:120],
                'error_type':type(error).__name__,'message':str(error)[:500]}


class _Session(Protocol):
    def run(self, payload: _Wire, seed: int) -> GenerationResult | PooledFeatures: ...
    def close(self) -> dict[str, Any]: ...


class _Transport:
    """Bounded local IPC; pipe waits never hold the pool condition.

    Two daemon pumps isolate platform blocking pipe operations. A partial or
    stalled child response therefore cannot block cancellation/status or the
    caller's close timeout. No worker is killed: an unverified worker retains
    its reservation until a verified exit, and its slot stays quarantined.
    """
    def __init__(self, connection: Connection, route: ModelRoute) -> None:
        self.connection, self.route = connection, route
        self.incoming: thread_queue.Queue[_Wire | Exception] = thread_queue.Queue(2)
        self.outgoing: thread_queue.Queue[_Wire] = thread_queue.Queue(1)
        self.stopped = threading.Event()
        # This event is a producer fence: after it is set, every reader message
        # has either entered incoming or was abandoned because stop was called.
        # Thread.is_alive() is not a queue-drained acknowledgement.
        self.reader_done = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True, name='aureon-pool-read')
        self.writer = threading.Thread(target=self._write, daemon=True, name='aureon-pool-write')
        self.reader.start(); self.writer.start()

    def _deliver(self, value: _Wire | Exception) -> None:
        while not self.stopped.is_set():
            try:
                self.incoming.put(value, timeout=0.05)
                return
            except thread_queue.Full:
                pass

    def _read(self) -> None:
        try:
            while not self.stopped.is_set():
                value = _receive(self.connection, self.route.resources.max_output_bytes+_META_LIMIT)
                self._deliver(value)
                value = {}
        except Exception as error:
            self._deliver(PoolError('Worker transport read failed: '+str(error)[:1000]))
        finally:
            self.reader_done.set()

    def _write(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    value = self.outgoing.get(timeout=0.05)
                except thread_queue.Empty:
                    continue
                _send(self.connection, value, self.route.resources.max_input_bytes+_META_LIMIT)
                value = {}
        except Exception as error:
            self._deliver(PoolError('Worker transport write failed: '+str(error)[:1000]))

    def send(self, value: _Wire) -> None:
        try:
            self.outgoing.put_nowait(value)
        except thread_queue.Full as error:
            raise PoolError('Worker IPC command credit exhausted') from error

    def stop(self) -> None:
        self.stopped.set()
        self.connection.close()
        # Process exit is verified before stop. Pumps may still be unwinding;
        # bound these joins and clear queued payloads before releasing credits.
        self.reader.join(0.2); self.writer.join(0.2)
        for messages in (self.incoming, self.outgoing):
            while not messages.empty():
                try:
                    messages.get_nowait()
                except thread_queue.Empty:
                    break


class _NativeSession:
    def __init__(self, route: ModelRoute, slot: DeviceSlot, work: Path) -> None:
        self.route, self.slot = route, slot
        self.runtime: GenerationEngine | TerraMindEncoder | None
        if isinstance(route, GenerationRoute):
            from .generation_engine import GenerationEngine
            self.runtime = GenerationEngine(route.model_id, model_file=route.model_file,
                tokenizer_file=route.tokenizer_file, work=work, execute_model=True,
                execution_profile=route.execution_profile)
        else:
            from .terramind_encoder import TerraMindEncoder
            self.runtime = TerraMindEncoder(route.model_id, model_file=route.model_file,
                modalities=route.modalities, tokenizer_files=dict(route.tokenizer_files),
                tim_modalities=route.tim_modalities, device='cuda:0' if slot.cuda_visible_device else 'cpu',
                execute_model=True, merge_method=route.merge_method,
                tim_decoding_steps=route.tim_decoding_steps, tim_temperature=route.tim_temperature,
                tim_top_p=route.tim_top_p, tim_top_k=route.tim_top_k,
                tim_chained_generations=route.tim_chained_generations)

    def run(self, payload: _Wire, seed: int) -> GenerationResult | PooledFeatures:
        import numpy as np
        if isinstance(self.route, GenerationRoute):
            metadata = payload['metadata']
            value = GenerationInput(np.frombuffer(payload['image'], dtype=np.float32).reshape(12,224,224),
                bands=metadata['bands'], units=metadata['units'],
                valid_mask=np.frombuffer(payload['mask'], dtype=np.bool_).reshape(224,224),
                grid=metadata['grid'], source_id=metadata.get('source_id'))
            if value.identity != payload['identity']:
                raise PoolError('Generation input identity changed during IPC')
            engine = cast('GenerationEngine', self.runtime)
            return engine.generate(value, recipe=replace(self.route.recipe, seed=seed))
        import torch
        tensors = {name: torch.from_numpy(np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy())
                   for name, shape, dtype, data in payload['items']}
        native = cast('TerraMindEncoder', self.runtime).predict(tensors, seed=seed)
        arrays: list[FeatureArray] = []
        try:
            for layer in native['features']:
                if isinstance(layer, dict):
                    arrays.append({name: tensor.detach().cpu().numpy().copy() for name, tensor in layer.items()})
                else:
                    arrays.append(layer.detach().cpu().numpy().copy())
            return {'features': arrays, 'provenance': native['provenance']}
        finally:
            native['features'].clear()
            tensors.clear()

    def close(self) -> dict[str, Any]:
        runtime, self.runtime = self.runtime, None
        if runtime is None:
            return {'passed': True, 'already_closed': True}
        reference = weakref.ref(runtime)
        if isinstance(self.route, GenerationRoute):
            receipt = cast('GenerationEngine', runtime).close()
        else:
            cast('TerraMindEncoder', runtime).close()
            receipt = {'passed': cast('TerraMindEncoder', runtime).state == 'closed'}
        runtime = None
        gc.collect()
        memory = {'allocated': 0, 'reserved': 0}
        if self.slot.cuda_visible_device:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                clear = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                if clear is not None:
                    clear()
                torch.cuda.empty_cache()
                memory = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
        return {**receipt, 'passed': receipt.get('passed') is True and reference() is None and not any(memory.values()),
                'owned_session_released': reference() is None, 'cuda': memory}


def _device_attestation(slot: DeviceSlot) -> dict[str, Any]:
    if slot.cuda_visible_device is None:
        return {'kind': 'cpu', 'selector': None, 'uuid': None, 'pid': os.getpid()}
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PoolError('Worker must see exactly its one configured CUDA device')
    props = torch.cuda.get_device_properties(0)
    identity = getattr(props, 'uuid', None)
    if identity is None:
        raise PoolError('CUDA UUID is required to prevent device-slot aliases')
    if slot.memory_bytes > props.total_memory:
        raise PoolError('Declared device budget exceeds physical capacity')
    return {'kind': 'cuda', 'selector': slot.cuda_visible_device, 'uuid': str(identity),
            'pid': os.getpid(), 'name': props.name, 'total_memory': props.total_memory}


def _validated_output(value: GenerationResult | PooledFeatures, route: ModelRoute) -> int:
    import numpy as np
    if isinstance(route, GenerationRoute):
        generated = cast(GenerationResult, value)
        arrays = [generated['array'], generated['valid_mask']]
        if (any(not isinstance(array,np.ndarray) or isinstance(array,np.ma.MaskedArray) for array in arrays)
                or arrays[0].shape != (2,224,224) or arrays[0].dtype != np.float32 or
                arrays[1].shape != (224,224) or arrays[1].dtype != np.bool_):
            raise PoolError('Generation output violates the registered contract')
    else:
        features = cast(PooledFeatures, value)['features']
        if not isinstance(features, list) or not 1 <= len(features) <= 128:
            raise PoolError('Encoder returned no feature arrays')
        for layer in features:
            if isinstance(layer,dict) and (not 1 <= len(layer) <= 16 or
                    any(not isinstance(name,str) or not 1 <= len(name) <= 128 for name in layer)):
                raise PoolError('Feature modality labels exceed their bounded contract')
        arrays = [array for layer in features for array in (layer.values() if isinstance(layer, dict) else (layer,))]
        if any(not isinstance(array, np.ndarray) or isinstance(array,np.ma.MaskedArray)
               or array.dtype != np.float32 or array.ndim != 3 or any(d<=0 for d in array.shape) for array in arrays):
            raise PoolError('Expected owned CPU FP32 feature arrays')
    size = sum(array.nbytes for array in arrays)
    if size > route.resources.max_output_bytes or any(not np.isfinite(array).all() for array in arrays):
        raise PoolError('Output exceeds its byte credit or contains nonfinite values')
    metadata = {key: item for key, item in value.items() if key not in ('array','valid_mask','features')}
    if len(json.dumps(metadata, allow_nan=False).encode()) > _META_LIMIT//2:
        raise PoolError('Output metadata exceeds its reserved IPC budget')
    return size


def _worker_main(connection: Connection, route: ModelRoute, slot: DeviceSlot, work: Path,
                 generation: str) -> None:
    """Private spawned entrypoint accepting only typed local execution requests."""
    os.environ['CUDA_VISIBLE_DEVICES'] = slot.cuda_visible_device or ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    session: _Session | None = None
    request = value = None
    identity = _route_identity(route)
    try:
        imported_torch = sys.modules.get('torch')
        if imported_torch is not None and imported_torch.cuda.is_initialized():
            raise PoolError('Spawned application initialized CUDA before device isolation; guard execution with __main__')
        attestation = _device_attestation(slot)
        session = _NativeSession(route, slot, work)
        _send(connection, {'type':'ready','identity':identity,'generation':generation,'device':attestation}, _META_LIMIT)
        while True:
            request = _receive(connection, route.resources.max_input_bytes+_META_LIMIT)
            if request.get('type') == 'close':
                break
            if request.get('type') != 'run' or request.get('identity') != identity:
                raise PoolError('Worker received a mismatched execution route')
            value = session.run(request['input'], _seed(request['seed']))
            _validated_output(value, route)
            _send(connection, {'type':'result','id':request['id'],'value':value}, route.resources.max_output_bytes+_META_LIMIT)
            value = request = None
            _send(connection, {'type':'idle'}, _META_LIMIT)
    except BaseException as error:
        message = str(error)[:2000]
        traceback.clear_frames(error.__traceback__)
        request = value = None
        try:
            _send(connection, {'type':'error','error_type':type(error).__name__,'message':message}, _META_LIMIT)
        except (OSError, EOFError):
            pass
    finally:
        request = value = None
        try:
            cleanup = session.close() if session is not None else {'passed': True, 'model_loaded': False}
            session = None
            _send(connection, {'type':'closed','identity':identity,'generation':generation,'cleanup':cleanup}, _META_LIMIT)
        except BaseException as error:
            try:
                _send(connection, {'type':'closed','identity':identity,'generation':generation,
                      'cleanup':{'passed':False,'error_type':type(error).__name__,'message':str(error)[:2000]}}, _META_LIMIT)
            except (OSError, EOFError):
                pass
        finally:
            connection.close()


class PoolHandle:
    """Single-consumer result credit; cancellation never preempts a running model."""
    def __init__(self, owner: ModelPool, identifier: int, route: ModelRoute, value: PoolInput, seed: int, tenant: str, affinity: str | None=None) -> None:
        self._owner, self._id, self._route = owner, identifier, route
        self._input: PoolInput | None = value
        self._seed, self._tenant = seed, tenant
        self._affinity=affinity
        self._state: _State = 'queued'
        self._result: PoolResult | None = None
        self._error: str | None = None
        self._discard = False
        self._input_credit = _COPY_FACTOR*_input_size(value)+2*_META_LIMIT
        self._output_credit = _COPY_FACTOR*route.resources.max_output_bytes+2*_META_LIMIT

    @property
    def state(self) -> _State:
        with self._owner._condition:
            return self._state

    def done(self) -> bool:
        return self.state not in ('queued', 'running')

    def cancel(self) -> bool:
        with self._owner._condition:
            return self._owner._cancel(self)

    def discard(self) -> None:
        with self._owner._condition:
            if self._owner._cancel(self):
                return
            if self._state == 'running':
                self._discard = True
            elif self._state in ('completed', 'failed'):
                self._result = None
                self._state = 'discarded'
                self._owner._release_result(self)

    def result(self, timeout: float | None = None) -> PoolResult:
        duration = validate_timeout(timeout)
        deadline = None if duration is None else monotonic()+duration
        with self._owner._condition:
            while self._state in ('queued','running'):
                remaining = None if deadline is None else deadline-monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('Pool result remains admitted')
                self._owner._condition.wait(remaining)
            if self._state == 'cancelled':
                raise CancelledError('Request cancelled before execution')
            if self._state in ('consumed','discarded'):
                raise PoolError('Result ownership already transferred or discarded')
            value, error = self._result, self._error
            self._result = self._error = None
            self._state = 'consumed'
            self._owner._release_result(self)
        if error is not None:
            raise PoolExecutionError(error)
        return cast(PoolResult, value)


def _input_size(value: PoolInput) -> int:
    return value.nbytes if isinstance(value, EncoderInput) else cast(int, value.identity['bytes'])


@dataclass
class _Resident:
    route: ModelRoute
    slot: DeviceSlot
    generation: str
    process: BaseProcess
    connection: Connection
    transport: _Transport | None = None
    state: Literal['loading','idle','running','closing','quarantined'] = 'loading'
    active: PoolHandle | None = None
    incoming: GenerationResult | PooledFeatures | None = None
    last_used: int = 0
    close_ack: dict[str, Any] | None = None
    device: dict[str, Any] | None = None
    held_host_bytes: int = 0
    held_device_bytes: int = 0
    transport_fault: str | None = None
    transport_fault_at: float | None = None
    exit_observed_at: float | None = None


class ModelPool:
    """Exact-route resident dispatch with tenant round-robin and idle-only LRU.

    Routes, slots and assets are explicit and immutable for one pool lifetime.
    Model execution stays in spawned owned processes, isolating global RNG,
    precision, decoder state and CUDA cleanup. Admission is nonblocking. Results
    must be consumed/discarded; close drains or cancels queued work and discards
    remaining owned results. A timeout never kills an active CUDA call.
    """
    def __init__(self, routes: Sequence[ModelRoute], devices: Sequence[DeviceSlot], *, work: str | Path,
                 host_memory_bytes: int, max_pending: int = 16, max_results: int = 32,
                 max_pending_per_tenant: int = 4, max_results_per_tenant: int = 8,
                 execute_model: bool = False) -> None:
        if execute_model is not True:
            raise ExecutionRequired('Pass execute_model=True; pool construction never runs a model implicitly')
        if not routes or not devices or any(not isinstance(r,(GenerationRoute,EncoderRoute)) for r in routes) or any(not isinstance(d,DeviceSlot) for d in devices):
            raise ValueError('Declare typed routes and device slots')
        self._routes = {route.name:route for route in routes}
        self._devices = {slot.name:slot for slot in devices}
        if len(self._routes) != len(routes) or len(self._devices) != len(devices):
            raise ValueError('Route and device names must be unique')
        selectors = [slot.cuda_visible_device for slot in devices if slot.cuda_visible_device is not None]
        if len(set(selectors)) != len(selectors):
            raise ValueError('CUDA selectors must be unique across device slots')
        self._host_limit = positive(host_memory_bytes, 'host_memory_bytes')
        self._max_pending, self._max_results = positive(max_pending,'max_pending'), positive(max_results,'max_results')
        self._tenant_pending = positive(max_pending_per_tenant,'max_pending_per_tenant')
        self._tenant_results = positive(max_results_per_tenant,'max_results_per_tenant')
        for route in routes:
            if any(name not in self._devices for name in route.devices):
                raise ValueError('Route references an unknown device slot')
            if not any(self._slot_compatible(route,self._devices[name]) for name in route.devices):
                raise ValueError('No compatible slot can hold this registered model/request budget')
            minimum = route.resources.resident_host_bytes+route.resources.transient_host_bytes+_COPY_FACTOR*(route.resources.max_input_bytes+route.resources.max_output_bytes)+4*_META_LIMIT
            if minimum > self._host_limit:
                raise ValueError('Host budget cannot admit one worst-case request for '+route.name)
        self.work = Path(work).absolute()
        if self.work.exists() or self.work.is_symlink() or not self.work.parent.is_dir():
            raise ValueError('Pool work must be a fresh directory under an existing parent')
        self.work = self.work.parent.resolve()/self.work.name
        self._condition = threading.Condition()
        self._queues: dict[str, deque[PoolHandle]] = {}
        self._tenants: deque[str] = deque()
        self._handles: set[PoolHandle] = set()
        self._residents: list[_Resident] = []
        self._failed_routes: dict[str,str] = {}
        self._quarantined: dict[str,str] = {}
        self._draining: set[str] = set()
        self._failed_placements: set[tuple[str,str]] = set()
        self._placement_counts={name:0 for name in self._devices}
        self._cleanup_records: deque[dict[str,Any]] = deque(maxlen=128)
        self._cleanup_count = 0
        self._pending = self._host_used = self._serial = self._clock = 0
        self._last_tenant: str | None = None
        self._accepting = True
        self._closed = False
        self._thread: threading.Thread | None = None
        self._close_report: dict[str,Any] | None = None
        self._context = mp.get_context('spawn')

    @staticmethod
    def _slot_compatible(route: ModelRoute, slot: DeviceSlot) -> bool:
        resources = route.resources
        if not slot.capabilities.admits(route.model_id,'generation' if isinstance(route,GenerationRoute) else 'features',
                                       'float32',route.required_capabilities):return False
        if slot.cuda_visible_device is None:
            return isinstance(route,EncoderRoute) and resources.resident_device_bytes == resources.transient_device_bytes == 0
        return resources.resident_device_bytes+resources.transient_device_bytes+resources.max_output_bytes <= slot.memory_bytes

    def submit(self, route: str, value: PoolInput, *, seed: int = 17, tenant: str = 'default', affinity: str | None=None) -> PoolHandle:
        _name(tenant,'tenant'); _seed(seed)
        if route not in self._routes:
            raise ValueError('Unknown exact route; no model substitution is performed')
        spec = self._routes[route]
        if affinity is not None and affinity not in spec.devices:raise ValueError('Affinity must name a configured route slot')
        if isinstance(spec,GenerationRoute):
            if not isinstance(value,GenerationInput):
                raise TypeError('Generation route requires GenerationInput')
        elif not isinstance(value,EncoderInput) or set(value.modalities) != set(spec.modalities):
            raise TypeError('Encoder route requires EncoderInput with exactly its registered modalities')
        if _input_size(value) > spec.resources.max_input_bytes:
            raise PoolBusy('Input exceeds route byte credit')
        with self._condition:
            if not self._accepting:
                raise PoolClosed('Pool admission is closed')
            if route in self._failed_routes:
                raise PoolExecutionError(self._failed_routes[route])
            if all(name in self._quarantined for name in spec.devices):
                raise PoolCleanupError('Every configured slot for this route is quarantined')
            if not self._available_slots(spec):raise PoolBusy('No eligible undrained slot for this route')
            tenant_handles = [h for h in self._handles if h._tenant == tenant]
            if (self._pending >= self._max_pending or len(self._handles) >= self._max_results
                    or sum(h._state in ('queued','running') for h in tenant_handles) >= self._tenant_pending
                    or len(tenant_handles) >= self._tenant_results):
                raise PoolBusy('Pending/result or tenant credits exhausted')
            handle = PoolHandle(self,self._serial,spec,value,seed,tenant,affinity)
            # Preserve enough cold/transient headroom for the largest admitted
            # request, preventing a queue of payloads from starving every loader.
            wanted = {spec.name: spec}
            wanted.update({h._route.name:h._route for h in self._handles if h._state=='queued'})
            def headroom_for(candidate: ModelRoute) -> int:
                exists = any(r.route.name==candidate.name and r.state in ('loading','idle','running')
                             and r.slot.name not in self._quarantined for r in self._residents)
                return candidate.resources.transient_host_bytes+(0 if exists else candidate.resources.resident_host_bytes)
            headroom = max(headroom_for(candidate) for candidate in wanted.values())
            reclaimable = sorted((r for r in self._residents if r.state=='idle' and r.active is None
                                  and r.route.name not in wanted and r.slot.name not in self._quarantined),
                                 key=lambda r:(r.last_used,r.slot.name,r.route.name))
            reclaim = sum(r.route.resources.resident_host_bytes for r in reclaimable)
            payload_total = self._host_used+handle._input_credit+handle._output_credit
            if payload_total > self._host_limit or payload_total+headroom-reclaim > self._host_limit:
                # A different idle model can occupy the room needed even for
                # request payloads. Begin a verified idle-only retirement, but
                # do not grant credits early; the caller receives backpressure.
                if reclaimable:
                    self._close_resident(reclaimable[0])
                raise PoolBusy('Host payload/workspace reservation exhausted')
            self._serial += 1
            self._handles.add(handle)
            self._pending += 1
            self._host_used += handle._input_credit+handle._output_credit
            if tenant not in self._queues:
                self._queues[tenant] = deque(); self._tenants.append(tenant)
            self._queues[tenant].append(handle)
            if self._thread is None:
                try:
                    self.work.mkdir(exist_ok=False)
                    self._thread = threading.Thread(target=self._dispatch,name='aureon-model-pool',daemon=True)
                    self._thread.start()
                except BaseException:
                    self._thread = None
                    self._cancel(handle)
                    self._accepting = False
                    raise
            self._condition.notify_all()
            return handle

    def _release_result(self, handle: PoolHandle) -> None:
        if handle in self._handles:
            self._handles.remove(handle)
            self._host_used -= handle._output_credit
            self._condition.notify_all()

    def _cancel(self, handle: PoolHandle) -> bool:
        if handle._owner is not self or handle._state != 'queued':
            return False
        self._queues[handle._tenant].remove(handle)
        handle._input = None; handle._state = 'cancelled'
        self._pending -= 1; self._host_used -= handle._input_credit
        self._release_result(handle)
        return True

    def _finish(self, resident: _Resident, error: str | None = None) -> None:
        handle = resident.active
        if handle is None:
            return
        route = resident.route
        self._host_used -= route.resources.transient_host_bytes+handle._input_credit
        self._pending -= 1
        handle._input = None
        if handle._discard:
            handle._state = 'discarded'; self._release_result(handle)
        else:
            handle._error = error
            if error is None:
                handle._result = PoolResult(route.name,_route_identity(route),_execution_key(route,handle._seed),
                    resident.slot.name,resident.generation,handle._seed,
                    'generation' if isinstance(route,GenerationRoute) else 'features',cast(GenerationResult|PooledFeatures,resident.incoming))
            handle._state = 'failed' if error else 'completed'
        resident.active = None; resident.incoming = None
        self._clock += 1; resident.last_used = self._clock
        self._condition.notify_all()

    def _fail_route(self, route: ModelRoute, message: str) -> None:
        self._failed_routes[route.name] = message
        for queue in self._queues.values():
            for handle in list(queue):
                if handle._route.name == route.name:
                    queue.remove(handle); handle._input = None; handle._state = 'failed'; handle._error = message
                    self._pending -= 1; self._host_used -= handle._input_credit
        self._condition.notify_all()

    def _close_resident(self, resident: _Resident) -> None:
        if resident.state != 'idle' or resident.active is not None:
            return
        assert resident.transport is not None
        resident.transport.send({'type':'close'})
        resident.state = 'closing'

    def _record_cleanup(self, value: dict[str, Any]) -> None:
        self._cleanup_count += 1
        self._cleanup_records.append(value)

    def _poll(self) -> None:
        for resident in list(self._residents):
            assert resident.transport is not None
            try:
                while True:
                    try:
                        received = resident.transport.incoming.get_nowait()
                    except thread_queue.Empty:
                        break
                    if isinstance(received, Exception):
                        # A writer failure/EOF may race the reader's final close
                        # acknowledgement. Retain the fault, then drain all
                        # produced messages before deciding cleanup failed.
                        resident.transport_fault = str(received)[:1000]
                        if resident.transport_fault_at is None:
                            resident.transport_fault_at = monotonic()
                        continue
                    message = received
                    kind = message.get('type')
                    if resident.state == 'quarantined' and kind != 'closed':
                        continue
                    if kind == 'ready' and resident.state == 'loading':
                        if message.get('identity') != _route_identity(resident.route) or message.get('generation') != resident.generation:
                            raise PoolError('Worker route/generation attestation mismatch')
                        device = message['device']
                        physical = device.get('uuid')
                        if physical and any(r.slot.name != resident.slot.name and r.device and r.device.get('uuid')==physical for r in self._residents):
                            raise PoolError('Distinct slots alias the same physical CUDA device')
                        resident.device = device
                        resident.state = 'idle'
                    elif kind == 'result' and resident.state == 'running':
                        if resident.active is None or message.get('id') != resident.active._id:
                            raise PoolError('Worker response request identity mismatch')
                        _validated_output(message['value'],resident.route)
                        resident.incoming = message['value']
                    elif kind == 'idle' and resident.state == 'running' and resident.incoming is not None:
                        self._finish(resident); resident.state = 'idle'
                    elif kind == 'error':
                        reason = str(message.get('error_type','WorkerError'))+': '+str(message.get('message',''))[:2000]
                        self._placement_failed(resident,reason)
                        self._finish(resident,reason); resident.state = 'closing'
                    elif kind == 'closed':
                        if message.get('identity') != _route_identity(resident.route) or message.get('generation') != resident.generation:
                            raise PoolError('Worker close identity mismatch')
                        resident.close_ack = _cleanup_receipt(message.get('cleanup'))
                        if resident.state != 'quarantined':
                            resident.state = 'closing'
                    else:
                        raise PoolError('Unexpected owned-worker protocol state')
            except (OSError,EOFError,PoolError,ValueError,TypeError,KeyError) as error:
                # EOF after close acknowledgement is normal, but capacity stays
                # held until process exit and the acknowledgement are verified.
                if resident.close_ack is None and resident.state != 'quarantined':
                    self._quarantine(resident,'Worker protocol/death: '+str(error)[:1000])
            alive = resident.process.is_alive()
            reader_done = resident.transport.reader_done.is_set()
            # The queue may have become nonempty after the get_nowait snapshot
            # above, just before the reader exited. A producer fence PLUS an
            # empty queue is required; is_alive alone drops that final receipt.
            if reader_done and not resident.transport.incoming.empty():
                continue
            if not alive:
                resident.process.join(0)
                if resident.exit_observed_at is None:
                    resident.exit_observed_at = monotonic()
                if not reader_done:
                    if monotonic()-resident.exit_observed_at >= _EXIT_DRAIN_SECONDS:
                        self._quarantine(resident,'Post-exit transport drain timed out without a verified reader fence')
                    continue
                if resident.close_ack is not None and resident.close_ack.get('passed') is True and resident.process.exitcode==0:
                    if resident.active is not None:
                        self._finish(resident,'Worker exited before completing its request')
                    self._host_used -= resident.route.resources.resident_host_bytes+resident.held_host_bytes
                    self._record_cleanup({'route':resident.route.name,'slot':resident.slot.name,
                        'generation':resident.generation,'passed':True,'receipt':resident.close_ack})
                    resident.transport.stop(); resident.process.close(); self._residents.remove(resident)
                elif resident.state != 'quarantined':
                    reason = ('Worker reported unsuccessful cleanup' if resident.close_ack is not None
                              and resident.close_ack.get('passed') is not True else
                              'Worker exited without a verified successful close')
                    self._quarantine(resident,reason)
            elif resident.close_ack is None and (reader_done or
                    (resident.transport_fault_at is not None and monotonic()-resident.transport_fault_at >= _EXIT_DRAIN_SECONDS)):
                self._quarantine(resident,'Worker transport ended without a verified close: '+str(resident.transport_fault or 'reader ended'))

    def _quarantine(self, resident: _Resident, message: str) -> None:
        if resident.state == 'quarantined':
            return
        self._quarantined[resident.slot.name] = message
        if resident.active is not None:
            handle = resident.active
            transient = resident.route.resources.transient_host_bytes
            resident.held_host_bytes += handle._input_credit+handle._output_credit+transient
            # Move the output reservation from the failed result to the unsafe
            # worker. The failed handle carries only a bounded error string.
            self._host_used += handle._input_credit+transient
            handle._output_credit = 0
            if resident.slot.cuda_visible_device is not None:
                resident.held_device_bytes += resident.route.resources.transient_device_bytes+resident.route.resources.max_output_bytes
        self._placement_failed(resident,message); self._finish(resident,message)
        resident.state = 'quarantined'
        self._record_cleanup({'route':resident.route.name,'slot':resident.slot.name,
            'generation':resident.generation,'passed':False,'reason':message,
            'receipt':_cleanup_receipt(resident.close_ack) if resident.close_ack is not None else None,
            'process_exitcode':resident.process.exitcode,
            'reader_done':resident.transport.reader_done.is_set() if resident.transport is not None else False,
            'incoming_drained':resident.transport.incoming.empty() if resident.transport is not None else False,
            'transport_fault':resident.transport_fault})
        # Do not kill/preempt a model or reuse its unverified memory reservation.
        if resident.process.is_alive() and resident.transport is not None:
            try:
                resident.transport.send({'type':'close'})
            except PoolError:
                pass

    def _device_used(self, slot: DeviceSlot) -> int:
        if slot.cuda_visible_device is None:
            return 0
        return sum(r.route.resources.resident_device_bytes+r.held_device_bytes+
            (r.route.resources.transient_device_bytes+r.route.resources.max_output_bytes if r.active is not None else 0)
            for r in self._residents if r.slot.name==slot.name)

    def _available_slots(self,route: ModelRoute) -> list[str]:
        return [name for name in route.devices if name not in self._quarantined and name not in self._draining
                and (route.name,name) not in self._failed_placements and self._slot_compatible(route,self._devices[name])]

    def _placement_failed(self,resident: _Resident,message: str) -> None:
        # Fail the in-flight request once; a healthy replica may serve other jobs.
        # No automatic replay, and the failed route/slot pair cannot restart-loop.
        self._failed_placements.add((resident.route.name,resident.slot.name))
        viable=[name for name in resident.route.devices if name not in self._quarantined
                and (resident.route.name,name) not in self._failed_placements
                and self._slot_compatible(resident.route,self._devices[name])]
        if not viable:self._fail_route(resident.route,message)

    def drain_device(self,name: str) -> dict[str,Any]:
        """Stop assigning to a slot; finish active work and verify idle close normally."""
        with self._condition:
            if name not in self._devices:raise ValueError('Unknown device slot')
            self._draining.add(name)
            for resident in self._residents:
                if resident.slot.name==name:self._close_resident(resident)
            self._condition.notify_all()
            return {'slot':name,'draining':True,'active_not_preempted':True}

    def resume_device(self,name: str) -> None:
        with self._condition:
            if name not in self._devices:raise ValueError('Unknown device slot')
            if name in self._quarantined:raise PoolCleanupError('Cannot resume an unverified quarantined slot')
            if not self._accepting:raise PoolClosed('Pool admission is closed')
            self._draining.discard(name);self._condition.notify_all()

    def _rank_slots(self,handle: PoolHandle,*,warm: bool) -> list[str]:
        route=handle._route;offers=[]
        for name in self._available_slots(route):
            slot=self._devices[name]
            occupants=[r for r in self._residents if r.slot.name==name]
            ready=any(r.route.name==route.name and r.state=='idle' for r in occupants)
            if warm and not ready:continue
            if not warm and any(r.route.name==route.name for r in occupants):continue
            needed=route.resources.transient_device_bytes+route.resources.max_output_bytes
            if not warm:needed+=route.resources.resident_device_bytes
            if slot.cuda_visible_device is not None and self._device_used(slot)+needed>slot.memory_bytes:continue
            offers.append(WorkerOffer(name,slot.capabilities,max(0,slot.memory_bytes-self._device_used(slot)),
                slot.relative_capacity,sum(r.active is not None for r in occupants),warm=ready))
        return [o.worker_id for o in rank_worker_offers(offers,model_id=route.model_id,
            task='generation' if isinstance(route,GenerationRoute) else 'features',precision='float32',
            required_bytes=0,required=route.required_capabilities,affinity=handle._affinity)]

    def _select(self, handle: PoolHandle) -> _Resident | None:
        route = handle._route
        ranked=self._rank_slots(handle,warm=True)
        warm = sorted((r for r in self._residents if r.route.name==route.name and r.state=='idle'
                       and r.slot.name in ranked),key=lambda r:ranked.index(r.slot.name))
        for resident in warm:
            extra = route.resources.transient_device_bytes+route.resources.max_output_bytes if resident.slot.cuda_visible_device else 0
            if self._device_used(resident.slot)+extra <= resident.slot.memory_bytes and self._host_used+route.resources.transient_host_bytes <= self._host_limit:
                return resident
        # A loading replica already reserves capacity for one queued request.
        # Repeated dispatcher polls must not fan one cold request out to every GPU.
        eligible=set(self._available_slots(route))
        loading=sum(r.route.name==route.name and r.state=='loading' and r.slot.name in eligible
                    for r in self._residents)
        queued=sum(h._route.name==route.name for queue in self._queues.values() for h in queue)
        if loading >= queued and queued > 0:
            return None
        # Include budget-constrained eligible slots last so idle-only eviction
        # can still make progress; capability/draining failures never enter here.
        ranked=self._rank_slots(handle,warm=False)
        for name in ranked+sorted(set(self._available_slots(route))-set(ranked)):
            slot = self._devices[name]
            if name in self._quarantined or not self._slot_compatible(route,slot):
                continue
            occupants = [r for r in self._residents if r.slot.name==name]
            # Avoid duplicate loads for a route while it is starting or busy.
            if any(r.route.name==route.name for r in occupants):
                continue
            extra_device = route.resources.resident_device_bytes+(route.resources.transient_device_bytes+route.resources.max_output_bytes if slot.cuda_visible_device else 0)
            extra_host = route.resources.resident_host_bytes+route.resources.transient_host_bytes
            if (len(occupants) < slot.max_residents and self._device_used(slot)+extra_device <= slot.memory_bytes
                    and self._host_used+extra_host <= self._host_limit):
                parent, child = self._context.Pipe(duplex=True)
                generation = uuid.uuid4().hex
                process = self._context.Process(target=_worker_main,args=(child,route,slot,self.work/generation,generation),daemon=True)
                resident = _Resident(route,slot,generation,process,parent)
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close(); raise
                child.close()
                resident.transport = _Transport(parent, route)
                self._residents.append(resident)
                self._host_used += route.resources.resident_host_bytes
                return None
            idle = sorted((r for r in self._residents if r.state=='idle' and r.active is None
                           and (r.last_used > 0 or not any(h._route.name==r.route.name for q in self._queues.values() for h in q))
                           and (r.slot.name==name or self._host_used+extra_host>self._host_limit)),
                          key=lambda r:(r.last_used,r.slot.name,r.route.name))
            if idle:
                self._close_resident(idle[0])
                return None
        return None

    def _schedule(self) -> None:
        for retiring in self._residents:
            if retiring.slot.name in self._draining:self._close_resident(retiring)
        # Preserve fairness when an active tenant's temporarily empty queue was
        # removed and then recreated while its previous request was running.
        if len(self._tenants)>1 and self._tenants[0] == self._last_tenant:
            self._tenants.rotate(-1)
        for _ in range(len(self._tenants)):
            tenant = self._tenants[0]; self._tenants.rotate(-1)
            queue = self._queues[tenant]
            if not queue:
                self._tenants.remove(tenant); del self._queues[tenant]; continue
            handle = queue[0]
            if all(name in self._quarantined for name in handle._route.devices):
                self._fail_route(handle._route,'Every configured device slot is quarantined'); continue
            if not self._available_slots(handle._route):
                queue.popleft();self._pending-=1;self._host_used-=handle._input_credit
                handle._input=None;handle._state='failed';handle._error='No undrained healthy compatible slot; request not executed'
                self._condition.notify_all();continue
            resident = self._select(handle)
            if resident is None:
                continue
            queue.popleft(); handle._state = 'running'
            self._last_tenant = tenant
            resident.active = handle; resident.state = 'running'
            self._placement_counts[resident.slot.name]+=1
            self._host_used += handle._route.resources.transient_host_bytes
            try:
                payload = _input_wire(cast(PoolInput,handle._input))
                assert resident.transport is not None
                resident.transport.send({'type':'run','identity':_route_identity(handle._route),
                    'id':handle._id,'seed':handle._seed,'input':payload})
            except BaseException as error:
                self._quarantine(resident,'Dispatch failed: '+str(error)[:1000])
            # Keep the next tenant at the head after an actual assignment.
            # A full scan must not reset fairness to the same busy tenant.
            return

    def _dispatch(self) -> None:
        try:
            while True:
                with self._condition:
                    self._poll()
                    self._schedule()
                    if not self._accepting and self._pending == 0:
                        for resident in self._residents:
                            self._close_resident(resident)
                        if all(r.state=='quarantined' and not r.process.is_alive() for r in self._residents):
                            for resident in self._residents:
                                if resident.transport is not None:
                                    resident.transport.stop()
                            self._closed = True; self._condition.notify_all(); return
                    self._condition.wait(0.01)
        except BaseException as error:
            with self._condition:
                self._accepting = False
                for route in self._routes.values():
                    self._fail_route(route,'Pool dispatcher failed: '+str(error)[:1000])
                for resident in self._residents:
                    self._quarantine(resident,'Pool dispatcher failed')
                self._closed = True; self._condition.notify_all()

    def metadata(self) -> dict[str, Any]:
        with self._condition:
            return {'schema':'aureon.model_pool.v1','accepting':self._accepting,'closed':self._closed,
                'pending':self._pending,'reserved_results':len(self._handles),
                'host_reserved_bytes':self._host_used,'host_budget_bytes':self._host_limit,
                'devices':{name:{'reserved_bytes':self._device_used(slot),'budget_bytes':slot.memory_bytes,
                    'quarantined':name in self._quarantined,'draining':name in self._draining,
                    'relative_capacity':slot.relative_capacity,'capabilities':asdict(slot.capabilities),
                    'assignments':self._placement_counts[name]} for name,slot in self._devices.items()},
                'residents':[{'route':r.route.name,'identity':_route_identity(r.route),'slot':r.slot.name,
                    'generation':r.generation,'state':r.state,'active':r.active is not None,
                    'device':dict(r.device) if r.device else None} for r in self._residents],
                'failed_routes':dict(self._failed_routes),'quarantined_slots':dict(self._quarantined),
                'cleanup_event_count':self._cleanup_count,'cleanup_history_capacity':128,
                'policy':'exact_route; capability/memory admission; warm affinity and declared load/capacity; tenant round-robin; idle-only LRU',
                'failed_placements':sorted(self._failed_placements),'transport':'trusted same-host owned-process IPC only; remote protocol is separate',
                'memory_scope':'Declared resident/transient workspaces plus 8-copy payload credits and bounded metadata; not physical RSS/VRAM enforcement. Caller-owned/consumed outputs excluded.',
                'performance_qualified':False}

    def close(self, *, cancel_pending: bool = False, timeout: float | None = None) -> dict[str, Any]:
        if type(cancel_pending) is not bool:
            raise TypeError('cancel_pending must be bool')
        duration = validate_timeout(timeout)
        with self._condition:
            if self._close_report is not None:
                return cast(dict[str, Any], json.loads(json.dumps(self._close_report)))
            self._accepting = False
            if cancel_pending:
                for handle in list(self._handles):
                    self._cancel(handle)
            self._condition.notify_all()
            thread = self._thread
        if thread is threading.current_thread():
            raise PoolBusy('Cannot close pool from its dispatcher')
        if thread is not None:
            thread.join(duration)
            if thread.is_alive():
                raise TimeoutError('Pool close is still draining; active model calls are not preempted')
        with self._condition:
            for handle in list(self._handles):
                handle.discard()
            self._closed = True
            alive = [r.generation for r in self._residents if r.process.is_alive()]
            for resident in self._residents:
                if not resident.process.is_alive():
                    resident.process.close()
            self._close_report = {'passed':not self._quarantined and not alive,'closed':True,
                'quarantined_slots':dict(self._quarantined),'unverified_live_workers':alive,
                'pending':self._pending,'reserved_results':len(self._handles),
                'host_reserved_bytes':self._host_used,'cleanup':list(self._cleanup_records),
                'cleanup_event_count':self._cleanup_count,'cleanup_history_capacity':128,
                'cleanup_events_omitted':self._cleanup_count-len(self._cleanup_records),
                'scope':'Owned process/close acknowledgements; actual hardware qualification is separate'}
            return cast(dict[str, Any], json.loads(json.dumps(self._close_report)))

    def __enter__(self) -> ModelPool:
        if not self._accepting:
            raise PoolClosed('Pool is closed')
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> Literal[False]:
        receipt = close_preserving_error(lambda:self.close(cancel_pending=exc is not None),exc,context='Model pool cleanup failed')
        if exc is None and receipt is not None and receipt.get('passed') is not True:
            raise PoolCleanupError('Model pool closed with quarantined/unverified workers; inspect close receipt')
        return False
