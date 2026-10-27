"""Array-only scene adapters; construction never imports or executes a model.

The caller explicitly loads and configures the predictor. FixedEightAdapter
adapts arbitrary final tile groups to an existing fixed-eight callable; that
callable remains responsible for ordered batch-one model arithmetic. Padding
is normalized zero and its outputs never contribute to the scene. The eager
adapter invokes its independent reference execution path one tile at a time.
Neither adapter changes precision, acquires weights, retries, or falls back.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import threading

import numpy as np


CAPACITY = 8
TILE_INPUT_SHAPE = (12, 224, 224)
TILE_OUTPUT_SHAPE = (10, 224, 224)


def _input_copy(values):
    if (not isinstance(values, np.ndarray) or values.ndim != 4
            or tuple(values.shape[1:]) != TILE_INPUT_SHAPE
            or not 1 <= values.shape[0] <= CAPACITY
            or values.dtype != np.dtype("float32")
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite normalized float32 [N,12,224,224], 1 <= N <= 8")
    return np.array(values, dtype=np.float32, order="C", copy=True)


def _logits(values, count):
    if (not isinstance(values, np.ndarray)
            or values.shape != (count, *TILE_OUTPUT_SHAPE)
            or values.dtype != np.dtype("float32")
            or not np.isfinite(values).all()):
        raise ValueError(f"Predictor must return finite float32 [{count},10,224,224] logits")
    return values


def _describe(predictor):
    get_metadata = getattr(predictor, "metadata", None)
    if callable(get_metadata):
        value = get_metadata()
        if not isinstance(value, dict):
            raise ValueError("Predictor metadata must be a dictionary")
        return deepcopy(value)
    value = {name: getattr(predictor, name, None)
             for name in ("execution_backend", "model_id", "device", "dtype")}
    bundle = getattr(predictor, "bundle", None)
    if isinstance(bundle, dict):
        value["bundle_receipt_sha256"] = bundle.get("receipt_sha256")
        value["bundle_files_sha256"] = deepcopy(bundle.get("files_sha256"))
    return value


class _Adapter:
    def __init__(self, predictor, *, owns_predictor=False):
        if not callable(predictor):
            raise ValueError("Supply an explicitly constructed callable predictor")
        if type(owns_predictor) is not bool:
            raise ValueError("owns_predictor must be a boolean")
        self._predictor = predictor
        self._owns_predictor = owns_predictor
        self._lock = threading.Lock()
        self._closed_backend = None
        self.state = "ready"
        self.calls_attempted = self.calls_completed = self.backend_calls_attempted = 0
        self.real_tiles_submitted = self.real_tiles_completed = 0
        self.padding_slots_submitted = self.padding_outputs_discarded = 0
        self.failure = None

    def _ready(self):
        if self.state != "ready":
            raise RuntimeError(f"Scene predictor adapter is {self.state}; no retry or fallback")

    def _failed(self, error):
        self.state = "failed"
        self.failure = {"type": type(error).__name__, "message": str(error)}

    def metadata(self):
        with self._lock:
            backend = (_describe(self._predictor) if self._predictor is not None
                       else deepcopy(self._closed_backend))
            return {
                "schema": "aureon.scene_predictor.v1",
                "adapter": type(self).__name__, "state": self.state,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "capacity": CAPACITY, "input_dtype": "float32", "output_dtype": "float32",
                "calls_attempted": self.calls_attempted,
                "calls_completed": self.calls_completed,
                "backend_calls_attempted": self.backend_calls_attempted,
                "real_tiles_submitted": self.real_tiles_submitted,
                "real_tiles_completed": self.real_tiles_completed,
                "padding_slots_submitted": self.padding_slots_submitted,
                "padding_outputs_discarded": self.padding_outputs_discarded,
                "padding_value": 0.0 if isinstance(self, FixedEightAdapter) else None,
                "padding_units": "normalized" if isinstance(self, FixedEightAdapter) else None,
                "input_ownership": "independent writable copy for each backend invocation",
                "output_ownership": "independent writable C-contiguous NumPy array per successful call",
                "owns_predictor": self._owns_predictor,
                "output_cache": False, "eager_fallback": False,
                "numerical_parity_verified": False,
                "failure": deepcopy(self.failure), "backend": backend,
                "execution_scope": self.execution_scope,
                "capture_and_warmup_scope": "Owned by injected backend; inspect backend metadata and caller timing receipts",
            }

    def close(self):
        with self._lock:
            if self.state == "closed":
                return
            # Preserve metadata before releasing the wrapper's own reference.
            self._closed_backend = _describe(self._predictor)
            if self._owns_predictor:
                close = getattr(self._predictor, "close", None)
                if callable(close):
                    close()
                else:
                    synchronize = getattr(self._predictor, "synchronize", None)
                    if callable(synchronize):
                        synchronize()
            # The adapter never deletes external references or clears a global
            # CUDA allocator. With owns_predictor=False the caller closes it.
            self._predictor = None
            self.state = "closed"

    def __enter__(self):
        with self._lock:
            self._ready()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class FixedEightAdapter(_Adapter):
    """Pad one group of 1..8 tiles and discard only the unused output slots.

    Intended backend: FullGroupGraphPipeline.from_reference(...), which
    executes eight ordered batch-one forwards. This wrapper itself performs
    no model operations and cannot attest an arbitrary injected callable.
    """

    execution_scope = "Fixed-eight backend call; backend must preserve ordered batch-one arithmetic"

    def __call__(self, normalized_batch):
        values = _input_copy(normalized_batch)
        count = len(values)
        with self._lock:
            self._ready()
            self.calls_attempted += 1
            self.real_tiles_submitted += count
            self.padding_slots_submitted += CAPACITY - count
            try:
                group = np.zeros((CAPACITY, *TILE_INPUT_SHAPE), dtype=np.float32)
                group[:count] = values
                self.backend_calls_attempted += 1
                output = _logits(self._predictor(group), CAPACITY)
                result = np.array(output[:count], dtype=np.float32, order="C", copy=True)
            except BaseException as error:
                self._failed(error)
                raise
            self.calls_completed += 1
            self.real_tiles_completed += count
            self.padding_outputs_discarded += CAPACITY - count
            return result


class SequentialEagerAdapter(_Adapter):
    """Call an injected strict reference separately for every tile, in order."""

    execution_scope = "Independent eager execution path, one tile per backend call in input order"

    def __call__(self, normalized_batch):
        values = _input_copy(normalized_batch)
        count = len(values)
        with self._lock:
            self._ready()
            self.calls_attempted += 1
            self.real_tiles_submitted += count
            try:
                result = np.empty((count, *TILE_OUTPUT_SHAPE), dtype=np.float32)
                for index in range(count):
                    tile = np.array(values[index:index + 1], dtype=np.float32, order="C", copy=True)
                    self.backend_calls_attempted += 1
                    result[index:index + 1] = _logits(self._predictor(tile), 1)
            except BaseException as error:
                self._failed(error)
                raise
            self.calls_completed += 1
            self.real_tiles_completed += count
            return result


class FullEightEagerTailAdapter(_Adapter):
    """Use the fixed-eight graph only for eight real tiles; execute tails eagerly.

    A count of 1..7 selects ordered batch-one calls to ``eager_predictor``
    before any execution. This is planned dispatch, never an error fallback.
    The caller supplies both explicit backends, owns their shared strict model,
    sets/checks precision, and closes the backends. Construction/import performs
    no runtime import, graph capture or model execution.
    """

    execution_scope = ("Eight real tiles: existing fixed-eight backend; one to seven real tiles: "
                       "ordered eager batch-one calls; zero padding forwards")

    def __init__(self, graph_predictor, eager_predictor):
        if not callable(eager_predictor):
            raise ValueError("Supply an explicitly constructed eager callable")
        super().__init__(graph_predictor, owns_predictor=False)
        # metadata() composes the unchanged base receipt under the same lock.
        self._lock = threading.RLock()
        self._eager_predictor = eager_predictor
        self._closed_eager_backend = None
        self.graph_groups_attempted = self.graph_groups_completed = 0
        self.tail_groups_attempted = self.tail_groups_completed = 0
        self.eager_tile_calls_attempted = self.eager_tile_calls_completed = 0
        self.backend_calls_completed = 0
        self.neural_tile_forwards_submitted = self.neural_tile_forwards_completed = 0

    def __call__(self, normalized_batch):
        values = _input_copy(normalized_batch)
        count = len(values)
        with self._lock:
            self._ready()
            self.calls_attempted += 1
            self.real_tiles_submitted += count
            try:
                if count == CAPACITY:
                    self.graph_groups_attempted += 1
                    self.backend_calls_attempted += 1
                    self.neural_tile_forwards_submitted += CAPACITY
                    output = _logits(self._predictor(values), CAPACITY)
                    result = np.array(output, dtype=np.float32, order="C", copy=True)
                    self.backend_calls_completed += 1
                    self.neural_tile_forwards_completed += CAPACITY
                    self.graph_groups_completed += 1
                else:
                    self.tail_groups_attempted += 1
                    result = np.empty((count, *TILE_OUTPUT_SHAPE), dtype=np.float32)
                    for index in range(count):
                        tile = np.array(values[index:index + 1], dtype=np.float32, order="C", copy=True)
                        self.backend_calls_attempted += 1
                        self.eager_tile_calls_attempted += 1
                        self.neural_tile_forwards_submitted += 1
                        result[index:index + 1] = _logits(self._eager_predictor(tile), 1)
                        self.backend_calls_completed += 1
                        self.eager_tile_calls_completed += 1
                        self.neural_tile_forwards_completed += 1
                    self.tail_groups_completed += 1
            except BaseException as error:
                self._failed(error)
                raise
            self.calls_completed += 1
            self.real_tiles_completed += count
            return result

    def _describe_eager(self):
        value = _describe(self._eager_predictor)
        value["validate_on_cpu"] = getattr(self._eager_predictor, "validate_on_cpu", None)
        return value

    def metadata(self):
        with self._lock:
            value = super().metadata()
            value.update({
                "dispatch_policy": "count==8: fixed-eight graph; count in 1..7: ordered eager batch-one",
                "graph_groups_attempted": self.graph_groups_attempted,
                "graph_groups_completed": self.graph_groups_completed,
                "tail_groups_attempted": self.tail_groups_attempted,
                "tail_groups_completed": self.tail_groups_completed,
                "eager_tile_calls_attempted": self.eager_tile_calls_attempted,
                "eager_tile_calls_completed": self.eager_tile_calls_completed,
                "backend_calls_completed": self.backend_calls_completed,
                "neural_tile_forwards_submitted": self.neural_tile_forwards_submitted,
                "neural_tile_forwards_completed": self.neural_tile_forwards_completed,
                "forward_counter_scope": "Requested logical tile forwards and validated returned logits; excludes internal capture/warmup; a failed invocation may have executed only part of its submitted work",
                "owns_eager_predictor": False,
                "eager_backend": (self._describe_eager() if self._eager_predictor is not None
                                  else deepcopy(self._closed_eager_backend)),
                "tail_validation_policy": "Inherited injected eager predictor policy; this adapter also validates CPU FP32 shape/finiteness",
            })
            return value

    def close(self):
        with self._lock:
            if self.state == "closed":
                return
            self._closed_eager_backend = self._describe_eager()
            super().close()
            self._eager_predictor = None
