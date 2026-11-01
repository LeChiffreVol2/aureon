"""Shared pair ownership for independent lane execution.

The base class has no executable denoiser implementation. Concrete lane sessions
supply their admitted executor; the retired fused-pair session is not exported.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast
from collections.abc import Callable, Sequence
from ..terramind_native._types import Metadata, PathInput, InputSnapshot
from concurrent.futures import CancelledError
from copy import deepcopy
import threading
from time import perf_counter
import uuid

from .generation_bridges import GenerationJob
from .generation_queue import INPUT_BYTES, OUTPUT_BYTES
from ..terramind_native.common import binding, write_json
from ..terramind_native.session import (TerraMindSession, SessionBusy, SessionClosed,
    SessionFailed, _NativeRuntime, _PROCESS_OWNER, _identifier)
from ..terramind_native.denoiser_batch import BatchIneligible, WORKSPACE_BYTES, compatible_recipes


class PairCredits:
    """One active pair; CPU input/result credits transfer at synchronous return.

These credits bound engine-owned snapshots/results, not caller-retained arrays
or neural activation memory. Cancellation releases the whole pair's credits.
"""
    def __init__(self, max_input_bytes: int, max_output_bytes: int) -> None:
        if (type(max_input_bytes) is not int or max_input_bytes < 2 * INPUT_BYTES
                or type(max_output_bytes) is not int or max_output_bytes < 2 * OUTPUT_BYTES):
            raise ValueError('Credits must admit exactly two fixed-profile inputs/results')
        self._lock = threading.Lock()
        self._active = False

    def reserve(self) -> None:
        with self._lock:
            if self._active: raise SessionBusy('One batch pair already owns the input/result credits')
            self._active = True

    def release(self) -> None:
        with self._lock:
            if not self._active: raise RuntimeError('Pair credits already released')
            self._active = False

    def metadata(self) -> Metadata:
        with self._lock:
            return {'active_pairs': int(self._active), 'input_bytes_reserved': int(self._active)*2*INPUT_BYTES,
                    'output_bytes_reserved': int(self._active)*2*OUTPUT_BYTES,
                    'retention': 'ownership transfers to caller at synchronous return'}


class _PairSession(TerraMindSession, ABC):
    """Private pair admission, result ownership and lifecycle for lane subclasses."""

    def __init__(self, config: PathInput, work: PathInput, *, execute_model: bool=False, allow_unqualified_batch: bool=False,
                 reuse: str='auto', max_input_bytes: int=2*INPUT_BYTES, max_output_bytes: int=2*OUTPUT_BYTES,
                 max_workspace_bytes: int=64*1024**2, min_free_cuda_bytes: int=512*1024**2) -> None:
        if allow_unqualified_batch is not True:
            raise ValueError('Pass allow_unqualified_batch=True for this unqualified source candidate')
        self._pair_credits = PairCredits(max_input_bytes, max_output_bytes)
        if type(max_workspace_bytes) is not int or max_workspace_bytes < WORKSPACE_BYTES:
            raise ValueError('Insufficient explicit denoiser tensor workspace budget')
        if type(min_free_cuda_bytes) is not int or min_free_cuda_bytes < 0:
            raise ValueError('min_free_cuda_bytes must be nonnegative')
        self._pair_workspace = max_workspace_bytes
        self._pair_free = min_free_cuda_bytes
        # Existing B1 path remains native here. GenerationEngine recommended and
        # all accepted runtime files are unchanged; no B2 graph is installed.
        super().__init__(config, work, execute_model=execute_model, reuse=reuse)

    def metadata(self) -> Metadata:
        return {**super().metadata(), 'batch_candidate': {'max_batch_size': 2,
                'stage': 'deterministic_denoiser', 'graph': False, 'qualified': False,
                'credits': self._pair_credits.metadata()}}

    def _runtime_factory(self) -> _NativeRuntime:
        return _NativeRuntime()

    def _pair_admission(self, recipes: list[Metadata]) -> Metadata:
        from ..terramind_native.denoiser_batch import eligibility
        return eligibility(cast("_NativeRuntime", self._runtime), recipes, max_workspace_bytes=self._pair_workspace,
                           min_free_cuda_bytes=self._pair_free)

    @abstractmethod
    def _execute_pair(self, snapshots: list[InputSnapshot], recipes: list[Metadata], ids: list[tuple[str, str]], cancel: Callable[[], None], admission: Metadata) -> list[Metadata]:
        raise NotImplementedError("Concrete lane executor required")

    def generate_pair(self, jobs: Sequence[GenerationJob], *, cancel_event: threading.Event | None=None) -> list[Metadata]:
        if not isinstance(jobs, (tuple, list)) or len(jobs) != 2 or not all(isinstance(j, GenerationJob) for j in jobs):
            raise TypeError('Supply a finite list/tuple of exactly two GenerationJob objects')
        if any(j.execution_profile is not None for j in jobs):
            raise BatchIneligible('Pair owns its eager experimental profile; omit B1 execution profiles')
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError('cancel_event must be threading.Event or None')
        recipes = compatible_recipes([j.recipe.native() for j in jobs])
        ids = [(_identifier(j.job_id or uuid.uuid4().hex, 'job_id'), _identifier(j.attempt_id, 'attempt_id')) for j in jobs]
        if len(set(ids)) != 2: raise ValueError('Pair job/attempt identifiers must be distinct')
        def cancel() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError('Pair cancelled at a safe stage boundary; no partial result delivered')
        if not self._gate.acquire(blocking=False): raise SessionBusy('The session already owns an active call')
        reserved = False
        try:
            if self._state in ('failed', 'closed'): raise SessionClosed('Session is terminal')
            if any(identity in self._attempts for identity in ids): raise FileExistsError('Pair attempt already admitted')
            cancel()
            self._pair_credits.reserve(); reserved = True
            snapshots = [j.input_data.snapshot() for j in jobs]
            return self._executor.submit(self._pair_owned, snapshots, recipes, ids, cancel).result()
        finally:
            if reserved: self._pair_credits.release()
            self._gate.release()

    def _pair_owned(self, snapshots: list[InputSnapshot], recipes: list[Metadata], ids: list[tuple[str, str]], cancel: Callable[[], None]) -> list[Metadata]:
        self._owner_thread = threading.get_ident()
        if binding(self._config_path) != self._config_identity: raise ValueError('Session config changed')
        cancel()
        outputs = [self.work/'jobs'/job/attempt for job, attempt in ids]
        for output in outputs:
            if any(p.is_symlink() for p in (self.work, self.work/'jobs', output.parent, output)):
                raise ValueError('Output paths cannot contain symlinks')
            if output.exists(): raise FileExistsError(output)
        if not self._owns_process:
            if not _PROCESS_OWNER.acquire(blocking=False): raise SessionBusy('Another session owns the process')
            self._owns_process = True
        try:
            if cast("_NativeRuntime", self._runtime) is None:
                self._load_attempts += 1
                self._runtime = self._runtime_factory()
                self._runtime.load(self._config_path, self.work)
                self._loads += 1
            cancel()
            # Admission happens before creating job evidence or consuming attempts.
            admission = self._pair_admission(recipes)
            for output in outputs:
                output.parent.mkdir(exist_ok=True); output.mkdir(exist_ok=False)
            self._attempts.update(ids)
            self._state = 'running'
            start = perf_counter()
            results = self._execute_pair(snapshots, recipes, ids, cancel, admission)
            commit = self.work/('batch-' + uuid.uuid4().hex + '.json')
            for result, output in zip(results, outputs):
                result['pair_complete_seconds'] = perf_counter() - start
                result['model_load_count'] = self._loads
                result['owner_thread_id'] = self._owner_thread
                write_json(output/'receipt.json', {**{k:v for k,v in result.items() if k not in ('array','valid_mask')},
                    'status':'prepared', 'requires_pair_commit':commit.name})
            # A per-job prepared receipt alone is never a successful pair. This
            # final record binds both identities; no fsync/durable-service claim.
            write_json(commit, {'status':'completed', 'jobs':[binding(p/'receipt.json') for p in outputs],
                                'scope':'local synchronous memory delivery; not a durable service transaction'})
            for result in results: result['pair_commit'] = binding(commit)
            self._completed += 2; self._state = 'ready'
            return [deepcopy(result) for result in results]
        except (CancelledError, BatchIneligible) as exc:
            from ..terramind_native.harness import capture_failure
            message = capture_failure(exc)
            for output in outputs:
                if output.exists() and not (output/'receipt.json').exists():
                    write_json(output/'cancelled.json', {'status': 'cancelled', 'reason': message,
                                                       'partial_result_delivered': False})
            self._state = 'ready' if cast("_NativeRuntime", self._runtime) is not None else 'new'
            raise type(exc)(str(exc)) from None
        except BaseException as exc:
            from ..terramind_native.harness import capture_failure
            error = capture_failure(exc)
            self._state = 'failed'
            try:
                for output in outputs:
                    if output.exists() and not (output/'receipt.json').exists():
                        write_json(output/'failure.json', {'status': 'failed', 'error': error})
            finally:
                self._close_owned()
            raise SessionFailed('Batch candidate failed; session terminated. ' + error) from None
