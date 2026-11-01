"""Small resident TerraMind facade; imports/unused construction never load a model."""
from __future__ import annotations
from typing import TYPE_CHECKING, cast
from .contracts import NativeOptions, RecipeValues, GroupRequests
if TYPE_CHECKING:
    from ..terramind_native.session import TerraMindSession
    from .generation_parallel import LaneSession

from collections.abc import Callable, Iterable
from copy import deepcopy
from enum import Enum
import math
from os import PathLike
from pathlib import Path
import threading
from types import TracebackType
from typing import Any, Literal

from .contracts import (
    INPUT_BYTES, OUTPUT_BYTES, EngineMetadata, EngineState, ExecutionProfile,
    GenerationResult, ParallelFallback, Parallelism, QueueLimits, positive,
)
from .lifecycle import close_preserving_error
from .memory_input import GenerationInput
from .generation_bridges import GenerationRecipe, GenerationJob, generation_capabilities
from .generation_queue import GenerationHandle, GenerationQueue, GenerationStream
from .parallel_policy import ParallelPolicy

from .models import GENERATION_MODEL, open_model_session
from .v1 import ExecutionRequired
from ..terramind_native.execution import validate_selection
from ..terramind_native.families import GENERATION_MODELS
from ..terramind_native.session import SessionBusy, SessionClosed, SessionFailed

class _DefaultSeed(Enum):
    OMITTED = "omitted"


_DEFAULT_SEED = _DefaultSeed.OMITTED


def _record_parallel_selection(result: GenerationResult, selection: dict[str, Any]) -> None:
    result['parallel_selection']=deepcopy(selection)
    result.setdefault('provenance',{})['parallel_selection']=deepcopy(selection)

class GenerationEngine:
    """One explicit local resident engine, with lazy asset binding and owned close.

    Pinned tiny/small/base/large S2L2A224 -> S1GRD generation is exposed. Explicit
    GenerationRecipe supports the existing finite sampling variants. ``work`` must
    be new; its parent must exist. No download or environment installation occurs.
    """

    _session: TerraMindSession | LaneSession | None
    _queue: GenerationQueue | None
    _cleanup: dict[str, Any] | None

    def __init__(self, model: str = GENERATION_MODEL, *, model_file: str | PathLike[str],
                 tokenizer_file: str | PathLike[str], work: str | PathLike[str],
                 execute_model: bool = False, execution_profile: ExecutionProfile = "recommended",
                 max_pending: int | None = None, max_results: int | None = None,
                 max_input_bytes: int | None = None, max_output_bytes: int | None = None,
                 parallel: Literal['off', 'two-lane'] | None = None, pair_wait_ms: float = 5.0,
                 parallelism: Parallelism = 1, parallel_wait_ms: float | None = None,
                 allow_unqualified_parallelism: bool = False, parallel_fallback: ParallelFallback = 'serial',
                 parallel_workspace_bytes: int = 2*1024**3,
                 parallel_resident_bytes: int = 3*1024**3,
                 parallel_policy: ParallelPolicy | None = None) -> None:
        if model not in GENERATION_MODELS:
            raise ValueError("GenerationEngine requires a TerraMind generation model_id")
        if execute_model is not True:
            raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded or created")
        if execution_profile is None:
            raise ValueError("GenerationEngine requires recommended or reference; use open_model_session for expert controls")
        validate_selection(execution_profile)
        if parallel_policy is not None:
            if not isinstance(parallel_policy,ParallelPolicy):raise TypeError('parallel_policy requires ParallelPolicy')
            if (parallelism!=1 or parallel is not None or allow_unqualified_parallelism or parallel_fallback!='serial'
                    or parallel_workspace_bytes!=2*1024**3 or parallel_resident_bytes!=3*1024**3):
                raise ValueError('Choose ParallelPolicy or legacy parallel controls, not both')
            parallelism=parallel_policy.requested
            workload=getattr(parallel_policy,'workload',None)
            if workload is not None and (workload.batch_size!=1 or workload.inputs.shape!=(12,224,224)
                    or workload.outputs.shape!=(2,224,224) or workload.inputs.dtype!='float32' or workload.outputs.dtype!='float32'):
                raise ValueError('Lane policy workload must match the implemented S2L2A224 FP32 B1 route')
        if parallel not in (None,'off','two-lane'):
            raise ValueError('parallel must be off or two-lane')
        if parallel is not None:
            if parallelism!=1:raise ValueError('Choose parallelism or the legacy parallel alias, not both')
            parallelism={'off':1,'two-lane':2}[parallel]
        if model != GENERATION_MODEL and parallelism not in (1, 'auto'):
            raise ValueError("Multiple generation lanes are implemented only for terramind-v1-base; other sizes are not qualified")
        self._parallel_policy=parallel_policy or ParallelPolicy(parallelism,allow_unqualified_parallelism,parallel_fallback,
                                              parallel_workspace_bytes,parallel_resident_bytes)
        if parallel_wait_ms is not None:
            if pair_wait_ms!=5.0:raise ValueError('Choose parallel_wait_ms or legacy pair_wait_ms')
            pair_wait_ms=parallel_wait_ms
        if type(pair_wait_ms) not in (int,float) or not math.isfinite(pair_wait_ms) or not 0 <= pair_wait_ms <= 1000:
            raise ValueError('pair_wait_ms must be finite in [0, 1000]')
        self._parallel=parallel or ('off' if parallelism==1 else 'groups')
        if model != GENERATION_MODEL:
            self._parallel='off'
        self._pair_wait_ms=pair_wait_ms
        self.model = model
        self._model_file = Path(model_file).absolute()
        self._tokenizer_file = Path(tokenizer_file).absolute()
        self.work = Path(work).absolute()
        if self.work.exists() or self.work.is_symlink() or not self.work.parent.is_dir():
            raise ValueError("work must be a fresh directory with an existing parent")
        self.work = self.work.parent.resolve() / self.work.name
        self._execution_profile = execution_profile
        self._session = None
        self._state: EngineState = "new"
        self._cleanup = None
        self._gate = threading.Lock()
        self._queue_lock = threading.Lock()
        self._queue = None
        self._direct_active = False
        self._closing = False
        from .generation_parallel import qualification
        evidence=qualification(model)
        estimate=getattr(self._parallel_policy,'estimate',None)
        if estimate is not None:
            per_lane=positive(estimate.workspace_bytes,'estimated lane workspace')
            desired=max(1,self._parallel_policy.workspace_bytes//per_lane) if parallelism=='auto' else parallelism
        else:
            # Compatibility with the immutable 17c policy. The integrated OSS
            # policy supplies estimate; no hardware name or duplicated byte constant.
            bootstrap=self._parallel_policy.plan(dict(max_pending=1,max_results=1,max_input_bytes=INPUT_BYTES,max_output_bytes=OUTPUT_BYTES),evidence or None)
            per_lane=positive(bootstrap['per_lane_reservation_bytes'],'declared lane workspace')
            desired=(evidence['lanes'] if evidence else 1) if parallelism=='auto' else parallelism
        default_credits=max(2,min(desired,max(1,self._parallel_policy.workspace_bytes//per_lane)))
        max_pending = default_credits if max_pending is None else positive(max_pending, "max_pending")
        max_results = default_credits if max_results is None else positive(max_results, "max_results")
        self._limits: QueueLimits = dict(max_pending=max_pending, max_results=max_results,
            max_input_bytes=max_pending * INPUT_BYTES if max_input_bytes is None else positive(max_input_bytes, "max_input_bytes"),
            max_output_bytes=max_results * OUTPUT_BYTES if max_output_bytes is None else positive(max_output_bytes, "max_output_bytes"))
        if self._limits["max_input_bytes"] < INPUT_BYTES or self._limits["max_output_bytes"] < OUTPUT_BYTES:
            raise ValueError("Byte credits must admit at least one fixed-profile input and result")
        self._limits=self._parallel_policy.bound_queue_limits(self._limits)
        self._parallel_plan=self._parallel_policy.plan(self._limits,evidence or None)
        if model != GENERATION_MODEL:
            # Hardware capacity does not add a native lane adapter for this size.
            self._parallel_plan['planned']=1
            self._parallel_plan['fallback_reason']='model_size_requires_serial_adapter'

    def _open(self) -> None:
        # mkdir is exclusive: a directory created by someone else after
        # construction is never adopted, modified or removed.
        self.work.mkdir(exist_ok=False)
        from ..terramind_native.offline import create_config
        create_config(model=self._model_file, tokenizer=self._tokenizer_file,
                      output=self.work / "assets.json", model_id=self.model)
        if self._parallel != 'off':
            from .generation_parallel import open_lane_session
            self._session = open_lane_session(config=self.work/'assets.json',work=self.work/'session',
                policy=self._parallel_policy,plan=self._parallel_plan,limits=self._limits)
        else:
            self._session = cast("TerraMindSession", open_model_session(self.model, config=self.work / "assets.json",
                                              work=self.work / "session", execution_profile=self._execution_profile,
                                              input_kind="array", execute_model=True))

    def _validate(self, input_data: GenerationInput, *, seed: int | _DefaultSeed=_DEFAULT_SEED, recipe: GenerationRecipe | None=None,
                  execution_profile: ExecutionProfile | None=None, job_id: str | None=None, attempt_id: str="1") -> NativeOptions:
        if not isinstance(input_data, GenerationInput):
            raise TypeError("generate requires GenerationInput with explicit input semantics")
        if recipe is not None:
            if seed is not _DEFAULT_SEED:
                raise ValueError("Choose seed or GenerationRecipe, not both")
            if not isinstance(recipe, GenerationRecipe):
                raise TypeError("recipe requires GenerationRecipe")
            native = recipe.native()
        else:
            native = GenerationRecipe(seed=17 if seed is _DEFAULT_SEED else seed).native()
        profile = self._execution_profile if execution_profile is None else execution_profile
        validate_selection(profile)
        from ..terramind_native.session import _identifier
        if job_id is not None: _identifier(job_id, "job_id")
        _identifier(attempt_id, "attempt_id")
        return dict(recipe=native, execution_profile=profile, job_id=job_id, attempt_id=attempt_id)

    def _submission_queue(self) -> GenerationQueue:
        with self._queue_lock:
            if self._closing or self._state in ("closed", "failed"):
                raise SessionClosed("GenerationEngine is terminal or closing")
            if self._direct_active:
                raise SessionBusy("A synchronous call owns the engine; retry submission after it completes")
            if self._queue is None:
                if self._parallel != 'off':
                    from .generation_group_queue import GenerationGroupQueue
                    self._queue = GenerationGroupQueue(self._generate_one,
                        lambda: self._state in ('closed','failed'),execute_group=self._generate_group,
                        max_group_size=self._parallel_plan['planned'],max_wait_ms=self._pair_wait_ms,**self._limits)
                else:
                    self._queue = GenerationQueue(self._generate_one,
                        lambda: self._state in ("closed", "failed"), **self._limits)
            return self._queue

    def _prepare_submission(self, value: GenerationInput | GenerationJob, options: dict[str, Any]) -> tuple[GenerationInput, NativeOptions]:
        if isinstance(value, GenerationJob):
            if (options.get("seed", _DEFAULT_SEED) is not _DEFAULT_SEED
                    or options.get("recipe") is not None or options.get("execution_profile") is not None
                    or options.get("job_id") is not None or options.get("attempt_id", "1") != "1"):
                raise ValueError("GenerationJob owns its recipe/profile/identifiers; omit shared overrides")
            return value.input_data, self._validate(value.input_data, **value.options())
        return value, self._validate(value, **options)

    def submit(self, input_data: GenerationInput | GenerationJob, *,
               seed: int | _DefaultSeed = _DEFAULT_SEED, recipe: GenerationRecipe | None = None,
               execution_profile: ExecutionProfile | None = None, job_id: str | None = None,
               attempt_id: str = "1") -> GenerationHandle:
        """Nonblocking bounded admission. Consume or discard every returned handle."""
        input_data, options = self._prepare_submission(input_data, dict(seed=seed, recipe=recipe,
            execution_profile=execution_profile, job_id=job_id, attempt_id=attempt_id))
        queue = self._submission_queue()
        handle = queue.reserve()
        try:
            return queue.commit(handle, input_data, options)
        except BaseException:
            handle.discard()
            raise

    def generate(self, input_data: GenerationInput | GenerationJob, *,
                 seed: int | _DefaultSeed = _DEFAULT_SEED, recipe: GenerationRecipe | None = None,
                 execution_profile: ExecutionProfile | None = None, job_id: str | None = None,
                 attempt_id: str = "1") -> GenerationResult:
        """Synchronous owned result; seed defaults to17, or supply GenerationRecipe.

        Legacy direct calls retain busy rejection. Once submit/generate_many is
        used, synchronous calls share its FIFO and result credits.
        """
        input_data, options = self._prepare_submission(input_data, dict(seed=seed, recipe=recipe,
            execution_profile=execution_profile, job_id=job_id, attempt_id=attempt_id))
        with self._queue_lock:
            if self._closing or self._state in ("closed", "failed"):
                raise SessionClosed("GenerationEngine is terminal or closing")
            queue = self._queue
            if queue is None:
                if self._direct_active: raise SessionBusy("One GenerationEngine call is already in flight")
                self._direct_active = True
        if queue is not None:
            handle = queue.reserve()
            try:
                queue.commit(handle, input_data, options)
                return handle.result()
            except BaseException:
                handle.discard()
                raise
        try:
            return self._generate_one(input_data, **options)
        finally:
            with self._queue_lock: self._direct_active = False

    def generate_many(self, iterable: Iterable[Any], *, max_in_flight: int | None = None,
                      adapter: Callable[[Any], GenerationInput | GenerationJob] | None = None,
                      seed: int | _DefaultSeed = _DEFAULT_SEED, recipe: GenerationRecipe | None = None,
                      execution_profile: ExecutionProfile | None = None) -> GenerationStream:
        """Return a lazy ordered context-managed stream with reserved output credits.

        Use ``with engine.generate_many(...) as results`` when breaking early.
        CPU adapters execute on the consumer while the prior GPU call may run.
        """
        if adapter is not None and not callable(adapter): raise TypeError("adapter must be callable")
        window = min(self._limits["max_pending"], self._limits["max_results"]) if max_in_flight is None else positive(max_in_flight, "max_in_flight")
        if window > min(self._limits["max_pending"], self._limits["max_results"]):
            raise ValueError("max_in_flight cannot exceed input/result count credits")
        return GenerationStream(self, iterable, max_in_flight=window, adapter=adapter,
            options=dict(seed=seed, recipe=recipe, execution_profile=execution_profile))

    def _generate_one(self, input_data: GenerationInput, *, recipe: RecipeValues, execution_profile: ExecutionProfile, job_id: str | None, attempt_id: str) -> GenerationResult:
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("One GenerationEngine call is already in flight")
        try:
            if self._state in ("closed", "failed"):
                raise SessionClosed("GenerationEngine is terminal; create a new engine")
            if self._session is None:
                try:
                    self._open()
                except BaseException:
                    self._state = "failed"
                    raise
            try:
                result = cast("TerraMindSession", self._session).generate(
                    input_data, recipe=recipe, execution_profile=execution_profile, job_id=job_id, attempt_id=attempt_id)
            except SessionFailed:
                self._state = "failed"
                raise
            self._state = "ready"
            _record_parallel_selection(result,{**deepcopy(self._parallel_plan),'selected':'serial',
                'admitted':1,'actual':1,'group_actual':1,'neural_batch_size':1,
                'qualification':('existing_B1_execution_profile' if self.model==GENERATION_MODEL
                                 else 'model_size_not_gpu_qualified'),
                'fallback_reason':self._parallel_plan['fallback_reason'] or
                    ('direct call, singleton, unsupported recipe/profile, or capacity limit' if self._parallel!='off' else None)})
            return result
        finally:
            self._gate.release()

    def _generate_group(self, requests: GroupRequests) -> list[GenerationResult]:
        from .generation_parallel import PairAdmissionRejected
        if not self._gate.acquire(blocking=False):
            raise SessionBusy('One GenerationEngine call is already in flight')
        fallback = None
        try:
            if self._state in ('closed','failed'):
                raise SessionClosed('GenerationEngine is terminal; create a new engine')
            if self._session is None:
                try:self._open()
                except BaseException:
                    self._state='failed'
                    raise
            jobs=[GenerationJob(data,recipe=GenerationRecipe(**options['recipe']),
                                job_id=options['job_id'],attempt_id=options['attempt_id'])
                  for data,options in requests]
            try:
                results=cast("list[GenerationResult]", cast("LaneSession", self._session).generate_group(jobs))
            except PairAdmissionRejected as exc:
                if self._parallel_policy.fallback=='error':
                    self._state='ready'
                    raise ValueError('Parallel admission rejected before execution: '+str(exc)) from None
                fallback=str(exc)
            except SessionFailed:
                self._state='failed'
                raise
            else:
                self._state='ready'
                for result in results:
                    result.setdefault('parallel_selection',{**deepcopy(self._parallel_plan),
                        'selected':'graph_lanes','admitted':len(results),'actual':len(results),
                        'group_actual':len(results),'qualification':'executor_receipt_required',
                        'neural_batch_size':1})
                    _record_parallel_selection(result,result['parallel_selection'])
                return results
        finally:self._gate.release()
        # Only the dedicated pre-execution admission rejection reaches here.
        # A failed or partly executed pair is never retried as B1.
        results=[]
        for data,options in requests:
            result=self._generate_one(data,**options)
            result['parallel_selection'].update(selected='serial',fallback_reason=fallback,
                reason=fallback,before_group_execution=True,before_pair_execution=True,
                collected_group_size=len(requests))
            _record_parallel_selection(result,result['parallel_selection'])
            results.append(result)
        return results

    def binding(self) -> dict[str, object]:
        """Detached configured paths/profile; not a loaded-weight attestation.

        Reading this binding never hashes assets, opens a session or probes CUDA.
        Native loading still verifies the pinned files before model execution.
        """
        return {"schema": "aureon.generation_engine_binding.v1",
                "model_id": self.model,
                "model_file": str(self._model_file.resolve()),
                "tokenizer_file": str(self._tokenizer_file.resolve()),
                "execution_profile": self._execution_profile,
                "work": str(self.work.resolve()),
                "asset_identity": "configured_paths_only; not loaded-weight attestation"}

    def metadata(self) -> EngineMetadata:
        queue = self._queue.metadata() if self._queue is not None else None
        return {"state": "failed" if queue and queue["failed"] else self._state, "model_id": self.model,
                "execution_profile": self._execution_profile, "work": str(self.work),
                "session": self._session.metadata() if self._session is not None else None,
                "queue": queue,
                "parallel": self._parallel,
                "parallelism":deepcopy(self._parallel_plan),
                "closing": self._closing}

    def capabilities(self) -> dict[str, Any]:
        value = generation_capabilities(model_id=self.model)
        value["credits"] = dict(self._limits)
        if self.model == GENERATION_MODEL:
            from .._reference.integrity import MANIFEST_SHA256
            value["numerical_core"] = {"namespace": "aureon._reference.terramind",
                "manifest_sha256": MANIFEST_SHA256, "binding": "verified at resident construction",
                "adapter": "owned-memory delivery; independent B1 graph lanes when admitted",
                "resident_opened": self._session is not None,
                "qualification": "new artifact/route requires its own parity and performance evidence"}
        from .generation_parallel import qualification
        accepted=qualification(self.model)
        value['parallel']={**deepcopy(self._parallel_plan),'qualification_profiles':[accepted] if accepted else [],
            'allow_unqualified':self._parallel_policy.allow_unqualified,'fallback_policy':self._parallel_policy.fallback,
            'parallel_wait_ms':self._pair_wait_ms,'performance_claim':False,
            'scope':'planned capacity; device admission and actual execution are recorded per result'}
        if self._parallel != 'off':
            value['batching'].update(step_interleaving=True,gpu_ownership='single_owner_bounded_B1_graph_lanes')
            value['stages'][1].update(inner_stage_scheduling=True)
        return value

    def close(self, *, cancel_pending: bool = False, timeout: float | None = None) -> dict[str, Any]:
        """Stop admission and drain/cancel queued work; never preempt a model call.

        timeout bounds dispatcher draining. Existing native cleanup then runs
        synchronously; it is not interrupted or given a new timeout policy.
        """
        from .generation_queue import deadline
        deadline(timeout)
        if type(cancel_pending) is not bool: raise TypeError("cancel_pending must be bool")
        with self._queue_lock:
            if self._direct_active:
                raise SessionBusy("Cannot close GenerationEngine during an in-flight call")
            self._closing = True
            queue = self._queue
        if queue is not None:
            queue.shutdown(cancel_pending=cancel_pending, timeout=timeout)
            if queue.metadata()["failed"]: self._state = "failed"
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("Cannot close GenerationEngine during an in-flight call")
        try:
            if self._cleanup is None:
                if self._session is not None:
                    try:
                        self._cleanup = self._session.close()
                    except Exception:
                        self._state = "failed"
                        raise
                else:
                    self._cleanup = {"passed": True, "model_load_count": 0,
                                     "execution_failed": self._state == "failed"}
                if self._state != "failed":
                    self._state = "closed"
            return deepcopy(self._cleanup)
        finally:
            self._gate.release()

    def __enter__(self) -> GenerationEngine:
        if self._closing or self._state in ("closed", "failed"):
            raise SessionClosed("GenerationEngine is terminal or closing")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> Literal[False]:
        close_preserving_error(self.close, exc, context="Generation engine cleanup failed")
        return False
