"""Bounded FIFO group collection around separate single/pair executors.

This module only dispatches work. A pair dispatch is not proof of neural batch
execution or parallel GPU work; those observations belong to runtime receipts.
The existing GenerationQueue remains the default serial path unchanged.
"""
from __future__ import annotations
from collections.abc import Callable
from typing import cast, Unpack
from .contracts import NativeOptions, ExecuteGeneration, ExecuteGroup, QueueLimits, QueueMetadata, GroupRequests, GenerationResult
from .generation_queue import GenerationHandle
from .memory_input import GenerationInput
from ..terramind_native._types import Array
import math
from time import monotonic

from .generation_queue import GenerationQueue, INPUT_BYTES, OUTPUT_BYTES
from ..terramind_native.session import SessionBusy, SessionFailed


def pair_compatible(left: NativeOptions, right: NativeOptions) -> bool:
    """Only the documented recommended 8-token/10-step recipe; seeds stay local."""
    if left.get('execution_profile') != 'recommended' or right.get('execution_profile') != 'recommended':
        return False
    one, two = left.get('recipe', {}), right.get('recipe', {})
    if one.get('token_steps') != 8 or one.get('diffusion_steps') != 10:
        return False
    if {k:v for k,v in one.items() if k != 'seed'} != {k:v for k,v in two.items() if k != 'seed'}:
        return False
    if (left.get('job_id') is not None and left.get('job_id') == right.get('job_id')
            and left.get('attempt_id') == right.get('attempt_id')):
        return False
    return True


class GenerationGroupQueue(GenerationQueue):
    """Collect a bounded compatible FIFO prefix; timeout/singletons use the B1 executor.

Reuse individual GenerationHandles and input/result credits. No unbounded lookahead
or reordering; cancellation before dispatch removes only that handle. Once running,
discard drops only that request's eventual result and never preempts the other row.
"""
    def __init__(self, execute: ExecuteGeneration, terminal: Callable[[], bool], *, execute_group: ExecuteGroup, max_group_size: int, max_wait_ms: float=5.0,
                 compatible: Callable[[NativeOptions,NativeOptions],bool]=pair_compatible, **limits: Unpack[QueueLimits]) -> None:
        if not callable(execute_group): raise TypeError('execute_group must be callable')
        if type(max_wait_ms) not in (int,float) or not math.isfinite(max_wait_ms) or not 0 <= max_wait_ms <= 1000:
            raise ValueError('max_wait_ms must be finite in [0, 1000]')
        if type(max_group_size) is not int or max_group_size<1:raise ValueError('max_group_size must be positive')
        super().__init__(execute, terminal, **limits)
        self.requested_group_size=max_group_size
        self.max_group_size=min(max_group_size,self.max_pending,self.max_results,
            self.max_input_bytes//INPUT_BYTES,self.max_output_bytes//OUTPUT_BYTES)
        self._execute_group = execute_group
        self._compatible = compatible
        self._wait_seconds = max_wait_ms/1000
        self._can_collect=self.max_group_size>1
        self._collection_counts = {'group_dispatches':0,'pair_dispatches':0,'single_dispatches':0,
                                   'timeout_singletons':0,'incompatible_neighbors':0}

    def _take(self) -> list[GenerationHandle]:
        """Bounded compatible FIFO prefix; cancellation restarts the decision."""
        while True:
            while not self._queue and self._accepting:self._condition.wait()
            if not self._queue:return []
            first=self._queue[0]
            eligible=self._can_collect and self._compatible(cast("NativeOptions", first._options),{**cast("NativeOptions", first._options),'job_id':None})
            count=1;incompatible=False;seen: set[tuple[str | None, str | None]]=set()
            if cast("NativeOptions", first._options).get('job_id') is not None:
                seen.add((cast("NativeOptions", first._options)['job_id'],cast("NativeOptions", first._options)['attempt_id']))
            if eligible:
                from itertools import islice
                for handle in islice(self._queue,1,self.max_group_size):
                    options=cast("NativeOptions", handle._options)
                    identity=(options.get('job_id'),options.get('attempt_id'))
                    if not self._compatible(cast("NativeOptions", first._options),options) or (identity[0] is not None and identity in seen):
                        incompatible=True;break
                    if identity[0] is not None:seen.add(identity)
                    count+=1
            remaining=first._submitted+self._wait_seconds-monotonic()
            if not eligible or incompatible or count==self.max_group_size or not self._accepting or remaining<=0:
                if incompatible:self._collection_counts['incompatible_neighbors']+=1
                if eligible and count==1 and remaining<=0:self._collection_counts['timeout_singletons']+=1
                chosen=[self._queue.popleft() for _ in range(count)]
                break
            self._condition.wait(remaining)
        for handle in chosen:handle._state='running'
        self._active=tuple(chosen)
        key='group_dispatches' if len(chosen)>1 else 'single_dispatches'
        self._collection_counts[key]+=1
        if len(chosen)==2:self._collection_counts['pair_dispatches']+=1  # Compatibility statistic only.
        return chosen

    def _run(self) -> None:
        handles: list[GenerationHandle] | None
        requests: GroupRequests | None
        data: GenerationInput | None
        options: NativeOptions | None
        result: GenerationResult | None
        while True:
            with self._condition:
                handles = self._take()
                if not handles: return
                requests = [(cast("GenerationInput", h._input),cast("NativeOptions", h._options)) for h in handles]
                for handle in handles: handle._input = handle._options = None
            results = error = None
            fatal = False
            try:
                if len(handles)>1:
                    results = self._execute_group(requests)
                    if not isinstance(results,(tuple,list)) or len(results)!=len(handles):
                        raise SessionFailed('Group executor must return one ordered result per request')
                else:
                    data, options = requests[0]
                    results = [self._execute(data,**options)]
                    data = options = None
                for result in results:
                    size = sum(cast("dict[str, Array]", result)[key].nbytes for key in ('array','valid_mask'))
                    if size > OUTPUT_BYTES:
                        raise SessionFailed('Runtime result exceeds its individual reserved payload credit')
                result = None
            except BaseException as exc:
                fatal = self._terminal() or type(exc) not in (ValueError,TypeError,FileExistsError,SessionBusy)
                kind = type(exc) if not fatal else SessionFailed
                error = (kind,str(exc)); results = None
            finally:
                requests = data = options = result = None
            with self._condition:
                self._active = None
                self._pending -= len(handles)
                for i, handle in enumerate(handles):
                    if handle._discard:
                        handle._state = 'discarded'; self._release_result(handle)
                    else:
                        handle._state = 'failed' if error else 'completed'
                        handle._result = None if error else cast("list[GenerationResult]", results)[i]
                        handle._error = error
                handles = results = error = None
                if fatal:
                    self._failed = True; self._accepting = False
                    for waiting in list(self._handles):
                        if waiting._state in ('reserved','queued'):
                            if waiting._state == 'queued': self._queue.remove(waiting)
                            waiting._input = waiting._options = None
                            waiting._state = 'failed'
                            waiting._error = (SessionFailed,'Resident execution failed; queued request was not executed')
                            self._pending -= 1
                self._condition.notify_all()

    def metadata(self) -> QueueMetadata:
        value = super().metadata()
        with self._condition:
            value['collection'] = {**self._collection_counts,'max_group_size':self.max_group_size,'requested_group_size':self.requested_group_size,
                'max_pair_size':self.max_group_size,
                'max_wait_ms':self._wait_seconds*1000, 'enabled_by_credits':self._can_collect,
                'running_requests':len(cast("tuple[GenerationHandle, ...]", self._active)) if self._active else 0,
                'scope':'dispatch groups, not attestation of neural batch/parallel execution'}
        return value
