"""Transport-independent capability and relative-capacity placement hooks.

Declarations are admission inputs, not measured GPU speed. The caller must fence
worker generations and reserve capacity atomically after selecting an offer.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class WorkerCapabilities:
    model_ids: tuple[str,...] = ()
    tasks: tuple[str,...] = ('generation','features')
    precisions: tuple[str,...] = ('float32',)
    tags: tuple[str,...] = ()

    def __post_init__(self) -> None:
        for values in (self.model_ids,self.tasks,self.precisions,self.tags):
            if (not isinstance(values,tuple) or len(values)>128 or len(set(values))!=len(values)
                    or any(not isinstance(v,str) or not 1<=len(v)<=256 for v in values)):
                raise ValueError('Capabilities require bounded immutable unique labels')

    def admits(self,model_id: str,task: str,precision: str,required: tuple[str,...]=()) -> bool:
        return ((not self.model_ids or model_id in self.model_ids) and task in self.tasks
                and precision in self.precisions and set(required)<=set(self.tags))


@dataclass(frozen=True)
class WorkerOffer:
    worker_id: str
    capabilities: WorkerCapabilities
    available_bytes: int
    relative_capacity: float = 1.0
    queued_units: int = 0
    warm: bool = False
    ready: bool = True
    draining: bool = False
    failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id,str) or not 1<=len(self.worker_id)<=256:raise ValueError('Invalid worker ID')
        if any(type(v) is not int or v<0 for v in (self.available_bytes,self.queued_units)):raise ValueError('Invalid capacity')
        if type(self.relative_capacity) not in (int,float) or not math.isfinite(self.relative_capacity) or self.relative_capacity<=0:
            raise ValueError('relative_capacity must be finite and positive')
        if not isinstance(self.capabilities,WorkerCapabilities):raise TypeError('Use WorkerCapabilities')


def rank_worker_offers(offers: Sequence[WorkerOffer], *,model_id: str,task: str,precision: str,
                       required_bytes: int,required: tuple[str,...]=(),affinity: str | None=None) -> list[WorkerOffer]:
    """Return eligible warm-first offers ranked by declared load/capacity and affinity.

No reservation, dispatch, network I/O, heartbeat or retry is performed here. A
remote coordinator can reuse this pure policy without exposing local pickle IPC.
"""
    if type(required_bytes) is not int or required_bytes<0:raise ValueError('required_bytes must be nonnegative')
    if len({o.worker_id for o in offers})!=len(offers):raise ValueError('Duplicate worker offers')
    eligible=[o for o in offers if o.ready and not o.draining and not o.failed
              and o.available_bytes>=required_bytes and o.capabilities.admits(model_id,task,precision,required)]
    def score(o: WorkerOffer) -> tuple[bool,float,str]:
        load=(o.queued_units+1)/o.relative_capacity
        if o.worker_id==affinity:load*=0.9  # Bounded soft affinity; never overrides eligibility.
        return not o.warm,load,o.worker_id
    return sorted(eligible,key=score)
