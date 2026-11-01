"""Source-only explicit two-lane candidate, separate from default GenerationEngine."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Callable
from ..terramind_native._types import Metadata, InputSnapshot
if TYPE_CHECKING:
    from ..terramind_native.denoiser_lanes import LaneRuntime
from .batch_generation import _PairSession
from ..terramind_native.denoiser_lanes import LaneRuntime, lane_eligibility, run_lane_pair


class ExperimentalLaneSession(_PairSession):
    def __init__(self,*args: Any,allow_unqualified_parallel: bool=False,**kwargs: Any) -> None:
        if allow_unqualified_parallel is not True:
            raise ValueError('Pass allow_unqualified_parallel=True for this source candidate')
        if 'allow_unqualified_batch' in kwargs:
            raise ValueError('Use allow_unqualified_parallel, not the fused B2 flag')
        super().__init__(*args,allow_unqualified_batch=True,**kwargs)

    def _runtime_factory(self) -> LaneRuntime:return LaneRuntime()

    def _pair_admission(self,recipes: list[Metadata]) -> Metadata:
        return lane_eligibility(cast("LaneRuntime", self._runtime),recipes,max_workspace_bytes=self._pair_workspace,
                                min_free_cuda_bytes=self._pair_free)

    def _execute_pair(self,snapshots: list[InputSnapshot],recipes: list[Metadata],ids: list[tuple[str, str]],cancel: Callable[[], None],admission: Metadata) -> list[Metadata]:
        return run_lane_pair(cast("LaneRuntime", self._runtime),snapshots,recipes,ids,reuse=self._reuse,cancel=cancel,
            session_identity=self._identity,max_workspace_bytes=self._pair_workspace,
            min_free_cuda_bytes=self._pair_free,admission=admission)

    def metadata(self) -> Metadata:
        value=super().metadata()
        value['batch_candidate'].update(max_batch_size=1,logical_pair_size=2,graph=True,
                                        stage='two_independent_B1_graph_lanes',qualified=False)
        return value
