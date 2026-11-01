"""Compatibility entry point for the generic bounded group collector."""
from __future__ import annotations
from collections.abc import Callable
from typing import Unpack
from .contracts import ExecuteGeneration, ExecuteGroup, QueueLimits
from .generation_group_queue import GenerationGroupQueue, pair_compatible as pair_compatible


class GenerationPairQueue(GenerationGroupQueue):
    def __init__(self,execute: ExecuteGeneration,terminal: Callable[[], bool],*,execute_pair: ExecuteGroup,max_wait_ms: float=5.0,**limits: Unpack[QueueLimits]) -> None:
        super().__init__(execute,terminal,execute_group=execute_pair,max_group_size=2,
                         max_wait_ms=max_wait_ms,**limits)

    @property
    def _execute_pair(self) -> ExecuteGroup:return self._execute_group

    @_execute_pair.setter
    def _execute_pair(self,value: ExecuteGroup) -> None:self._execute_group=value
