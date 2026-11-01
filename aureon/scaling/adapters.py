"""One resident worker lifecycle for different model-specific inference contracts.

Trusted loaders/deliverers define their explicit EO route, grid, units, masks and
artifact schema. They never deserialize Python objects received over the network.
No general any-to-any route is inferred from these reusable execution hooks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.contracts import GenerationResult
    from ..api.eo import EORequest
    from ..api.generation_engine import GenerationEngine
    from ..api.memory_input import GenerationInput
    from ..copgen_native.contracts import CopgenInput
    from ..copgen_native.engine import CopgenEngine

from .worker import WorkContext

Input = TypeVar("Input")
Output = TypeVar("Output")


class ResidentAdapter(Generic[Input, Output]):
    """Borrow existing resident model state; leave model creation/close to its owner.

    The three hooks run on the worker owner thread. CPU stages can instead use
    Pipeline with explicit thread-safe stages; native CUDA ownership is not moved
    implicitly. Cancellation is checked before and after each native operation.
    """
    def __init__(self, *, prepare: Callable[[WorkContext], Input],
                 infer: Callable[[Input, WorkContext], Output],
                 deliver: Callable[[Output, WorkContext], Mapping[str, object]]) -> None:
        self._prepare, self._infer, self._deliver = prepare, infer, deliver

    def __call__(self, context: WorkContext) -> Mapping[str, object]:
        context.check()
        input_data = self._prepare(context)
        context.check()
        output = self._infer(input_data, context)
        context.check()
        result = self._deliver(output, context)
        context.check()
        return result


def terramind_worker_adapter(engine: GenerationEngine, *,
        prepare: Callable[[WorkContext], tuple[EORequest, GenerationInput]],
        deliver: Callable[[GenerationResult, WorkContext], Mapping[str, object]],
        ) -> ResidentAdapter[tuple[EORequest, GenerationInput], GenerationResult]:
    """Use the existing EO route checks and resident engine on the worker thread."""
    def infer(value: tuple[EORequest, GenerationInput], context: WorkContext) -> GenerationResult:
        from ..api.eo import generate_eo
        request, data = value
        if request.sampling.seed != context.job.seed:
            raise ValueError("Worker job seed differs from the EO sampling contract")
        return generate_eo(engine, request, data, job_id=context.job.job_id, attempt_id=str(context.attempt))
    return ResidentAdapter(prepare=prepare, infer=infer, deliver=deliver)


def copgen_worker_adapter(engine: CopgenEngine, *,
        prepare: Callable[[WorkContext], tuple[EORequest, CopgenInput]],
        deliver: Callable[[Mapping[str, object], WorkContext], Mapping[str, object]],
        ) -> ResidentAdapter[tuple[EORequest, CopgenInput], Mapping[str, object]]:
    """Borrow COP-GEN's explicit native route; preserve its own sampling/RNG policy."""
    def infer(value: tuple[EORequest, CopgenInput], context: WorkContext) -> Mapping[str, object]:
        request, data = value
        if request.sampling.seed != context.job.seed:
            raise ValueError("Worker job seed differs from the EO sampling contract")
        return engine.generate(request, data)
    return ResidentAdapter(prepare=prepare, infer=infer, deliver=deliver)
