"""Opt-in CPU fixture conformance for contributor-owned adapters, not benchmarks."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Protocol

from .lifecycle import close_preserving_error

if TYPE_CHECKING:
    from torch import Tensor


class ServingAdapter(Protocol):
    def predict(self, inputs: Mapping[str, Tensor], *, seed: int = 17,
                cancel: threading.Event | None = None) -> dict[str, Tensor]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class AdapterConformance:
    requests: int
    exact_fixture_parity: bool
    input_preserved: bool
    output_ownership: bool
    closed: bool
    scope: str = 'Caller-supplied CPU fixture; no pretrained, GPU, performance or scientific qualification'


def check_adapter_conformance(factory: Callable[[], ServingAdapter], inputs: Mapping[str, Tensor],
                              expected: Mapping[str, Tensor], *, seed: int = 17) -> AdapterConformance:
    """Construct/close one owned adapter and execute the same CPU fixture twice.

    Mutates the first returned output to test isolation from subsequent requests.
    Factory is trusted, must create a fresh disposable owner and must not download
    assets or initialize CUDA. Production caller models must not be supplied here.
    """
    import torch
    if not inputs or not expected:
        raise ValueError('Provide named input and expected output fixtures')
    for value in (*inputs.values(), *expected.values()):
        if value.device.type != 'cpu' or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Conformance fixtures must be finite CPU FP32 tensors')
    owned = {k: v.detach().clone() for k, v in inputs.items()}
    baseline = {k: v.clone() for k, v in owned.items()}
    adapter = factory()
    error: BaseException | None = None
    try:
        first = adapter.predict(owned, seed=seed)
        def compare(values: Mapping[str, Tensor]) -> None:
            if set(values) != set(expected):
                raise AssertionError('Output names differ from the fixture')
            for name, value in values.items():
                if (value.device.type != 'cpu' or value.dtype != torch.float32
                        or value.requires_grad or not torch.equal(value, expected[name])):
                    raise AssertionError('Output differs from the exact CPU fixture')
                if any(value.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in owned.values()):
                    raise AssertionError('Output aliases caller input storage')
        compare(first)
        for value in first.values():
            value.add_(1)
        second = adapter.predict(owned, seed=seed)
        compare(second)
        if any(v.untyped_storage().data_ptr() == old.untyped_storage().data_ptr()
               for v in second.values() for old in first.values()):
            raise AssertionError('Successive results share output storage')
        if any(not torch.equal(v, baseline[k]) for k, v in owned.items()):
            raise AssertionError('Adapter modified caller inputs')
    except BaseException as exc:
        error = exc
        raise
    finally:
        close_preserving_error(adapter.close, error, context='adapter conformance cleanup')
    return AdapterConformance(2, True, True, True, True)
