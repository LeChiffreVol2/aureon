"""Best resident with integrated features and no forwarding run method.

The original run, owner thread, RNG, graph/reuse and cleanup are inherited
unchanged. Constructor policy and metadata are outside the numerical hot path.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Self
from .._reference.integrity import CoreFingerprint, verify_core
from ..value_types import Pathish

BestProfile = Literal["best_fp32", "reference_fp32"]

if TYPE_CHECKING:
    class BestSessionBusy(RuntimeError): pass
    class _NativeSession:
        """Typed boundary for the verified, unmodified original implementation."""
        work: Path
        def __init__(self, config: Pathish, work: Pathish, *, execute_model: bool,
                     reuse: str, denoiser: str) -> None: raise NotImplementedError
        def run(self, input_manifest: Pathish | None = None, *, job_id: str, attempt_id: str,
                recipe: Mapping[str, object] | None = None,
                capture_logits: bool = False) -> dict[str, object]: raise NotImplementedError
        def metadata(self) -> dict[str, object]: raise NotImplementedError
        def close(self) -> dict[str, object]: raise NotImplementedError
        def __enter__(self) -> Self: raise NotImplementedError
        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                     tb: TracebackType | None) -> Literal[False]: raise NotImplementedError
else:
    from .._reference.terramind.session import TerraMindSession as _NativeSession
    from .._reference.terramind.session import SessionBusy as BestSessionBusy


class BestTerraMindSession(_NativeSession):
    """The Best runtime itself; scene, queue and service features borrow this owner.

    No child interpreter, transport, alternate installed wheel or proxy object.
    run/close/context-manager methods are the original function objects.
    """
    def __init__(self, config: Pathish, work: Pathish, *, execute_model: bool = False,
                 profile: BestProfile = "best_fp32") -> None:
        if execute_model is not True:
            raise ValueError("Pass execute_model=True explicitly")
        if profile not in ("best_fp32", "reference_fp32"):
            raise ValueError("profile must be best_fp32 or reference_fp32")
        self._fingerprint = verify_core()
        self._profile = profile
        super().__init__(config, work, execute_model=True,
                         reuse="auto" if profile == "best_fp32" else "off",
                         denoiser="graph" if profile == "best_fp32" else "native")

    @property
    def core_fingerprint(self) -> CoreFingerprint:
        return self._fingerprint

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "profile": self._profile,
                "core": self._fingerprint.metadata(), "direct_best_run": True}
