"""Opt-in registered-buffer controller for one fixed same-UID slot pair."""
import atexit
import hashlib
from pathlib import Path

from .resident_worker import ResidentEngineForward
from .registered_graph import RegisteredGroupGraph


class RegisteredEngineForward(ResidentEngineForward):
    def _acquire_registered(self, descriptor):
        identity={key:descriptor[key] for key in ('model','client','capacity','dtype','slots')}
        if self._registered_identity is not None and self._registered_identity!=identity:
            raise ValueError('Registered profile binds one client slot pair; restart to change it')
        lease=self.slot_cache.acquire(descriptor)
        self._registered_identity=identity
        return lease

    def metadata(self):
        value=super().metadata()
        value.update(registered_host_buffers=True,worker_cpu_staging_copies=0,
            slot_pair_policy='one fixed pair for service lifetime; reject pair changes before registration or eviction')
        if hasattr(self, 'finite_scan'): value['finite_scan'] = dict(self.finite_scan)
        if hasattr(self, 'control_ack'): value['control_ack'] = dict(self.control_ack)
        if hasattr(self, 'cpu_control'): value['cpu_control'] = dict(self.cpu_control)
        return value

    def close(self):
        if self._pipeline is not None: self._pipeline.close()
        super().close()


def install_registered_worker(worker):
    prior=worker._aureon_resident_forward
    if type(prior) is not ResidentEngineForward or prior.completed_calls or prior._pipeline is not None:
        raise ValueError('Install registered buffers before any resident request')
    controller=object.__new__(RegisteredEngineForward)
    controller.__dict__.update(prior.__dict__)
    controller._registered_identity=None
    controller._slots_factory=controller._acquire_registered
    controller._pipeline_factory=RegisteredGroupGraph.from_verified_worker
    controller.source_sha256.update({name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                    for name in ('registered_worker.py','registered_graph.py','registered_precision_worker.py')})
    # Preserve the base installed event; this separate opt-in receipt records
    # the final controller before requests and does not replace old evidence.
    controller._event('registered-installed')
    prior.model.forward=controller
    prior.model._aureon_resident_forward=controller
    worker._aureon_resident_forward=controller
    atexit.register(controller.close)
    return controller
