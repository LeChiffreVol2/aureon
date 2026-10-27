"""Optional one-pair mmap reuse inside the existing native vLLM worker.

Math, CUDA graph, pinned staging, descriptor/ACK and per-request locking stay
the same as the prior worker route. Only mappings survive a completed request.
"""
import atexit
import hashlib
from pathlib import Path

from .resident_protocol import DESCRIPTOR_SHAPE, decode_descriptor
from .resident_slots import ResidentSlotCache
from .shm_engine_worker import ShmEngineForward


class ResidentEngineForward(ShmEngineForward):
    descriptor_shape = DESCRIPTOR_SHAPE
    decode_descriptor = staticmethod(decode_descriptor)

    def __init__(self, model, folder, **kwargs):
        self.slot_cache = ResidentSlotCache()
        self._reuse_evidence_written = False
        super().__init__(model, folder, slots_factory=self.slot_cache.acquire, **kwargs)
        self.source_sha256.update({name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ('resident_worker.py', 'resident_protocol.py', 'resident_slots.py')})

    def metadata(self):
        value = super().metadata()
        value.update(schema='aureon.resident_engine_worker.v1',
            slot_cache=self.slot_cache.metadata(),
            completion='pipeline synchronization and output validation/write; release flock; then ACK; retain bounded mappings',
            mapping_retention='At most one pair; revalidate FD and named inode/owner/mode/size and lock every request; evict on pair change or close')
        return value

    def _ack(self, descriptor, statuscode):
        if statuscode == 0 and not self._reuse_evidence_written and self.slot_cache.metadata()['reuses'] > 0:
            self._event('first-reuse', request=descriptor['request'])
            self._reuse_evidence_written = True
        return super()._ack(descriptor, statuscode)

    def close(self):
        self.slot_cache.close()


def install_resident_forward(model, folder, *, rank=0, worker_record, verified_bundle):
    if hasattr(model, '_aureon_resident_forward'):
        raise ValueError('Resident worker route already installed')
    destination = Path(folder) / 'resident-worker'
    destination.mkdir(exist_ok=False)
    controller = ResidentEngineForward(model, destination, rank=rank,
        worker_record=worker_record, verified_bundle=verified_bundle)
    controller._event('installed')
    model.forward = controller
    model._aureon_resident_forward = controller
    atexit.register(controller.close)
    return controller
