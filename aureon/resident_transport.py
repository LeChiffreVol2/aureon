"""Explicit same-Linux client and gateway for one resident worker mapping pair."""
import os

from .shm_engine_transport import SharedMemoryEnginePoolingPredictor
from .shm_engine_gateway import SharedMemoryEnginePoolingMiddleware
from .resident_protocol import PATH, encode_descriptor


class ResidentSharedMemoryPredictor(SharedMemoryEnginePoolingPredictor):
    transport = 'shared_memory_resident'
    request_path = PATH

    def metadata(self):
        value = super().metadata()
        value.update(schema='aureon.resident_shm_client.v1',
                     worker_mapping_capacity=1, worker_mapping_reuse=True,
                     completion_scope='GPU/output completion and flock release; worker mapping remains cached until eviction or shutdown')
        return value


class ResidentSharedMemoryMiddleware(SharedMemoryEnginePoolingMiddleware):
    request_path = PATH
    encode_control = staticmethod(encode_descriptor)

    async def __call__(self, scope, receive, send):
        if (scope.get('type') == 'http' and scope.get('path') == self.request_path
                and os.environ.get('AUREON_RESIDENT_SLOTS') != '1'):
            return await self._respond(send, 403, b'{"error":"Resident worker mappings are not enabled."}',
                                       [(b'content-type', b'application/json')])
        return await super().__call__(scope, receive, send)
