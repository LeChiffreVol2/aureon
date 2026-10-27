"""Explicit same-buffer adapters for the matched binary HTTP service."""
import time
from .finite_scan import FiniteScanner
from .shm_files import SharedSlots


class QueuedBackend:
    def __init__(self, kind, bundle, folder):
        self.kind, self.bundle = kind, bundle
        self.scanner = FiniteScanner()
        self.slots = SharedSlots.create(8)
        self.engine = self.graph = self.cache = None
        self.calls = 0
        try:
            if kind == 'vllm':
                from .inproc_engine import InprocessPoolingEngine
                self.engine = InprocessPoolingEngine(bundle, folder)
            elif kind == 'pytorch':
                import torch
                from .reference import PyTorchPredictor
                from .finite_profile import FiniteRegisteredGraph
                from .resident_slots import ResidentSlotCache
                reference = PyTorchPredictor(bundle['bundle_dir'], device='cuda:0', dtype='float32', validate_on_cpu=True)
                torch.set_float32_matmul_precision('highest')
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cudnn.benchmark = False
                graph_dir = folder/'direct-graph'
                graph_dir.mkdir()
                self.graph = FiniteRegisteredGraph.from_reference(reference, graph_dir)
                self.cache = ResidentSlotCache(bundle['model_id'])
            else:
                raise ValueError('Explicit pytorch or vllm backend required')
        except BaseException:
            self.close()
            raise

    def predict_into(self, values, output, request_id):
        start = time.perf_counter()
        view = self.slots._array('input')
        try:
            if not self.scanner.copy(view, values):
                raise ValueError('Nonfinite input')
        finally:
            del view
        copied = time.perf_counter()
        spec = self.slots.descriptor(self.bundle['model_id'], 8, request_id)
        if self.engine:
            self.engine.execute(spec)
        else:
            with self.cache.acquire(spec) as lease:
                inp, out = lease._array('input'), lease._array('output')
                try:
                    self.graph.predict_into(inp, out)
                finally:
                    del inp, out
        executed = time.perf_counter()
        view = self.slots._array('output')
        try:
            if not self.scanner.copy(output, view):
                raise ValueError('Nonfinite output')
        finally:
            del view
        finished = time.perf_counter()
        self.calls += 1
        return {'input_copy_finite_ms': (copied-start)*1000,
                'backend_ms': (executed-copied)*1000,
                'output_copy_finite_ms': (finished-executed)*1000}

    def metadata(self):
        return {'kind': self.kind, 'calls': self.calls, 'slots': self.slots.identities,
                'scanner': self.scanner.metadata(),
                'backend': self.engine.metadata() if self.engine else self.graph.metadata()}

    def close(self):
        if self.engine:
            self.engine.close()
        if self.graph:
            self.graph.close()
        if self.cache:
            self.cache.close()
        self.slots.close(unlink=True)
