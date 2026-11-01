"""Owner-thread, single-signature tensor graph lifecycle, independent of a decoder.

A binding supplies input admission/staging and the exact forward. There is one
capture and one bounded buffer set, never a signature cache or generated-output
cache. Injected backends are a CPU test seam, not CUDA qualification.
"""
from contextlib import contextmanager
from time import perf_counter
import threading
import torch


class CudaGraphBackend:
    def capture(self, forward, buffers, stats):
        device = next(iter(buffers.values())).device
        graph = None
        with torch.cuda.device(device):
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            try:
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    stats['warmup_stage_forwards'] += 1
                    warm = forward(buffers)
                    del warm
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                stats['capture_stage_forwards'] += 1
                with torch.cuda.graph(graph, stream=stream):
                    output = forward(buffers)
                torch.cuda.synchronize(device)
                if (not torch.equal(torch.get_rng_state(), cpu_rng)
                        or not torch.equal(torch.cuda.get_rng_state(device), cuda_rng)):
                    raise RuntimeError('Graph setup consumed RNG; stochastic capture is unsupported')
                return graph, output
            except BaseException:
                if graph is not None:
                    graph.reset()
                raise
            finally:
                # Request RNG capsules also restore their enclosing state. Setup
                # preserves its own boundary even on a failed warmup/capture.
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)


class FixedTensorGraph:
    def __init__(self, binding, *, backend=None):
        self.binding = binding
        self.backend = backend if backend is not None else CudaGraphBackend()
        self.graph = self.output = None
        self.buffers = {}
        self.signature = None
        self.owner_thread = threading.get_ident()
        self.started = self.failed = self.closed = self.active = False
        self.total = {'warmup_stage_forwards': 0, 'capture_stage_forwards': 0,
                      'replay_stage_forwards': 0, 'setup_seconds': 0.,
                      'extra_full_generations': 0, 'released': False, 'failed': False,
                      'owner_thread_id': self.owner_thread}

    def owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Graph owner thread differs')
        if self.closed:
            raise RuntimeError('Graph is closed')
        if self.failed:
            raise RuntimeError('Failed graph cannot be reused')

    @staticmethod
    def _signature(inputs):
        return tuple((name, tuple(value.shape), value.dtype, value.device)
                     for name, value in inputs.items())

    def call(self, inputs):
        self.owner()
        try:
            self.binding.validate(inputs)
            signature = self._signature(inputs)
            if self.signature is not None and signature != self.signature:
                raise ValueError('Graph input signature changed; recapture is unsupported')
            if not self.started:
                self.started = True
                begin = perf_counter()
                try:
                    self.signature = signature
                    self.buffers = self.binding.stage(inputs, self.total)
                    if set(self.buffers) != set(inputs):
                        raise ValueError('Binding changed staged input names')
                    self.graph, self.output = self.backend.capture(self.binding.forward, self.buffers, self.total)
                    if not isinstance(self.output, torch.Tensor):
                        raise ValueError('Graph forward must return one tensor')
                    self.total['captured_shapes'] = {name: list(t.shape) for name, t in self.buffers.items()}
                    self.total['input_buffer_bytes'] = sum(t.numel()*t.element_size() for t in self.buffers.values())
                finally:
                    self.total['setup_seconds'] = perf_counter() - begin
            # Refresh every input even on the first replay after setup. Outputs
            # are copied before returning, so later replays cannot mutate them.
            for name, value in inputs.items():
                self.buffers[name].copy_(value)
            self.graph.replay()
            self.total['replay_stage_forwards'] += 1
            return self.output.clone()
        except BaseException:
            self.failed = True
            self.total['failed'] = True
            raise

    @contextmanager
    def request(self):
        self.owner()
        if self.active:
            raise RuntimeError('Graph request is already active')
        self.active = True
        try:
            yield
        except BaseException:
            self.failed = True
            self.total['failed'] = True
            raise
        finally:
            self.active = False

    def close(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Graph owner thread differs')
        if self.closed:
            return dict(self.total)
        if self.active:
            raise RuntimeError('Cannot close an active graph request')
        try:
            if self.graph is not None:
                self.graph.reset()
        finally:
            try:
                self.binding.close()
            finally:
                self.buffers.clear()
                self.graph = self.output = self.binding = self.backend = None
                self.signature = None
                self.total['released'] = True
                self.closed = True
        return dict(self.total)
