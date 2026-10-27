"""Explicit, unqualified precision experiments; never selected by a default API.

Import is runtime-free. Every worker owns one immutable policy and one GPU thread.
Stored weights, normalized inputs, CPU outputs and blending remain FP32.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np

PROFILES = ('fp32_reference', 'tf32', 'amp_fp16', 'amp_bf16')
ROUTES = ('direct_eager', 'aureon_optimized')


def precision_contract(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown explicit experimental precision profile')
    return {'schema': 'aureon.experimental_precision.v1', 'profile': profile,
            'weight_dtype': 'float32', 'input_dtype': 'float32', 'output_dtype': 'float32',
            'autocast_dtype': {'amp_fp16': 'float16', 'amp_bf16': 'bfloat16'}.get(profile),
            'autocast_cache_enabled': True, 'autocast_scope': 'one real group, cache cleared on exit',
            'matmul_tf32': profile == 'tf32', 'convolution_tf32': profile == 'tf32',
            'neural_batch': 1, 'padding': False, 'experimental': True}


def validate_contract(value):
    if not isinstance(value, dict) or value != precision_contract(value.get('profile')):
        raise ValueError('Precision contract must match the complete versioned profile')
    return dict(value)


@dataclass(frozen=True)
class PrecisionPolicy:
    profile: str

    def __post_init__(self):
        precision_contract(self.profile)

    def apply(self, torch):
        if self.profile == 'tf32' and os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
            raise ValueError('TF32 profile is disabled by NVIDIA_TF32_OVERRIDE=0')
        if self.profile == 'amp_bf16' and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('Native BF16 is unavailable; no emulation or dtype fallback')
        # Use only the legacy family, consistent with the accepted runtime.
        torch.set_float32_matmul_precision('high' if self.profile == 'tf32' else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = self.profile == 'tf32'
        torch.backends.cudnn.allow_tf32 = self.profile == 'tf32'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        if self.profile.startswith('amp_'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        return self.readback(torch)

    def readback(self, torch):
        from .cuda_graph import _precision
        return {**_precision(torch),
            'fp16_reduced_precision_reduction': bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
            'bf16_reduced_precision_reduction': bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            'math_sdp_reduced_precision_reduction': bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'),
            'default_dtype': str(torch.get_default_dtype())}

    @contextmanager
    def scope(self, torch):
        dtype = {'amp_fp16': torch.float16, 'amp_bf16': torch.bfloat16}.get(self.profile)
        with torch.autocast('cuda', enabled=dtype is not None, dtype=dtype, cache_enabled=True):
            yield


class PrecisionExecutor:
    """One strictly loaded model; independent eager and fixed-eight graph routes.

    Construction attests actual resident tensor bytes. Every call returns owned
    FP32 output after completion. Capture/parity failures never fall back.
    """
    def __init__(self, reference, profile, route):
        from .reference import PyTorchPredictor
        from .scene_api.identity import TensorSeal
        from .vllm_bundle import _tensor_record
        if not isinstance(reference, PyTorchPredictor) or reference.dtype != 'float32':
            raise ValueError('Supply the strict FP32-loaded reference')
        if route not in ROUTES or reference._device.type != 'cuda' or not reference._as_dict:
            raise ValueError('An explicit supported CUDA S2L2A route is required')
        self.reference, self.torch, self.model = reference, reference._torch, reference.model
        self.device = next(self.model.parameters()).device
        self.owner = threading.get_ident()
        self.policy = PrecisionPolicy(profile)
        self.actual = self.policy.apply(self.torch)
        self.contract = precision_contract(profile)
        self.route = route
        self.state = 'ready'
        self.raw_output_dtypes = set()
        self.calls = self.forwards = self.graph_groups = self.eager_groups = self.captures = 0
        self.stream = self.graph = self.done = None
        self.host_input = self.host_output = self.device_input = self.device_output = None
        self.input_array = self.output_array = None
        self.loaded_manifest = {name: _tensor_record(tensor) for name, tensor in self.model.state_dict().items()}
        if self.loaded_manifest != reference.bundle['receipt']['tensor_manifest']:
            raise ValueError('Actual resident tensor values differ from the strict bundle')
        self.loaded_manifest_sha256 = hashlib.sha256(json.dumps(self.loaded_manifest, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        self.seal = TensorSeal(self.model)
        self.guard()

    def guard(self):
        if threading.get_ident() != self.owner or self.state != 'ready':
            raise RuntimeError('Wrong GPU owner or unavailable precision executor')
        if self.policy.readback(self.torch) != self.actual:
            raise RuntimeError('Precision policy changed inside the worker')
        if self.torch.is_autocast_enabled('cuda'):
            raise RuntimeError('Caller autocast scope is unsupported')
        self.seal.verify()

    def _logits(self, output):
        torch = self.torch
        value = output if isinstance(output, torch.Tensor) else getattr(output, 'output', None)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1,10,224,224)
                or value.device != self.device or not value.is_floating_point()):
            raise ValueError('Model must return one CUDA [1,10,224,224] floating output')
        self.raw_output_dtypes.add(str(value.dtype))
        return value.float()

    def _eager(self, values):
        torch = self.torch
        result = np.empty((len(values),10,224,224), dtype=np.float32)
        with self.policy.scope(torch):
            for index in range(len(values)):
                tensor = torch.from_numpy(values[index:index+1]).to(self.device)
                logits = self._logits(self.model({'S2L2A': tensor}))
                result[index:index+1] = logits.to('cpu').contiguous().numpy()
        return result

    def _body(self):
        self.device_input.copy_(self.host_input, non_blocking=True)
        with self.policy.scope(self.torch):
            for index in range(8):
                logits = self._logits(self.model({'S2L2A': self.device_input[index:index+1]}))
                self.device_output[index:index+1].copy_(logits, non_blocking=True)
        self.host_output.copy_(self.device_output, non_blocking=True)

    def _graph_call(self, values):
        torch = self.torch
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device)
            self.done = torch.cuda.Event()
            self.host_input = torch.empty((8,12,224,224),dtype=torch.float32,pin_memory=True)
            self.host_output = torch.empty((8,10,224,224),dtype=torch.float32,pin_memory=True)
            with torch.cuda.stream(self.stream):
                self.device_input = torch.empty_like(self.host_input,device=self.device)
                self.device_output = torch.empty_like(self.host_output,device=self.device)
            self.input_array = self.host_input.numpy()
            self.output_array = self.host_output.numpy()
        np.copyto(self.input_array, values, casting='no')
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        if self.graph is None:
            with torch.cuda.stream(self.stream):
                for _ in range(3): self._body()
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream,capture_error_mode='global'):
                self._body()
            self.captures += 1
        with torch.cuda.stream(self.stream):
            self.graph.replay()
            self.done.record(self.stream)
        self.done.synchronize()
        return self.output_array.copy()

    def __call__(self, values):
        self.guard()
        if (not isinstance(values,np.ndarray) or values.dtype != np.float32
                or values.ndim != 4 or values.shape[1:] != (12,224,224)
                or not 1 <= len(values) <= 8 or not np.isfinite(values).all()):
            raise ValueError('Expected one to eight finite normalized FP32 tiles')
        values = np.array(values, copy=True, order='C')
        try:
            with self.torch.inference_mode(), self.torch.cuda.device(self.device):
                graph_call = self.route == 'aureon_optimized' and len(values) == 8
                result = self._graph_call(values) if graph_call else self._eager(values)
            if not np.isfinite(result).all(): raise ValueError('Nonfinite neural output')
            self.guard()
        except BaseException:
            self.state = 'failed'
            raise
        self.calls += 1
        self.forwards += len(values)
        self.graph_groups += int(graph_call)
        self.eager_groups += int(not graph_call)
        return result

    def metadata(self):
        return {'schema':'aureon.precision_executor.v1','route':self.route,
            'contract':dict(self.contract),'effective_policy':self.policy.readback(self.torch),
            'state':self.state,'pid':os.getpid(),'owner_thread':self.owner,
            'model_id':self.reference.model_id,'device':str(self.device),
            'loaded_tensor_manifest_sha256':self.loaded_manifest_sha256,
            'loaded_tensor_count':len(self.loaded_manifest),'loaded_values_match':True,
            'raw_neural_output_dtypes':sorted(self.raw_output_dtypes),
            'calls':self.calls,'forwards':self.forwards,'graph_groups':self.graph_groups,
            'eager_groups':self.eager_groups,'capture_count':self.captures,
            'padding_forwards':0,'output_cache':False,'fallback':False,
            'graph_key':{'model':self.reference.bundle['receipt_sha256'],
                'precision':self.contract,'effective_policy':self.actual,
                'device':str(self.device),'input':[8,12,224,224],
                'preprocessing_recipe_sha256':hashlib.sha256((Path(__file__).parent/'recipe.py').read_bytes()).hexdigest(),
                'input_contract':'S2L2A normalized FP32; original units/masks/source are independently bound by each input receipt',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'pinned_staging_bytes':sum(t.numel()*t.element_size() for t in
                (self.host_input,self.host_output) if t is not None)}

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def close(self):
        if threading.get_ident() != self.owner: raise RuntimeError('Wrong owner for cleanup')
        if self.state == 'closed': return
        self.synchronize()
        self.graph = self.stream = self.done = None
        self.input_array = self.output_array = None
        self.host_input = self.host_output = self.device_input = self.device_output = None
        self.state = 'closed'
