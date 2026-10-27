"""Explicit vLLM 0.19 worker for controlled local precision/weight evidence.

Imported only by the separately launched Linux service. It checks the fully
loaded model before use. Optional CUDA graph and diagnostic modes are installed
after startup only; their records remain separate from the weight attestation.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from .runtime import verify_bundle
from .vllm_bundle import _tensor_record


class PrecisionWorker(Worker):
    def compile_or_warm_up_model(self):
        # TerraTorch 1.2.13 DummyDataGenerator hardcodes float16 ones, even
        # for an FP32 service. Cast only its startup dummy call; remove the
        # hook before any real request so measured inputs are untouched.
        model = self.model_runner.model.inference_runner.model
        def profile_dtype(module, args):
            if len(args) != 1 or not isinstance(args[0], dict):
                raise RuntimeError('Unexpected TerraTorch startup input structure')
            return ({key: value.to(dtype=torch.float32) for key, value in args[0].items()},)
        hook = model.register_forward_pre_hook(profile_dtype)
        try:
            result = super().compile_or_warm_up_model()
        finally:
            hook.remove()
        folder = Path(os.environ['AUREON_SERVICE_RUN'])
        if os.environ.get('AUREON_CUDA_GRAPH') == '1':
            from .cuda_graph import install_cuda_graph
            self._aureon_cuda_graph = install_cuda_graph(
                model, folder, rank=self.rank, worker_record=self._aureon_worker_record)
        if os.environ.get('AUREON_PACKED_TILES') == '1':
            from .packed_forward import install_packed_forward
            self._aureon_packed_forward = install_packed_forward(
                model, folder, rank=self.rank, worker_record=self._aureon_worker_record)
        if os.environ.get('AUREON_SHM_ENGINE') == '1':
            from .shm_engine_worker import install_shm_engine_forward
            self._aureon_shm_engine = install_shm_engine_forward(
                model, folder, rank=self.rank, worker_record=self._aureon_worker_record,
                verified_bundle=self._aureon_verified_bundle)
            if os.environ.get('AUREON_PHASE5_DIAGNOSTICS') == '1':
                from .phase5_worker import install_shm_engine_diagnostics
                (folder / 'phase5-diagnostics').mkdir(exist_ok=False)
                self._aureon_shm_engine = install_shm_engine_diagnostics(
                    self._aureon_shm_engine, folder / 'phase5-diagnostics')
                model.forward = self._aureon_shm_engine
        if os.environ.get('AUREON_RESIDENT_SLOTS') == '1':
            from .resident_worker import install_resident_forward
            self._aureon_resident_forward = install_resident_forward(
                model, folder, rank=self.rank, worker_record=self._aureon_worker_record,
                verified_bundle=self._aureon_verified_bundle)
        if os.environ.get('AUREON_FORWARD_DIAGNOSTICS') == '1':
            from .forward_diagnostics import install_forward_diagnostics
            self._aureon_forward_diagnostics = install_forward_diagnostics(model, folder, limit=64)
        if os.environ.get('AUREON_ENGINE_DIAGNOSTICS') == '1':
            from .engine_diagnostics import install_worker_diagnostics
            self._aureon_engine_diagnostics = install_worker_diagnostics(self, folder)
        return result

    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError('AUREON benchmarks require real verified weights')
        super().load_model(load_dummy_weights=False)
        # TerraTorch imports may change Torch globals after Worker.__init__.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark = False
        settings = {
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_sdp_enabled': bool(torch.backends.cuda.cudnn_sdp_enabled()),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'flash_sdp_enabled': bool(torch.backends.cuda.flash_sdp_enabled()),
            'mem_efficient_sdp_enabled': bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            'math_sdp_enabled': bool(torch.backends.cuda.math_sdp_enabled()),
        }
        if (settings['float32_matmul_precision'] != 'highest'
                or any(settings[key] for key in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                       'cudnn_sdp_enabled', 'cudnn_benchmark'))):
            raise RuntimeError('Worker precision policy assertion failed')
        verified = verify_bundle(os.environ['AUREON_VERIFIED_BUNDLE'])
        wrapper = self.model_runner.model
        model = wrapper.inference_runner.model
        if model.training:
            raise RuntimeError('Inference model must be in evaluation mode')
        state = model.state_dict()
        # FP32 run: exact CPU tensor fingerprints, including learned buffers.
        actual = {key: _tensor_record(value.detach().cpu()) for key, value in state.items()}
        expected = verified['receipt']['tensor_manifest']
        if actual != expected:
            mismatches = sorted(set(actual) ^ set(expected)) + [
                key for key in sorted(set(actual) & set(expected)) if actual[key] != expected[key]]
            raise RuntimeError(f'Worker state differs from verified FP32 bundle: {mismatches[:20]}')
        record = {
            'schema': 'aureon.worker_attestation.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'pid': os.getpid(), 'rank': self.rank, 'model_id': verified['model_id'],
            'bundle_receipt_sha256': verified['receipt_sha256'],
            'bundle_files_sha256': verified['files_sha256'],
            'loaded_tensor_count': len(actual), 'loaded_tensor_manifest_matches': True,
            'precision_settings_actual': settings,
            'model_training': model.training, 'gpu_name': torch.cuda.get_device_name(),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'scope': 'local worker post-load state and precision; not remote cryptographic attestation',
            'startup_compatibility': 'float16 dummy inputs cast to float32 only during compile_or_warm_up_model; hook then removed',
        }
        folder = Path(os.environ['AUREON_SERVICE_RUN'])
        with (folder / f'worker-{self.rank}.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        self._aureon_worker_record = record
        self._aureon_verified_bundle = verified
        print('AUREON_WORKER_VERIFIED ' + json.dumps(record), flush=True)
