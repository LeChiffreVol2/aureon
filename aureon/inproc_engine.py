"""Explicit synchronous native vLLM engine; importing never loads a model.

One owner thread constructs, executes and closes the real LLM/EngineCore.
Only bounded descriptor/ACK frames are passed to native pooling. Bulk arrays
retain the existing guarded resident-slot protocol and FP32 graph pipeline.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

from .resident_protocol import encode_descriptor
from .shm_engine_protocol import validate_ack, ACK_SHAPE


class InprocessPoolingEngine:
    def __init__(self, bundle, folder):
        # These settings must precede the first vLLM import in this process.
        if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') != '0':
            raise ValueError('Explicitly disable vLLM V1 multiprocessing before construction')
        if any(os.environ.get(key) != '1' for key in ('AUREON_CUDA_GRAPH', 'AUREON_PACKED_TILES',
                'AUREON_SHM_ENGINE', 'AUREON_RESIDENT_SLOTS', 'AUREON_SHARED_MEMORY_API')):
            raise ValueError('Require the existing controlled FP32 resident-slot worker profile')
        if any(os.environ.get(key, '0') != '0' for key in ('AUREON_PHASE5_DIAGNOSTICS',
                'AUREON_ENGINE_DIAGNOSTICS', 'AUREON_FORWARD_DIAGNOSTICS')):
            raise ValueError('This clean service excludes diagnostic instrumentation')
        self.owner = threading.get_ident()
        self.folder = Path(folder).resolve(strict=True)
        self.bundle = bundle
        self.closed = self.failed = False
        self.completed_calls = 0
        self.llm = None
        self._restore_cpu_control = None
        import torch
        from vllm import LLM, PoolingParams
        from vllm.v1.engine.core_client import InprocClient
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        from .cuda_graph import _precision
        self.torch = torch
        self.params = PoolingParams(task='plugin', use_activation=False)
        self.llm = LLM(model=bundle['bundle_dir'], runner='pooling', model_impl='terratorch',
            skip_tokenizer_init=True, enable_mm_embeds=True, dtype='float32', enforce_eager=True,
            tensor_parallel_size=1, distributed_executor_backend='uni', async_scheduling=False,
            max_model_len=1, max_num_seqs=1, max_num_batched_tokens=1, gpu_memory_utilization=.4,
            mm_processor_cache_gb=0, disable_log_stats=True, seed=0,
            worker_cls=('aureon.registered_precision_worker.RegisteredPrecisionWorker'
                if os.environ.get('AUREON_REGISTERED_SLOTS')=='1' else 'aureon.vllm_worker.PrecisionWorker'))
        client = self.llm.llm_engine.engine_core
        if not isinstance(client, InprocClient):
            raise RuntimeError('Expected actual native InprocClient, not an external engine')
        core = client.engine_core
        executor = core.model_executor
        if not isinstance(executor, UniProcExecutor):
            raise RuntimeError('Expected actual native UniProcExecutor')
        worker = executor.driver_worker.worker
        record = json.loads((self.folder/'worker-0.json').read_bytes())
        if (record['pid'] != os.getpid() or worker._aureon_worker_record != record
                or record['loaded_tensor_manifest_matches'] is not True
                or record['bundle_receipt_sha256'] != bundle['receipt_sha256']
                or record['precision_settings_actual'] != _precision(torch)):
            raise RuntimeError('In-process worker provenance differs')
        self.worker, self.precision = worker, _precision(torch)
        objects = {'client': client, 'core': core, 'executor': executor, 'scheduler': core.scheduler,
                   'worker': worker, 'runner': worker.model_runner}
        self.topology = {}
        for name, obj in objects.items():
            path = Path(inspect.getfile(type(obj)))
            self.topology[name] = {'class': type(obj).__module__+'.'+type(obj).__name__,
                'source_path': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if os.environ.get('AUREON_CPU_CONTROL') == '1':
            from .cpu_control import install_processor
            self._restore_cpu_control = install_processor(self)

    def _check(self):
        from .cuda_graph import _precision
        if threading.get_ident() != self.owner or self.closed or self.failed:
            raise RuntimeError('Engine requires its live, unpoisoned owner thread')
        if _precision(self.torch) != self.precision:
            raise RuntimeError('In-process FP32 settings changed')

    def execute(self, descriptor):
        self._check()
        if descriptor.get('model') != self.bundle['model_id']:
            raise ValueError('Descriptor must name the loaded KAO-TerraMind model')
        values = encode_descriptor(descriptor)
        try:
            prompts = [{'prompt_token_ids': [1], 'multi_modal_data': {
                'image': {'S2L2A': self.torch.from_numpy(values)}}}]
            outputs = self.llm.encode(prompts, pooling_params=self.params, pooling_task='plugin', use_tqdm=False)
            if len(outputs) != 1 or not outputs[0].finished:
                raise RuntimeError('Expected one completed native pooling result')
            data = outputs[0].outputs.data
            if (not isinstance(data, self.torch.Tensor) or data.is_cuda or data.dtype != self.torch.float32
                    or tuple(data.shape) != ACK_SHAPE):
                raise RuntimeError('Native pooling must return the exact CPU FP32 acknowledgement')
            validate_ack(data.numpy(), descriptor)
            self._check()
            self.completed_calls += 1
        except BaseException:
            self.failed = True
            raise

    def metadata(self):
        import importlib.metadata
        return {'schema': 'aureon.inproc_engine.v1', 'pid': os.getpid(), 'owner_thread': self.owner,
            'model_id': self.bundle['model_id'], 'bundle_receipt_sha256': self.bundle['receipt_sha256'],
            'topology': self.topology, 'precision_settings_actual': self.precision,
            'completed_calls': self.completed_calls, 'failed': self.failed, 'closed': self.closed,
            'versions': {name: importlib.metadata.version(name) for name in ('vllm', 'terratorch', 'torch')},
            'native_scheduler_used': True, 'native_pooler_used': True, 'engine_request_ipc': False,
            'service_topology': 'synchronous HTTP and native vLLM EngineCore in one owner thread',
            'output_cache': False, 'automatic_fallback': False,
            'resident_worker': self.worker._aureon_resident_forward.metadata()}

    def close(self):
        if self.closed:
            return
        if threading.get_ident() != self.owner:
            raise RuntimeError('Close the native engine on its owner thread')
        try:
            if self.llm is not None:
                self.torch.cuda.synchronize()
                self.worker._aureon_resident_forward.close()
                self.llm.llm_engine.engine_core.shutdown()
        finally:
            if self._restore_cpu_control is not None:
                self._restore_cpu_control()
                self._restore_cpu_control = None
            self.closed = True
