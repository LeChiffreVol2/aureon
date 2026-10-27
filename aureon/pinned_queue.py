"""Explicit PyTorch candidate: HTTP owns bounded pinned buffers through send.

Only the GPU owner touches Torch. Socket threads read/write NumPy views of
preallocated pinned storage. Device addresses and eight batch-one forwards
remain fixed; host transfers surround a compute-only graph on one stream.
"""
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .finite_scan import FiniteScanner
from .graph_pipeline import FullGroupGraphPipeline, INPUT_SHAPE, OUTPUT_SHAPE, COUNT


class PinnedQueueGraph(FullGroupGraphPipeline):
    execution_backend='pytorch_pinned_queue_compute_graph'

    def __init__(self,*args,**kwargs):
        self.scanner=FiniteScanner()
        self.pool=[]
        self.owner=threading.get_ident()
        self.pinned_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)

    def _event(self,event,**extra):
        return super()._event(event,captured_host_transfers=False,
            pinned_queue_source_sha256=self.pinned_source_sha256,**extra)

    def create_jobs(self,capacity):
        from .queued_http import Job
        if self.pool or capacity!=8 or threading.get_ident()!=self.owner:
            raise ValueError('Create exactly eight owned pinned pairs once on the GPU owner')
        torch=self.torch
        with torch.cuda.device(self._device):
            self._stream=torch.cuda.Stream(device=self._device)
            self._completion=torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self._stream):
                self._device_input=torch.empty(INPUT_SHAPE,dtype=torch.float32,device=self._device)
                self._device_output=torch.empty(OUTPUT_SHAPE,dtype=torch.float32,device=self._device)
            for _ in range(capacity):
                inp=torch.zeros(INPUT_SHAPE,dtype=torch.float32,device='cpu',pin_memory=True)
                out=torch.zeros(OUTPUT_SHAPE,dtype=torch.float32,device='cpu',pin_memory=True)
                if not inp.is_pinned() or not out.is_pinned():
                    raise RuntimeError('Pinned allocation is mandatory; no fallback')
                self.pool.append((inp,out,inp.numpy(),out.numpy()))
        # Metadata compatibility only; the compute graph has no host pointers.
        self._host_input,self._host_output=self.pool[0][:2]
        return [Job(values,output) for _,_,values,output in self.pool]

    def _body(self):
        for index in range(COUNT):
            result=self._forward({'S2L2A':self._device_input[index:index+1]})
            self._device_output[index:index+1].copy_(self._logits(result),non_blocking=True)

    def _owned_pair(self,values,output):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Only the original owner can execute GPU work')
        for inp,out,owned_in,owned_out in self.pool:
            if values is owned_in and output is owned_out:
                return inp,out
        raise ValueError('Input and output must be the exact same owned pinned pair')

    def _predict(self,values,destination):
        # Never fall back to the parent's pageable staging or allocating API.
        if destination is None:
            raise ValueError('Pinned inference requires an owned output destination')
        self.predict_stages(values,destination)
        return destination

    def predict_stages(self,values,output):
        inp,out=self._owned_pair(values,output)
        start=time.perf_counter()
        with self._lock,self.torch.inference_mode():
            if self.state in ('failed','closed'):
                raise RuntimeError('Pinned graph is unavailable')
            try:
                self._check_contract()
                if not self.scanner(values):
                    raise ValueError('Nonfinite input')
                checked=time.perf_counter()
                with self.torch.cuda.device(self._device):
                    current=self.torch.cuda.current_stream(self._device)
                    if current!=self._stream:
                        self._stream.wait_stream(current)
                    with self.torch.cuda.stream(self._stream):
                        self._device_input.copy_(inp,non_blocking=True)
                    if self._graph is None:
                        self._capture()
                    with self.torch.cuda.stream(self._stream):
                        self._graph.replay()
                        self.replay_count+=1
                        out.copy_(self._device_output,non_blocking=True)
                        self._completion.record(self._stream)
                    self._completion.synchronize()
                executed=time.perf_counter()
                self._check_contract()
                if not self.scanner(output):
                    raise ValueError('Nonfinite output')
                self.completed_calls+=1
                self.completed_tiles+=COUNT
                if self.state!='active':
                    self.state='active'
                    self._event('first-replay-completed')
                finished=time.perf_counter()
                return {'input_copy_finite_ms':(checked-start)*1000,
                        'backend_ms':(executed-checked)*1000,
                        'output_copy_finite_ms':(finished-executed)*1000,
                        'cpu_staging_copy_bytes':0}
            except BaseException as error:
                self._fail(error)
                raise

    def metadata(self):
        value=super().metadata()
        value.update(pinned_queue_source_sha256=self.pinned_source_sha256,
            host_pool_pairs=len(self.pool),host_pool_bytes=sum(t.numel()*t.element_size() for pair in self.pool for t in pair[:2]),
            pin_memory_actual=bool(self.pool) and all(t.is_pinned() for pair in self.pool for t in pair[:2]),
            cpu_staging_copy_bytes_per_request=0,finite_scan=self.scanner.metadata(),
            captured_host_transfers=False,
            transfer_scope='Pinned owned HTTP input -> H2D -> eight original batch-one forwards in compute-only CUDA graph -> D2H into owned HTTP output; one stream and completion event',
            memory_scope='Eight pinned HTTP pairs, one device pair, one model and graph; no shared-memory staging pair',
            output_ownership='HTTP slot remains exclusive until full response send or disconnect after completed GPU work')
        return value

    def close(self):
        super().close()
        self.pool.clear()


class PinnedQueuedBackend:
    def __init__(self,bundle,folder):
        import torch
        from .reference import PyTorchPredictor
        # Validate the native helper before model or buffer acquisition.
        self.scanner=FiniteScanner()
        self.kind,self.bundle,self.calls='pytorch',bundle,0
        self.engine=self.graph=None
        self.slots=SimpleNamespace(identities={})
        reference=PyTorchPredictor(bundle['bundle_dir'],device='cuda:0',dtype='float32',validate_on_cpu=True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark=False
        graph_dir=folder/'direct-graph'
        graph_dir.mkdir()
        self.graph=PinnedQueueGraph.from_reference(reference,graph_dir)

    def create_jobs(self,capacity):
        return self.graph.create_jobs(capacity)

    def predict_into(self,values,output,request_id):
        stages=self.graph.predict_stages(values,output)
        self.calls+=1
        return stages

    def metadata(self):
        return {'kind':self.kind,'io_mode':'pinned','calls':self.calls,'slots':{},
                'scanner':self.scanner.metadata(),'backend':self.graph.metadata()}

    def close(self):
        if self.graph:
            self.graph.close()
