"""Bounded request-correlated HTTP diagnostics; no clean speed claims."""
import hashlib
import http.client
import json
from pathlib import Path
from .inproc_engine import InprocessPoolingEngine
from .phase8_diagnostics import HostSpans,SegmentedRegisteredGraph
from .resident_transport import ResidentSharedMemoryPredictor
from .shm_files import SharedSlots


class DiagnosticNativeEngine(InprocessPoolingEngine):
    def __init__(self,*args,**kwargs):
        self.traces=[]
        self.active_trace=None
        self.diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)
        controller=self.worker._aureon_resident_forward
        def factory(*args,**kwargs):
            value=SegmentedRegisteredGraph.from_verified_worker(*args,**kwargs)
            original=value.predict_into
            def predict(*args,**kwargs):
                with self.active_trace.span('worker_pipeline'): return original(*args,**kwargs)
            value.predict_into=predict
            return value
        controller._pipeline_factory=factory

    def execute(self,descriptor):
        if len(self.traces)>=64: raise RuntimeError('Bounded diagnostic request limit reached')
        self.active_trace=HostSpans()
        with self.active_trace.span('native_encode'): super().execute(descriptor)
        self.traces.append({'request':descriptor['request'],'host':self.active_trace.rows,
            'gpu_ms':self.worker._aureon_resident_forward._pipeline.gpu_stages()})

    def metadata(self):
        value=super().metadata()
        value.update(diagnostic_only=True,clean_speed_eligible=False,diagnostic_source_sha256=self.diagnostic_source_sha256)
        return value

    def close(self):
        if self.closed: return
        super().close()
        with (self.folder/'http-traces.json').open('x') as stream:
            json.dump({'schema':'aureon.phase8_http_worker_trace.v1','diagnostic_only':True,
                'traces':self.traces,'closed':self.closed,'source_sha256':self.diagnostic_source_sha256},stream,indent=2,allow_nan=False)


class ObservedConnection(http.client.HTTPConnection):
    def __init__(self,owner,*args,**kwargs):
        self.owner=owner
        super().__init__(*args,**kwargs)

    def request(self,*args,**kwargs):
        with self.owner.active_trace.span('http_request_send'): return super().request(*args,**kwargs)

    def getresponse(self):
        with self.owner.active_trace.span('http_response_headers'): response=super().getresponse()
        original=response.read
        def read(*args,**kwargs):
            with self.owner.active_trace.span('http_response_body'): return original(*args,**kwargs)
        response.read=read
        return response


class DiagnosticResidentClient(ResidentSharedMemoryPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.active_trace=None
        self._slots=SharedSlots.create(8)
        self._connection=ObservedConnection(self,'127.0.0.1',self.port,timeout=self.timeout)
        original_write,original_snapshot=self._slots.write,self._slots.snapshot
        def write(*args,**kwargs):
            with self.active_trace.span('client_input_copy'): return original_write(*args,**kwargs)
        def snapshot(*args,**kwargs):
            with self.active_trace.span('client_output_copy'): return original_snapshot(*args,**kwargs)
        self._slots.write,self._slots.snapshot=write,snapshot

    def __call__(self,values):
        self.active_trace=HostSpans()
        with self.active_trace.span('client_total'): result=super().__call__(values)
        return result
