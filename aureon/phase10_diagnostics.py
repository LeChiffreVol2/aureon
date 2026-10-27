"""Explicit bounded instrumentation of the phase-nine native service."""
import hashlib
import inspect
import json
from pathlib import Path
from .inproc_engine import InprocessPoolingEngine
from .phase8_diagnostics import HostSpans, SegmentedRegisteredGraph
from .phase8_http import ObservedConnection
from .finite_profile import FiniteRegisteredGraph, FiniteResidentPredictor
from .shm_files import SharedSlots

class SegmentedFiniteGraph(SegmentedRegisteredGraph, FiniteRegisteredGraph):
    pass

class DiagnosticNativeEngine(InprocessPoolingEngine):
    def __init__(self,*args,**kwargs):
        self.traces=[]; self.active_trace=None; self.instrumented=[]
        super().__init__(*args,**kwargs)
        controller=self.worker._aureon_resident_forward
        def factory(*args,**kwargs):
            value=SegmentedFiniteGraph.from_verified_worker(*args,**kwargs)
            self.observe(value,'predict_into','worker_pipeline')
            self.observe(value,'_all_finite','worker_finite')
            return value
        controller._pipeline_factory=factory
        engine=self.llm.llm_engine; core=engine.engine_core.engine_core
        for obj,name,label in (
            (self.llm,'_render_and_add_requests','render_and_add'),
            (self.llm,'_run_engine','run_engine'),
            (engine,'add_request','engine_add_request'),
            (engine,'step','engine_step'),
            (engine.output_processor,'process_outputs','output_processing'),
            (core.scheduler,'schedule','scheduler_schedule'),
            (core.scheduler,'update_from_output','scheduler_update'),
            (core.model_executor,'execute_model','executor_execute'),
            (self.worker.model_runner,'execute_model','runner_execute'),
            (self.worker.model_runner,'_prepare_inputs','runner_prepare_inputs'),
            (self.worker.model_runner,'_preprocess','runner_preprocess'),
            (self.worker.model_runner,'_pool','runner_pool'),
            (controller,'decode_descriptor','worker_descriptor_decode'),
            (controller,'_slots_factory','worker_slots_acquire'),
            (controller,'_ack','worker_ack')):
            self.observe(obj,name,label)
        original=controller.model.forward
        def dispatch(*args,**kwargs):
            with self.active_trace.span('worker_dispatch'): return original(*args,**kwargs)
        controller.model.forward=dispatch
        self.diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def observe(self,obj,name,label):
        original=getattr(obj,name)
        path=Path(inspect.getfile(original))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        target=self.folder/'diagnostic-source'/f'{digest}.py'
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            with target.open('xb') as stream: stream.write(path.read_bytes())
        self.instrumented.append({'label':label,'method':name,'path':str(path),'sha256':digest})
        def wrapped(*args,**kwargs):
            if self.active_trace is None: return original(*args,**kwargs)
            with self.active_trace.span(label): return original(*args,**kwargs)
        setattr(obj,name,wrapped)

    def execute(self,descriptor):
        if len(self.traces)>=64: raise RuntimeError('Bounded diagnostic request limit')
        self.active_trace=HostSpans()
        try:
            with self.active_trace.span('native_encode'): super().execute(descriptor)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        self.traces.append({'request':descriptor['request'],'host':self.active_trace.rows,
            'gpu_ms':self.worker._aureon_resident_forward._pipeline.gpu_stages()})
        self.active_trace=None

    def metadata(self):
        return {**super().metadata(),'diagnostic_only':True,'clean_speed_eligible':False,
            'instrumented':self.instrumented,'diagnostic_source_sha256':self.diagnostic_source_sha256}

    def close(self):
        if self.closed:return
        super().close()
        with (self.folder/'http-traces.json').open('x') as stream:
            json.dump({'schema':'aureon.phase10_http_worker_trace.v1','closed':self.closed,
                'diagnostic_only':True,'traces':self.traces,'instrumented':self.instrumented,
                'source_sha256':self.diagnostic_source_sha256},stream,indent=2,allow_nan=False)

class DiagnosticResidentClient(FiniteResidentPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.active_trace=None; self._slots=SharedSlots.create(8)
        self._connection=ObservedConnection(self,'127.0.0.1',self.port,timeout=self.timeout)

    def _write_input(self,values):
        with self.active_trace.span('client_input_copy_finite'): return super()._write_input(values)

    def _read_output(self):
        with self.active_trace.span('client_output_copy_finite'): return super()._read_output()

    def __call__(self,values):
        self.active_trace=HostSpans()
        with self.active_trace.span('client_total'): return super().__call__(values)
