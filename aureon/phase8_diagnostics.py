"""Explicit in-memory diagnostics; never a clean performance profile.

CUDA event nodes change the captured graph. Stage times include device waits
and contention, not just active kernel execution. No files are written inside
the timed model call. The unchanged eight batch-one FP32 forwards are retained.
"""
from contextlib import contextmanager
import time
from .registered_graph import RegisteredGroupGraph


class HostSpans:
    def __init__(self): self.rows=[]

    @contextmanager
    def span(self,name):
        import resource
        before=resource.getrusage(resource.RUSAGE_THREAD)
        cpu=time.thread_time_ns()
        start=time.perf_counter_ns()
        try: yield
        finally:
            end=time.perf_counter_ns()
            cpu_end=time.thread_time_ns()
            after=resource.getrusage(resource.RUSAGE_THREAD)
            self.rows.append({'stage':name,'start_ns':start,'end_ns':end,
                'wall_ms':(end-start)/1e6,'thread_cpu_ms':(cpu_end-cpu)/1e6,
                'minor_faults':after.ru_minflt-before.ru_minflt,
                'major_faults':after.ru_majflt-before.ru_majflt,
                'voluntary_switches':after.ru_nvcsw-before.ru_nvcsw,
                'involuntary_switches':after.ru_nivcsw-before.ru_nivcsw})


class SegmentedRegisteredGraph(RegisteredGroupGraph):
    def __init__(self,*args,**kwargs):
        self.stage_events=None
        super().__init__(*args,**kwargs)

    def _body(self):
        if self.stage_events is None:
            self.stage_events=[self.torch.cuda.Event(enable_timing=True,external=True) for _ in range(19)]
        events=self.stage_events
        events[0].record(self._stream)
        self._device_input.copy_(self._host_input,non_blocking=True)
        events[1].record(self._stream)
        for index in range(8):
            result=self._forward({'S2L2A':self._device_input[index:index+1]})
            events[2+index*2].record(self._stream)
            self._device_output[index:index+1].copy_(self._logits(result),non_blocking=True)
            events[3+index*2].record(self._stream)
        self._host_output.copy_(self._device_output,non_blocking=True)
        events[18].record(self._stream)

    def gpu_stages(self):
        # The inherited prediction returned after its completion event.
        labels=['h2d']+[label for i in range(8) for label in (f'forward_{i}',f'output_d2d_{i}')]+['d2h']
        result={name:self.stage_events[i].elapsed_time(self.stage_events[i+1]) for i,name in enumerate(labels)}
        result['total']=self.stage_events[0].elapsed_time(self.stage_events[-1])
        return result

    def close(self):
        super().close()
        self.stage_events=None
