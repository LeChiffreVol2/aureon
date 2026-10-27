"""Fixed-pair registered host buffers with unchanged FP32 neural arithmetic.

One controlled same-UID client owns its input snapshot through completion.
The worker pins existing private mmap pages and captures H2D/D2H against their
stable addresses. It performs no CPU staging copies. A service restart is
required for another slot pair; no eviction or pageable fallback is allowed.
"""
import ctypes
import hashlib
import mmap
from pathlib import Path

import numpy as np

from .graph_pipeline import FullGroupGraphPipeline, INPUT_SHAPE, OUTPUT_SHAPE


class HostRegistration:
    def __init__(self, torch, device):
        self.torch, self.pointers = torch, []
        path=Path(torch.__file__).resolve().parents[1]/'nvidia/cuda_runtime/lib/libcudart.so.12'
        self.library_path=str(path)
        self.library_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.lib=ctypes.CDLL(str(path))
        self.lib.cudaDeviceGetAttribute.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
        self.lib.cudaDeviceGetAttribute.restype=ctypes.c_int
        self.lib.cudaHostRegister.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint]
        self.lib.cudaHostRegister.restype=ctypes.c_int
        self.lib.cudaHostUnregister.argtypes=[ctypes.c_void_p]
        self.lib.cudaHostUnregister.restype=ctypes.c_int
        self.lib.cudaGetErrorString.argtypes=[ctypes.c_int]
        self.lib.cudaGetErrorString.restype=ctypes.c_char_p
        self.capabilities={}
        for name,attribute in (('host_register',99),('host_page_tables',100),('read_only_register',113)):
            value=ctypes.c_int()
            self.check(self.lib.cudaDeviceGetAttribute(ctypes.byref(value),attribute,device.index or 0))
            self.capabilities[name]=value.value
        if not self.capabilities['host_register'] or not (self.capabilities['host_page_tables'] or self.capabilities['read_only_register']):
            raise RuntimeError('Registered read-only host input is unsupported; no fallback')

    def check(self, code):
        if code: raise RuntimeError(str(code)+': '+self.lib.cudaGetErrorString(code).decode())

    def add(self, values, *, readonly):
        if len(self.pointers)>=2 or not values.flags.c_contiguous:
            raise ValueError('Register at most one bounded contiguous input/output pair')
        pointer=int(values.ctypes.data)
        if pointer in [item[0] for item in self.pointers]: raise ValueError('Overlapping registrations are forbidden')
        flags=8 if readonly and self.capabilities['read_only_register'] else 0
        self.check(self.lib.cudaHostRegister(pointer,values.nbytes,flags))
        self.pointers.append((pointer,int(values.nbytes),flags))

    def close(self):
        # Remove records only after a successful unregister. A failed close
        # retains the address and prevents the owner from unmapping storage.
        while self.pointers:
            self.check(self.lib.cudaHostUnregister(self.pointers[-1][0]))
            self.pointers.pop()

    def metadata(self):
        return {'capabilities':dict(self.capabilities),'runtime_library':self.library_path,
            'runtime_library_sha256':self.library_sha256,'regions':[{'bytes':n,'flags':f} for _,n,f in self.pointers],
            'registered_bytes':sum(n for _,n,_ in self.pointers),'pageable_fallback':False}


class RegisteredGroupGraph(FullGroupGraphPipeline):
    @staticmethod
    def _all_finite(values):
        return bool(np.isfinite(values).all())

    def __init__(self, *args, **kwargs):
        self.registration=None
        self._addresses=None
        self.registered_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)

    def _event(self,event,**extra):
        return super()._event(event,registered_implementation_sha256=self.registered_source_sha256,
            registered_host_buffers=True,worker_cpu_staging_copies=0,
            registration=self.registration.metadata() if self.registration else None,**extra)

    @staticmethod
    def _persistent_view(values, shape, *, writable):
        if (not isinstance(values,np.ndarray) or values.shape!=shape or values.dtype!=np.dtype('<f4')
                or not values.flags.c_contiguous or not isinstance(values.base,mmap.mmap)
                or bool(values.flags.writeable)!=writable):
            raise ValueError('Require the exact private mmap view with expected access mode')
        # A separate ndarray over the same mmap does not retain the temporary
        # lease view. The controller retains the mapping until graph close.
        result=np.ndarray(shape,dtype=np.float32,buffer=values.base)
        if result.ctypes.data!=values.ctypes.data: raise ValueError('Unexpected mmap offset')
        return result

    def _allocate_registered(self, values, destination):
        torch=self.torch
        self._input_numpy=self._persistent_view(values,INPUT_SHAPE,writable=False)
        self._output_numpy=self._persistent_view(destination,OUTPUT_SHAPE,writable=True)
        self._addresses=(int(values.ctypes.data),int(destination.ctypes.data))
        with torch.cuda.device(self._device):
            self.registration=HostRegistration(torch,self._device)
            self.registration.add(self._input_numpy,readonly=True)
            self.registration.add(self._output_numpy,readonly=False)
            # Input is registered read-only, and inherited _body only reads
            # it. No operation writes through this non-writable Torch view.
            self._host_input=torch.from_numpy(self._input_numpy)
            self._host_output=torch.from_numpy(self._output_numpy)
            if not self._host_input.is_pinned() or not self._host_output.is_pinned():
                raise RuntimeError('Registered views must be recognized as pinned; no fallback')
            self._stream=torch.cuda.Stream(device=self._device)
            self._completion=torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self._stream):
                self._device_input=torch.empty(INPUT_SHAPE,dtype=torch.float32,device=self._device)
                self._device_output=torch.empty(OUTPUT_SHAPE,dtype=torch.float32,device=self._device)

    def _predict(self, values, destination):
        self._validate_input(values)
        if (destination is None or destination.shape!=OUTPUT_SHAPE or destination.dtype!=np.dtype('<f4')
                or not destination.flags.c_contiguous or not destination.flags.writeable
                or np.shares_memory(values,destination)):
            raise ValueError('Registered graph requires the owned mapped output destination')
        with self._lock,self.torch.inference_mode():
            if self.state in ('failed','closed'): raise RuntimeError('Registered graph is unavailable')
            try:
                self._check_contract()
                if self._addresses is None: self._allocate_registered(values,destination)
                if self._addresses!=(int(values.ctypes.data),int(destination.ctypes.data)):
                    raise ValueError('Registered graph addresses changed; restart for a different client')
                if not self._all_finite(self._input_numpy): raise ValueError('Input contains nonfinite values')
                with self.torch.cuda.device(self._device):
                    current=self.torch.cuda.current_stream(self._device)
                    if current!=self._stream: self._stream.wait_stream(current)
                    if self._graph is None: self._capture()
                    with self.torch.cuda.stream(self._stream):
                        self._graph.replay()
                        self.replay_count+=1
                        self._completion.record(self._stream)
                    self._completion.synchronize()
                self._check_contract()
                if not self._all_finite(self._output_numpy): raise ValueError('Output contains nonfinite values')
                self.completed_calls+=1
                self.completed_tiles+=8
                if self.state!='active':
                    self.state='active'
                    self._event('first-replay-completed')
                return destination
            except BaseException as error:
                self._fail(error)
                raise

    def metadata(self):
        value=super().metadata()
        value.update(registered_host_buffers=True,registered_implementation_sha256=self.registered_source_sha256,
            registration=self.registration.metadata() if self.registration else None,
            worker_cpu_staging_copies=0,
            memory_scope='Host buffers are registered existing mmap regions; device buffers and graph remain separately allocated',
            input_ownership='Same-UID serialized client owns snapshot until worker completion; fixed slot pair for service lifetime')
        return value

    def close(self):
        with self._lock:
            if self.state=='closed': return
            if self._stream is not None: self._stream.synchronize()
            self._graph=None
            self._host_input=self._host_output=None
            self._input_numpy=self._output_numpy=None
            if self.registration is not None: self.registration.close()
            self._device_input=self._device_output=None
            self._stream=self._completion=None
            self.state='closed'
