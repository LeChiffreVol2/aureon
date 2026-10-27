"""Per-engine CPU control tensors using vLLM's keep_on_cpu field contract."""
import hashlib
import os
from pathlib import Path
from types import MethodType
from .resident_protocol import DESCRIPTOR_SHAPE
from .registered_worker import RegisteredEngineForward

def metadata():
    return {'descriptor_device':'cpu','keep_on_cpu':True,'frame_shape':list(DESCRIPTOR_SHAPE),
        'scope':'bounded control frame only; image and logit transfers remain in the CUDA graph',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def cpu_device_valid(self,value): return not value.is_cuda and value.device.type=='cpu'

def install_cpu_descriptor(controller):
    if (type(controller) is not RegisteredEngineForward or controller.completed_calls
            or controller._pipeline is not None or controller.pid!=os.getpid()
            or os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!='0'):
        raise ValueError('CPU descriptor requires unused verified native in-process worker')
    controller._descriptor_device_valid=MethodType(cpu_device_valid,controller)
    controller.cpu_control=metadata()
    controller._event('cpu-control-installed')

def install_processor(engine):
    from vllm.model_executor.models.terratorch import TerratorchMultiModalProcessor
    from vllm.multimodal.inputs import MultiModalFieldConfig
    processor=engine.llm.llm_engine.renderer.get_mm_processor()
    if (type(processor) is not TerratorchMultiModalProcessor
            or not hasattr(engine.worker._aureon_resident_forward,'cpu_control')):
        raise ValueError('CPU control requires the matching TerraTorch processor and worker')
    original=processor._get_mm_fields_config
    torch=engine.torch
    def fields(self,hf_inputs,hf_processor_mm_kwargs,*,is_shared=True):
        value=hf_inputs.get('S2L2A')
        if (is_shared or set(hf_inputs)!={'S2L2A'} or hf_processor_mm_kwargs
                or not isinstance(value,torch.Tensor) or value.is_cuda
                or value.dtype!=torch.float32 or tuple(value.shape)!=(1,*DESCRIPTOR_SHAPE)):
            raise ValueError('CPU-control engine only admits the exact processed FP32 descriptor')
        return {'S2L2A':MultiModalFieldConfig.batched('image',keep_on_cpu=True)}
    processor._get_mm_fields_config=MethodType(fields,processor)
    def restore(): processor._get_mm_fields_config=original
    return restore
