"""CPU control ACK through the actual native vLLM identity pooler.

The registered graph has already synchronized D2H and checked all logits.
This changes only the device of the fresh bounded control tensor. No result
is cached, and the existing request-bound ACK decoder remains unchanged.
"""
import hashlib
import copy
import os
from pathlib import Path
from types import MethodType
from .registered_worker import RegisteredEngineForward

def cpu_ack_tensor(self, values):
    return self.torch.from_numpy(values).unsqueeze(0)

def install_cpu_pool_cursor(runner):
    """Keep the native pooler and its checks, supplying matching CPU offsets.

    vLLM 0.19 builds offsets on CPU before copying them to GPU. For this fixed
    one-token control request, reuse the current CPU [0,1] mirror. No GPU field
    is replaced, and the original per-request metadata object is not modified.
    """
    original_pool=runner._pool
    def pool(self,hidden_states,num_scheduled_tokens,num_scheduled_tokens_np,kv_connector_output):
        if (hidden_states.is_cuda or tuple(hidden_states.shape)!=(1,512)
                or str(hidden_states.dtype)!='torch.float32' or num_scheduled_tokens!=1
                or num_scheduled_tokens_np.tolist()!=[1] or self.input_batch.num_reqs!=1
                or self.use_async_scheduling):
            raise ValueError('CPU ACK pooling requires exactly one completed FP32 control frame')
        offsets=self.query_start_loc.cpu[:2]
        if offsets.device.type!='cpu' or offsets.tolist()!=[0,1]:
            raise ValueError('Current CPU query offsets differ from the fixed single-token request')
        original_get=self.input_batch.get_pooling_metadata
        def get_metadata():
            value=copy.copy(original_get())
            original_build=type(value).build_pooling_cursor
            def build(meta,counts,lengths,device,query_start_loc_gpu=None):
                if (device.type!='cpu' or query_start_loc_gpu is None
                        or query_start_loc_gpu.device!=self.device
                        or tuple(query_start_loc_gpu.shape)!=(2,)
                        or query_start_loc_gpu.data_ptr()!=self.query_start_loc.gpu.data_ptr()):
                    raise ValueError('Native query-offset provenance changed')
                return original_build(meta,counts,lengths,device,query_start_loc_gpu=offsets)
            value.build_pooling_cursor=MethodType(build,value)
            return value
        self.input_batch.get_pooling_metadata=get_metadata
        try:
            return original_pool(hidden_states,num_scheduled_tokens,num_scheduled_tokens_np,kv_connector_output)
        finally:
            self.input_batch.get_pooling_metadata=original_get
    runner._pool=MethodType(pool,runner)

def install_cpu_ack(worker,controller):
    from vllm.model_executor.layers.pooler import IdentityPooler
    if (type(controller) is not RegisteredEngineForward or controller.completed_calls
            or controller._pipeline is not None or controller.pid!=os.getpid()
            or os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!='0'
            or worker.model_runner.use_async_scheduling
            or type(worker.model_runner.model.pooler) is not IdentityPooler):
        raise ValueError('CPU ACK requires an unused verified synchronous in-process identity-pooling worker')
    controller._ack_tensor=MethodType(cpu_ack_tensor,controller)
    install_cpu_pool_cursor(worker.model_runner)
    controller.control_ack={'device':'cpu','fresh_tensor_per_request':True,'native_identity_pooler':True,
        'cursor':'current native CPU [0,1] offsets; native build_pooling_cursor checks retained',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    controller._event('cpu-ack-installed')
