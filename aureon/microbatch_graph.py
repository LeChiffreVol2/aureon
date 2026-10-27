"""Explicit FP32 microbatch experiment; parity must pass before any timing.

This changes GPU batch size and may change floating-point accumulation. It is
not equivalent by construction and is never selected automatically.
"""
from dataclasses import fields,is_dataclass
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from .graph_pipeline import FullGroupGraphPipeline,_original,COUNT,INPUT_SHAPE,OUTPUT_SHAPE
from .reference import PyTorchPredictor


class MicrobatchGroupGraph(FullGroupGraphPipeline):
    @classmethod
    def from_reference(cls,reference,evidence_dir,*,neural_batch_size):
        if (not isinstance(reference,PyTorchPredictor) or reference.dtype!='float32'
                or reference._device.type!='cuda' or reference._as_dict is not True):
            raise ValueError('Require strict CUDA FP32 S2L2A reference')
        return cls(reference.model,_original(reference.model),reference.bundle,evidence_dir,
            torch_module=reference._torch,provenance={'kind':'strict_pytorch_reference'},rank=0,
            neural_batch_size=neural_batch_size)

    def __init__(self,*args,neural_batch_size,**kwargs):
        if type(neural_batch_size) is not int or neural_batch_size not in (2,4):
            raise ValueError('Explicit experimental batch size must be two or four')
        self.neural_batch_size=neural_batch_size
        self.microbatch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)

    def _logits(self,value):
        if isinstance(value,self.torch.Tensor): result=value
        elif (is_dataclass(value) and {field.name for field in fields(value)}=={'output','auxiliary_heads'}
                and value.auxiliary_heads in (None,{})): result=value.output
        else: raise ValueError('Expected tensor or TerraTorch ModelOutput without auxiliary heads')
        if (not isinstance(result,self.torch.Tensor) or tuple(result.shape)!=(self.neural_batch_size,*OUTPUT_SHAPE[1:])
                or result.dtype!=self.torch.float32 or not result.is_cuda or result.device!=self._device):
            raise ValueError('Wrong FP32 microbatch logits shape/device')
        return result

    def _body(self):
        self._device_input.copy_(self._host_input,non_blocking=True)
        for index in range(0,COUNT,self.neural_batch_size):
            stop=index+self.neural_batch_size
            result=self._forward({'S2L2A':self._device_input[index:stop]})
            self._device_output[index:stop].copy_(self._logits(result),non_blocking=True)
        self._host_output.copy_(self._device_output,non_blocking=True)

    def _event(self,event,**extra):
        # Use an explicit new schema: the inherited batch-one event fields
        # would misstate this experiment's arithmetic and forward count.
        self._event_count+=1
        if 'warmup_model_forwards' in extra: extra['warmup_model_forwards']=3*COUNT//self.neural_batch_size
        if 'captured_model_forwards' in extra: extra['captured_model_forwards']=COUNT//self.neural_batch_size
        record={'schema':'aureon.microbatch_graph_event.v1','event':event,'created_utc':datetime.now(timezone.utc).isoformat(),
            'model_id':self.model_id,'bundle_receipt_sha256':self.bundle_receipt_sha256,'state':self.state,
            'base_source_sha256':self.source_sha256,'microbatch_source_sha256':self.microbatch_source_sha256,
            'precision':self.precision,'neural_batch_size':self.neural_batch_size,
            'ordered_forwards_per_replay':COUNT//self.neural_batch_size,'input_shape':list(INPUT_SHAPE),'output_shape':list(OUTPUT_SHAPE),
            'capture_count':self.capture_count,'replay_count':self.replay_count,'output_cache':False,
            'parity_required':True,'automatic_selection':False,**extra}
        with (self.folder/f'microbatch-{self._event_count:04d}-{event}.json').open('x') as stream:
            json.dump(record,stream,indent=2,allow_nan=False)

    def metadata(self):
        value=super().metadata()
        value.update(schema='aureon.microbatch_graph.v1',neural_batch_size=self.neural_batch_size,
            ordered_forwards_per_replay=COUNT//self.neural_batch_size,
            microbatch_source_sha256=self.microbatch_source_sha256,automatic_selection=False,parity_required=True)
        return value
