"""Explicit bounded streaming execution with sequential on-disk FP32 reference."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .core import array_hash
from .job import file_hash, write_json, canonical_hash
from .stream_io import (WindowSource, CategorySink, DiskBlend, StripeBlend, deliver,
                        check_cancelled, GDAL_CACHE_BYTES)

LOGIT_SHAPE=(10,224,224)
SCRATCH_BUDGET=8*1024**3
APPLICATION_BUDGET=384*1024**2


def budget(source,destination):
    # Includes separate reference, disk accumulator, representative candidate,
    # categorical staging/COGs and conservative TIFF/overview overhead.
    disk=source.total*10*224*224*4*2 + source.width*source.height*64 + 64*1024**2
    # Rolling accumulator plus overlap-copy, IO, two predictor copies, logits,
    # reference read, arithmetic temporaries and one bounded GDAL cache.
    ram=2*42*224*source.width + 5*8*12*224*224*4 + GDAL_CACHE_BYTES + 48*1024**2
    if disk>SCRATCH_BUDGET or ram>APPLICATION_BUDGET:
        raise ValueError('Streaming allocation plan exceeds declared budget')
    if shutil.disk_usage(Path(destination).parent).free<disk+2*1024**3:
        raise ValueError('Insufficient scratch space before model execution')
    return {'scratch_estimate_bytes':disk,'scratch_budget_bytes':SCRATCH_BUDGET,
            'application_estimate_bytes':ram,'application_budget_bytes':APPLICATION_BUDGET,
            'gdal_cache_bytes':GDAL_CACHE_BYTES,'maximum_group_tiles':8,
            'blend_rows':224,'width':source.width,'model_memory_included':False,
            'scope':'Application buffer plan; sampled RSS also includes interpreter/runtime/allocator retention'}


def _output(values,count):
    if (not isinstance(values,np.ndarray) or values.dtype!=np.float32
            or values.shape!=(count,*LOGIT_SHAPE) or not np.isfinite(values).all()):
        raise ValueError('Expected finite FP32 outputs for the real group')
    return values


def _line(stream,record):
    stream.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n')


def _cleanup_owned(folder,names):
    # Only exact temporary files in this new attempt, never arbitrary folders.
    folder=Path(folder).resolve(strict=True)
    for name in names:
        path=folder/name
        if path.parent!=folder or path.is_symlink(): raise ValueError('Unsafe scratch cleanup target')
        if path.exists(): path.unlink()


def _precision_contract(value):
    if value is None: return 'highest FP32; TF32 off'
    from ..experimental_precision import validate_contract
    return validate_contract(value)


def build_reference(manifest,eager,destination,*,model_identity,cancelled=None,precision_contract=None):
    """Separate eager calls plus independent disk accumulation; outside warm cache timing."""
    precision=_precision_contract(precision_contract)
    destination=Path(destination).absolute()
    if destination.exists(): raise FileExistsError('Reference directory must be new')
    started=perf_counter(); executed=groups=0; logit_digest=hashlib.sha256()
    with WindowSource(manifest) as source:
        limits=budget(source,destination); destination.mkdir()
        try:
            with ExitStack() as stack:
                logits_file=stack.enter_context((destination/'logits.bin').open('xb'))
                ledger=stack.enter_context((destination/'groups.jsonl').open('x'))
                blend=stack.enter_context(DiskBlend(destination,source))
                sink=stack.enter_context(CategorySink(destination/'categories',source))
                for inputs,selected in source.groups(cancelled):
                    outputs=_output(eager(inputs),len(inputs))
                    data=memoryview(np.ascontiguousarray(outputs)).cast('B')
                    logits_file.write(data); logit_digest.update(data)
                    _line(ledger,{'windows':selected,'input_sha256':array_hash(inputs),
                                  'logits_sha256':array_hash(outputs),'count':len(inputs)})
                    for window,output in zip(selected,outputs): blend.add(window,output)
                    executed+=len(inputs); groups+=1
                if not executed: raise ValueError('All-invalid scene has no reference tiles')
                blend.finish(sink,cancelled)
                valid_count=sink.valid_pixels
                if valid_count!=source.manifest['valid_mask']['valid_pixels']:
                    raise ValueError('Manifest valid count differs from streamed source')
            source.verify()
            result={'schema':'aureon.streaming_reference.v1','status':'completed',
                'manifest_path':str(Path(manifest).resolve()),
                'source_manifest_sha256':source.manifest_hash,'model_identity':model_identity,
                'precision':precision,'tile_count':executed,'tiles_total':source.total,
                'tiles_skipped':source.skipped,'groups':groups,'valid_pixels':valid_count,
                'grid':source.grid,'source_identity':source.identity,'logits_sha256':logit_digest.hexdigest(),
                'logits_bytes':(destination/'logits.bin').stat().st_size,
                'groups_sha256':file_hash(destination/'groups.jsonl'),
                'class_sha256':file_hash(destination/'categories/classes.tif'),
                'mask_sha256':file_hash(destination/'categories/mask.tif'),
                'limits':limits,'seconds':perf_counter()-started,
                'scope':'Independent eager outputs, sequential FP32 disk-window accumulation; no whole-scene array'}
            write_json(destination/'reference.json',result)
            return result
        except BaseException as error:
            write_json(destination/'failure.json',{'status':'failed','error_type':type(error).__name__,
                'error':str(error),'partial_artifacts_not_publishable':True})
            raise
        finally:
            _cleanup_owned(destination,('accumulator.tif','coverage.tif'))


class ReferenceCursor:
    def __init__(self,folder,source,model_identity,*,precision_contract=None):
        self.folder=Path(folder); self.meta=json.loads((self.folder/'reference.json').read_text())
        m=self.meta
        if (m['status']!='completed' or m['source_manifest_sha256']!=source.manifest_hash
                or m['model_identity']!=model_identity or m['precision']!=_precision_contract(precision_contract)
                or (self.folder/'logits.bin').stat().st_size!=m['logits_bytes']
                or file_hash(self.folder/'groups.jsonl')!=m['groups_sha256']
                or file_hash(self.folder/'categories/classes.tif')!=m['class_sha256']
                or file_hash(self.folder/'categories/mask.tif')!=m['mask_sha256']):
            raise ValueError('Reference identity changed or belongs to another execution')
        self.stack=ExitStack()
        try:
            self.logits=self.stack.enter_context((self.folder/'logits.bin').open('rb'))
            self.ledger=self.stack.enter_context((self.folder/'groups.jsonl').open())
        except BaseException:
            self.stack.close(); raise
        self.digest=hashlib.sha256(); self.tiles=0

    def next(self,inputs,selected):
        return self.next_hashed(array_hash(inputs),selected)

    def next_hashed(self,input_sha256,selected):
        """Consume a prepared group's sealed hash after its input lease is free."""
        if (not isinstance(input_sha256,str) or len(input_sha256)!=64
                or any(c not in '0123456789abcdef' for c in input_sha256)):
            raise ValueError('Canonical input SHA-256 required')
        line=self.ledger.readline(65537)
        if not line or len(line)>65536: raise ValueError('Reference group ledger missing/oversized')
        record=json.loads(line)
        if record['windows']!=[list(w) for w in selected] or record['input_sha256']!=input_sha256:
            raise ValueError('Normalized pixels or tile order differ from independent reference')
        count=len(selected)
        if record['count']!=count: raise ValueError('Reference group length changed')
        outputs=np.fromfile(self.logits,dtype=np.float32,count=count*10*224*224)
        if outputs.size!=count*10*224*224: raise ValueError('Truncated reference logits')
        outputs=outputs.reshape(count,*LOGIT_SHAPE)
        if array_hash(outputs)!=record['logits_sha256']: raise ValueError('Reference logits changed')
        self.digest.update(memoryview(outputs).cast('B')); self.tiles+=count
        return outputs,record

    def finish(self):
        if (self.logits.read(1) or self.ledger.read(1) or self.tiles!=self.meta['tile_count']
                or self.digest.hexdigest()!=self.meta['logits_sha256']):
            raise ValueError('Reference coverage or whole-logit identity changed')

    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def run_streaming(manifest,candidate,reference,destination,*,model_identity,provenance,
                  cancelled=None,save_logits=False,precision_contract=None,
                  execution_route='graph_eight_eager_tail'):
    """Warm delivery with a verified disk reference; API builds one per job."""
    precision=_precision_contract(precision_contract)
    if execution_route not in ('graph_eight_eager_tail','direct_eager'):
        raise ValueError('Unknown explicit scene execution route')
    destination=Path(destination).absolute()
    if destination.exists(): raise FileExistsError('Streaming output must be new')
    started=perf_counter(); stages={}; total=group_count=0
    with WindowSource(manifest) as source:
        stages['source_admission_seconds']=perf_counter()-started
        limits=budget(source,destination); destination.mkdir()
        try:
            with ExitStack() as stack:
                checked=stack.enter_context(ReferenceCursor(reference,source,model_identity,
                    precision_contract=precision_contract))
                ledger=stack.enter_context((destination/'groups.jsonl').open('x'))
                raw=stack.enter_context((destination/'logits.bin').open('xb')) if save_logits else None
                sink=stack.enter_context(CategorySink(destination/'categories',source))
                blend=StripeBlend(source,sink,cancelled)
                iterator=iter(source.groups(cancelled))
                for key in ('read_normalize','prediction','tile_parity','blend','evidence'):
                    stages[key+'_seconds']=0.0
                while True:
                    tick=perf_counter()
                    try: inputs,selected=next(iterator)
                    except StopIteration:
                        stages['read_normalize_seconds']+=perf_counter()-tick; break
                    stages['read_normalize_seconds']+=perf_counter()-tick
                    check_cancelled(cancelled); tick=perf_counter()
                    outputs=_output(candidate(inputs),len(inputs))
                    stages['prediction_seconds']+=perf_counter()-tick
                    tick=perf_counter(); expected,record=checked.next(inputs,selected)
                    if not np.array_equal(outputs,expected) or array_hash(outputs)!=record['logits_sha256']:
                        raise ValueError('Bitwise FP32 parity failed; no output may be published')
                    stages['tile_parity_seconds']+=perf_counter()-tick
                    tick=perf_counter()
                    for window,output in zip(selected,outputs): blend.add(window,output)
                    stages['blend_seconds']+=perf_counter()-tick
                    tick=perf_counter()
                    if raw is not None: raw.write(memoryview(np.ascontiguousarray(outputs)).cast('B'))
                    _line(ledger,record)
                    stages['evidence_seconds']+=perf_counter()-tick
                    total+=len(inputs); group_count+=1
                if not total: raise ValueError('All-invalid scene rejected')
                checked.finish(); tick=perf_counter(); blend.finish()
                stages['blend_seconds']+=perf_counter()-tick
                valid_count=sink.valid_pixels
                if valid_count!=source.manifest['valid_mask']['valid_pixels']:
                    raise ValueError('Streamed source valid count changed')
            tick=perf_counter()
            parity={'passed':True,'bitwise_equal':True,'max_abs_error':0.0,
                    'tile_count':total,'normalization_and_tile_order_equal':True,
                    'scene_class_map_equal':True,'scene_mask_equal':True,
                    'reference_sha256':file_hash(Path(reference)/'reference.json')}
            write_json(destination/'parity.json',parity)
            provenance={**provenance,'source_manifest_sha256':source.manifest_hash,
                'source':source.manifest['source'],'source_preprocessing':source.manifest.get('preprocessing'),
                'recipe':{'id':'aureon-landcover-v1','window':224,'stride':168,'order':'row_major',
                          'blend':'uniform_float32_logits','neural_batch_size':1,'padding_forwards':0},
                'parity_receipt_sha256':file_hash(destination/'parity.json'),'parity_passed':True,
                'model_identity':model_identity,'memory_plan':limits,
                'publication_gate':{'requires_parent_receipt':'../streaming.json','required_parent_status':'completed'}}
            if precision_contract is not None:
                provenance.update(precision_contract=precision,execution_route=execution_route)
            artifacts=deliver(destination/'artifacts',source,destination/'categories',
                              Path(reference)/'categories',provenance,cancelled)
            stages['cog_write_reopen_seconds']=perf_counter()-tick
            tick=perf_counter(); source.verify()
            stages['input_postflight_seconds']=perf_counter()-tick
            result={'schema':'aureon.streaming_scene.v1','status':'completed','parity':parity,
                'model_identity':model_identity,'source_manifest_sha256':source.manifest_hash,
                'tile_count':total,'tiles_total':source.total,'tiles_skipped':source.skipped,
                'actual_forwards':total,'padding_forwards':0,'groups':group_count,
                'tail_tiles':total%8,'graph_groups':total//8 if execution_route=='graph_eight_eager_tail' else 0,'valid_pixels':valid_count,
                'grid':source.grid,'stages_seconds':stages,'whole_seconds_before_receipt':perf_counter()-started,
                'memory_plan':limits,'artifacts':artifacts,'candidate_logits_retained':save_logits,
                'groups_sha256':file_hash(destination/'groups.jsonl'),
                'scope':'Warm streaming delivery including disk-reference parity; reference construction/startup separate'}
            if precision_contract is not None:
                result.update(precision_contract=precision,execution_route=execution_route)
            write_json(destination/'streaming.json',result)
            return result
        except BaseException as error:
            write_json(destination/'failure.json',{'status':'failed','error_type':type(error).__name__,
                'error':str(error),'partial_artifacts_not_publishable':True})
            raise
