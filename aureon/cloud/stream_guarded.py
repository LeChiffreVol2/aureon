"""Opt-in single-pass candidate; frozen guarded-v2 and its qualification remain intact."""
from contextlib import ExitStack
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter

import numpy as np

from ..scene_pipeline.stream_io import WindowSource, CategorySink, StripeBlend, deliver
from ..scene_pipeline.streaming import budget, _output, _line
from ..scene_pipeline.core import array_hash
from ..scene_pipeline.job import file_hash, write_json
from .single_pass_source import SinglePassSource

GUARDED = 'fp32-stream-guarded-v3'


def sources():
    # Numerical qualification binds the execution module. Control-plane/client
    # edits receive their own tests, without invalidating model qualification.
    return {name:file_hash(Path(__file__).parent/name) for name in ('stream_guarded.py','single_pass_source.py')}


def verify_qualification(record):
    if (record.get('schema')!='aureon.guarded_qualification.v2' or record.get('status')!='passed'
            or record.get('profile_id')!=GUARDED or record.get('source_sha256')!=sources()
            or record.get('model_id')!='kao-terramind-b' or len(record.get('cases',[]))!=3
            or not all(c.get('logits_and_groups_exact') and c.get('cog_pixels_masks_grid_exact') for c in record['cases'])):
        raise ValueError('Guarded profile needs the exact passing three-case source qualification')
    for name, value in record['packages'].items():
        if metadata.version(name)!=value:
            raise ValueError('Qualification dependency mismatch: '+name)
    from ..scene_api.worker import source_hashes
    if source_hashes()!=record.get('runtime_source_sha256'):
        raise ValueError('Qualified Engine runtime source changed')
    return record


def run_guarded(worker, manifest, identifier, output, *, qualification):
    """One candidate pass and FP32 spatial accumulation, no eager reference pass."""
    started = perf_counter()
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Guarded job output must be new')
    worker._assert_owner()
    tick = perf_counter(); worker._guard(); guard_before = perf_counter()-tick
    output.mkdir()
    stages = {'input_validation':perf_counter()-started,'read_normalize':0.,'prediction':0.,'blend':0.,'evidence':0.}
    validation = {'policy':'guarded','numerical':{'status':'not_evaluated','reason':'no_per_job_reference'},
                  'qualification':qualification,'checks':{'input_identity':True,'model_guard':True,
                  'finite_outputs':True,'spatial_coverage':True,'artifact_reopen':True}}
    worker.torch.cuda.reset_peak_memory_stats(0)
    count = groups = 0
    with SinglePassSource(manifest) as source:
        stages['input_validation'] = perf_counter()-started
        limits = budget(source,output)
        storage_paths = [output] + ([Path('/mnt/c')] if Path('/mnt/c').is_dir() else [])
        if min(shutil.disk_usage(p).free for p in storage_paths) < 32*1024**3+limits['scratch_estimate_bytes']:
            raise ValueError('Host storage floor and conservative scratch reserve unavailable')
        with ExitStack() as stack:
            sink = stack.enter_context(CategorySink(output/'categories',source))
            ledger = stack.enter_context((output/'groups.jsonl').open('x'))
            blend = StripeBlend(source,sink)
            iterator = iter(source.groups())
            while True:
                tick = perf_counter()
                try:
                    inputs, windows = next(iterator)
                except StopIteration:
                    stages['read_normalize'] += perf_counter()-tick
                    break
                stages['read_normalize'] += perf_counter()-tick
                tick = perf_counter()
                outputs = _output(worker.candidate(inputs),len(inputs))
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 3*1024**3:
                    raise ValueError('Guarded process RSS cap exceeded')
                stages['prediction'] += perf_counter()-tick
                tick = perf_counter()
                for window, logits in zip(windows,outputs):
                    blend.add(window,logits)
                stages['blend'] += perf_counter()-tick
                tick = perf_counter()
                _line(ledger,{'windows':windows,'input_sha256':array_hash(inputs),'logits_sha256':array_hash(outputs),'count':len(inputs)})
                stages['evidence'] += perf_counter()-tick
                count += len(inputs); groups += 1
            if not count:
                raise ValueError('All-invalid scene')
            tick = perf_counter(); blend.finish(); stages['blend'] += perf_counter()-tick
            if sink.valid_pixels != source.manifest['valid_mask']['valid_pixels'] or sink.valid_pixels < 1:
                raise ValueError('Valid spatial coverage differs')
            valid_count = sink.valid_pixels
        identity = {'model_id':worker.model_id,'bundle_receipt_sha256':worker.bundle['receipt_sha256'],
                    'bundle_files_sha256':worker.bundle['files_sha256'],'precision':worker.precision,
                    'startup_receipt_sha256':worker.startup_receipt_hash}
        provenance = {'schema':'aureon.guarded_provenance.v2','job_id':identifier,
            'model_identity':identity,'source_manifest_sha256':source.manifest_hash,
            'source':source.manifest['source'],'source_preprocessing':source.manifest.get('preprocessing'),
            'validation':validation,'code_sha256':sources(),
            'recipe':{'id':'aureon-landcover-v1','window':224,'stride':168,'order':'row_major',
                      'blend':'uniform_float32_logits','neural_batch_size':1,'padding_forwards':0},
            'publication_gate':{'requires_parent_receipt':'../job.json','required_parent_status':'completed'}}
        tick = perf_counter()
        # Reuse the accepted COG writer to compare encoded pixels with the
        # candidate's category rasters. This is serialization verification,
        # explicitly NOT an independent numerical/model reference comparison.
        delivered = deliver(output/'artifacts',source,output/'categories',output/'categories',provenance)
        stages['cog_write_reopen'] = perf_counter()-tick
        tick = perf_counter(); source.verify()
        source_post = perf_counter()-tick
        guard_tick = perf_counter(); worker._guard(); guard_after = perf_counter()-guard_tick
        stages['postflight'] = perf_counter()-tick
        allocated, reserved = worker.torch.cuda.max_memory_allocated(), worker.torch.cuda.max_memory_reserved()
        if allocated >= 1024**3 or reserved >= 2*1024**3:
            raise ValueError('Guarded CUDA resource cap exceeded')
        delivered = dict(delivered, schema='aureon.cloud.provenance.v2', validation=validation,
                         reopen_validation={'status':'passed','scope':'COG serialization against candidate category rasters'})
        write_json(output/'provenance.json',delivered)
        receipt = {'schema':'aureon.guarded_scene_job.v2','status':'completed','job_id':identifier,
            'model_id':worker.model_id,'profile_id':GUARDED,'validation':validation,'model_identity':identity,
            'source_manifest_sha256':source.manifest_hash,'code_sha256':sources(), 'tile_count':count,'groups':groups,
            'actual_candidate_forwards':count,'actual_eager_reference_forwards':0,'padding_forwards':0,
            'stages_seconds':stages,'elapsed_seconds':perf_counter()-started,'identity_guard_before_seconds':guard_before,
            'validation_subspans_seconds':dict(source.timings,model_guard_before=guard_before,
                model_guard_after=guard_after,source_postflight=source_post),
            'validation_strategy':'single owned source; finite/mask checks in inference and output-coverage stream',
            'memory':{'cuda_peak_allocated_bytes':allocated,'cuda_peak_reserved_bytes':reserved},'memory_plan':limits,
            'valid_pixels':valid_count,'grid':source.grid,'groups_sha256':file_hash(output/'groups.jsonl'),
            'cost_scenario':{'THB_per_hour':1,'whole_job_THB':(perf_counter()-started)/3600},
            'scope':'FP32 guarded candidate; no per-job numerical parity or task accuracy evaluation; startup/queue excluded'}
        write_json(output/'job.json',receipt)
    worker.jobs_completed += 1
    return {name:output/relative for name,relative in {'class_map.tif':'artifacts/class_map.tif',
            'valid_mask.tif':'artifacts/valid_mask.tif','provenance.json':'provenance.json','job.json':'job.json'}.items()}


class ResidentExecutor:
    def __init__(self, bundle, work, qualification, *, execute_model=False, qualification_mode=False):
        if not execute_model or sys.platform!='linux':
            raise ValueError('Explicit Linux model execution is required')
        self.bundle, self.work = Path(bundle).resolve(), Path(work).resolve()
        if self.work.exists() or not self.work.parent.is_dir():
            raise ValueError('Worker workspace must be new')
        self.work.mkdir()
        self.qualification = qualification
        self.qualification_mode = qualification_mode
        if not qualification_mode:
            verify_qualification(qualification['record'])
        self.worker = None
        self.code = sources()

    def execute(self, manifest, job_id, *, qualification=None):
        if sources()!=self.code:
            raise ValueError('Executor source changed during session')
        if self.worker is None:
            from ..scene_api.worker import ResidentSceneWorker
            path = Path(manifest)
            with path.open('rb') as stream:
                raw = stream.read(4*1024**2+1)
            if len(raw)>4*1024**2:
                raise ValueError('Scene manifest must be bounded')
            scene = json.loads(raw)['scene_id']
            # This is only a registration key. ResidentSceneWorker immediately
            # validates the full manifest/files before model startup, and the
            # job's owned source validates again before reading pixels.
            self.worker = ResidentSceneWorker(self.bundle,{scene:manifest},self.work/'owner',
                execute_model=True,streaming=True,model_id='kao-terramind-b')
            if not self.qualification_mode:
                record = self.qualification['record']
                if (self.worker.bundle['receipt_sha256']!=record['bundle_receipt_sha256']
                        or self.worker.precision!=record['precision']
                        or self.worker.hardware['gpu_name']!=record['gpu_name']):
                    self.close()
                    raise ValueError('Live worker differs from qualified model/precision/hardware')
        declaration = qualification or {'sha256':self.qualification.get('sha256'),
            'profile_id':GUARDED,'status':'qualification_in_progress' if self.qualification_mode else 'qualified',
            'scope':'Release/profile qualification, not numerical comparison of this job'}
        return run_guarded(self.worker,manifest,job_id,self.work/job_id,qualification=declaration)

    def close(self):
        if self.worker is not None:
            result = self.worker.close()
            self.worker = None
            if not result['cleanup_passed'] or not result['full_identity_check_passed'] or result['errors']:
                raise RuntimeError('Resident cleanup failed')
            return result
