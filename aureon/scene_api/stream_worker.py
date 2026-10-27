"""Resident API path with independent eager disk reference for every scene job."""
from pathlib import Path
from time import perf_counter

from ..scene_pipeline.job import (write_json,file_hash,_snapshot,cost_report)
from ..scene_pipeline.streaming import build_reference,run_streaming
from .identity import WorkerIntegrityError


def model_identity(worker):
    return {'model_id':worker.bundle['model_id'],'bundle_receipt_sha256':worker.bundle['receipt_sha256'],
            'bundle_files_sha256':worker.bundle['files_sha256'],'precision':worker.precision}


def run_streaming_job(worker,scene_id,job_id,output_dir):
    worker._assert_owner()
    if scene_id not in worker.scenes: raise ValueError('Scene is not registered')
    if not isinstance(job_id,str) or len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid service-owned job ID')
    output=Path(output_dir).resolve()
    if output.exists() or not output.parent.is_dir(): raise ValueError('Use a fresh job directory')
    output.mkdir(); started=perf_counter()
    try:
        guard_before=worker._guard()
        _snapshot(output,worker.sources)
        manifest_path,registered_hash=worker.scenes[scene_id]
        manifest,_,digest=worker._manifest(manifest_path)
        if digest!=registered_hash:raise ValueError('Registered manifest changed')
        with (output/'input-manifest.json').open('xb') as stream:stream.write(manifest_path.read_bytes())
        identity=model_identity(worker)
        write_json(output/'worker-startup.json',worker.startup_receipt)
        write_json(output/'launch.json',{'job_id':job_id,'scene_id':scene_id,'source_sha256':worker.sources,
                    'manifest_path':str(manifest_path),
                    'model_identity':identity,'source_manifest_sha256':digest})
        before=worker.candidate.metadata()
        worker._gpu_call(worker.torch.cuda.reset_peak_memory_stats,0)
        reference=build_reference(manifest_path,worker.eager,output/'reference',model_identity=identity)
        result=run_streaming(manifest_path,worker.candidate,output/'reference',output/'delivery',
            model_identity=identity,save_logits=False,provenance={
                'schema':'aureon.streaming_provenance.v1','job_id':job_id,'scene_id':scene_id,
                'model_name':worker.model_name,'model_id':worker.model_id,
                'bundle_receipt_sha256':worker.bundle['receipt_sha256'],
                'worker_id':worker.worker_id,'worker_startup_receipt_sha256':worker.startup_receipt_hash,
                'precision_actual':worker.precision,'code_sha256':worker.sources,
                'quality':{'task_accuracy_evaluated':False},
                'job_publication_gate':{'required_job_status':'completed','receipt':'../../job.json'}})
        guard_after=worker._guard()
        import rasterio
        transform=result['grid']['transform']
        crs=rasterio.crs.CRS.from_user_input(result['grid']['crs'])
        if not crs.is_projected:raise ValueError('Projected grid required for area/cost accounting')
        factor=crs.linear_units_factor[1]
        area=abs(transform[0]*transform[4]-transform[1]*transform[3])*factor**2*result['valid_pixels']/1e6
        whole=perf_counter()-started
        receipt={'schema':'aureon.streaming_scene_job.v1','status':'completed','job_id':job_id,
            'scene_id':scene_id,'model_name':worker.model_name,'model_id':worker.model_id,
            'source_manifest_sha256':digest,'bundle_receipt_sha256':worker.bundle['receipt_sha256'],
            'precision':worker.precision,'hardware':worker.hardware,'source_sha256':worker.sources,
            'worker':{'worker_id':worker.worker_id,'model_load_count':worker.model_load_count,
                      'worker_startup_receipt_sha256':worker.startup_receipt_hash},
            'parity':result['parity'],'tile_count':result['tile_count'],'tiles_total':result['tiles_total'],
            'tiles_skipped':result['tiles_skipped'],'groups':result['groups'],'padding_forwards_in_job':0,
            'actual_candidate_forwards':result['actual_forwards'],
            'actual_eager_reference_forwards':reference['tile_count'],
            'warm_job_seconds':result['whole_seconds_before_receipt'],
            'worker_job_seconds_before_receipt':whole,'reference_seconds':reference['seconds'],
            'worker_identity_guard_seconds':guard_before+guard_after,
            'area':{'valid_pixels':result['valid_pixels'],'valid_area_km2':area},
            'cost':cost_report(whole,0,reference['seconds'],area,hourly_rate=worker.hourly_rate,currency=worker.currency),
            'memory':{'cuda_peak_allocated_bytes':worker._gpu_call(worker.torch.cuda.max_memory_allocated,0),
                      'cuda_peak_reserved_bytes':worker._gpu_call(worker.torch.cuda.max_memory_reserved,0),
                      'scope':'Owner process whole job including reference, model and graph'},
            'memory_plan':result['memory_plan'],'adapter_before_job':before,
            'adapter_after_job':worker.candidate.metadata(),'artifacts':result['artifacts'],
            'files_sha256':{name:file_hash(output/name) for name in
                ('reference/reference.json','reference/groups.jsonl','delivery/streaming.json','delivery/groups.jsonl',
                 'delivery/parity.json','delivery/artifacts/class_map.tif','delivery/artifacts/valid_mask.tif',
                 'delivery/artifacts/receipt.json')},
            'scope':{'job_kind':'resident_streaming_scene','reference':'Fresh eager disk reference for each job',
                'whole_timer':'Admission/snapshots, eager reference, candidate, disk parity, COG/reopen and guard; excludes queue/startup/final receipt',
                'max_scene_side':4096,'task_accuracy_evaluated':False,'public_api_deployed':False}}
        write_json(output/'job.json',receipt);worker.jobs_completed+=1
        downloads={public:{'path':member,'sha256':file_hash(output/member),'bytes':(output/member).stat().st_size}
            for public,member in {'class_map.tif':'delivery/artifacts/class_map.tif',
                'valid_mask.tif':'delivery/artifacts/valid_mask.tif',
                'provenance.json':'delivery/artifacts/receipt.json','job.json':'job.json'}.items()}
        return {'status':'completed','job_id':job_id,'worker_id':worker.worker_id,
                'summary':{'warm_job_seconds':result['whole_seconds_before_receipt'],
                           'worker_job_seconds':perf_counter()-started,'parity_passed':True,
                           'bitwise_equal':True,'model_load_count':worker.model_load_count},
                'download_artifacts':downloads}
    except BaseException as error:
        if getattr(worker.candidate,'state',None)=='failed' or getattr(worker.eager,'state',None)=='failed':
            worker.state='failed'
        if isinstance(error,WorkerIntegrityError) or 'parity' in str(error).lower():worker.state='failed'
        write_json(output/'failure.json',{'status':'failed','job_id':job_id,'error_type':type(error).__name__,
                  'error':str(error),'partial_artifacts_not_publishable':True})
        raise
