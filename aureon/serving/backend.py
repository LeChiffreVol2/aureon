"""Thin adapter over installed public Engine APIs; contains no model forward code."""
from importlib import metadata
import json
from pathlib import Path
import sys
import threading
import uuid
from ..cloud.control import require,file_hash,encoded,write_file
from .control import GENERATION
from .ownership import safe_error

def installed_binding():
    import aureon.api as api
    dist=metadata.distribution('aureon-inference');root=Path(dist.locate_file('aureon')).resolve(strict=True)
    direct=dist.read_text('direct_url.json')
    require(not direct or not json.loads(direct).get('dir_info',{}).get('editable'),'noneditable_package_required')
    modules={}
    for name,module in tuple(sys.modules.items()):
        if (name=='aureon' or name.startswith('aureon.')) and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();require(path.is_relative_to(root),'installed_package_required')
            modules[name]={'path':str(path),'sha256':file_hash(path)}
    return {'version':dist.version,'root':str(root),'modules':modules}

class Backend:
    def __init__(self,control,work,*,bundle=None,config=None,execute_model=False):
        require(execute_model is True and control.execution=='engine','explicit_execution_required')
        self.control=control;self.route=control.selected;self.work=Path(work);self.owner=threading.get_ident()
        self.session=None;self.last_cleanup=None;self.calls=0;self.loads=0
        self.binding=installed_binding()
        from ..api.profiles import require_profile
        require_profile(self.route['environment'])
        if self.route['task']=='generation':
            require(config is not None and bundle is None,'generation_config_required')
            from ..terramind_native.offline import inspect_config
            self.config=Path(config).resolve(strict=True);inspect_config(self.config)
            self.config_hash=file_hash(self.config)
        else:
            require(bundle is not None and config is None,'scene_bundle_required')
            from ..runtime import verify_bundle
            self.bundle=Path(bundle).resolve(strict=True);verified=verify_bundle(self.bundle)
            require(verified['model_id']==self.route['model_id'],'model_bundle_mismatch')
        write_file(self.work/'installed-startup.json',encoded(self.binding))

    def _owner(self):require(threading.get_ident()==self.owner,'wrong_worker_owner')

    def run(self,lease):
        self._owner()
        from ..api import run_model,SceneRequest,open_model_session
        job,attempt=lease['job_id'],lease['attempt'];folder=self.work/(job+'-'+str(attempt));folder.mkdir()
        manifest=Path(lease['manifest']);source_hash=file_hash(manifest)
        if self.route['task']=='generation':
            require(file_hash(self.config)==self.config_hash,'config_changed')
            if self.session is None:
                self.session=open_model_session(GENERATION,config=self.config,work=self.work/'session',environment='generation',
                  reuse='auto',denoiser=self.route['denoiser'],execute_model=True)
            recipe={'seed':lease['request']['parameters']['seed'],'token_steps':8,'diffusion_steps':10,'temperature':1.0,'top_p':0.8,'top_k':0}
            record=self.session.run(manifest,job_id=job,attempt_id='attempt-'+str(attempt),recipe=recipe)
            native=self.session.work/'jobs'/job/('attempt-'+str(attempt))/'receipt.json'
            files={'synthetic-s1grd.tif':Path(record['cog']['file']['path']),'raw.npy':Path(record['raw']['path']),
              'tokens.npy':Path(record['trace']['tokens']['path']),'provenance.json':Path(record['provenance']['path']),'native-job.json':native}
            numerical='not_evaluated';self.loads=record['model_load_count']
        else:
            if self.route['profile_id']=='fp32-full-reference-v1':
                result=run_model(SceneRequest(manifest,self.bundle,folder/'engine',self.route['model_id'],self.route['profile_id'],self.route['environment']),execute_model=True)
                self.last_cleanup=json.loads((result.output/'worker-cleanup.json').read_bytes());self.loads+=1
            else:
                if self.session is None:
                    self.session=open_model_session(self.route['model_id'],bundle=self.bundle,work=self.work/'session',profile_id=self.route['profile_id'],
                      environment=self.route['environment'],qualification=self.control.root/'engine-qualification.json',execute_model=True)
                result=self.session.run(manifest,job_id=job,attempt_id=uuid.uuid5(uuid.NAMESPACE_OID,job+':'+str(attempt)).hex)
                self.loads=self.session.metadata().get('model_load_count',0)
            files={'class_map.tif':result.class_map,'valid_mask.tif':result.valid_mask,'provenance.json':result.provenance,'native-job.json':result.job_receipt}
            numerical=result.numerical_status
        require(file_hash(manifest)==source_hash,'input_changed')
        artifacts={name:{'original_path':str(path),'sha256':file_hash(path),'bytes':Path(path).stat().st_size} for name,path in files.items()}
        receipt={'schema':'aureon.async_result.v1','status':'completed','job_id':job,'attempt':attempt,**self.route,
          'input_manifest_sha256':source_hash,'artifacts':artifacts,'numerical_status':numerical,'scientific_quality':'not_evaluated',
          'synthetic':self.route['task']=='generation','package_version':self.binding['version'],'model_load_count':self.loads,
          'native_receipts_preserved':True,'scope':'Local package API execution; separate service shutdown receipt; no hosted SLA'}
        write_file(folder/'job.json',encoded(receipt));files['job.json']=folder/'job.json';self.calls+=1
        return files

    def close(self):
        self._owner()
        close_error=None
        if self.session is not None:
            try:self.last_cleanup=self.session.close()
            except Exception as exc:close_error=safe_error(exc,'engine_close_failed')
            finally:self.session=None
        binding=installed_binding()
        torch=sys.modules.get('torch');allocated=reserved=0
        if torch is not None and torch.cuda.is_initialized():
            allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
        result={'schema':'aureon.async_worker_cleanup.v1','jobs_completed':self.calls,'model_load_count':self.loads,
          'engine_cleanup':self.last_cleanup,'installed':binding,'passed':close_error is None and allocated==reserved==0,
          'close_error':close_error,'cuda_allocated_bytes':allocated,'cuda_reserved_bytes':reserved}
        if self.last_cleanup is not None:
            result['passed']=result['passed'] and self.last_cleanup.get('passed',self.last_cleanup.get('cleanup_passed',False)) and not self.last_cleanup.get('errors')
        write_file(self.work/'cleanup.json',encoded(result));require(result['passed'],'cleanup_failed',500)
        return result
