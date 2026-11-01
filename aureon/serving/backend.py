"""Thin adapter over installed public Engine APIs; contains no model forward code."""
from __future__ import annotations
from copy import deepcopy
import gc
from typing import TYPE_CHECKING, Protocol, cast
from ..value_types import JsonObject
if TYPE_CHECKING:
    from ..api.session import EngineSession
    from ..terramind_native.session import TerraMindSession
    from ..api.generation_engine import GenerationEngine
    from ..api.terramind_encoder import TerraMindEncoder
    from ..api.best import BestTerraMindSession
    from .workflow_config import WorkflowConfig
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import threading
import uuid
from ..cloud.control import require,file_hash,encoded,write_file
from ..cloud.contracts import ServiceRoute
from .ownership import safe_error
from .contracts import ArtifactFiles, AttemptLease, BackendCleanup, InstalledBinding, PathInput, ModuleBinding, DeviceAttestation, TorchProcess


class BackendControl(Protocol):
    """Operator configuration needed by a worker; no control database access."""
    @property
    def root(self) -> Path: ...
    @property
    def selected(self) -> ServiceRoute: ...
    @property
    def execution(self) -> str: ...

def installed_binding() -> InstalledBinding:
    import aureon.api as api
    dist=metadata.distribution('aureon-inference');root=Path(cast(PathInput,dist.locate_file('aureon'))).resolve(strict=True)
    direct=dist.read_text('direct_url.json')
    require(not direct or not json.loads(direct).get('dir_info',{}).get('editable'),'noneditable_package_required')
    modules: dict[str,ModuleBinding]={}
    for name,module in tuple(sys.modules.items()):
        if (name=='aureon' or name.startswith('aureon.')) and getattr(module,'__file__',None):
            path=Path(cast(str,module.__file__)).resolve();require(path.is_relative_to(root),'installed_package_required')
            modules[name]={'path':str(path),'sha256':file_hash(path)}
    return {'version':dist.version,'root':str(root),'modules':modules}

class Backend:
    session: EngineSession | TerraMindSession | GenerationEngine | TerraMindEncoder | BestTerraMindSession | None
    last_cleanup: JsonObject | None
    device: DeviceAttestation | None

    def __init__(self, control: BackendControl, work: PathInput, *, bundle: PathInput | None = None,
                 config: PathInput | None = None, execute_model: bool = False,
                 device: str | None = None) -> None:
        require(execute_model is True and control.execution=='engine','explicit_execution_required')
        self.control=control;self.route=control.selected;self.work=Path(work);self.owner=threading.get_ident()
        self.session=None;self.last_cleanup=None;self.calls=0;self.loads=0
        self._closed = False
        self._close_result: BackendCleanup | None = None
        self.cancel_event: threading.Event | None = None
        self.workflow: WorkflowConfig | None = None
        self.binding=installed_binding()
        from ..api.profiles import require_profile
        require_profile(self.route['environment'])
        self.device=None
        if device is not None:
            from .devices import attest_device
            self.device=attest_device(device)
        if 'workflow' in self.route:
            from .workflow_config import load, binding
            require(bundle is None,'workflow_bundle_not_supported')
            self.config=Path(config) if config is not None else control.root/'workflow-config.json'
            self.workflow=load(self.config)
            require(binding(self.workflow)==self.route['workflow'],'workflow_config_mismatch')
            self.config_hash=file_hash(self.config)
        elif self.route['task']=='generation':
            require(config is not None and bundle is None,'generation_config_required')
            from ..terramind_native.offline import inspect_config
            self.config=Path(cast(PathInput,config)).resolve(strict=True)
            inspected=inspect_config(self.config)
            require(inspected.get('model_id','terramind-v1-base')==self.route['model_id'],'model_config_mismatch')
            self.config_hash=file_hash(self.config)
        else:
            require(bundle is not None and config is None,'scene_bundle_required')
            from ..runtime import verify_bundle
            self.bundle=Path(cast(PathInput,bundle)).resolve(strict=True);verified=verify_bundle(self.bundle)
            require(verified['model_id']==self.route['model_id'],'model_bundle_mismatch')
        write_file(self.work/'installed-startup.json',encoded(self.binding))

    def _owner(self) -> None:require(threading.get_ident()==self.owner,'wrong_worker_owner')

    def _workflow_run(self, manifest: Path, folder: Path, seed: int) -> ArtifactFiles:
        """Borrow the configured resident engine; native helpers own request I/O."""
        configured=self.workflow;assert configured is not None
        require(file_hash(self.config)==self.config_hash,'config_changed')
        options=configured['options'];limits=configured['limits']
        value=json.loads(manifest.read_bytes())
        if configured['task']=='generation_scene':
            from ..api.best import BestTerraMindSession
            from ..api.best_workflows import BestSceneRequest, run_best_scene
            from ..api.selected_scene import native_config
            from ..api.spatial import InvalidTilePolicy
            if configured['model_id']!='terramind-v1-base':
                raise ValueError('This selected service executes only original TerraMind base; no candidate fallback')
            if self.session is None:
                native=native_config(manifest.parent/value['image']['path'],manifest.parent/value['mask']['path'],
                                     configured,self.work/'selected-admission')
                self.session=BestTerraMindSession(native,self.work/'session',execute_model=True)
            selected=cast(BestTerraMindSession,self.session)
            request=BestSceneRequest(image=manifest.parent/value['image']['path'],valid_mask=manifest.parent/value['mask']['path'],
                output=folder/'engine',job_id=folder.name,seed=seed,stride=cast(int,options['stride']),
                invalid_policy=cast(InvalidTilePolicy,options['invalid_policy']),max_buffer_bytes=limits['max_buffer_bytes'],
                max_decode_bytes=limits['max_decode_bytes'],max_workspace_bytes=limits['max_scratch_bytes'],
                max_output_bytes=limits['max_output_bytes'],min_free_bytes=limits['min_free_bytes'])
            run_best_scene(selected,request,execute_model=True,cancel_event=self.cancel_event)
            destination=folder/'engine'
            files={'synthetic-s1grd.tif':destination/'synthetic-s1grd.tif','provenance.json':destination/'provenance.json',
                   'tiles.jsonl':destination/'tiles.jsonl','preflight.json':destination/'preflight.json','native-job.json':destination/'completion.json'}
            self.loads=cast(int,selected.metadata()['model_load_count'])
            return files
        else:
            raise ValueError('Selected original route does not enable another model executor')

    def run(self, lease: AttemptLease) -> ArtifactFiles:
        self._owner()
        require(not self._closed, 'backend_closed', 409)
        from ..api import run_model,SceneRequest,open_model_session
        job,attempt=lease['job_id'],lease['attempt'];folder=self.work/(job+'-'+str(attempt));folder.mkdir()
        manifest=Path(lease['manifest']);source_hash=file_hash(manifest)
        if self.workflow is not None:
            files=self._workflow_run(manifest,folder,lease['request']['parameters']['seed']);numerical='not_evaluated'
        elif self.route['task']=='generation':
            require(file_hash(self.config)==self.config_hash,'config_changed')
            require(self.route['model_id']=='terramind-v1-base','selected_original_base_only')
            if self.session is None:
                # Let the public factory select the preserved best core. Explicit
                # expert/non-base recipes retain their own dispatch contract.
                selection = ({'execution_profile':'recommended'}
                    if self.route['model_id']=='terramind-v1-base'
                    and self.route.get('reuse')=='auto' and self.route['denoiser']=='graph'
                    else {'execution_profile':'reference'}
                    if self.route['model_id']=='terramind-v1-base'
                    and self.route.get('reuse')=='off' and self.route['denoiser']=='native'
                    else {'reuse':self.route.get('reuse') or 'off','denoiser':cast(str,self.route['denoiser'])})
                self.session=cast('TerraMindSession', open_model_session(self.route['model_id'],config=self.config,work=self.work/'session',environment='generation',
                  execute_model=True,**selection))
            recipe={'seed':lease['request']['parameters']['seed'],'token_steps':8,'diffusion_steps':10,'temperature':1.0,'top_p':0.8,'top_k':0}
            record=cast('TerraMindSession',self.session).run(manifest,job_id=job,attempt_id='attempt-'+str(attempt),recipe=recipe)
            native=cast('TerraMindSession',self.session).work/'jobs'/job/('attempt-'+str(attempt))/'receipt.json'
            files={'synthetic-s1grd.tif':Path(record['cog']['file']['path']),'raw.npy':Path(record['raw']['path']),
              'tokens.npy':Path(record['trace']['tokens']['path']),'provenance.json':Path(record['provenance']['path']),'native-job.json':native}
            numerical='not_evaluated';self.loads=record['model_load_count']
        else:
            if self.route['profile_id']=='fp32-full-reference-v1':
                result=run_model(SceneRequest(manifest,self.bundle,folder/'engine',self.route['model_id'],self.route['profile_id'],self.route['environment']),execute_model=True)
                self.last_cleanup=json.loads((result.output/'worker-cleanup.json').read_bytes());self.loads+=1
            else:
                if self.session is None:
                    self.session=cast('EngineSession', open_model_session(self.route['model_id'],bundle=self.bundle,work=self.work/'session',profile_id=self.route['profile_id'],
                      environment=self.route['environment'],qualification=self.control.root/'engine-qualification.json',execute_model=True))
                result=cast('EngineSession',self.session).run(manifest,job_id=job,attempt_id=uuid.uuid5(uuid.NAMESPACE_OID,job+':'+str(attempt)).hex)
                self.loads=cast('EngineSession',self.session).metadata().get('model_load_count',0)
            files={'class_map.tif':result.class_map,'valid_mask.tif':result.valid_mask,'provenance.json':result.provenance,'native-job.json':result.job_receipt}
            numerical=result.numerical_status
        require(file_hash(manifest)==source_hash,'input_changed')
        artifacts={name:{'original_path':str(path),'sha256':file_hash(path),'bytes':Path(path).stat().st_size} for name,path in files.items()}
        receipt={'schema':'aureon.async_result.v1','status':'completed','job_id':job,'attempt':attempt,**self.route,
          'input_manifest_sha256':source_hash,'artifacts':artifacts,'numerical_status':numerical,'scientific_quality':'not_evaluated',
          'synthetic':self.route['task'] in {'generation','generation_scene'},'package_version':self.binding['version'],'model_load_count':self.loads,
          'native_receipts_preserved':True,'device':self.device,'scope':'Local package API execution; separate service shutdown receipt; no hosted SLA'}
        if self.workflow is not None:
            receipt['workflow_config_sha256']=self.route['workflow']['config_sha256'];receipt['parameters']=lease['request']['parameters']
        write_file(folder/'job.json',encoded(receipt));files['job.json']=folder/'job.json';self.calls+=1
        return files

    def close(self) -> BackendCleanup:
        """Close once on the owner thread; replay detached receipts, never execution."""
        self._owner()
        if self._close_result is not None:
            require(self._close_result['passed'], 'cleanup_failed', 500)
            return deepcopy(self._close_result)
        self._closed = True
        close_error=None
        features=self.workflow is not None and self.workflow['task']=='features'
        if self.session is not None:
            try:
                self.last_cleanup=self.session.close()
                if features:
                    state=cast('TerraMindEncoder',self.session).state
                    self.last_cleanup={'schema':'aureon.encoder_owner_cleanup.v1','encoder_state':state,
                                       'passed':state=='closed','native_close_returns_receipt':False}
            except Exception as exc:close_error=safe_error(exc,'engine_close_failed')
            finally:self.session=None
        try:
            binding=installed_binding()
        except Exception as exc:
            # Cleanup has already run. Record an inspection failure without
            # treating the startup binding as a successful fresh attestation.
            binding=self.binding
            close_error=close_error or safe_error(exc,'installed_binding_failed')
        torch=cast(TorchProcess | None,sys.modules.get('torch'));allocated=reserved=0
        if features:gc.collect()
        try:
            if torch is not None and torch.cuda.is_initialized():
                if features:
                    torch.cuda.synchronize()
                    clear=getattr(getattr(torch,'_C',None),'_cuda_clearCublasWorkspaces',None)
                    if callable(clear):clear()
                    torch.cuda.empty_cache()
                    if self.last_cleanup is not None:
                        self.last_cleanup.update(gc_collected=True,cublas_workspaces_cleared=callable(clear),cuda_cache_emptied=True)
                allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
        except Exception as exc:
            close_error=close_error or safe_error(exc,'cuda_cleanup_failed');allocated=reserved=-1
        result: BackendCleanup={'schema':'aureon.async_worker_cleanup.v1','jobs_completed':self.calls,'model_load_count':self.loads,
          'engine_cleanup':self.last_cleanup,'installed':binding,'passed':close_error is None and allocated==reserved==0,
          'close_error':close_error,'cuda_allocated_bytes':allocated,'cuda_reserved_bytes':reserved}
        if self.last_cleanup is not None:
            result['passed']=result['passed'] and self.last_cleanup.get('passed',self.last_cleanup.get('cleanup_passed',False)) and not self.last_cleanup.get('errors')
        write_file(self.work/'cleanup.json',encoded(result))
        self._close_result=deepcopy(result)
        require(result['passed'],'cleanup_failed',500)
        return deepcopy(result)
