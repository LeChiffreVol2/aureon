"""Pinned native COP-GEN, explicit FP32 compatibility and request-owned RNG."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch

from .contracts import INPUT, TARGET, CopgenPlan, digest
from .profiles import validate_profile
from ..terramind_native.rng import rng_capsule


def file_identity(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':sha,'bytes':path.stat().st_size}


def verify_assets(source: Path, assets: Path) -> dict[str, Any]:
    pins=json.loads(Path(__file__).with_name('pins.json').read_text())
    bound: dict[str, Any] = {'code_revision':pins['code_revision'],'sources':{},'assets':{},'licenses':{'code':'Apache-2.0','weights':'CC-BY-4.0'}}
    for group, root in (('source',source),('assets',assets)):
        for name, expected in pins[group].items():
            path=(root/name).resolve(strict=True)
            if not path.is_relative_to(root.resolve(strict=True)):
                raise ValueError('Pinned asset escaped its explicit root')
            actual=file_identity(path)
            if any(actual[key]!=expected[key] for key in ('bytes','sha256')):
                raise ValueError('Pinned source/checkpoint identity differs: '+name)
            bound['sources' if group=='source' else 'assets'][name]={**expected,**actual}
    extra={p.relative_to(source).as_posix() for p in source.rglob('*.py')}-set(pins['source'])
    if extra: raise ValueError('Unpinned Python source under the native import root')
    return bound


def import_pinned(source: Path) -> tuple[ModuleType, ModuleType, Any]:
    root=source.resolve(strict=True)
    for name in ('libs','libs.autoencoder','libs.copgen','utils','datasets','dpm_solver_pp'):
        cached=sys.modules.get(name)
        if cached is not None:
            paths=list(getattr(cached,'__path__',[])) or [getattr(cached,'__file__','')]
            if not paths or any(not Path(path).resolve().is_relative_to(root) for path in paths):
                raise RuntimeError('Native import namespace is already owned by another source')
    sys.path.insert(0,str(root))
    try:
        native=importlib.import_module('libs.copgen')
        # Only this previously hashed local Python configuration is executable.
        spec=importlib.util.spec_from_file_location('_aureon_pinned_copgen_config',root/'configs/copgen/discrete/cop_gen_base.py')
        if spec is None or spec.loader is None: raise RuntimeError('Pinned config loader unavailable')
        config_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(config_module)
        post_spec=importlib.util.spec_from_file_location('_aureon_pinned_copgen_prepost',root/'ddm/pre_post_process_data.py')
        if post_spec is None or post_spec.loader is None: raise RuntimeError('Pinned preprocessing loader unavailable')
        preprocess=importlib.util.module_from_spec(post_spec);post_spec.loader.exec_module(preprocess)
        return native,preprocess,config_module.get_config()
    finally:
        sys.path.remove(str(root))


class _FP32Torch:
    """Module-local view; no mutation of the global torch autocast functions."""
    def __init__(self, torch: ModuleType, receipt: dict[str, Any]) -> None:
        self._torch=torch;self._receipt=receipt
        self.amp=self
    def __getattr__(self,name: str) -> Any:return getattr(self._torch,name)
    def autocast(self,device_type: str,**kwargs: Any) -> Any:
        self._receipt['inner_autocast_disabled']+=1
        return self._torch.autocast(device_type=device_type,**dict(kwargs,enabled=False))
    def load(self,*args: Any,**kwargs: Any) -> Any:
        self._receipt['restricted_checkpoint_loads']+=1
        return self._torch.load(*args,**dict(kwargs,weights_only=True))


@contextmanager
def fp32_scope(native: ModuleType) -> Iterator[dict[str, Any]]:
    import torch
    ae=importlib.import_module('libs.autoencoder')
    old={'precision':torch.get_float32_matmul_precision(),'tf32':torch.backends.cuda.matmul.allow_tf32,
         'cudnn_tf32':torch.backends.cudnn.allow_tf32,'benchmark':torch.backends.cudnn.benchmark,
         'threads':torch.get_num_threads(),'dtype':torch.get_default_dtype(),
         'sdp':(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]  # PyTorch backend query annotations are incomplete.
                torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled())}  # type: ignore[no-untyped-call]
    receipt={'inner_autocast_disabled':0,'restricted_checkpoint_loads':0,'restored':False,
             'profile':'FP32 highest; TF32/autocast off; math SDPA; native draw/solver order unchanged'}
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
        with patch.object(native,'torch',_FP32Torch(torch,receipt)),patch.object(ae,'torch',_FP32Torch(torch,receipt)),torch.no_grad(),torch.autocast('cuda',enabled=False):
            yield receipt
    finally:
        torch.set_float32_matmul_precision(old['precision']);torch.backends.cuda.matmul.allow_tf32=old['tf32']
        torch.backends.cudnn.allow_tf32=old['cudnn_tf32'];torch.backends.cudnn.benchmark=old['benchmark']
        torch.set_num_threads(old['threads']);torch.set_default_dtype(old['dtype'])
        for setter,value in zip((torch.backends.cuda.enable_flash_sdp,torch.backends.cuda.enable_mem_efficient_sdp,
                                 torch.backends.cuda.enable_math_sdp,torch.backends.cuda.enable_cudnn_sdp),old['sdp']): setter(value)
        receipt['restored']=True


class NativeRuntime:
    def __init__(self, source: Path, assets: Path, *, dependency_profile: str='torch260') -> None:
        self.model: Any=None
        self._owns_cuda=False
        self._dependency_profile=dependency_profile
        try:
            self._initialize(source,assets)
        except BaseException as primary:
            # Unwound upstream constructor frames can retain temporary CUDA VAEs
            # that never reached self.model.autoencoders. Keep the traceback's
            # location/error, but release its local tensor references before close.
            traceback.clear_frames(primary.__traceback__)
            try:
                cleanup_receipt=self.close()
                primary.add_note('Partial COP-GEN construction cleanup: '+repr(cleanup_receipt))
            except BaseException as cleanup_error:
                primary.add_note('Partial COP-GEN construction cleanup failed: '+repr(cleanup_error))
            raise

    def _initialize(self, source: Path, assets: Path) -> None:
        import torch
        profile=validate_profile(self._dependency_profile,torch_version=torch.__version__,cuda_build=torch.version.cuda)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Pinned profile requires exactly one visible CUDA GPU')
        if torch.cuda.memory_allocated() or torch.cuda.memory_reserved():
            raise RuntimeError('Native COP-GEN needs an empty allocator in its owned process')
        # Only successful profile/device/empty-allocator admission permits global
        # CUDA cleanup. Rejected callers may already own a foreign model/context.
        self._owns_cuda=True
        self.identity=verify_assets(source,assets)
        self.identity['dependency_profile']=profile
        with rng_capsule(0,devices=(0,)) as import_rng:
            self.native,self.preprocess,config=import_pinned(source)
        self.identity['import_rng']=import_rng
        # Keep all twelve joint transformer modalities. Only VAE residency changes;
        # both native reference and adapter use this exact disclosed compatibility.
        config.autoencoders={name:config.autoencoders[name] for name in (INPUT,TARGET)}
        for name in (INPUT,TARGET):
            relative=Path(config.autoencoders[name].pretrained_path).relative_to('models/vae')
            config.autoencoders[name].pretrained_path=str(assets/relative)
        self.config=config
        with rng_capsule(0,devices=(0,)) as rng,fp32_scope(self.native) as precision:
            # get_nnet lazily imports libs.copgen_multi_post_ln under the pinned root.
            sys.path.insert(0,str(source.resolve()))
            try:
                # Retain the partially initialized owner if native __init__ fails.
                self._construct_model(self.native.CopgenModel,str(assets/'500000_nnet_ema.pth'),config,device='cuda',seed=None)
            finally:sys.path.remove(str(source.resolve()))
        self.identity['construction']={'rng':rng,'precision':precision,'resident_vaes':[INPUT,TARGET],
                                       'joint_modalities':list(self.model.modality_names),'seeded_after_construction_per_request':True}
        modules=[self.model.nnet,*self.model.autoencoders.values()]
        for module in modules:
            module.eval().requires_grad_(False)
            if any(t.is_floating_point() and t.dtype!=torch.float32 for t in module.state_dict().values()):
                raise ValueError('Loaded COP-GEN state is not entirely FP32')
        self.versions=self._versions()
        self.identity['parameter_counts']=[sum(p.numel() for p in module.parameters()) for module in modules]
        self.identity['dependencies']=dict(profile['dependencies'])
        if verify_assets(source,assets)['assets']!=self.identity['assets']:raise ValueError('Assets changed during loading')
        torch.cuda.synchronize()

    def _construct_model(self, cls: Any, *args: Any, **kwargs: Any) -> None:
        self.model=cls.__new__(cls)
        cls.__init__(self.model,*args,**kwargs)

    def _versions(self) -> tuple[Any,...]:
        return tuple((id(t),t._version,t.data_ptr(),tuple(t.shape),t.dtype,t.device)
                     for module in [self.model.nnet,*self.model.autoencoders.values()]
                     for t in list(module.parameters())+list(module.buffers()))

    def generate(self, image: Any, plan: CopgenPlan) -> dict[str, Any]:
        import numpy as np
        import torch
        if self.model is None or self._versions()!=self.versions:raise RuntimeError('Resident model state changed or closed')
        trace={'nnet_calls':0,'vae_encoder_calls':0,'vae_decoder_calls':0,'floating_tensors_checked':0}
        def tensors(value: Any) -> Iterator[Any]:
            if torch.is_tensor(value):yield value
            elif isinstance(value,(tuple,list)):
                for item in value:yield from tensors(item)
            elif isinstance(value,dict):
                for item in value.values():yield from tensors(item)
        def observer(name: str) -> Any:
            def check(module: Any,args: Any,result: Any) -> None:
                trace[name]+=1
                if torch.is_autocast_enabled():raise RuntimeError('Inner autocast escaped the FP32 policy')
                for tensor in tensors((args,result)):
                    if tensor.is_floating_point():
                        trace['floating_tensors_checked']+=1
                        if tensor.dtype!=torch.float32:raise RuntimeError('Non-FP32 native stage tensor')
            return check
        handles=[self.model.nnet.register_forward_hook(observer('nnet_calls')),
                 self.model.autoencoders[INPUT].encoder.register_forward_hook(observer('vae_encoder_calls')),
                 self.model.autoencoders[TARGET].decoder.register_forward_hook(observer('vae_decoder_calls'))]
        try:
            with rng_capsule(plan.seed,devices=(0,)) as rng,fp32_scope(self.native) as precision:
                normalized=self.preprocess.pre_process_data(image,INPUT,self.config.dataset)
                if normalized.dtype!=np.float32:raise ValueError('Preprocessing changed precision')
                condition=torch.from_numpy(normalized).unsqueeze(0).to('cuda')
                decoded,latents=self.model.generate(TARGET,conditions={INPUT:condition},n_samples=1,
                                                   sample_steps=plan.sample_steps,return_latents=True)
                values=self.preprocess.post_process_data(decoded,TARGET,self.config.dataset)
                if any(t.dtype!=torch.float32 for t in (decoded,values,latents[TARGET])):
                    raise ValueError('Native output differs from FP32')
                output={'array':values.detach().cpu().numpy()[0].copy(),
                        'decoded':decoded.detach().cpu().numpy()[0].copy(),
                        'latent':latents[TARGET].detach().cpu().numpy()[0].copy()}
                del condition,decoded,latents,values
                torch.cuda.synchronize()
            if not rng['restored'] or not precision['restored'] or self._versions()!=self.versions:
                raise RuntimeError('Native request state was not restored')
            if output['array'].shape!=(2,192,192) or not np.isfinite(output['array']).all() or np.any(output['array']<=0):
                raise ValueError('Invalid generated linear S1RTC output')
            return {**output,'rng':rng,'precision':precision,'native_trace':trace}
        finally:
            for handle in handles:handle.remove()

    def close(self) -> dict[str, Any]:
        if not self._owns_cuda:
            return {'cuda_owned':False,'cleanup_performed':False,'cuda_memory':'not_measured'}
        # Relinquish once, including cleanup failures. A later close must never
        # clean a context subsequently acquired by another owner.
        self._owns_cuda=False
        self.model=None;gc.collect()
        torch=sys.modules.get('torch')
        if torch is None or not torch.cuda.is_initialized():
            return {'allocated_bytes':0,'reserved_bytes':0,'cuda_initialized':False}
        torch.cuda.synchronize()
        clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
        if clear is not None:clear()
        torch.cuda.empty_cache()
        return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
