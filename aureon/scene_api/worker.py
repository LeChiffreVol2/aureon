"""Explicit resident FP32 worker; the async service owns its single thread.

No model imports at module import. Every job retains eager numerical validation
and the existing scene-job evidence format. Model/code hashes move to startup
and shutdown, with pinned snapshots and cheap fail-closed lifecycle guards.
"""
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as packages
import gc
import importlib
import os
from pathlib import Path
import platform
import sys
import threading
from time import perf_counter
import uuid
import weakref

import numpy as np

from ..catalog import CLASS_NAMES
from ..runtime import verify_bundle
from ..bundle_storage import snapshot_bundle, finish_bundle_storage
from ..scene_pipeline.core import array_hash, prepare_tiles, blend_logits, compare_logits, cost_report
from ..scene_pipeline.predictor import FullEightEagerTailAdapter, SequentialEagerAdapter
from ..scene_pipeline.job import (ROOT, file_hash, write_json, load_manifest, _read_source,
    _sources, _snapshot, _save_array, _reference_scene, _valid_area, _verify_input_files,
    _close_resources)
from ..scene_pipeline.georaster import write_segmentation_geotiffs
from .identity import FileSeal, TensorSeal, WorkerIntegrityError


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    hashes = _sources()
    hashes['aureon/bundle_storage.py'] = file_hash(ROOT/'aureon/bundle_storage.py')
    hashes.update({p.relative_to(ROOT).as_posix(): file_hash(p)
                   for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')})
    return hashes


def _prime_model_imports():
    # Import hooks can retain the frame that first imports TerraTorch. Import
    # before the predictor constructor owns model tensors, so that such a
    # retained frame cannot keep the model/CUDA allocation alive at shutdown.
    importlib.import_module('terratorch.models')


class ResidentSceneWorker:
    def __init__(self, bundle_dir, scenes, evidence_dir, *, execute_model=False,
                 hourly_rate=1.0, currency='THB', streaming=False, model_id='kao-terramind-b'):
        if execute_model is not True or sys.platform != 'linux':
            raise ValueError('Explicit model execution in the isolated Linux runtime is required')
        cost_report(0, 0, 0, 0, hourly_rate=hourly_rate, currency=currency)
        if type(streaming) is not bool or model_id not in ('kao-terramind-a','kao-terramind-b','kao-terramind-c'):
            raise ValueError('Select an explicit KAO-TerraMind A/B/C model and boolean streaming mode')
        if model_id!='kao-terramind-b' and not streaming:
            raise ValueError('A/C use the explicit common streaming serving contract')
        self.streaming=streaming
        self.model_id=model_id
        self.model_name='KAO-TerraMind '+model_id[-1].upper()
        if not isinstance(scenes, dict) or not 1 <= len(scenes) <= 16:
            raise ValueError('Register one to sixteen explicit local scene manifests')
        self.owner = threading.get_ident()
        self.worker_id = uuid.uuid4().hex
        self.state = 'initializing'
        self.model_load_count = self.jobs_completed = 0
        self.reference = self.graph = self.candidate = self.eager = None
        self.hourly_rate, self.currency = hourly_rate, currency
        self.folder = Path(evidence_dir).resolve()
        self.folder.mkdir(exist_ok=False)
        self.scenes = {}
        self.started_utc = utc_now()
        startup_start = start = perf_counter()
        try:
            self.sources = source_hashes()
            _snapshot(self.folder, self.sources)
            for scene_id, path in scenes.items():
                manifest, _, digest = self._manifest(path)
                if scene_id != manifest.get('scene_id'):
                    raise ValueError('Registered scene ID must match its manifest')
                self.scenes[scene_id] = (Path(path).resolve(), digest)
            # Own a pinned bundle copy. It is never exposed through HTTP.
            self.bundle_dir = self.folder/'bundle'
            self.original_bundle_dir = Path(bundle_dir).absolute()
            self.bundle, self.bundle_storage = snapshot_bundle(
                self.original_bundle_dir, self.bundle_dir, model_id=model_id)
            preparation_seconds = perf_counter()-start
            start = perf_counter()
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_HUB_DISABLE_TELEMETRY='1', NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='1')
            from ..process_memory import configure_process_memory
            memory_policy = configure_process_memory('disable')
            import torch
            from ..cuda_graph import _precision
            from ..reference import PyTorchPredictor
            from ..graph_pipeline import FullGroupGraphPipeline
            self.torch, self._precision = torch, _precision
            torch.set_num_threads(1)
            self._configure_precision()
            _prime_model_imports()
            runtime_seconds = perf_counter()-start
            start = perf_counter()
            self.reference = PyTorchPredictor(self.bundle_dir, device='cuda:0', dtype='float32')
            self.model_load_count = 1
            self._configure_precision()  # TerraTorch import may change globals.
            self.reference.synchronize()
            self.precision = _precision(torch)
            load_seconds = perf_counter()-start
            start = perf_counter()
            loaded_identity = self._attest_loaded_model()
            loaded_identity_seconds = perf_counter()-start
            self.eager = SequentialEagerAdapter(self.reference)
            start = perf_counter()
            first_path, _ = next(iter(self.scenes.values()))
            if self.streaming:
                from ..scene_pipeline.stream_io import WindowSource
                with WindowSource(first_path) as source:
                    iterator=source.groups()
                    try: tiles=next(iterator)[0].copy()
                    except StopIteration: tiles=np.empty((0,12,224,224),dtype=np.float32)
                    finally: iterator.close()
            else:
                manifest, members, _ = load_manifest(first_path)
                source = _read_source(manifest, members)
                tiles = prepare_tiles(source.image, source.valid_mask).values[:8]
            if not len(tiles):
                raise ValueError('Startup scene has no valid tiles')
            baseline = self.eager(tiles)
            (self.folder/'graph').mkdir()
            self.graph = FullGroupGraphPipeline.from_reference(self.reference, self.folder/'graph')
            self.candidate = FullEightEagerTailAdapter(self.graph, self.reference)
            first_parity = compare_logits(baseline, self.candidate(tiles))
            reversed_parity = compare_logits(baseline[::-1], self.candidate(tiles[::-1]))
            if not first_parity['bitwise_equal'] or not reversed_parity['bitwise_equal']:
                raise WorkerIntegrityError('Resident startup graph/freshness parity failed')
            # Exercise every reachable partial count before advertising readiness.
            tail_parity = []
            for count in range(1, min(7, len(tiles))+1):
                checked_tail = compare_logits(baseline[:count], self.candidate(tiles[:count]))
                checked_reverse = compare_logits(baseline[:count][::-1], self.candidate(tiles[:count][::-1]))
                if not checked_tail['bitwise_equal'] or not checked_reverse['bitwise_equal']:
                    raise WorkerIntegrityError('Resident startup eager-tail parity/freshness failed')
                tail_parity.append({'real_tiles': count, 'normal': checked_tail, 'reverse': checked_reverse})
            capture_seconds = perf_counter()-start
            # Full verification after loading/capture, before trusting cheap guards.
            checked = verify_bundle(self.bundle_dir)
            if (checked['files_sha256'] != self.bundle['files_sha256']
                    or checked['receipt_sha256'] != self.bundle['receipt_sha256']
                    or source_hashes() != self.sources):
                raise WorkerIntegrityError('Startup identity changed')
            self.file_seal = FileSeal([self.bundle_dir/name for name in
                ('config.json', 'pytorch_model.bin', 'bundle_receipt.json')]
                + [ROOT/name for name in self.sources])
            self.tensor_seal = TensorSeal(self.reference.model)
            self.hardware = {'platform': platform.platform(), 'python': platform.python_version(),
                'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'packages': {key: packages.version(key) for key in ('terratorch','rasterio','numpy')},
                'process_memory': memory_policy}
            final_identity_seconds = perf_counter()-start-capture_seconds
            receipt = {'schema': 'aureon.resident_scene_worker.v1', 'worker_id': self.worker_id,
                'status': 'ready', 'pid': os.getpid(), 'started_utc': self.started_utc,
                'ready_utc': utc_now(), 'model_name': self.model_name, 'model_load_count': 1,
                'model_id':self.model_id,'execution_mode':'streaming' if self.streaming else 'materialized',
                'bundle_receipt_sha256': self.bundle['receipt_sha256'],
                'bundle_files_sha256': self.bundle['files_sha256'], 'code_sha256': self.sources,
                'bundle_storage': self.bundle_storage,
                'registered_scenes': {key: digest for key, (_, digest) in self.scenes.items()},
                'precision': self.precision, 'hardware': self.hardware,
                'loaded_tensor_identity': loaded_identity, 'model_runtime_imports_primed': True,
                'startup_seconds': {'admission_snapshot': preparation_seconds,
                    'runtime_import_config': runtime_seconds, 'model_load': load_seconds,
                    'graph_capture_eager_validation': capture_seconds,
                    'loaded_tensor_value_attestation': loaded_identity_seconds,
                    'final_identity_seals_hardware': final_identity_seconds},
                'startup_observed_seconds_before_receipt': perf_counter()-startup_start,
                'startup_timer_scope': 'Constructor preparation through final identity/seals/hardware; excludes final startup receipt serialization/hash and HTTP listener startup',
                'startup_parity': first_parity, 'reversed_input_probe': reversed_parity,
                'startup_tail_parity': tail_parity, 'padding_policy': 'No padding forwards; explicit graph-eight/eager-tail dispatch',
                'file_seal': self.file_seal.records(),
                'identity_policy': {'full_hash': 'startup and controlled shutdown',
                    'per_job': 'file metadata, tensor storage/shape/dtype/version/eval mode and precision guards; all input evidence remains hashed',
                    'scope': 'One controlled local process with owned bundle snapshot; not remote attestation or defense against privileged same-user tampering',
                    'model_mutation_endpoints': False, 'full_checkpoint_hash_per_job': False}}
            write_json(self.folder/'startup.json', receipt)
            self.startup_receipt_hash = file_hash(self.folder/'startup.json')
            self.startup_receipt = receipt
            self.state = 'ready'
        except BaseException as error:
            self.state = 'failed'
            cleanup_errors = self._release_resources()
            try:
                write_json(self.folder/'startup-failure.json', {'error_type': type(error).__name__,
                    'error': str(error), 'cleanup_errors': cleanup_errors})
            except Exception:
                pass  # Preserve the primary failure even if the evidence volume is full.
            raise

    def _manifest(self,path):
        return load_manifest(path,max_evidence_bytes=2*1024**3) if self.streaming else load_manifest(path)

    def _configure_precision(self):
        self.torch.set_float32_matmul_precision('highest')
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.backends.cudnn.allow_tf32 = False
        self.torch.backends.cuda.enable_cudnn_sdp(False)
        self.torch.backends.cudnn.benchmark = False

    def _attest_loaded_model(self):
        self._assert_owner()
        from ..vllm_bundle import _tensor_record, _manifest_sha256
        actual = {name: _tensor_record(value) for name, value in self.reference.model.state_dict().items()}
        expected = self.bundle['receipt']['tensor_manifest']
        if actual != expected:
            raise WorkerIntegrityError('Loaded model tensor values differ from the verified bundle')
        return {'tensor_count': len(actual), 'loaded_tensor_manifest_sha256': _manifest_sha256(actual),
                'matched_bundle_tensor_values': True,
                'scope': 'Owner-thread hash of every live tensor value; not remote attestation'}

    def _assert_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('All worker operations must stay on its owner thread')

    def _guard(self):
        self._assert_owner()
        start = perf_counter()
        try:
            if self.state != 'ready':
                raise WorkerIntegrityError('Resident worker is not ready')
            self.file_seal.verify()
            self.tensor_seal.verify()
            if self._precision(self.torch) != self.precision:
                raise WorkerIntegrityError('Resident precision settings changed')
        except BaseException:
            self.state = 'failed'
            raise
        return perf_counter()-start

    def metadata(self):
        self._assert_owner()
        value = {'worker_id': self.worker_id, 'state': self.state,
            'model_name': getattr(self,'model_name','KAO-TerraMind B'), 'model_load_count': self.model_load_count,
            'execution_mode':'streaming' if getattr(self,'streaming',False) else 'materialized',
            'jobs_completed': self.jobs_completed,
            'startup_receipt_sha256': getattr(self, 'startup_receipt_hash', None),
            'precision': deepcopy(getattr(self, 'precision', {})),
            'full_checkpoint_hash_per_job': False}
        if self.state == 'ready':
            # Called by the owner after initialization/job completion, never by
            # an HTTP thread or concurrently with a forward. No allocator reset.
            value['settled_cuda_memory'] = {
                'allocated_bytes': self._gpu_call(self.torch.cuda.memory_allocated, 0),
                'reserved_bytes': self._gpu_call(self.torch.cuda.memory_reserved, 0),
                'after_completed_jobs': self.jobs_completed,
                'scope': 'Owner-thread snapshot after initialization or job return; not live device-wide usage'}
        return value

    def _gpu_call(self, function, *args):
        try:
            return function(*args)
        except BaseException:
            self.state = 'failed'
            raise

    def run_job(self, scene_id, job_id, output_dir):
        if self.streaming:
            from .stream_worker import run_streaming_job
            return run_streaming_job(self,scene_id,job_id,output_dir)
        self._assert_owner()
        if scene_id not in self.scenes:
            raise ValueError('Scene is not registered')
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('Invalid service-owned job ID')
        output = Path(output_dir).resolve()
        if output.exists() or not output.parent.is_dir():
            raise ValueError('Use a new job directory under an existing parent')
        output.mkdir()
        whole_start = perf_counter()
        started_utc = utc_now()
        stages, validation = {}, {}
        try:
            guard_before = self._guard()
            _snapshot(output, self.sources)
            manifest_path, registered_hash = self.scenes[scene_id]
            manifest, members, manifest_hash = load_manifest(manifest_path)
            if manifest_hash != registered_hash:
                raise ValueError('Registered input manifest changed; restart with explicit admission')
            with (output/'input-manifest.json').open('xb') as stream:
                stream.write(manifest_path.read_bytes())
            if file_hash(output/'input-manifest.json') != manifest_hash:
                raise ValueError('Input manifest changed during snapshot')
            write_json(output/'launch.json', {'schema': 'aureon.scene_job_launch.v1',
                'job_id': job_id, 'started_utc': started_utc, 'worker_id': self.worker_id,
                'source_sha256': self.sources, 'manifest_sha256': manifest_hash,
                'bundle_receipt_sha256': self.bundle['receipt_sha256']})
            # Every submitted job receives independent eager reference execution.
            start = perf_counter()
            source = _read_source(manifest, members)
            prepared = prepare_tiles(source.image, source.valid_mask)
            if not len(prepared.windows):
                raise ValueError('All-invalid scenes are rejected')
            area = _valid_area(source)
            _save_array(output/'normalized_tiles.npy', prepared.values)
            write_json(output/'tile-plan.json', {'windows': prepared.windows, 'total': prepared.total,
                'skipped': prepared.skipped, 'executed': len(prepared.windows),
                'input_array_sha256': array_hash(prepared.values),
                'height': prepared.height, 'width': prepared.width})
            validation['reference_preparation_seconds'] = perf_counter()-start
            start = perf_counter()
            reference_logits = np.concatenate([self.eager(prepared.values[i:i+1])
                for i in range(len(prepared.windows))], axis=0)
            reference_result = _reference_scene(source, prepared, reference_logits)
            _save_array(output/'eager_logits.npy', reference_logits)
            _save_array(output/'eager_classes.npy', reference_result.class_map)
            validation['eager_reference_and_recipe_seconds'] = perf_counter()-start
            write_json(output/'worker-startup.json', self.startup_receipt)
            before_adapter = self.candidate.metadata()
            self._gpu_call(self.torch.cuda.reset_peak_memory_stats, 0)
            warm_start = perf_counter()
            start = perf_counter()
            actual_source = _read_source(manifest, members)
            stages['raster_read_validate_seconds'] = perf_counter()-start
            actual = prepare_tiles(actual_source.image, actual_source.valid_mask)
            stages.update(actual.timings)
            start = perf_counter()
            if actual.windows != prepared.windows or not np.array_equal(actual.values, prepared.values):
                raise ValueError('Input changed between reference and scene execution')
            stages['input_parity_seconds'] = perf_counter()-start
            predicted = np.empty_like(reference_logits)
            groups = []
            for i in range(0, len(actual.windows), 8):
                count = min(8, len(actual.windows)-i)
                start = perf_counter()
                predicted[i:i+count] = self.candidate(actual.values[i:i+count])
                groups.append({'first_tile': i, 'real_tiles': count, 'padding_tiles': 0, 'neural_forwards': count,
                               'elapsed_seconds': perf_counter()-start})
            stages['predict_seconds'] = sum(row['elapsed_seconds'] for row in groups)
            start = perf_counter()
            result = blend_logits(actual, predicted, actual_source.valid_mask)
            stages['stitch_argmax_seconds'] = perf_counter()-start
            start = perf_counter()
            parity = compare_logits(reference_logits, predicted)
            parity.update(scene_class_map_equal=bool(np.array_equal(result.class_map, reference_result.class_map)),
                scene_mask_equal=bool(np.array_equal(result.valid_mask, reference_result.valid_mask)),
                normalization_and_tile_order_equal=True,
                reference_scope='Separate eager batch-one execution for every job; same strictly loaded resident model')
            parity['passed'] &= parity['scene_class_map_equal'] and parity['scene_mask_equal']
            stages['output_parity_seconds'] = perf_counter()-start
            if not parity['passed']:
                self.state = 'failed'
                raise WorkerIntegrityError('Resident job failed fixed FP32 parity')
            start = perf_counter()
            _save_array(output/'candidate_logits.npy', predicted)
            write_json(output/'groups.json', groups)
            write_json(output/'parity.json', parity)
            stages['evidence_write_seconds'] = perf_counter()-start
            worker_identity = {'worker_id': self.worker_id, 'model_load_count': self.model_load_count,
                'worker_startup_receipt_sha256': self.startup_receipt_hash,
                'full_checkpoint_hash_per_job': False,
                'identity_policy': self.startup_receipt['identity_policy']}
            provenance = {'schema': 'aureon.scene_provenance.v1', 'job_id': job_id,
                'publication_gate': {'requires_parent_receipt': '../job.json', 'required_parent_status': 'completed',
                    'absence_or_failed_parent_means_not_publishable': True},
                'model_name': 'KAO-TerraMind B', 'model_id': self.bundle['model_id'],
                'bundle_receipt_sha256': self.bundle['receipt_sha256'],
                'bundle_files_sha256': self.bundle['files_sha256'], 'worker': worker_identity,
                'model_identity_scope': 'Strict same-process resident load from owned verified snapshot; no per-job weight value rehash or remote attestation',
                'scene_id': scene_id, 'source': manifest['source'],
                'source_manifest_sha256': manifest_hash, 'prepared_input_identity': actual_source.input_identity,
                'source_preprocessing': manifest.get('preprocessing'),
                'recipe': {'id': 'aureon-landcover-v1', 'source_sha256': self.sources['aureon/recipe.py'],
                    'dtype': 'float32', 'window': 224, 'stride': 168, 'blend': 'uniform_float32_logits',
                    'order': 'row_major', 'neural_batch_size': 1, 'group_capacity': 8,
                    'last_group': 'explicit ordered eager batch-one tail; zero padding forwards'},
                'precision_actual': self.precision, 'parity_passed': True,
                'parity_receipt_sha256': file_hash(output/'parity.json'), 'class_schema': list(CLASS_NAMES),
                'quality': {'task_accuracy_evaluated': False, 'ground_truth_labels_used': False,
                    'confidence_calibrated': False}, 'code_sha256': self.sources}
            start = perf_counter()
            artifacts = write_segmentation_geotiffs(output/'artifacts', result, actual_source,
                provenance=provenance, cog=True)
            stages['geotiff_cog_write_reopen_seconds'] = perf_counter()-start
            start = perf_counter()
            _, _, final_hash = load_manifest(manifest_path)
            _verify_input_files(actual_source.input_identity)
            if final_hash != manifest_hash:
                raise ValueError('Input changed during job')
            stages['input_identity_postflight_seconds'] = perf_counter()-start
            guard_after = self._guard()
            stages['worker_identity_guard_seconds'] = guard_after
            warm_seconds = perf_counter()-warm_start
            stages['orchestration_residual_seconds'] = max(0.0, warm_seconds-sum(stages.values()))
            worker_identity['identity_guard_seconds'] = guard_before+guard_after
            receipt = {'schema': 'aureon.scene_job.v1', 'status': 'completed', 'job_id': job_id,
                'started_utc': started_utc, 'completed_utc': utc_now(), 'model_name': 'KAO-TerraMind B',
                'scene_id': scene_id, 'source_manifest_sha256': manifest_hash,
                'bundle_receipt_sha256': self.bundle['receipt_sha256'], 'worker': worker_identity,
                'hardware': self.hardware, 'precision': self.precision, 'area': area,
                'tile_count': len(actual.windows), 'tiles_total': actual.total, 'tiles_skipped': actual.skipped,
                'groups': len(groups), 'padding_forwards_in_job': sum(row['padding_tiles'] for row in groups),
                'stages_seconds': stages, 'warm_job_seconds': warm_seconds,
                'startup': {'resident_worker_already_initialized_seconds': 0.0},
                'validation_overhead': validation, 'parity': parity, 'artifacts': artifacts,
                'adapter_before_job': before_adapter, 'adapter_after_job': self.candidate.metadata(),
                'source_sha256': self.sources,
                'memory': {'cuda_peak_allocated_bytes': self._gpu_call(self.torch.cuda.max_memory_allocated, 0),
                    'cuda_peak_reserved_bytes': self._gpu_call(self.torch.cuda.max_memory_reserved, 0),
                    'scope': 'This process warm-job interval including resident model/graph; not device-wide peak'},
                'cost': cost_report(warm_seconds, 0.0, sum(validation.values()), area['valid_area_km2'],
                    hourly_rate=self.hourly_rate, currency=self.currency),
                'scope': {'job_kind': 'resident_local_async_scene_job', 'public_api_deployed': False,
                    'task_accuracy_evaluated': False, 'repeated_speedup_comparison': False,
                    'production_sla_established': False, 'source_acquisition_in_job_timer': False,
                    'startup_amortized': 'Full resident startup cost recorded once in worker-startup.json; not allocated to this job',
                    'warm_job_timer': 'Includes raster read through COG/input checks and resident guard; excludes eager-reference validation, queue, model startup and final receipt',
                    'max_scene_side': 1024}}
            receipt['files_sha256'] = {p.relative_to(output).as_posix(): file_hash(p)
                for p in output.rglob('*') if p.is_file() and 'source-snapshot' not in p.parts}
            receipt['worker_job_seconds_before_receipt'] = perf_counter()-whole_start
            write_json(output/'job.json', receipt)
            self.jobs_completed += 1
            downloads = {}
            for public, member in {'class_map.tif':'artifacts/class_map.tif',
                    'valid_mask.tif':'artifacts/valid_mask.tif', 'provenance.json':'artifacts/receipt.json',
                    'job.json':'job.json'}.items():
                downloads[public] = {'path': member, 'sha256': file_hash(output/member),
                                     'bytes': (output/member).stat().st_size}
            return {'status': 'completed', 'job_id': job_id, 'worker_id': self.worker_id,
                'summary': {'warm_job_seconds': warm_seconds, 'worker_job_seconds': perf_counter()-whole_start,
                    'parity_passed': True, 'bitwise_equal': parity['bitwise_equal'], 'model_load_count': 1},
                'download_artifacts': downloads}
        except BaseException as error:
            if (getattr(self.candidate, 'state', None) == 'failed'
                    or getattr(self.eager, 'state', None) == 'failed'):
                self.state = 'failed'
            try:
                write_json(output/'failure.json', {'job_id': job_id, 'status': 'failed',
                    'error_type': type(error).__name__, 'error': str(error), 'at_utc': utc_now(),
                    'partial_artifacts_not_publishable': True})
            except Exception:
                pass
            raise

    def _release_resources(self):
        retained = {name: weakref.ref(value) for name, value in (
            ('model', getattr(self.reference, 'model', None)), ('predictor', self.reference),
            ('graph', self.graph), ('candidate', self.candidate), ('eager', self.eager)) if value is not None}
        errors = _close_resources((('graph', self.graph), ('candidate', self.candidate), ('eager', self.eager)))
        if self.reference is not None:
            try:
                self.reference.synchronize()
            except Exception as error:
                errors.append({'stage':'synchronize', 'error_type':type(error).__name__, 'error':str(error)})
        self.tensor_seal = self.reference = self.graph = self.candidate = self.eager = None
        gc.collect()
        self.cleanup_objects_alive = {name: ref() is not None for name, ref in retained.items()}
        if any(self.cleanup_objects_alive.values()):
            errors.append({'stage':'owned_objects', 'error_type':'WorkerIntegrityError',
                           'error':'Owned model or execution objects remain referenced after release'})
        if hasattr(self, 'torch'):
            # Only this process's sole GPU owner reaches this lifecycle point.
            # PyTorch 2.10 keeps a cuBLAS workspace per handle/stream after model
            # tensors are freed. Release it after synchronization, then cache.
            if self.torch.cuda.is_initialized():
                try:
                    self.torch.cuda.synchronize(0)
                except Exception as error:
                    errors.append({'stage':'cuda_synchronize', 'error_type':type(error).__name__, 'error':str(error)})
                try:
                    clear = getattr(self.torch._C, '_cuda_clearCublasWorkspaces', None)
                    if not callable(clear):
                        raise RuntimeError('Pinned runtime cuBLAS workspace cleanup API unavailable')
                    clear()
                except Exception as error:
                    errors.append({'stage':'cublas_workspace_clear', 'error_type':type(error).__name__, 'error':str(error)})
            try:
                self.torch.cuda.empty_cache()
            except Exception as error:
                errors.append({'stage':'empty_cache', 'error_type':type(error).__name__, 'error':str(error)})
        return errors

    def close(self):
        self._assert_owner()
        if self.state == 'closed':
            return
        errors = []
        start = perf_counter()
        previous_state = self.state
        loaded_identity = None
        try:
            self.file_seal.verify()
            self.tensor_seal.verify()
            if self._precision(self.torch) != self.precision:
                raise WorkerIntegrityError('Resident precision settings changed before shutdown')
        except Exception as error:
            errors.append({'stage':'final_resident_guard', 'error_type':type(error).__name__, 'error':str(error)})
        try:
            loaded_identity = self._attest_loaded_model()
            final = verify_bundle(self.bundle_dir)
            if (final['receipt_sha256'] != self.bundle['receipt_sha256']
                    or final['files_sha256'] != self.bundle['files_sha256'] or source_hashes() != self.sources):
                raise WorkerIntegrityError('Full shutdown identity verification failed')
        except Exception as error:
            errors.append({'stage':'final_identity', 'error_type':type(error).__name__, 'error':str(error)})
        identity_passed = not errors
        cleanup_errors = self._release_resources()
        self.state = 'closed'
        memory = {}
        try:
            memory = {'cuda_allocated_bytes_after_cleanup':self.torch.cuda.memory_allocated(0),
                'cuda_reserved_bytes_after_cleanup':self.torch.cuda.memory_reserved(0)}
            if any(memory.values()):
                cleanup_errors.append({'stage':'allocator_not_empty', 'error_type':'WorkerIntegrityError',
                    'error':'CUDA allocator still owns memory after sole-owner shutdown'})
        except Exception as error:
            cleanup_errors.append({'stage':'memory_after_cleanup', 'error_type':type(error).__name__, 'error':str(error)})
        errors += cleanup_errors
        result = {'schema':'aureon.resident_scene_worker_shutdown.v1', 'worker_id':self.worker_id,
            'at_utc':utc_now(), 'jobs_completed':self.jobs_completed, 'model_load_count':self.model_load_count,
            'previous_worker_state':previous_state,
            'full_identity_check_passed':identity_passed, 'cleanup_passed':not cleanup_errors,
            'loaded_tensor_identity':loaded_identity,
            'owned_objects_alive_after_cleanup':self.cleanup_objects_alive,
            'cublas_workspace_clear_scope':'Sole GPU owner after graph/model release and synchronization',
            'errors':errors, 'cleanup_seconds':perf_counter()-start, **memory}
        write_json(self.folder/'shutdown.json', result)
        if getattr(self, 'original_bundle_dir', None) is not None:
            finish_bundle_storage(self.folder, self.original_bundle_dir, result)
        return result
