"""One explicit local Sentinel-2 -> KAO-TerraMind B -> COG job with evidence.

No downloads or installations occur here. The CLI requires --execute-model and
the prepared Linux runtime. This is a local job contract, not a public server.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import uuid

import numpy as np

from ..recipe import S2_BANDS, INPUT_UNITS, segment_tiled
from ..catalog import CLASS_NAMES
from ..runtime import verify_bundle
from .core import array_hash, prepare_tiles, blend_logits, compare_logits, cost_report
from .predictor import FullEightEagerTailAdapter, SequentialEagerAdapter


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _relative_file(folder, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Scene files must use relative paths within the manifest directory')
    path = folder / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder):
        raise ValueError('Scene input must be an existing regular file within its manifest directory')
    return path.resolve()


def load_manifest(path, *, max_evidence_bytes=128*1024*1024):
    if type(max_evidence_bytes) is not int or not 0 < max_evidence_bytes <= 2*1024**3:
        raise ValueError('Manifest evidence budget must be positive and at most 2 GiB')
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= 4*1024*1024:
        raise ValueError('Scene manifest must be a bounded JSON file')
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get('schema') != 'aureon.sentinel2_scene.v1':
        raise ValueError('Expected aureon.sentinel2_scene.v1 manifest')
    raster, mask = manifest['raster'], manifest['valid_mask']
    if raster['bands'] != list(S2_BANDS) or raster['units'] != INPUT_UNITS:
        raise ValueError('Prepared scene must declare the exact 12-band reflectance_x10000 contract')
    if not isinstance(raster.get('crs'), str) or not raster['crs']:
        raise ValueError('Scene manifest must explicitly declare its CRS')
    members = {}
    computed = {}
    def content_digest(member):
        # Content was read in this invocation; never reuse across phases/jobs.
        if member not in computed:
            computed[member] = file_hash(member)
        return computed[member]
    declarations = []
    def collect(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                declarations.append(value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(manifest)
    if len(declarations) > 64:
        raise ValueError('Too many local evidence members in bounded scene manifest')
    total = 0
    for declaration in declarations:
        member = _relative_file(path.parent, declaration['path'])
        total += member.stat().st_size
        if total > max_evidence_bytes:
            raise ValueError('Local scene evidence exceeds its declared admission budget')
        if ('bytes' in declaration and (type(declaration['bytes']) is not int
                or declaration['bytes'] != member.stat().st_size)):
            raise ValueError('Scene evidence size mismatch: '+declaration['path'])
        if content_digest(member) != declaration['sha256']:
            raise ValueError('Scene evidence hash mismatch: '+declaration['path'])
    for key, declaration in (('raster', raster), ('valid_mask', mask)):
        member = _relative_file(path.parent, declaration['path'])
        digest = content_digest(member)
        if digest != declaration.get('sha256'):
            raise ValueError('Scene member hash mismatch: '+key)
        members[key] = member
    return manifest, members, hashlib.sha256(raw).hexdigest()


def _read_source(manifest, members):
    from .georaster import read_s2_geotiff
    source = read_s2_geotiff(members['raster'], bands=manifest['raster']['bands'],
        units=manifest['raster']['units'], expected_crs=manifest['raster']['crs'],
        mask_policy='all_bands_valid', valid_mask_path=members['valid_mask'])
    declared = manifest['raster']
    raw = source.input_identity['source']['raster_metadata']
    if (declared.get('shape') != list(source.image.shape)
            or declared.get('dtype') != 'float32' or raw['dtypes'] != ['float32']*12
            or declared.get('transform') != source.metadata['transform']
            or manifest['valid_mask'].get('valid_pixels') != int(source.valid_mask.sum())):
        raise ValueError('Prepared raster grid, dtype or valid count differs from manifest')
    return source


def _close_resources(resources):
    errors = []
    for name, resource in resources:
        if resource is not None:
            try:
                resource.close()
            except BaseException as error:
                errors.append({'resource': name, 'error_type': type(error).__name__, 'error': str(error)})
    return errors


def _verify_input_files(identity):
    """Include GDAL sidecars discovered during admission in postflight checks."""
    if isinstance(identity, dict):
        if 'path' in identity and 'sha256' in identity:
            if file_hash(identity['path']) != identity['sha256']:
                raise ValueError('Admitted source or supporting file changed during job')
        for child in identity.values():
            _verify_input_files(child)
    elif isinstance(identity, list):
        for child in identity:
            _verify_input_files(child)


def _sources():
    selected = [ROOT/'aureon'/name for name in (
        'recipe.py', 'reference.py', 'runtime.py', 'catalog.py', 'vllm_bundle.py',
        'cuda_graph.py', 'graph_pipeline.py', 'process_memory.py', 'preflight.py', 'scene_cli.py')]
    selected += [p for p in (ROOT/'aureon'/'prepare').glob('*.py') if not p.name.startswith('test_')]
    selected += [p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in selected}


def _snapshot(folder, hashes):
    for name, digest in hashes.items():
        target = folder/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Source changed while making snapshot')
        with target.open('xb') as stream:
            stream.write(data)


def _save_array(path, values):
    with Path(path).open('xb') as stream:
        np.save(stream, values, allow_pickle=False)


def _reference_scene(source, prepared, reference_logits):
    """Use the unchanged recipe as a separate CPU tiling/stitching reference."""
    position = 0
    def saved_prediction(values):
        nonlocal position
        end = position + len(values)
        if not np.array_equal(values, prepared.values[position:end]):
            raise ValueError('New scene tiling differs from the original recipe inputs')
        result = reference_logits[position:end].copy()
        position = end
        return result
    result = segment_tiled(source.image, source.valid_mask, saved_prediction,
        bands=S2_BANDS, units=INPUT_UNITS, batch_size=1, include_max_score=False)
    if position != len(prepared.windows):
        raise ValueError('Reference recipe tile coverage differs')
    return result


def _valid_area(source):
    # Geographic/angular grids are rejected for this area/cost calculation.
    import rasterio
    crs = rasterio.crs.CRS.from_user_input(source.metadata['crs'])
    if not crs.is_projected:
        raise ValueError('This bounded job requires a projected grid for valid-area accounting')
    transform = source.metadata['transform']
    a, b, unused_c, d, e, unused_f = transform[:6]
    unit_name, metres_per_unit = crs.linear_units_factor
    area = abs(a*e-b*d)*metres_per_unit**2
    return {'pixel_area_m2': area, 'valid_pixels': int(source.valid_mask.sum()),
            'valid_area_km2': float(source.valid_mask.sum())*area/1e6,
            'method': 'affine determinant in projected CRS, converted to square metres; grid area',
            'crs_linear_units': unit_name, 'geodesic_area': False}


def run_job(manifest_path, bundle_dir, output_dir, *, execute_model=False,
            hourly_rate=1.0, currency='THB'):
    if execute_model is not True:
        raise ValueError('Explicit execute_model=True / --execute-model is required')
    if sys.platform != 'linux':
        raise ValueError('Run model jobs in the existing isolated WSL/Linux environment')
    # Validate the price scenario before creating a directory or importing torch.
    cost_report(0, 0, 0, 0, hourly_rate=hourly_rate, currency=currency)
    experiment_start = perf_counter()
    output = Path(output_dir).resolve()
    if not output.parent.is_dir():
        raise ValueError('Create the job parent directory explicitly before starting')
    output.mkdir(exist_ok=False)
    job_id = uuid.uuid4().hex
    graph = eager = candidate = None
    completed = False
    stages, validation = {}, {}
    try:
        sources = _sources()
        _snapshot(output, sources)
        started = datetime.now(timezone.utc).isoformat()
        manifest, members, manifest_hash = load_manifest(manifest_path)
        raw_manifest = Path(manifest_path).read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != manifest_hash:
            raise ValueError('Input manifest changed during snapshot')
        with (output/'input-manifest.json').open('xb') as stream:
            stream.write(raw_manifest)
        bundle = verify_bundle(bundle_dir)
        if bundle['model_id'] != 'kao-terramind-b':
            raise ValueError('This initial scene job explicitly supports KAO-TerraMind B only')
        write_json(output/'launch.json', {'schema': 'aureon.scene_job_launch.v1',
            'job_id': job_id, 'started_utc': started, 'pid': os.getpid(),
            'source_sha256': sources, 'manifest_sha256': manifest_hash,
            'bundle_receipt_sha256': bundle['receipt_sha256'],
            'model_execution_authorized': True, 'network_acquisition': False,
            'cost_assumption': {'hourly_rate': hourly_rate, 'currency': currency}})
        runtime_start = perf_counter()
        # Strict offline policy prevents any loader from acquiring missing weights.
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_TELEMETRY='1', NVIDIA_TF32_OVERRIDE='0',
                          OMP_NUM_THREADS='1')
        from ..process_memory import configure_process_memory
        process_memory = configure_process_memory('disable')
        import torch
        from ..cuda_graph import _precision
        from ..reference import PyTorchPredictor
        from ..graph_pipeline import FullGroupGraphPipeline
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark = False
        precision = _precision(torch)
        runtime_init_seconds = perf_counter()-runtime_start
        start = perf_counter()
        source = _read_source(manifest, members)
        prepared = prepare_tiles(source.image, source.valid_mask)
        if not len(prepared.windows):
            raise ValueError('The evidence job needs at least one valid tile; all-invalid scene rejected before model loading')
        area = _valid_area(source)
        _save_array(output/'normalized_tiles.npy', prepared.values)
        write_json(output/'tile-plan.json', {'windows': prepared.windows,
            'total': prepared.total, 'skipped': prepared.skipped, 'executed': len(prepared.windows),
            'input_array_sha256': array_hash(prepared.values),
            'height': prepared.height, 'width': prepared.width})
        validation['reference_preparation_seconds'] = perf_counter()-start
        start = perf_counter()
        reference = PyTorchPredictor(bundle_dir, device='cuda:0', dtype='float32')
        # TerraTorch imports may change the process matmul setting. Reassert
        # the contract after construction and before any reference forward.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark = False
        precision = _precision(torch)
        reference.synchronize()
        model_load_seconds = perf_counter()-start
        eager = SequentialEagerAdapter(reference)
        start = perf_counter()
        reference_logits = np.empty((len(prepared.windows), 10, 224, 224), dtype=np.float32)
        for index in range(len(prepared.windows)):
            reference_logits[index:index+1] = eager(prepared.values[index:index+1])
        reference_result = _reference_scene(source, prepared, reference_logits)
        _save_array(output/'eager_logits.npy', reference_logits)
        _save_array(output/'eager_classes.npy', reference_result.class_map)
        validation['eager_reference_and_recipe_seconds'] = perf_counter()-start
        write_json(output/'eager.json', eager.metadata())
        start = perf_counter()
        (output/'graph').mkdir()
        graph = FullGroupGraphPipeline.from_reference(reference, output/'graph')
        candidate = FullEightEagerTailAdapter(graph, reference)
        first_count = min(8, len(prepared.windows))
        first = prepared.values[:first_count]
        first_logits = candidate(first)
        capture_seconds = perf_counter()-start
        start = perf_counter()
        first_check = compare_logits(reference_logits[:first_count], first_logits)
        if not first_check['passed']:
            write_json(output/'warmup-parity-failure.json', first_check)
            raise ValueError('CUDA graph capture does not preserve reference logits')
        reversed_check = compare_logits(reference_logits[:first_count][::-1], candidate(first[::-1]))
        if not reversed_check['passed']:
            raise ValueError('Reversed-input freshness probe failed')
        validation['freshness_probe_seconds'] = perf_counter()-start
        before_adapter = candidate.metadata()
        hardware = {'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'packages': {key: package_metadata.version(key) for key in ('terratorch', 'rasterio', 'numpy')},
            'process_memory': process_memory}
        torch.cuda.reset_peak_memory_stats(0)
        # This contiguous interval is the actual measured warm local job.
        job_start = perf_counter()
        start = perf_counter()
        actual_source = _read_source(manifest, members)
        stages['raster_read_validate_seconds'] = perf_counter()-start
        actual = prepare_tiles(actual_source.image, actual_source.valid_mask)
        stages.update(actual.timings)
        start = perf_counter()
        if actual.windows != prepared.windows or not np.array_equal(actual.values, prepared.values):
            raise ValueError('Inputs changed between eager reference and scene execution')
        stages['input_parity_seconds'] = perf_counter()-start
        predicted = np.empty_like(reference_logits)
        groups = []
        for index in range(0, len(actual.windows), 8):
            count = min(8, len(actual.windows)-index)
            start = perf_counter()
            predicted[index:index+count] = candidate(actual.values[index:index+count])
            groups.append({'first_tile': index, 'real_tiles': count, 'padding_tiles': 0, 'neural_forwards': count,
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
                      warmup=first_check, reversed_input_probe=reversed_check,
                      reference_scope='Independent eager batch-one execution and unchanged array recipe; same strictly loaded model allocation')
        parity['passed'] = parity['passed'] and parity['scene_class_map_equal'] and parity['scene_mask_equal']
        stages['output_parity_seconds'] = perf_counter()-start
        if not parity['passed']:
            write_json(output/'parity-failure.json', parity)
            raise ValueError('Scene output failed the unchanged FP32 logit and class gate')
        start = perf_counter()
        _save_array(output/'candidate_logits.npy', predicted)
        write_json(output/'groups.json', groups)
        write_json(output/'parity.json', parity)
        stages['evidence_write_seconds'] = perf_counter()-start
        provenance = {'schema': 'aureon.scene_provenance.v1', 'job_id': job_id,
            'publication_gate': {'requires_parent_receipt': '../job.json',
                                 'required_parent_status': 'completed',
                                 'absence_or_failed_parent_means_not_publishable': True},
            'model_name': 'KAO-TerraMind B', 'model_id': bundle['model_id'],
            'bundle_receipt_sha256': bundle['receipt_sha256'],
            'bundle_files_sha256': bundle['files_sha256'],
            'model_identity_scope': 'Strict same-process local load; not remote server attestation',
            'scene_id': manifest.get('scene_id'), 'source': manifest.get('source'),
            'source_manifest_sha256': manifest_hash,
            'prepared_input_identity': actual_source.input_identity,
            'source_preprocessing': manifest.get('preprocessing'),
            'recipe': {'id': 'aureon-landcover-v1', 'source_sha256': sources['aureon/recipe.py'],
                'dtype': 'float32', 'window': 224, 'stride': 168, 'blend': 'uniform_float32_logits',
                'order': 'row_major', 'neural_batch_size': 1, 'group_capacity': 8,
                'last_group': 'explicit ordered eager batch-one tail; zero padding forwards'},
            'precision_actual': precision, 'parity_passed': True,
            'parity_receipt_sha256': file_hash(output/'parity.json'),
            'class_schema': list(CLASS_NAMES),
            'quality': {'task_accuracy_evaluated': False, 'ground_truth_labels_used': False,
                        'confidence_calibrated': False,
                        'scope': 'Numerical/geospatial engineering evidence for one scene crop'},
            'code_sha256': sources}
        start = perf_counter()
        from .georaster import write_segmentation_geotiffs
        artifacts = write_segmentation_geotiffs(output/'artifacts', result, actual_source,
                                                provenance=provenance, cog=True)
        stages['geotiff_cog_write_reopen_seconds'] = perf_counter()-start
        start = perf_counter()
        _, _, final_manifest_hash = load_manifest(manifest_path)
        _verify_input_files(actual_source.input_identity)
        final_bundle = verify_bundle(bundle_dir)
        if (final_manifest_hash != manifest_hash or final_bundle['receipt_sha256'] != bundle['receipt_sha256']
                or final_bundle['files_sha256'] != bundle['files_sha256'] or _precision(torch) != precision
                or _sources() != sources):
            raise ValueError('Source, input, checkpoint or precision changed during job')
        stages['identity_postflight_seconds'] = perf_counter()-start
        warm_job_seconds = perf_counter()-job_start
        stage_sum = sum(stages.values())
        stages['orchestration_residual_seconds'] = max(0.0, warm_job_seconds-stage_sum)
        peak_memory = {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                       'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                       'scope': 'This process warm-job interval, including resident model and graph; not device-wide peak'}
        metadata_after = candidate.metadata()
        cleanup_errors = _close_resources((('graph', graph), ('candidate', candidate), ('eager', eager)))
        if cleanup_errors:
            write_json(output/'cleanup-errors.json', cleanup_errors)
            raise RuntimeError('Resource cleanup failed; job is not publishable')
        cleanup = {'graph_state': graph.state, 'candidate_state': candidate.state,
                   'eager_adapter_state': eager.state, 'gpu_process_exits_when_cli_exits': True}
        receipt = {'schema': 'aureon.scene_job.v1', 'status': 'completed', 'job_id': job_id,
            'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'model_name': 'KAO-TerraMind B', 'scene_id': manifest.get('scene_id'),
            'source_manifest_sha256': manifest_hash, 'bundle_receipt_sha256': bundle['receipt_sha256'],
            'hardware': hardware, 'precision': precision, 'area': area,
            'tile_count': len(actual.windows), 'tiles_total': actual.total,
            'tiles_skipped': actual.skipped, 'groups': len(groups),
            'padding_forwards_in_job': sum(row['padding_tiles'] for row in groups),
            'stages_seconds': stages, 'warm_job_seconds': warm_job_seconds,
            'startup': {'runtime_import_config_seconds': runtime_init_seconds,
                        'model_load_seconds': model_load_seconds,
                        'graph_initialization_capture_first_replay_seconds': capture_seconds},
            'validation_overhead': validation, 'parity': parity,
            'memory': peak_memory, 'artifacts': artifacts,
            'adapter_before_job': before_adapter, 'adapter_after_job': metadata_after,
            'cleanup': cleanup, 'source_sha256': sources,
            'cost': cost_report(warm_job_seconds, runtime_init_seconds+model_load_seconds+capture_seconds,
                sum(validation.values()), area['valid_area_km2'], hourly_rate=hourly_rate, currency=currency),
            'experiment_wall_seconds_before_receipt': perf_counter()-experiment_start,
            'scope': {'job_kind': 'explicit_local_offline_scene_job', 'public_api_deployed': False,
                      'source_acquisition_in_job_timer': False, 'task_accuracy_evaluated': False,
                      'repeated_speedup_comparison': False, 'production_sla_established': False,
                      'warm_job_timer': 'Contiguous local interval including source read/validation, inference, parity, evidence/COG writes and identity postflight; input pages may already be cached',
                      'max_scene_side': 1024}}
        receipt['files_sha256'] = {p.relative_to(output).as_posix(): file_hash(p)
            for p in output.rglob('*') if p.is_file() and 'source-snapshot' not in p.parts}
        write_json(output/'job.json', receipt)
        completed = True
        return receipt
    except BaseException as error:
        try:
            write_json(output/'failure.json', {'schema': 'aureon.scene_job_failure.v1',
                'job_id': job_id, 'status': 'failed', 'error_type': type(error).__name__,
                'error': str(error), 'at_utc': datetime.now(timezone.utc).isoformat(),
                'partial_artifacts_not_publishable': True, 'stages_seconds': stages})
        except Exception:
            pass
        raise
    finally:
        if not completed:
            errors = _close_resources((('graph', graph), ('candidate', candidate), ('eager', eager)))
            try:
                write_json(output/'failure-cleanup.json', {'errors': errors,
                    'attempted_all_resources': True, 'primary_exception_preserved': True})
            except Exception:
                # Never replace the primary computation failure with a logger failure.
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Explicit FP32 KAO-TerraMind B georeferenced scene job')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--hourly-rate', type=float, default=1.0)
    parser.add_argument('--currency', default='THB')
    args = parser.parse_args(argv)
    receipt = run_job(args.manifest, args.bundle, args.output,
        execute_model=args.execute_model, hourly_rate=args.hourly_rate, currency=args.currency)
    print(json.dumps({'job': str(Path(args.output).resolve()/'job.json'), 'status': receipt['status'],
        'parity_passed': receipt['parity']['passed'], 'warm_job_seconds': receipt['warm_job_seconds'],
        'scenario_cost': receipt['cost']['warm_job_cost'], 'currency': receipt['cost']['currency']}), flush=True)


if __name__ == '__main__':
    main()
