"""Explicit array inference and delivery against an independently started service.

Bundle verification checks local artifact identities, not the weights loaded by
the remote process. No function downloads weights, starts a server or loads a
model. NumPy outputs retain pixel positions but contain no georeferencing.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .catalog import get_catalog
from .recipe import INPUT_UNITS, S2_BANDS, _validate_input, segment_tiled
from .transport import VLLMPoolingPredictor
from .vllm_bundle import (TENSOR_HASH_FORMAT, _legacy_export_key, _manifest_sha256,
                         _validate_legacy_manifest, build_vllm_config,
                         legacy_scale_migration, pinned_legacy_model_args)


_BUNDLE_FILES = {'config.json', 'pytorch_model.bin'}
_IDENTITY_KEYS = ('repo_id', 'revision', 'checkpoint', 'published_checkpoint_sha256',
                  'published_checkpoint_bytes')
_MAX_JSON_BYTES = 4 * 1024 * 1024


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path):
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError('Bundle JSON exceeds its metadata budget.')
    with path.open('rb') as stream:
        data = stream.read(_MAX_JSON_BYTES + 1)
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError('Bundle JSON exceeds its metadata budget.')
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Invalid bundle JSON.') from exc
    if not isinstance(value, dict):
        raise ValueError('Bundle metadata must be a JSON object.')
    return value


def _member(folder, name):
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:
        raise ValueError('Bundle members must be regular local files inside the bundle.')
    return path


def _verify_tensor_metadata(receipt, config):
    """Validate preparation metadata without deserializing weights.

    For migrated published artifacts the source manifest is independently pinned
    in code. The file hash binds the exported bytes; tensor metadata describes the
    preparation operation and is not an attestation of a running server.
    """
    source_args = receipt.get('source_model_args')
    source_manifest = receipt.get('source_tensor_manifest')
    exported_manifest = receipt.get('tensor_manifest')
    if receipt.get('tensor_hash_format') != TENSOR_HASH_FORMAT:
        raise ValueError('Unsupported tensor hash format in bundle receipt.')
    try:
        args = config['pretrained_cfg']['model']['init_args']['model_args']
        source_config = build_vllm_config(source_args, as_dict=receipt['source_as_dict'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid source model arguments in bundle receipt.') from exc
    if source_config != config or receipt.get('resolved_model_args') != args:
        raise ValueError('Bundle configuration differs from source/resolved model arguments.')
    for field, manifest in (('source_tensor_manifest', source_manifest), ('tensor_manifest', exported_manifest)):
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError('Bundle requires a nonempty ' + field)
        for name, record in manifest.items():
            if (not isinstance(name, str) or not name or not isinstance(record, dict)
                    or set(record) != {'shape', 'dtype', 'sha256'}
                    or not isinstance(record['shape'], list)
                    or any(type(n) is not int or n < 0 for n in record['shape'])
                    or not isinstance(record['dtype'], str) or not re.fullmatch(r'torch\.[a-z0-9_]+', record['dtype'])
                    or not isinstance(record['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
                raise ValueError('Invalid tensor record in ' + field)
        if receipt.get(field + '_sha256') != _manifest_sha256(manifest):
            raise ValueError('Tensor manifest identity changed: ' + field)
    migration = receipt.get('architecture_migration')
    if source_args.get('decoder_scale_modules') is True:
        if (source_args != pinned_legacy_model_args(receipt['model_id'])
                or receipt['source_as_dict'] is not True
                or migration != legacy_scale_migration()):
            raise ValueError('Unsupported KAO-TerraMind architecture migration.')
        _validate_legacy_manifest(receipt['model_id'], source_manifest)
    elif migration is not None:
        raise ValueError('Architecture migration is not supported by the source configuration.')
    expected_manifest = {}
    expected_rewrites = []
    for name, record in source_manifest.items():
        target = _legacy_export_key(name) if migration else name
        if target in expected_manifest:
            raise ValueError('Tensor key migration collides with another source tensor.')
        expected_manifest[target] = record
        if target != name:
            expected_rewrites.append({'source': name, 'target': target})
    expected_rewrites.sort(key=lambda item: item['source'])
    if exported_manifest != expected_manifest or receipt.get('state_key_rewrites') != expected_rewrites:
        raise ValueError('Bundle tensor mapping, values, shapes or precision differ from the audited migration.')


def verify_bundle(bundle_dir):
    """Verify a generated candidate bundle using hashes and JSON only.

    Returns its receipt plus the expected pinned catalog identity. The original
    checkpoint is not reopened; its verified digest is a preparation receipt
    assertion. Local file verification is not server checkpoint attestation.
    """
    folder = Path(bundle_dir).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('bundle_dir must be a directory.')
    if {path.name for path in folder.iterdir()} != _BUNDLE_FILES | {'bundle_receipt.json'}:
        raise ValueError('Bundle directory must contain only its two declared artifacts and bundle_receipt.json.')
    receipt_path = _member(folder, 'bundle_receipt.json')
    receipt_hash = _file_hash(receipt_path)
    receipt = _json_file(receipt_path)
    if receipt.get('schema') != 'aureon.vllm_bundle.v1':
        raise ValueError('Unsupported bundle receipt schema.')
    model = next((row for row in get_catalog()['models']
                  if row['id'] == receipt.get('model_id') and row['status'] == 'candidate'), None)
    if model is None:
        raise ValueError('Bundle model is not a pinned KAO-TerraMind candidate.')
    expected = model['identity']
    claimed = receipt.get('source_identity')
    if not isinstance(claimed, dict) or any(claimed.get(key) != expected[key] for key in _IDENTITY_KEYS):
        raise ValueError('Bundle source identity differs from the pinned catalog.')
    if receipt.get('verified_source_sha256') != expected['published_checkpoint_sha256']:
        raise ValueError('Bundle source checkpoint digest differs from the catalog.')
    if (receipt.get('state_tensors_dropped') != []
            or 'dtype_conversion' not in receipt or receipt['dtype_conversion'] is not None
            or type(receipt.get('source_as_dict')) is not bool):
        raise ValueError('Bundle does not preserve source tensors and precision.')
    for field in ('model_executed', 'downloads_started', 'server_started'):
        if receipt.get(field) is not False:
            raise ValueError('Unexpected execution claim in preparation receipt: ' + field)
    hashes = receipt.get('files_sha256')
    if not isinstance(hashes, dict) or set(hashes) != _BUNDLE_FILES:
        raise ValueError('Bundle hashes must identify exactly config.json and pytorch_model.bin.')
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid bundle artifact SHA-256.')
        path = _member(folder, name)
        if _file_hash(path) != digest:
            raise ValueError('Bundle artifact changed: ' + name)
    config = _json_file(folder / 'config.json')
    try:
        args = config['pretrained_cfg']['model']['init_args']['model_args']
        expected_config = build_vllm_config(args, as_dict=receipt['source_as_dict'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid TerraTorch bundle configuration.') from exc
    if config != expected_config:
        raise ValueError('Bundle configuration does not match the declared tensor serving contract.')
    metadata_fields = {'source_model_args', 'resolved_model_args', 'source_tensor_manifest',
                       'source_tensor_manifest_sha256', 'tensor_manifest_sha256',
                       'tensor_hash_format', 'architecture_migration'}
    if metadata_fields.intersection(receipt):
        if not metadata_fields.issubset(receipt):
            raise ValueError('Bundle tensor migration metadata is incomplete.')
        _verify_tensor_metadata(receipt, config)
    elif receipt.get('state_key_rewrites') != []:
        # Original v1 receipts are admitted only with their original no-rewrite contract.
        raise ValueError('Bundle key rewrites require audited migration metadata.')
    if _file_hash(receipt_path) != receipt_hash:
        raise ValueError('Bundle receipt changed during verification.')
    return {
        'bundle_dir': str(folder),
        'model_id': model['id'],
        'model_identity': deepcopy(expected),
        'receipt': receipt,
        'receipt_sha256': receipt_hash,
        'files_sha256': deepcopy(hashes),
        'verification_scope': 'local_bundle_files_and_preparation_receipt',
        'server_checkpoint_attestation': 'not_available',
    }


def _array_identity(array):
    """Hash exact dtype, shape and contiguous C-order bytes, including invalids."""
    header = {'dtype': array.dtype.str, 'shape': list(array.shape), 'order': 'C'}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    digest.update(b'\0')
    digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return {**header, 'sha256': digest.hexdigest(), 'hash_format': 'json_header_nul_c_order_bytes_v1'}


def run_scene(bundle_dir, image, mask, *, output_dir, dtype='float32', batch_size=1,
              max_in_flight=1, base_url='http://127.0.0.1:8000', timeout=60,
              include_max_score=True, predictor=None, reuse_connections=False,
              json_backend='stdlib', response_format='base64', input_format='torch',
              base64_backend='stdlib', transport='json'):
    """Run declared S2 arrays and save an exclusive NumPy delivery directory.

    ``image`` is [12,H,W] surface_reflectance_x10000 in S2_BANDS order;
    ``mask`` is boolean [H,W]. An injected predictor is for adapter testing and
    is labeled separately in the receipt. Successful HTTP calls cannot prove
    the server loaded the local bundle. Failed inference creates no outputs;
    a filesystem failure during export can leave an incomplete new directory.
    Optional connection reuse is bounded by max_in_flight and always closed
    after prediction, including failures. It does not change input mathematics.
    """
    destination = Path(output_dir).absolute()
    if destination.exists():
        raise FileExistsError('Output directory must not already exist.')
    if not destination.parent.is_dir():
        raise ValueError('Output parent directory must already exist.')
    verified = verify_bundle(bundle_dir)
    if destination.resolve().is_relative_to(Path(verified['bundle_dir'])):
        raise ValueError('Output directory must be outside the model bundle.')
    _validate_input(image, mask, S2_BANDS, INPUT_UNITS)
    # Snapshot the arrays so caller changes cannot alter an in-flight receipt.
    source_image = np.array(image, copy=True, order='C')
    source_mask = np.array(mask, copy=True, order='C')
    source_image.flags.writeable = False
    source_mask.flags.writeable = False
    input_identity = {'image': _array_identity(source_image), 'valid_mask': _array_identity(source_mask),
                      'units': INPUT_UNITS, 'band_order': list(S2_BANDS)}
    if predictor is not None and not callable(predictor):
        raise ValueError('Injected predictor must be callable.')
    if transport == 'shared_memory':
        if dtype != 'float32' or type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError('Shared-memory inference requires FP32 and a group of at most eight tiles.')
        from .shm_transport import SharedMemoryPoolingPredictor
        client = SharedMemoryPoolingPredictor(verified['model_id'], base_url=base_url, timeout=timeout, capacity=8)
    else:
        client = VLLMPoolingPredictor(verified['model_id'], base_url=base_url, dtype=dtype,
                                  max_in_flight=max_in_flight, timeout=timeout,
                                  reuse_connections=reuse_connections,
                                  json_backend=json_backend, response_format=response_format,
                                  input_format=input_format, base64_backend=base64_backend, transport=transport)
    try:
        selected_predictor = client if predictor is None else predictor
        started = time.perf_counter()
        result = segment_tiled(source_image, source_mask, selected_predictor,
                               bands=S2_BANDS, units=INPUT_UNITS, batch_size=batch_size,
                               include_max_score=include_max_score)
        processing_seconds = time.perf_counter() - started
    finally:
        client.close()
    # Keep an artifact mutation during inference from receiving a valid receipt.
    after = verify_bundle(bundle_dir)
    if (after['receipt_sha256'] != verified['receipt_sha256']
            or after['files_sha256'] != verified['files_sha256']):
        raise ValueError('Bundle changed during inference; no delivery was written.')
    recipe = next(row for row in get_catalog()['recipes'] if row['id'] == result.recipe_id)
    outputs = {'class_map.npy': result.class_map, 'valid_mask.npy': result.valid_mask}
    if result.max_uncalibrated_score is not None:
        outputs['score.npy'] = result.max_uncalibrated_score
    destination.mkdir(parents=False, exist_ok=False)
    files = {}
    for name, array in outputs.items():
        path = destination / name
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        files[name] = {'sha256': _file_hash(path), 'bytes': path.stat().st_size,
                       'shape': list(array.shape), 'dtype': str(array.dtype)}
    predictor_invoked = result.tiles_executed > 0
    requested_server = predictor is None and predictor_invoked
    receipt = {
        'schema': 'aureon.array_delivery.v1',
        'status': 'completed',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'output_directory': str(destination),
        'execution_backend': 'vllm_pooling' if predictor is None else 'injected_predictor',
        'model_id': verified['model_id'],
        'expected_model_identity': verified['model_identity'],
        'model_identity_scope': 'expected_local_bundle_not_server_attested',
        'bundle_receipt_sha256': verified['receipt_sha256'],
        'bundle_files_sha256': verified['files_sha256'],
        'source_preservation': {'state_key_rewrites': deepcopy(verified['receipt']['state_key_rewrites']),
                                'state_tensors_dropped': [], 'dtype_conversion': None,
                                'architecture_migration': deepcopy(verified['receipt'].get('architecture_migration'))},
        'server_checkpoint_attestation': 'not_available',
        'recipe_identity': recipe['identity'],
        'recipe_status': result.recipe_status,
        'class_names': recipe['output']['class_names'],
        'input': input_identity,
        'output': {'format': 'numpy_arrays', 'shape': list(result.class_map.shape),
                   'georeferencing': 'not_supplied', 'nodata_class': 255,
                   'score_calibration': 'uncalibrated', 'invalid_score': 'NaN'},
        'runtime': {'base_url': base_url, 'served_model_name': verified['model_id'],
                    'input_dtype': dtype, 'server_dtype': 'caller_must_match_input_dtype',
                    'batch_size': result.batch_size,
                    'max_in_flight': client.max_in_flight if predictor is None else max_in_flight,
                    'reuse_connections': client.reuse_connections if predictor is None else None,
                    'transport': client.transport if predictor is None else None,
                    'json_backend': client.json_backend if predictor is None else None,
                    'response_format': client.response_format if predictor is None else None,
                    'input_format': client.input_format if predictor is None else None,
                    'base64_backend': client.base64_backend if predictor is None else None,
                    'socket_timeout_seconds': timeout},
        'tiles': {'total': result.tiles_total, 'executed': result.tiles_executed,
                  'skipped': result.tiles_skipped, 'predictor_calls': result.predictor_calls},
        'processing_seconds': processing_seconds,
        'timing_scope': 'normalization_tiling_predictor_and_stitching_excluding_bundle_verification_and_export',
        'files': files,
        'server_requested': requested_server,
        'predictor_invoked': predictor_invoked,
        'model_executed': None if predictor_invoked else False,
        'downloads_started': False,
        'server_started': False,
        'speedup_measured': False,
        'quality_evaluation': 'not_evaluated',
    }
    with (destination / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt
