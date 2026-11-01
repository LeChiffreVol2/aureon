"""Verify local checkpoint bundles without loading a model or contacting a service."""

from __future__ import annotations
from typing import cast
from .value_types import JsonObject, Pathish
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .catalog import get_catalog
from .checkpoint_bundle import (TENSOR_HASH_FORMAT, _legacy_export_key, _manifest_sha256,
                                _validate_legacy_manifest, build_checkpoint_config,
                                legacy_scale_migration, pinned_legacy_model_args)


_BUNDLE_FILES = {'config.json', 'pytorch_model.bin'}
_IDENTITY_KEYS = ('repo_id', 'revision', 'checkpoint', 'published_checkpoint_sha256',
                  'published_checkpoint_bytes')
_MAX_JSON_BYTES = 4 * 1024 * 1024


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path) -> JsonObject:
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
    return cast(JsonObject, value)


def _member(folder: Path, name: str) -> Path:
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:
        raise ValueError('Bundle members must be regular local files inside the bundle.')
    return path


def _verify_tensor_metadata(receipt: JsonObject, config: JsonObject) -> None:
    """Validate preparation metadata without deserializing weights.

    For migrated published artifacts the source manifest is independently pinned
    in code. The file hash binds the exported bytes; tensor metadata describes the
    preparation operation and is not an attestation of a running server.
    """
    source_args = cast(JsonObject, receipt.get('source_model_args'))
    source_manifest = cast(dict[str, JsonObject], receipt.get('source_tensor_manifest'))
    exported_manifest = receipt.get('tensor_manifest')
    if receipt.get('tensor_hash_format') != TENSOR_HASH_FORMAT:
        raise ValueError('Unsupported tensor hash format in bundle receipt.')
    try:
        args = config['pretrained_cfg']['model']['init_args']['model_args']
        source_config = build_checkpoint_config(source_args, as_dict=receipt['source_as_dict'])
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


def verify_bundle(bundle_dir: Pathish) -> JsonObject:
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
    # Preserve the exact on-disk schema of previously prepared public-alpha bundles.
    if receipt.get('schema') not in {'aureon.checkpoint_bundle.v1', 'aureon.vllm_bundle.v1'}:
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
        expected_config = build_checkpoint_config(args, as_dict=receipt['source_as_dict'])
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
