"""Offline, opt-in preparation of a KAO-TerraMind checkpoint for TerraTorch's vLLM wrapper.

This exports a candidate, never constructs a model or starts a service. The
source must match a pinned catalog identity. Numerical parity, strict model
loading, runtime compatibility and acceleration still require a separate run.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .catalog import get_catalog


VLLM_WRAPPER_BLOB_SHA = 'ce04acc72d379f0ebd8314d831a301ae02271a4f'
VLLM_WRAPPER_URL = 'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/terratorch.py'
_BACKBONE_ARGS = {
    'pretrained', 'modalities', 'out_indices', 'bands', 'img_size', 'patch_size',
    'merge_method', 'drop_path_rate', 'drop_rate', 'attn_drop_rate',
    'modality_drop_rate', 'encoder_norm', 'qk_norm',
}
_DECODER_ARGS = {'channels', 'pool_scales', 'align_corners', 'scale_modules'}
_HEAD_ARGS = {'dropout'}
_NECKS = {'SelectIndices', 'ReshapeTokensToImage', 'InterpolateToPyramidal',
          'MaxpoolToPyramidal', 'LearnedInterpolateToPyramidal'}
_TOP_ARGS = {'backbone', 'decoder', 'num_classes', 'necks', 'rescale',
             'image_size_out', 'backbone_kwargs', 'decoder_kwargs', 'head_kwargs'}
_TOP_ARGS |= {prefix + key for prefix, names in (
    ('backbone_', _BACKBONE_ARGS), ('decoder_', _DECODER_ARGS), ('head_', _HEAD_ARGS)
) for key in names}

LEGACY_SCALE_MIGRATION_ID = 'aureon.kao_terramind.learned_pyramid.v1'
LEGACY_SCALE_UPSTREAM_COMMIT = '89a3ed13d7a6272c009c4c5fbabf62461fcfb16c'
TENSOR_HASH_FORMAT = 'json_dtype_shape_nul_c_order_bytes_v1'
# Derived from the complete SHA-verified source artifacts, with TENSOR_HASH_FORMAT.
LEGACY_SCALE_SOURCE_MANIFEST_SHA256 = {
    'kao-terramind-a': 'b45873a7a4f2dd71a01f2d38a50204d0846da9cdd1cd496bd7c9ce46a5aded42',
    'kao-terramind-b': '72d4c28d1b02a6694190ee292c5f230c9d82903b02548cb370da8e7ca5c77f7c',
    'kao-terramind-c': '941b8ecc60f18d5edf5b9c81058e54a748f6d982bceb1f850ae40a0066d0a16d',
}
_LEGACY_SCALE_PATTERN = re.compile(r'^decoder\.fpn([1-4])\.')
_LEGACY_SCALE_SHAPES = {
    'decoder.fpn1.0.weight': [768, 384, 2, 2],
    'decoder.fpn1.0.bias': [384],
    'decoder.fpn1.1.weight': [384],
    'decoder.fpn1.1.bias': [384],
    'decoder.fpn1.1.running_mean': [384],
    'decoder.fpn1.1.running_var': [384],
    'decoder.fpn1.1.num_batches_tracked': [],
    'decoder.fpn1.3.weight': [384, 192, 2, 2],
    'decoder.fpn1.3.bias': [192],
    'decoder.fpn2.0.weight': [768, 384, 2, 2],
    'decoder.fpn2.0.bias': [384],
}


def pinned_legacy_model_args(model_id):
    """Arguments observed in the three SHA-bound published checkpoints."""
    if model_id not in ('kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'):
        raise ValueError('Legacy migration requires a pinned KAO-TerraMind variant')
    is_c = model_id == 'kao-terramind-c'
    return {
        'backbone': 'terramind_v1_base', 'backbone_pretrained': True,
        'backbone_modalities': ['S2L2A'], 'backbone_merge_method': 'mean',
        'num_classes': 10, 'head_dropout': 0.2 if is_c else 0.1,
        'decoder': 'UperNetDecoder', 'decoder_scale_modules': True,
        'decoder_channels': 512 if is_c else 256,
        'necks': [{'name': 'SelectIndices', 'indices': [3, 6, 9, 11] if is_c else [2, 5, 8, 11]},
                  {'name': 'ReshapeTokensToImage', 'remove_cls_token': False}],
    }


def legacy_scale_migration():
    return {'id': LEGACY_SCALE_MIGRATION_ID, 'upstream_commit': LEGACY_SCALE_UPSTREAM_COMMIT,
            'neck_index': 2, 'source_flag': 'decoder_scale_modules',
            'replacement_neck': 'LearnedInterpolateToPyramidal',
            'tensor_values_changed': False, 'numerical_parity_verified': False}


def _legacy_export_key(key):
    return _LEGACY_SCALE_PATTERN.sub(r'neck.2.fpn\1.', key, count=1)


def _manifest_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _tensor_record(tensor):
    import torch

    header = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    digest.update(b'\0')
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(raw))
    return {**header, 'sha256': digest.hexdigest()}


def _validate_legacy_manifest(model_id, manifest):
    if len(manifest) != 198 or _manifest_sha256(manifest) != LEGACY_SCALE_SOURCE_MANIFEST_SHA256.get(model_id):
        raise ValueError('Legacy source tensor manifest differs from the pinned checkpoint')
    scale_keys = {key for key in manifest if _LEGACY_SCALE_PATTERN.match(key)}
    if scale_keys != set(_LEGACY_SCALE_SHAPES):
        raise ValueError('Legacy migration must preserve exactly the eleven expected learned-pyramid tensors')
    for key, shape in _LEGACY_SCALE_SHAPES.items():
        dtype = 'torch.int64' if key.endswith('num_batches_tracked') else 'torch.float32'
        if manifest[key]['shape'] != shape or manifest[key]['dtype'] != dtype:
            raise ValueError('Unexpected learned-pyramid tensor shape or dtype: ' + key)


def _sha256(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    """Permit ordinary data only; no classes, tensors or executable objects."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) in (list, tuple):
        return [_json_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError('model_args must contain only JSON-compatible plain data')


def build_vllm_config(model_args, *, as_dict):
    """Translate actual checkpoint arguments, without guessing variant settings."""
    if type(model_args) is not dict or type(as_dict) is not bool:
        raise ValueError('checkpoint requires a model_args dict and an as_dict bool')
    args = _json_value(model_args)
    json.dumps(args, allow_nan=False)
    unknown = set(args) - _TOP_ARGS
    if unknown:
        raise ValueError(f'unsupported model_args fields require review: {sorted(unknown)}')
    if args.get('backbone') != 'terramind_v1_base' or args.get('decoder') != 'UperNetDecoder':
        raise ValueError('only terramind_v1_base with UperNetDecoder is admitted')
    if type(args.get('num_classes')) is not int or args['num_classes'] != 10:
        raise ValueError('KAO-TerraMind segmentation requires exactly 10 classes')
    for prefix, allowed in (('backbone', _BACKBONE_ARGS), ('decoder', _DECODER_ARGS), ('head', _HEAD_ARGS)):
        nested = args.get(prefix + '_kwargs', {})
        if type(nested) is not dict or set(nested) - allowed:
            raise ValueError(f'unsupported {prefix}_kwargs requires review')
        if nested and any(key.startswith(prefix + '_') and key != prefix + '_kwargs' for key in args):
            raise ValueError(f'ambiguous flattened and nested {prefix} arguments')
    backbone = args.get('backbone_kwargs') or {key[9:]: value for key, value in args.items() if key.startswith('backbone_') and key != 'backbone_kwargs'}
    if backbone.get('modalities') != ['S2L2A']:
        raise ValueError('the bundle requires explicit S2L2A-only modalities')
    if backbone.get('img_size', 224) not in (224, [224, 224]):
        raise ValueError('the input contract requires 224 x 224 model chips')
    necks = args.get('necks', [])
    if type(necks) is not list or any(type(neck) is not dict or neck.get('name') not in _NECKS for neck in necks):
        raise ValueError('unsupported neck configuration requires review')
    scale_modules = args.pop('decoder_scale_modules', args.get('decoder_kwargs', {}).get('scale_modules', False))
    if type(scale_modules) is not bool:
        raise ValueError('decoder_scale_modules must be a bool')
    if 'scale_modules' in args.get('decoder_kwargs', {}):
        args['decoder_kwargs'].pop('scale_modules')
    if scale_modules:
        # This is the exact legacy layout in the three pinned checkpoints.
        if (len(necks) != 2 or set(necks[0]) != {'name', 'indices'}
                or necks[0]['name'] != 'SelectIndices'
                or necks[0]['indices'] not in ([2, 5, 8, 11], [3, 6, 9, 11])
                or necks[1] != {'name': 'ReshapeTokensToImage', 'remove_cls_token': False}):
            raise ValueError('legacy scale_modules requires the audited two-neck KAO-TerraMind layout')
        args['necks'] = [*necks, {'name': 'LearnedInterpolateToPyramidal'}]
    # The checkpoint supplies every tensor; constructing a pretrained backbone
    # would trigger acquisition and could silently replace its original values.
    if args.get('backbone_kwargs'):
        args['backbone_kwargs']['pretrained'] = False
    else:
        args['backbone_pretrained'] = False
    inputs = {'data': {'S2L2A': {'type': 'torch.Tensor', 'shape': [12, 224, 224]}}}
    if not as_dict:
        inputs['target'] = 'S2L2A'
    return {
        'architectures': ['Terratorch'], 'num_classes': 0,
        'pretrained_cfg': {
            'model': {'class_path': 'terratorch.tasks.SemanticSegmentationTask', 'init_args': {
                'model_factory': 'EncoderDecoderFactory', 'model_args': args, 'ignore_index': 255,
            }},
            'input': inputs,
        },
    }


def serve_argv(bundle_dir, *, port=8000, dtype='float32', served_model_name=None,
               max_num_seqs=1, gpu_memory_utilization=0.4):
    """Return a bounded vLLM 0.19 candidate command; do not launch it.

    Each normalized tile uses one dummy token. Explicit TerraTorch selection
    also selects its tokenizer-free renderer. The conservative memory fraction
    leaves room for other GPU users; actual startup and parity still need tests.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('port must be an integer from 1 to 65535')
    if dtype not in ('float32', 'float16'):
        raise ValueError('dtype must be float32 or float16; lower precision requires parity validation')
    if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= 16:
        raise ValueError('max_num_seqs must be an integer from 1 to 16')
    if (type(gpu_memory_utilization) not in (int, float)
            or not 0 < gpu_memory_utilization <= 1):
        raise ValueError('gpu_memory_utilization must be a finite number in (0, 1]')
    if served_model_name is not None and (not isinstance(served_model_name, str) or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}', served_model_name)):
        raise ValueError('served_model_name must be a simple model identifier')
    argv = ['vllm', 'serve', str(Path(bundle_dir).resolve()), '--host', '127.0.0.1',
            '--port', str(port), '--skip-tokenizer-init', '--enforce-eager',
            '--enable-mm-embeds', '--dtype', dtype,
            '--model-impl', 'terratorch', '--runner', 'pooling',
            '--max-model-len', '1', '--max-num-seqs', str(max_num_seqs),
            '--max-num-batched-tokens', str(max_num_seqs),
            '--gpu-memory-utilization', str(gpu_memory_utilization),
            '--mm-processor-cache-gb', '0']
    if served_model_name is not None:
        argv.extend(['--served-model-name', served_model_name])
    return argv


def prepare_vllm_bundle(checkpoint_path, output_dir, *, model_id, expected_sha256):
    """Read a hash-bound local checkpoint with weights_only=True, export once.

    No fallback to unsafe pickle is allowed. Caller supplies an explicit local
    path and the expected catalog SHA-256. A new output directory is required;
    existing outputs and the source checkpoint are never overwritten. An error
    during writing may leave an incomplete directory for manual inspection.
    """
    model = next((row for row in get_catalog()['models'] if row['id'] == model_id and row['status'] == 'candidate'), None)
    if model is None:
        raise ValueError('model_id must identify a pinned KAO-TerraMind catalog candidate')
    identity = deepcopy(model['identity'])
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected_sha256 must be a lowercase SHA-256 digest')
    if expected_sha256 != identity['published_checkpoint_sha256']:
        raise ValueError('expected_sha256 must match the selected catalog identity')
    source = Path(checkpoint_path).resolve(strict=True)
    destination = Path(output_dir).absolute()
    if not source.is_file():
        raise ValueError('checkpoint_path must be a local regular file')
    if destination.exists():
        raise FileExistsError('output directory must not already exist')
    if source.stat().st_size != identity['published_checkpoint_bytes']:
        raise ValueError('checkpoint size does not match the selected catalog identity')
    import torch  # Lazy: importing metadata/planning never imports a neural runtime.
    with source.open('rb') as stream:
        if _sha256(stream) != expected_sha256:
            raise ValueError('checkpoint SHA-256 does not match the selected catalog identity')
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        if _sha256(stream) != expected_sha256:
            raise ValueError('checkpoint changed while it was being read')
    if not isinstance(checkpoint, Mapping):
        raise ValueError('checkpoint must contain model_args, as_dict and state_dict')
    config = build_vllm_config(checkpoint.get('model_args'), as_dict=checkpoint.get('as_dict'))
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping) or not state:
        raise ValueError('checkpoint requires a nonempty state_dict')
    if any(type(key) is not str or not key or type(value) is not torch.Tensor or value.layout != torch.strided for key, value in state.items()):
        raise ValueError('state_dict must map exact nonempty names to plain dense tensors')
    source_args = _json_value(checkpoint['model_args'])
    source_manifest = {key: _tensor_record(value) for key, value in state.items()}
    migration = None
    if source_args.get('decoder_scale_modules') is True:
        if source_args != pinned_legacy_model_args(model_id) or checkpoint['as_dict'] is not True:
            raise ValueError('Legacy source configuration differs from the pinned checkpoint')
        _validate_legacy_manifest(model_id, source_manifest)
        migration = legacy_scale_migration()
    elif source_args.get('decoder_kwargs', {}).get('scale_modules') is True:
        raise ValueError('Nested legacy scale_modules is not a pinned checkpoint configuration')
    export_state = {}
    rewrites = []
    for key, value in state.items():
        target_key = _legacy_export_key(key) if migration else key
        if target_key in export_state:
            raise ValueError('Tensor key migration would overwrite an existing tensor')
        export_state[target_key] = value
        if target_key != key:
            rewrites.append({'source': key, 'target': target_key})
    rewrites.sort(key=lambda item: item['source'])
    tensor_manifest = {key: _tensor_record(value) for key, value in export_state.items()}
    destination.mkdir(parents=False, exist_ok=False)
    weights_path = destination / 'pytorch_model.bin'
    # Use the wrapper's flat-tensor branch. The nested state_dict branch skips
    # positional embeddings and rewrites names; neither operation is acceptable.
    with weights_path.open('xb') as stream:
        torch.save(export_state, stream)
    config_path = destination / 'config.json'
    with config_path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write('\n')
    hashes = {}
    for path in (weights_path, config_path):
        with path.open('rb') as stream:
            hashes[path.name] = _sha256(stream)
    receipt = {
        'schema': 'aureon.vllm_bundle.v1', 'status': 'candidate_not_parity_verified',
        'model_id': model_id, 'source_identity': identity,
        'source_checkpoint': str(source), 'verified_source_sha256': expected_sha256,
        'bundle_directory': str(destination), 'files_sha256': hashes,
        'source_as_dict': checkpoint['as_dict'], 'tensor_manifest': tensor_manifest,
        'source_model_args': source_args,
        'resolved_model_args': config['pretrained_cfg']['model']['init_args']['model_args'],
        'source_tensor_manifest': source_manifest,
        'source_tensor_manifest_sha256': _manifest_sha256(source_manifest),
        'tensor_manifest_sha256': _manifest_sha256(tensor_manifest),
        'tensor_hash_format': TENSOR_HASH_FORMAT, 'architecture_migration': migration,
        'transforms': ['state_dict extracted as a flat raw tensor mapping',
                       'model_args copied into JSON; tuples represented as arrays',
                       'backbone pretrained disabled; all tensors come from checkpoint'],
        'state_key_rewrites': rewrites, 'state_tensors_dropped': [], 'dtype_conversion': None,
        'strict_model_load_verified': False, 'numerical_parity_verified': False,
        'runtime_versions_verified': False, 'speedup_measured': False,
        'audited_vllm_wrapper': {'url': VLLM_WRAPPER_URL, 'git_blob_sha': VLLM_WRAPPER_BLOB_SHA,
                                'note': 'Source-file identity, not a release or commit pin.'},
        'serve_argv_candidate': serve_argv(destination, served_model_name=model_id),
        'model_executed': False, 'downloads_started': False, 'server_started': False,
    }
    if migration:
        receipt['transforms'].append('legacy decoder scale modules moved to learned-pyramid neck 2; values and dtypes unchanged')
    with (destination / 'bundle_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt
