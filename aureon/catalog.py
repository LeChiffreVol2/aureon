"""Pinned KAO-TerraMind metadata and offline inference preparation plans.

Published metadata is not local checkpoint verification. Nothing in this module
downloads weights, imports a neural runtime, reads projects, or dispatches jobs.
"""
from copy import deepcopy
import hashlib
import json


BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
CLASS_NAMES = ('Maize', 'Sugarcane', 'Cassava', 'Rice', 'Other Crops', 'Water',
               'Trees', 'Flooded Veg.', 'Built Area', 'Other')
RECIPE_ID = 'aureon-landcover-v1'
UPSTREAM_REVISION = '2d8a9c8953e9bfbec01fae4d80ce5ffcba67c777'
UPSTREAM_URL = 'https://github.com/kaopanboonyuen/KAO-TerraMind/tree/' + UPSTREAM_REVISION
TERRATORCH_VLLM_URL = 'https://torchgeo.org/terratorch/stable/guide/vllm/intro/'
TERRATORCH_VLLM_CONFIG_URL = 'https://torchgeo.org/terratorch/stable/guide/vllm/prepare_your_model/'
NORMALIZATION_SOURCE_URL = 'https://huggingface.co/kaopanboonyuen/KAO-TerraMind-C/blob/4e7252d3c83b3e877744da214005184a4133883f/kao_terramind_inference.py'

_RECIPE_CONFIG = {
    'id': RECIPE_ID,
    'version': 1,
    'task': 'semantic-segmentation',
    'input': {
        'modality': 'Sentinel-2 L2A',
        'units': 'surface_reflectance_x10000',
        'band_order': list(BANDS),
        'source_validity_required': True,
        'normalization': {
            'operation': '(value - mean) / std',
            'dtype': 'float32',
            'mean': [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
                     2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648],
            'std': [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
                    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
            'source_url': NORMALIZATION_SOURCE_URL,
            'clipping': False,
            'additional_scale_or_offset': False,
        },
        'mask_policy': {
            'shape': '[H,W]',
            'dtype': 'bool',
            'meaning': 'caller_declared_validity_for_all_12_bands',
            'valid_samples': 'all_bands_finite',
            'zero_is_missing': False,
            'invalid_fill': 'normalized_zero',
            'invalid_input_samples': 'ignored',
        },
    },
    'tiling': {
        'window_size': 224,
        'stride': 168,
        'blend': 'uniform_logits',
        'anchors': 'start_at_zero_stride_then_append_far_edge_if_missing',
        'padding': {'edges': 'bottom_right', 'space': 'normalized', 'value': 0.0},
        'fully_invalid_windows': 'skip',
        'accumulation_dtype': 'float32',
        'accumulation_order': 'row_major_window_order',
        'max_side': 1024,
        'max_pixels': 1048576,
        'reference_batch_size': 1,
        'max_batch_size': 16,
    },
    'output': {
        'num_classes': 10,
        'class_names': list(CLASS_NAMES),
        'class_names_order': 'zero_based_class_index',
        'class_names_source_url': NORMALIZATION_SOURCE_URL,
        'class_names_verification': 'published_helper_not_local_checkpoint',
        'nodata': 255,
        'classes_dtype': 'uint8',
        'decision': 'argmax_after_logit_blending',
        'tie_policy': 'first_class_index',
        'score': 'softmax',
        'score_output': 'optional_maximum_after_logit_blending',
        'score_calibration': 'uncalibrated',
        'score_dtype': 'float32',
        'invalid_score': 'NaN',
        'preserve_source_grid': True,
    },
}
_RECIPE = {
    **_RECIPE_CONFIG,
    'status': 'draft',
    'identity': {
        'id': RECIPE_ID,
        'version': 1,
        'configuration_sha256': hashlib.sha256(json.dumps(
            _RECIPE_CONFIG, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode('utf-8')).hexdigest(),
        'hash_scope': 'draft_configuration_only',
    },
    'checkpoint_parity': 'not_verified',
    'quality_evaluation': 'not_evaluated',
    'improvement_claim': False,
    'intended_backend': 'vllm',
    'description': 'Draft AUREON tiling and delivery recipe. Synthetic recipe checks do not establish checkpoint parity or task quality.',
}

_VARIANTS = (
    ('a', '24a8ae9631f253d03e4f6b793315a7c15bf4c839',
     '6ccd4375fabc4fde5620ac27d663056009bf7b43f5a03ba5695c10521d58a496', 398251303, 40.81),
    ('b', '9698b737e6fdbdbc3f404a4a64cb70799472c2c8',
     'd5cd2c97ce853971c847eb1a423c6d5ba996eff8ebc0e1e64d116bed64e5e23a', 398251303, 40.97),
    ('c', '4e7252d3c83b3e877744da214005184a4133883f',
     '62ac6062623210661651e90010571c7a544cbb4c5693769716ccf470adb90fea', 487767335, 41.13),
)


def _candidate(letter, revision, digest, size, miou):
    name = 'KAO-TerraMind-' + letter.upper()
    repo_id = 'kaopanboonyuen/' + name
    url = 'https://huggingface.co/' + repo_id
    return {
        'id': 'kao-terramind-' + letter,
        'name': name,
        'status': 'candidate',
        'task': 'semantic-segmentation',
        'execution_enabled': False,
        'endpoint': None,
        'identity': {
            'repo_id': repo_id,
            'revision': revision,
            'checkpoint': 'terramind_v1_kao' + letter.upper() + '_best.pt',
            'published_checkpoint_sha256': digest,
            'published_checkpoint_bytes': size,
            'digest_source': 'Hugging Face published LFS metadata',
            'locally_verified': False,
        },
        'published_configuration': {
            'model_type': 'kao-terramind',
            'library': 'terratorch',
            'task': 'semantic-segmentation',
            'num_classes': 10,
            'class_names': list(CLASS_NAMES),
            'input_modality': 'Sentinel-2 L2A',
            'input_bands': 12,
            'input_size': 224,
            'auto_map_present': False,
        },
        'recipe_ids': [RECIPE_ID],
        'backend': 'vllm',
        'author_reported_metrics': {
            'miou_percent': miou,
            'reference_supervised_base_miou_percent': 40.82,
            'evidence': 'author_reported_not_locally_reproduced',
            'reference': 'Upstream task-trained supervised TerraMind Base benchmark; not an AUREON inference result.',
            'source_url': UPSTREAM_URL,
            'generalization_claim': False,
        },
        'sources': {
            'model_card': url + '/blob/' + revision + '/README.md',
            'configuration': url + '/blob/' + revision + '/config.json',
            'inference_helper': url + '/blob/' + revision + '/kao_terramind_inference.py',
            'repository': url + '/tree/' + revision,
            'upstream_repository': UPSTREAM_URL,
        },
    }


_MODELS = [_candidate(*variant) for variant in _VARIANTS]

_BACKENDS = [
    {
        'id': 'vllm',
        'name': 'vLLM through TerraTorch',
        'recommended': False,
        'recommended_for': ['kao-terramind-a', 'kao-terramind-c'],
        'status': 'supported_integration_kao_validation_required',
        'description': 'Official TerraTorch integration supports semantic segmentation. Prepare the KAO-TerraMind checkpoint/config bundle, then validate tensor-to-tensor serving with AUREON preprocessing and stitching.',
        'model_implementation': 'terratorch',
        'mode': 'tensor-to-tensor',
        'serving_endpoint': '/pooling',
        'io_processor_plugin': None,
        'input_modality': 'S2L2A',
        'input_transport': 'image_embeds',
        'config_contract': {
            'architectures': ['Terratorch'],
            'num_classes': 0,
            'pretrained_cfg_required': ['input', 'model'],
            'segmentation_classes_location': 'pretrained_cfg.model.init_args.model_args.num_classes',
            'auto_map_required': False,
        },
        'io_processor_note': 'The documented built-in TerraMind segmentation example uses an ImpactMesh multimodal pipeline. Do not apply it blindly to a KAO-TerraMind S2-only checkpoint; an AUREON IO processor is optional future integration.',
        'source_url': TERRATORCH_VLLM_URL,
        'configuration_source_url': TERRATORCH_VLLM_CONFIG_URL,
    },
    {
        'id': 'pytorch',
        'name': 'PyTorch / TerraTorch',
        'recommended': True,
        'recommended_for': ['kao-terramind-b'],
        'status': 'adapter_validation_required',
        'description': 'Primary KAO-TerraMind B service backend after matched HTTP/FP32 measurements; also the strict parity reference. Explicit local service startup and verified checkpoint loading remain required.',
    },
]

_BLOCKERS = [
    {
        'code': 'checkpoint_not_locally_verified',
        'detail': 'This static plan does not inspect local weights. Verify the pinned revision and digest; saved acquisition receipts separately record completed checkpoint checks.',
    },
    {
        'code': 'adapter_not_validated',
        'detail': 'Validate checkpoint structure, model_args, strict state loading and the segmentation output wrapper with the pinned upstream helper.',
    },
    {
        'code': 'preprocessing_and_class_contract_unverified',
        'detail': 'Confirm model normalization, exact band order, ten-class mapping, source units and nodata behavior against checkpoint metadata and upstream code.',
    },
    {
        'code': 'recipe_checkpoint_parity_unverified',
        'detail': 'Consult saved same-input, same-precision parity receipts for numerical evidence. Independent task-quality evaluation is still required; softmax scores are uncalibrated.',
    },
    {
        'code': 'live_service_not_verified',
        'detail': 'Explicit inference is available through python -m aureon infer after bundle preparation and manual service startup. This offline plan does not probe a service or establish its readiness.',
    },
]


def default_backend(model_id):
    return 'pytorch' if model_id == 'kao-terramind-b' else 'vllm'


_DEFAULT_BACKEND = object()


def get_catalog():
    """Return independent JSON-serializable metadata; perform no I/O or execution."""
    return deepcopy({
        'schema': 'aureon.inference_catalog.v1',
        'models': _MODELS,
        'recipes': [_RECIPE],
        'backends': _BACKENDS,
        'planning_only': True,
        'automatic_downloads': False,
        'automatic_execution': False,
        'primary_model': 'kao-terramind-b',
        'recommended_backend': 'pytorch',
        'recommended_backend_by_model': {m['id']: default_backend(m['id']) for m in _MODELS},
        'primary_service': {'command': 'python -m aureon serve-model', 'model_id': 'kao-terramind-b',
                            'backend': 'pytorch', 'precision': 'highest_fp32', 'explicit_start_required': True},
    })


def plan_inference(model_id, recipe_id=RECIPE_ID, backend=_DEFAULT_BACKEND):
    """Describe a candidate request without accepting or running an inference job."""
    if not isinstance(model_id, str) or model_id not in {m['id'] for m in _MODELS}:
        raise ValueError('Unknown model ID.')
    if not isinstance(recipe_id, str) or recipe_id != RECIPE_ID:
        raise ValueError('Unknown recipe ID.')
    if backend is _DEFAULT_BACKEND:
        backend = default_backend(model_id)
    if not isinstance(backend, str) or backend not in {'pytorch', 'vllm'}:
        raise ValueError('Unknown backend. Choose pytorch or vllm.')
    model = next(model for model in _MODELS if model['id'] == model_id)
    blockers = deepcopy(_BLOCKERS)
    if backend == 'vllm':
        blockers.extend([
            {
                'code': 'vllm_kao_bundle_unverified',
                'detail': 'Create a TerraTorch serving config and weights bundle from the actual pinned checkpoint model_args/state_dict. The published descriptive KAO-TerraMind config is not this serving bundle.',
                'source_url': TERRATORCH_VLLM_CONFIG_URL,
            },
            {
                'code': 'vllm_runtime_and_parity_unverified',
                'detail': 'Validate a compatible pinned vLLM/TerraTorch runtime and compare tensor-to-tensor outputs with the direct PyTorch reference. Installed plugin source alone does not establish a running vLLM service.',
                'source_url': TERRATORCH_VLLM_URL,
            },
            {
                'code': 'acceleration_not_measured',
                'detail': 'Measure warm-model latency, throughput and memory against a same-precision baseline before claiming a speedup. FP16 is a separate numerical-parity candidate.',
            },
        ])
    return deepcopy({
        'schema': 'aureon.inference_plan.v1',
        'model_id': model_id,
        'model_identity': model['identity'],
        'recipe_id': recipe_id,
        'recipe_identity': _RECIPE['identity'],
        'recipe': _RECIPE,
        'backend': backend,
        'backend_status': next(row['status'] for row in _BACKENDS if row['id'] == backend),
        'serving': {
            'mode': 'tensor-to-tensor' if backend == 'vllm' else 'direct_reference',
            'model_implementation': 'terratorch',
            'endpoint': '/pooling' if backend == 'vllm' else None,
            'io_processor_plugin': None,
            'input_modality': 'S2L2A',
            'preprocessing_owner': 'AUREON versioned recipe',
            'stitching_owner': 'AUREON versioned recipe',
        },
        'acceleration_recipe': {
            'status': 'unbenchmarked_candidate',
            'warm_model': True,
            'bounded_request_scheduling': True,
            'bounded_tile_batching': True,
            'precision_candidate': 'float16',
            'reference_comparison': 'same_precision_same_inputs_same_tiling',
            'overlap_stride': 168,
            'overlap_note': 'Overlapping 224-pixel windows add compute; this is a reference quality recipe, not a measured acceleration.',
            'speedup_measured': False,
        },
        'task': model['task'],
        'status': 'blocked',
        'ready': False,
        'execution_enabled': False,
        'endpoint': None,
        'blockers': blockers,
        'model_executed': False,
        'downloads_started': False,
        'output': None,
        'sources': model['sources'],
    })
