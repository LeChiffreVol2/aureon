"""Operator-owned finite TerraMind service workflows; no downloads/model imports."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

from .contracts import PathInput
from ..cloud.control import require, encoded, digest
from ..scene_api.durable import _plain_path
from ..terramind_native.families import get_terramind_spec, TOKENIZER_ASSETS

Task = Literal['generation_scene', 'features']
SCHEMA = 'aureon.terramind_service_workflow.v1'
PROFILES = {'generation_scene': 'fp32-generation-scene-v1', 'features': 'fp32-features-v1'}
DEFAULT_LIMITS = {'max_input_bytes': 1024**3, 'max_output_bytes': 2*1024**3,
    'max_artifact_bytes': 1024**3, 'max_input_side': 8192, 'max_batch': 8,
    'max_buffer_bytes': 128*1024**2, 'max_decode_bytes': 32*1024**2,
    'min_free_bytes': 1024**3, 'max_scratch_bytes': 8*1024**3,
    'max_tenant_output_bytes': 8*1024**3, 'max_global_output_bytes': 16*1024**3,
    'request_deadline_seconds':45}


class WorkflowConfig(TypedDict):
    schema: str
    model_id: str
    task: Task
    model_file: str
    tokenizer_files: dict[str, str]
    options: dict[str, object]
    limits: dict[str, int]


class WorkflowBinding(TypedDict):
    schema: str
    model_id: str
    task: Task
    config_sha256: str
    asset_pins: dict[str, object]
    options: dict[str, object]
    limits: dict[str, int]


def _mapping(value: object) -> dict[str, object]:
    require(isinstance(value, dict) and all(isinstance(k, str) for k in value), 'invalid_workflow_config')
    return cast(dict[str, object], value)


def normalize(value: object, base: Path) -> WorkflowConfig:
    value = _mapping(value)
    require(set(value) == {'schema','model_id','task','model_file','tokenizer_files','options','limits'}
            and value['schema'] == SCHEMA, 'invalid_workflow_config')
    require(isinstance(value['model_id'], str) and isinstance(value['task'], str) and value['task'] in PROFILES, 'unsupported_workflow')
    task = cast(Task, value['task']); spec = get_terramind_spec(cast(str, value['model_id']))
    require((task == 'generation_scene' and spec.mode == 'generation') or
            (task == 'features' and spec.mode in ('backbone','tim')), 'workflow_model_mismatch')
    def path(item: object) -> str:
        require(isinstance(item, str) and bool(item) and '://' not in item and not item.startswith('/vsi'), 'operator_asset_path_required')
        selected = Path(cast(str, item))
        return str(_plain_path(selected if selected.is_absolute() else base/selected).resolve())
    options = _mapping(value['options'])
    needed: tuple[str, ...]
    if task == 'generation_scene':
        defaults: dict[str, object] = {'stride':224,'invalid_policy':'reject','execution_profile':'recommended','max_in_flight':1}
        require(not set(options)-set(defaults), 'unsupported_scene_options')
        options = {**defaults, **options}
        require(type(options['stride']) is int and 1 <= options['stride'] <= 224, 'invalid_scene_stride')
        require(options['invalid_policy'] in ('reject','skip_invalid_tiles') and options['execution_profile'] in ('reference','recommended'), 'invalid_scene_recipe')
        require(type(options['max_in_flight']) is int and 1 <= options['max_in_flight'] <= 16, 'invalid_scene_queue_bound')
        needed = ('S1GRD',)
    else:
        from ..terramind_native.encoder_loader import validate_options, required_tokenizers, MergeMethod
        defaults = {'modalities':['S2L2A'], 'tim_modalities':None, 'merge_method':'mean',
                    'tim_decoding_steps':1,'tim_temperature':1.0,'tim_top_p':0.8,'tim_top_k':0,'tim_chained_generations':True}
        require(not set(options)-set(defaults), 'unsupported_feature_options')
        options = {**defaults, **options}
        modalities = options['modalities']; targets = options['tim_modalities']
        require(isinstance(modalities, list) and all(isinstance(x, str) for x in modalities)
                and (targets is None or isinstance(targets, list) and all(isinstance(x, str) for x in targets)), 'invalid_feature_modalities')
        _, checked = validate_options(spec.model_id, cast(list[str], modalities), cast(list[str] | None, targets),
            cast(MergeMethod, options['merge_method']), cast(int, options['tim_decoding_steps']),
            cast(float, options['tim_temperature']), cast(float, options['tim_top_p']),
            cast(int, options['tim_top_k']), cast(bool, options['tim_chained_generations']))
        options = asdict(checked)
        options['modalities'] = list(checked.modalities)
        options['tim_modalities'] = list(checked.tim_modalities) if spec.mode == 'tim' else None
        needed = required_tokenizers(checked)
    tokenizers = _mapping(value['tokenizer_files'])
    require(set(tokenizers) == set(needed), 'workflow_tokenizer_set')
    supplied_limits = _mapping(value['limits'])
    require(not set(supplied_limits)-set(DEFAULT_LIMITS), 'unknown_workflow_limit')
    limits = {**DEFAULT_LIMITS, **supplied_limits}
    require(all(type(v) is int and 0 < v <= 2**40 for v in limits.values()), 'finite_workflow_limits_required')
    bounded = cast(dict[str, int], limits)
    require(bounded['max_input_bytes'] <= 1024**3 and bounded['max_output_bytes'] <= 4*1024**3
            and bounded['max_artifact_bytes'] <= bounded['max_output_bytes']
            and bounded['max_scratch_bytes'] >= 2*bounded['max_output_bytes']
            and bounded['max_buffer_bytes'] <= 512*1024**2 and bounded['max_decode_bytes'] <= 128*1024**2
            and bounded['max_input_side'] <= 32768 and bounded['max_batch'] <= 64
            and bounded['request_deadline_seconds'] <= 600
            and bounded['max_output_bytes'] <= bounded['max_tenant_output_bytes'] <= bounded['max_global_output_bytes'], 'inconsistent_workflow_limits')
    return {'schema':SCHEMA,'model_id':spec.model_id,'task':task,'model_file':path(value['model_file']),
            'tokenizer_files':{k:path(v) for k,v in tokenizers.items()},'options':options,'limits':bounded}


def load(path: PathInput) -> WorkflowConfig:
    path = _plain_path(path).resolve(strict=True)
    require(path.stat().st_size <= 64*1024, 'workflow_config_too_large')
    return normalize(json.loads(path.read_bytes()), path.parent)


def binding(config: WorkflowConfig) -> WorkflowBinding:
    spec = get_terramind_spec(config['model_id'])
    pins: dict[str, object] = {'model':asdict(spec.checkpoint),
        'tokenizers':{k:asdict(TOKENIZER_ASSETS[k]) for k in config['tokenizer_files']}}
    return {'schema':SCHEMA,'model_id':config['model_id'],'task':config['task'],
        'config_sha256':digest(encoded(config)),'asset_pins':pins,
        'options':deepcopy(config['options']),'limits':dict(config['limits'])}


def artifacts(task: str) -> set[str]:
    common = {'job.json','native-job.json','provenance.json'}
    if task == 'generation_scene': return common | {'synthetic-s1grd.tif','tiles.jsonl','preflight.json'}
    if task == 'features': return common | {'features.npz'}
    if task == 'generation': return common | {'synthetic-s1grd.tif','raw.npy','tokens.npy'}
    if task == 'scene': return common | {'class_map.tif','valid_mask.tif'}
    raise ValueError('Unknown service task')
