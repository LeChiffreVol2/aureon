"""Published pinned asset identities. Listing never downloads or loads weights."""
from __future__ import annotations
from typing import Any
from ..catalog import get_catalog
from dataclasses import asdict
from ..terramind_native.families import list_terramind_specs, TOKENIZER_ASSETS


def list_assets(model_id: str | None=None) -> list[dict[str, Any]]:
    """Return official revision-specific sources for separately obtained assets."""
    assets = []
    for model in get_catalog()['models']:
        identity = model['identity']
        repo, revision, filename = (identity[k] for k in ('repo_id', 'revision', 'checkpoint'))
        assets.append({'model_id': model['id'], 'role': 'checkpoint', 'repo_id': repo,
                       'revision': revision, 'filename': filename,
                       'sha256': identity['published_checkpoint_sha256'],
                       'bytes': identity['published_checkpoint_bytes']})
    for spec in list_terramind_specs():
        assets.append({"model_id": spec.model_id, "role": "checkpoint", **asdict(spec.checkpoint)})
        modalities = ("S1GRD",) if spec.mode == "generation" else tuple(TOKENIZER_ASSETS) if spec.mode == "tim" else ("LULC", "NDVI")
        for modality in modalities:
            assets.append({"model_id": spec.model_id, "role": "tokenizer", "modality": modality,
                           "requirement": "only for selected generation/TiM targets or tokenized LULC/NDVI inputs",
                           **asdict(TOKENIZER_ASSETS[modality])})
    for asset in assets:
        root = 'https://huggingface.co/' + asset['repo_id']
        asset['source_url'] = root + '/resolve/' + asset['revision'] + '/' + asset['filename']
        asset['model_card_url'] = root + '/blob/' + asset['revision'] + '/README.md'
    if model_id is not None:
        assets = [a for a in assets if a['model_id'] == model_id]
        if not assets:
            raise ValueError('Unknown model_id: ' + str(model_id))
    return assets
