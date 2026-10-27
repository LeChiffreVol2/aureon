"""Published pinned asset identities. Listing never downloads or loads weights."""
from ..catalog import get_catalog
from ..terramind_native.common import MODEL_SHA, TOKENIZER_SHA


def list_assets(model_id=None):
    """Return official revision-specific sources for separately obtained assets."""
    assets = []
    for model in get_catalog()['models']:
        identity = model['identity']
        repo, revision, filename = (identity[k] for k in ('repo_id', 'revision', 'checkpoint'))
        assets.append({'model_id': model['id'], 'role': 'checkpoint', 'repo_id': repo,
                       'revision': revision, 'filename': filename,
                       'sha256': identity['published_checkpoint_sha256'],
                       'bytes': identity['published_checkpoint_bytes']})
    for role, repo, revision, filename, digest, size in (
        ('checkpoint', 'ibm-esa-geospatial/TerraMind-1.0-base',
         'fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a', 'TerraMind_v1_base.pt', MODEL_SHA, 1518713958),
        ('tokenizer', 'ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD',
         '38937fd35012c2dcf81987b400cec725b5eb0466', 'TerraMind_Tokenizer_S1GRD.pt', TOKENIZER_SHA, 1146593994),
    ):
        assets.append({'model_id': 'terramind-v1-base', 'role': role, 'repo_id': repo,
                       'revision': revision, 'filename': filename, 'sha256': digest, 'bytes': size})
    for asset in assets:
        root = 'https://huggingface.co/' + asset['repo_id']
        asset['source_url'] = root + '/resolve/' + asset['revision'] + '/' + asset['filename']
        asset['model_card_url'] = root + '/blob/' + asset['revision'] + '/README.md'
    if model_id is not None:
        assets = [a for a in assets if a['model_id'] == model_id]
        if not assets:
            raise ValueError('Unknown model_id: ' + str(model_id))
    return assets
