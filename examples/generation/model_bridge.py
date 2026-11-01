"""Offline CPU bridge example; no checkpoints, assets or service required."""
import torch
from aureon.api import ModelBridge, PreprocessingRecipe

torch.set_num_threads(1)
recipe = PreprocessingRecipe(('red', 'nir'), 'reflectance', 'reflectance')
prepared = recipe.apply(torch.ones(1, 2, 4, 4), bands=('red', 'nir'), units='reflectance')
model = torch.nn.Conv2d(2, 3, 1).eval()
with ModelBridge(model, forward=lambda m, x: m(x['image']),
                 output=lambda value: {'logits': value}) as bridge:
    result = bridge.predict({'image': prepared.tensor})
    assert torch.equal(result['logits'], model(prepared.tensor))
    print('CPU direct-forward parity passed:', tuple(result['logits'].shape))
