"""Offline CPU example: caller-owned model, explicit registry, upstream pooling.

Run with the installed generation profile. Uses random weights, no model assets.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', LIGHTLY_DID_VERSION_CHECK='True',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
import torch
from torch import nn
from aureon.api import (AdapterRegistry, AdapterSpec, AdapterRequest, DependencyPin,
                        BorrowedModelBridge, DeviceTensorPolicy, TemporalSpec,
                        TemporalInput, TemporalBridge, GeographicConditioning)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 3, 1)
        self.out_channels = [3]

    def forward(self, image):
        return [self.projection(image)]


def create(request, model):
    return BorrowedModelBridge(model,
        forward=lambda module, values: module(values['image']),
        output=lambda values: {'features': values[0]},
        policy=DeviceTensorPolicy(request.device, request.allow_unqualified_cuda))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = Encoder().eval()
    registry = AdapterRegistry()
    registry.register(AdapterSpec('example-encoder', ('image',), ('features',), ('cpu',),
        (DependencyPin('torch', ('2.10.0', '2.11.0')),)), create)
    request = AdapterRequest('image', 'features')
    assert registry.plan('example-encoder', request).admitted
    with registry.create('example-encoder', request, model) as bridge:
        result = bridge.predict({'image': torch.ones(1, 2, 4, 5)})
        assert result['features'].shape == (1, 3, 4, 5)
    spec = TemporalSpec(('red', 'nir'), 'caller_normalized',
                        ('2026-01-01T00:00:00Z', '2026-02-01T00:00:00Z'))
    frames = TemporalInput(torch.ones(1, 2, 2, 4, 5), torch.ones(1, 2, 4, 5, dtype=torch.bool), spec)
    with TemporalBridge(model, spec=spec, pooling='mean') as bridge:
        features = bridge.predict(frames)
        assert features.values[0].shape == (1, 3, 4, 5)
        assert not features.coordinate_conditioning_used
    coordinates = GeographicConditioning(((13., 100.),), 'unit_sphere_xyz')
    assert len(coordinates.encoded()[0]) == 3  # Separate; not passed into this encoder.
    assert not torch.cuda.is_initialized()
    print('CPU borrowed model, registered adapter and upstream temporal pooling passed.')


if __name__ == '__main__': main()
