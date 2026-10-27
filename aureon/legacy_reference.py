# AUREON review candidate - adaptation and modification notice.
# Learned pyramid layout and forward ordering adapted from TerraTorch commit
# 948b5535092f847e1ddc90e3f7ef7a67094725c0 (Apache-2.0).
# AUREON changes: validated lazy subclass/registration and delegation to the
# externally installed UPerNet core. The upstream decoder credits MMSegmentation.
# Migration reference: TerraTorch 89a3ed13d7a6272c009c4c5fbabf62461fcfb16c;
# Copyright contributors to the Terratorch project (migration-test attribution).
# See NOTICE and THIRD-PARTY-NOTICES.md for sources and license copies.

"""Explicit reference for the original KAO-TerraMind decoder checkpoint layout.

The scale-module layout and forward order follow TerraTorch at commit
948b5535092f847e1ddc90e3f7ef7a67094725c0. The unchanged UPerNet core is supplied
by the installed TerraTorch; the learned pyramid stays inside this decoder,
so the original checkpoint loads with no state-key rewrites. Imports are lazy.
This is a parity reference, not a serving architecture or a speed claim.
"""
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from .catalog import get_catalog
from .reference import PyTorchPredictor, _stream_hash


LEGACY_SOURCE_URL = (
    'https://github.com/torchgeo/terratorch/blob/'
    '948b5535092f847e1ddc90e3f7ef7a67094725c0/'
    'terratorch/models/decoders/upernet_decoder.py'
)
MIGRATION_TEST_SOURCE_URL = (
    'https://github.com/torchgeo/terratorch/blob/'
    '89a3ed13d7a6272c009c4c5fbabf62461fcfb16c/tests/test_legacy_scale_modules.py'
)
LEGACY_DECODER_NAME = 'AureonLegacyUperNetDecoder'


@lru_cache(maxsize=1)
def legacy_decoder_class():
    """Register the original learned pyramid under a separate decoder name.

    No existing registry entry is replaced. The parent UPerNet implementation
    receives the scaled channel dimensions, exactly as the historical decoder
    did after constructing fpn1..4. Its forward runs after those four modules.
    """
    from torch import nn
    from terratorch.models.decoders.upernet_decoder import UperNetDecoder
    from terratorch.registry import TERRATORCH_DECODER_REGISTRY

    class AureonLegacyUperNetDecoder(UperNetDecoder):
        def __init__(self, embed_dim, pool_scales=(1, 2, 3, 6), channels=256,
                     align_corners=True, scale_modules=True):
            if type(scale_modules) is not bool or not scale_modules:
                raise ValueError('The legacy KAO-TerraMind reference requires scale_modules=True.')
            if (not isinstance(embed_dim, (list, tuple)) or len(embed_dim) != 4
                    or any(type(n) is not int or n < 4 for n in embed_dim)
                    or embed_dim[0] % 4 or embed_dim[1] % 2):
                raise ValueError('Legacy learned scaling requires four compatible embedding dimensions.')
            original_dims = list(embed_dim)
            scaled_dims = [original_dims[0] // 4, original_dims[1] // 2,
                           original_dims[2], original_dims[3]]
            super().__init__(scaled_dims, pool_scales=pool_scales, channels=channels,
                             align_corners=align_corners)
            self.scale_modules = True
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(original_dims[0], original_dims[0] // 2, 2, 2),
                nn.BatchNorm2d(original_dims[0] // 2),
                nn.GELU(),
                nn.ConvTranspose2d(original_dims[0] // 2, original_dims[0] // 4, 2, 2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(original_dims[1], original_dims[1] // 2, 2, 2),
            )
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        def forward(self, inputs):
            if len(inputs) != 4:
                raise ValueError('Legacy KAO-TerraMind decoder requires four feature maps.')
            scaled_inputs = [self.fpn1(inputs[0]), self.fpn2(inputs[1]),
                             self.fpn3(inputs[2]), self.fpn4(inputs[3])]
            return super().forward(scaled_inputs)

    if LEGACY_DECODER_NAME in TERRATORCH_DECODER_REGISTRY:
        raise RuntimeError('The AUREON legacy decoder registry name is already occupied.')
    TERRATORCH_DECODER_REGISTRY.register(AureonLegacyUperNetDecoder)
    return AureonLegacyUperNetDecoder


def _reference_model_args(source_args):
    """Admit the original pinned architecture; keep its two original necks."""
    if type(source_args) is not dict:
        raise ValueError('Original checkpoint must contain a plain model_args dict.')
    args = deepcopy(source_args)
    common = {
        'backbone': 'terramind_v1_base', 'backbone_modalities': ['S2L2A'],
        'backbone_merge_method': 'mean', 'num_classes': 10,
        'decoder': 'UperNetDecoder', 'decoder_scale_modules': True,
    }
    if any(args.get(k) != v for k, v in common.items()):
        raise ValueError('Original checkpoint is not the admitted legacy KAO-TerraMind architecture.')
    expected_keys = set(common) | {'backbone_pretrained', 'head_dropout', 'decoder_channels', 'necks'}
    if set(args) != expected_keys:
        raise ValueError('Unexpected original model_args fields require source review.')
    necks = args.get('necks')
    if (type(necks) is not list or len(necks) != 2
            or necks[0] not in ({'name': 'SelectIndices', 'indices': [2, 5, 8, 11]},
                                {'name': 'SelectIndices', 'indices': [3, 6, 9, 11]})
            or necks[1] != {'name': 'ReshapeTokensToImage', 'remove_cls_token': False}
            or args['decoder_channels'] not in (256, 512)
            or args['head_dropout'] not in (0.1, 0.2)):
        raise ValueError('Unexpected original decoder or neck configuration requires source review.')
    args['backbone_pretrained'] = False
    args['decoder'] = LEGACY_DECODER_NAME
    return args


class LegacyPyTorchPredictor(PyTorchPredictor):
    """Strictly load an original hash-pinned checkpoint without key migration.

    The normalized-batch callable and synchronize API match PyTorchPredictor.
    Construction is explicit; no weights or pretrained assets are acquired.
    """
    execution_backend = 'pytorch_legacy_checkpoint_reference'

    def __init__(self, checkpoint_path, model_id, *, device='cuda', dtype='float32'):
        if not isinstance(device, str) or dtype not in ('float32', 'float16'):
            raise ValueError('Explicit cpu/cuda device and float32/float16 precision are required.')
        row = next((m for m in get_catalog()['models'] if m['id'] == model_id), None)
        if row is None:
            raise ValueError('model_id must identify a pinned KAO-TerraMind candidate.')
        identity = deepcopy(row['identity'])
        source = Path(checkpoint_path).resolve(strict=True)
        if not source.is_file() or source.stat().st_size != identity['published_checkpoint_bytes']:
            raise ValueError('Original checkpoint size does not match the catalog identity.')
        expected_hash = identity['published_checkpoint_sha256']
        import torch
        try:
            selected_device = torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError('device must be cpu, cuda or an indexed CUDA device.') from exc
        if (selected_device.type not in ('cpu', 'cuda')
                or (selected_device.type == 'cpu' and selected_device.index is not None)):
            raise ValueError('device must be cpu, cuda or an indexed CUDA device.')
        if selected_device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available; explicitly choose cpu to use it.')
            if selected_device.index is not None and selected_device.index >= torch.cuda.device_count():
                raise ValueError('Requested CUDA device index is not available.')
        with source.open('rb') as stream:
            if _stream_hash(stream) != expected_hash:
                raise ValueError('Original checkpoint SHA-256 does not match the catalog identity.')
            stream.seek(0)
            checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
            if _stream_hash(stream) != expected_hash:
                raise ValueError('Original checkpoint changed while it was being read.')
        if not isinstance(checkpoint, Mapping) or type(checkpoint.get('as_dict')) is not bool:
            raise ValueError('Original checkpoint requires model_args, as_dict and state_dict.')
        args = _reference_model_args(checkpoint.get('model_args'))
        state = checkpoint.get('state_dict')
        if (not isinstance(state, Mapping) or not state
                or any(type(k) is not str or type(v) is not torch.Tensor
                       or v.layout != torch.strided for k, v in state.items())):
            raise ValueError('Original state_dict must be a nonempty mapping of plain dense tensors.')
        legacy_decoder_class()
        from terratorch.models import EncoderDecoderFactory
        model = EncoderDecoderFactory().build_model('segmentation', **args)
        if set(model.state_dict()) != set(state):
            raise ValueError('Original state keys do not exactly match the unmigrated reference model.')
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError('Strict legacy loading reported missing or unexpected keys.')
        self._torch = torch
        self._device = selected_device
        self.device = str(selected_device)
        self.dtype = dtype
        self._torch_dtype = torch.float32 if dtype == 'float32' else torch.float16
        self.model_id = model_id
        self._as_dict = checkpoint['as_dict']
        self.source_identity = identity
        self.source_checkpoint = str(source)
        self.source_checkpoint_sha256 = expected_hash
        self.source_model_args = deepcopy(checkpoint['model_args'])
        self.reference_provenance = {
            'legacy_decoder_source': LEGACY_SOURCE_URL,
            'upstream_migration_tests': MIGRATION_TEST_SOURCE_URL,
            'state_key_rewrites': [], 'state_tensors_dropped': [],
            'strict_model_load_verified': True,
            'decoder_pyramid_location': 'decoder.fpn1..fpn4',
            'core_decoder': 'installed TerraTorch UperNetDecoder',
        }
        model.requires_grad_(False)
        model.eval()
        model.to(device=self.device, dtype=self._torch_dtype)
        self.model = model
