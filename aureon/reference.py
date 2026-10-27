"""Direct PyTorch reference for comparison with the TerraTorch/vLLM backend.

Importing this module does not import a neural runtime or construct a model.
Construction is explicit, accepts only a verified local candidate bundle, and
loads its complete state strictly without requesting pretrained weights.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .runtime import verify_bundle


_INPUT_SHAPE = (12, 224, 224)
_OUTPUT_SHAPE = (10, 224, 224)


def _stream_hash(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


class PyTorchPredictor:
    """Strict direct reference, consuming already normalized NumPy tile batches.

    ``device`` is explicitly ``cpu``, ``cuda`` or an indexed CUDA device such as
    ``cuda:0``. Input and parameter precision use the same declared dtype. Lower
    precision still needs parity validation; this adapter never silently falls
    back to a different device or precision.

    ``validate_on_cpu=True`` checks finite logits after their required CPU
    transfer, avoiding an additional device reduction and synchronization.
    The default retains both device and CPU checks for baseline comparisons.
    """

    execution_backend = 'pytorch_reference'
    validate_on_cpu = False

    def __init__(self, bundle_dir, *, device='cuda', dtype='float32', validate_on_cpu=False):
        if not isinstance(device, str):
            raise ValueError('device must be cpu, cuda or an indexed CUDA device.')
        if dtype not in ('float32', 'float16'):
            raise ValueError('dtype must be float32 or float16.')
        if type(validate_on_cpu) is not bool:
            raise ValueError('validate_on_cpu must be a boolean.')
        verified = verify_bundle(bundle_dir)
        folder = Path(verified['bundle_dir'])
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        args = deepcopy(config['pretrained_cfg']['model']['init_args']['model_args'])
        # verify_bundle checks this already; retain a local guard at construction.
        backbone = args.get('backbone_kwargs') or {}
        if args.get('backbone_pretrained', False) is not False or backbone.get('pretrained', False) is not False:
            raise ValueError('The reference must not request pretrained backbone acquisition.')
        import torch
        try:
            selected_device = torch.device(device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError('device must be cpu, cuda or an indexed CUDA device.') from exc
        if (selected_device.type not in ('cpu', 'cuda')
                or (selected_device.type == 'cpu' and selected_device.index is not None)):
            raise ValueError('device must be cpu, cuda or an indexed CUDA device.')
        if selected_device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available; explicitly choose cpu to use it.')
            if selected_device.index is not None and selected_device.index >= torch.cuda.device_count():
                raise ValueError('Requested CUDA device index is not available.')
        from terratorch.models import EncoderDecoderFactory

        self._torch = torch
        self._device = selected_device
        self.device = str(selected_device)
        self.dtype = dtype
        self.validate_on_cpu = validate_on_cpu
        self._torch_dtype = torch.float32 if dtype == 'float32' else torch.float16
        self.bundle = verified
        self.model_id = verified['model_id']
        self._as_dict = verified['receipt']['source_as_dict']
        expected_hash = verified['files_sha256']['pytorch_model.bin']
        with (folder / 'pytorch_model.bin').open('rb') as stream:
            if _stream_hash(stream) != expected_hash:
                raise ValueError('Reference bundle weights changed before loading.')
            stream.seek(0)
            state = torch.load(stream, map_location='cpu', weights_only=True)
            if _stream_hash(stream) != expected_hash:
                raise ValueError('Reference bundle weights changed during loading.')
        if not isinstance(state, Mapping) or not state:
            raise ValueError('Reference weights must be a nonempty flat state mapping.')
        if any(type(key) is not str or type(value) is not torch.Tensor for key, value in state.items()):
            raise ValueError('Reference weights must map exact names to plain tensors.')
        model = EncoderDecoderFactory().build_model('segmentation', **args)
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError('Strict reference loading reported missing or unexpected keys.')
        model.requires_grad_(False)
        model.eval()
        model.to(device=self.device, dtype=self._torch_dtype)
        self.model = model
        after = verify_bundle(folder)
        if (after['receipt_sha256'] != verified['receipt_sha256']
                or after['files_sha256'] != verified['files_sha256']):
            raise ValueError('Reference bundle changed during construction.')

    def __call__(self, normalized_batch):
        if (not isinstance(normalized_batch, np.ndarray) or normalized_batch.ndim != 4
                or tuple(normalized_batch.shape[1:]) != _INPUT_SHAPE
                or not 1 <= normalized_batch.shape[0] <= 16
                or normalized_batch.dtype.kind not in 'fiu'
                or not np.isfinite(normalized_batch).all()):
            raise ValueError('Expected finite normalized tiles [N,12,224,224], with 1 <= N <= 16.')
        with np.errstate(over='ignore', invalid='ignore'):
            # Own writable storage: the predictor never mutates caller arrays.
            values = np.array(normalized_batch, dtype=self.dtype, order='C', copy=True)
        if not np.isfinite(values).all():
            raise ValueError('Normalized input exceeds the selected reference precision.')
        tensor = self._torch.from_numpy(values).to(device=self.device, dtype=self._torch_dtype)
        with self._torch.inference_mode():
            result = self.model({'S2L2A': tensor} if self._as_dict else tensor)
            output = result if isinstance(result, self._torch.Tensor) else getattr(result, 'output', None)
            expected_shape = (len(values), *_OUTPUT_SHAPE)
            if not isinstance(output, self._torch.Tensor) or tuple(output.shape) != expected_shape:
                raise ValueError(f'Reference must return tensor logits with shape {expected_shape}.')
            if (not output.is_floating_point()
                    or (not self.validate_on_cpu and not self._torch.isfinite(output).all().item())):
                raise ValueError('Reference returned nonfinite or non-floating logits.')
            logits = output.detach().to(device='cpu', dtype=self._torch.float32).contiguous().numpy()
        if not np.isfinite(logits).all():
            raise ValueError('Reference output cannot be represented as finite float32.')
        return logits

    def synchronize(self):
        """Wait for this device before/after an explicitly timed benchmark."""
        if self._device.type == 'cuda':
            self._torch.cuda.synchronize(device=self._device)
