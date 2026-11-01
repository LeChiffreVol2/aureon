"""Explicit inference compute policies; weights and public tensors stay FP32.

Contexts must be entered inside the native RNG capsule, which serializes AUREON
global precision changes. No model, Torch or CUDA import/probe occurs on import.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
import os
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from ..terramind_native.encoder_loader import fp32_policy

if TYPE_CHECKING:
    import torch

ComputePrecision: TypeAlias = Literal['highest_fp32', 'tf32', 'amp_bf16', 'amp_fp16']
PrecisionRecord: TypeAlias = dict[str, object]


def validate_precision(precision: ComputePrecision, *, mode: str, device: str,
                       allow_experimental_precision: bool) -> None:
    if precision not in ('highest_fp32', 'tf32', 'amp_bf16', 'amp_fp16'):
        raise ValueError('Unknown compute precision policy')
    if type(allow_experimental_precision) is not bool:
        raise ValueError('allow_experimental_precision must be a bool')
    if precision != 'highest_fp32':
        if not allow_experimental_precision:
            raise ValueError('Non-FP32 compute requires allow_experimental_precision=True')
        if mode != 'backbone':
            raise ValueError('Experimental precision is limited to Backbone; TiM/generation remain FP32')
        if device != 'cuda' and not device.startswith('cuda:'):
            raise ValueError('Experimental precision requires an explicit CUDA device')


def precision_preflight(precision: ComputePrecision, device: torch.device) -> PrecisionRecord:
    """Reject unsupported hardware; no emulation, tensor allocation or trial forward."""
    import torch
    if precision == 'highest_fp32':
        return {'policy': precision, 'device': str(device), 'experimental': False}
    if precision not in ('tf32', 'amp_bf16', 'amp_fp16') or device.type != 'cuda':
        raise ValueError('Experimental precision requires CUDA')
    if precision == 'tf32' and os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise ValueError('TF32 is disabled by NVIDIA_TF32_OVERRIDE=0; no fallback is allowed')
    if torch.version.cuda is None or torch.version.hip is not None or not torch.cuda.is_available():
        raise ValueError('A native NVIDIA CUDA runtime is required; no fallback is allowed')
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError('Requested CUDA device is unavailable')
    capability = torch.cuda.get_device_capability(index)
    # Conservative supported architecture boundary for this experimental API.
    # AMP FP16 requires Volta+; TF32/BF16 require Ampere+. Older FP16-capable
    # architectures are deliberately not admitted, rather than assumed tested.
    minimum = (7, 0) if precision == 'amp_fp16' else (8, 0)
    if capability < minimum:
        raise ValueError(f'{precision} requires CUDA compute capability >= {minimum[0]}.{minimum[1]}')
    if precision == 'amp_bf16':
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError('Native BF16 is unavailable; emulation is not permitted')
    return {'policy': precision, 'device': f'cuda:{index}', 'experimental': True,
            'compute_capability': list(capability), 'cuda_build': torch.version.cuda,
            'native_support_checked': True, 'qualification': 'unqualified_experimental'}


@contextmanager
def _amp_reduction_policy() -> Iterator[PrecisionRecord]:
    """Own AMP reduction settings, including the paired split-k state."""
    import torch
    matmul = torch.backends.cuda.matmul
    fp16 = (bool(matmul.allow_fp16_reduced_precision_reduction),
            bool(matmul.allow_fp16_reduced_precision_reduction_split_k))
    bf16 = (bool(matmul.allow_bf16_reduced_precision_reduction),
            bool(matmul.allow_bf16_reduced_precision_reduction_split_k))
    math_sdp_allowed = cast(Callable[[], bool], torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed)
    math_sdp = bool(math_sdp_allowed())
    accumulation = getattr(matmul, 'allow_fp16_accumulation', None)
    # Register all restoration before changing any flag. ExitStack attempts each
    # restoration even if a setter fails. Bare bool writes would reset split-k.
    with ExitStack() as restore:
        restore.callback(setattr, matmul, 'allow_fp16_reduced_precision_reduction', fp16)
        restore.callback(setattr, matmul, 'allow_bf16_reduced_precision_reduction', bf16)
        restore.callback(torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp, math_sdp)
        if accumulation is not None:
            restore.callback(setattr, matmul, 'allow_fp16_accumulation', accumulation)
        # Disabling split-K requires cuBLASLt in Torch 2.10 and fails on the
        # default cuBLAS path. Permit its normal heuristic while prohibiting
        # reduced-precision accumulation; do not override the caller's backend.
        matmul.allow_fp16_reduced_precision_reduction = (False, True)
        matmul.allow_bf16_reduced_precision_reduction = (False, True)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        if accumulation is not None:
            matmul.allow_fp16_accumulation = False
        applied: PrecisionRecord = {
            'cuda_matmul_allow_fp16_reduced_precision_reduction':
                bool(matmul.allow_fp16_reduced_precision_reduction),
            'cuda_matmul_allow_fp16_reduced_precision_reduction_split_k':
                bool(matmul.allow_fp16_reduced_precision_reduction_split_k),
            'cuda_matmul_allow_bf16_reduced_precision_reduction':
                bool(matmul.allow_bf16_reduced_precision_reduction),
            'cuda_matmul_allow_bf16_reduced_precision_reduction_split_k':
                bool(matmul.allow_bf16_reduced_precision_reduction_split_k),
            'math_sdp_allow_fp16_bf16_reduction':
                bool(math_sdp_allowed()),
            'cuda_matmul_allow_fp16_accumulation':
                bool(matmul.allow_fp16_accumulation) if accumulation is not None else None,
        }
        expected: PrecisionRecord = {key: False for key in applied}
        expected['cuda_matmul_allow_fp16_reduced_precision_reduction_split_k'] = True
        expected['cuda_matmul_allow_bf16_reduced_precision_reduction_split_k'] = True
        expected['cuda_matmul_allow_fp16_accumulation'] = False if accumulation is not None else None
        if any(value is not expected[key] for key, value in applied.items()):
            raise RuntimeError('Requested AMP reduction policy was not applied; no fallback is allowed')
        yield applied


@contextmanager
def compute_precision_scope(precision: ComputePrecision, device: torch.device) -> Iterator[PrecisionRecord]:
    """One explicit forward scope inside an already-held RNG capsule.

    This changes no sampler or model weights. AMP may keep some operators in
    FP32; its selected dtype is an autocast policy, not an operator-level trace.
    """
    import torch
    receipt = precision_preflight(precision, device)
    amp = precision in ('amp_bf16', 'amp_fp16')
    dtype = torch.bfloat16 if precision == 'amp_bf16' else torch.float16 if precision == 'amp_fp16' else None
    # Reuse the unchanged reference policy and its finally-based restoration.
    # Autocast construction can query the current CUDA device (not the tensor's
    # device), so experimental policies hold the resolved target through forward.
    # The default path does not change device, reduction settings or cache policy.
    target: AbstractContextManager[object]
    if precision == 'highest_fp32':
        target = nullcontext()
    else:
        target = torch.cuda.device(str(receipt['device']))
    reductions: AbstractContextManager[PrecisionRecord] = _amp_reduction_policy() if amp else nullcontext({})
    with target, fp32_policy(), reductions as reduction_record:
        if precision == 'tf32':
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.allow_tf32 = True
        with torch.inference_mode(), torch.autocast(
                device.type, enabled=amp, dtype=dtype, cache_enabled=False if amp else None):
            actual_amp = torch.is_autocast_enabled(device.type)
            if actual_amp != amp or (amp and torch.get_autocast_dtype(device.type) != dtype):
                raise RuntimeError('Requested autocast policy was not applied; no fallback is allowed')
            if amp and torch.is_autocast_cache_enabled():
                raise RuntimeError('Requested AMP autocast cache policy was not applied')
            actual_precision = torch.get_float32_matmul_precision()
            if actual_precision != ('high' if precision == 'tf32' else 'highest'):
                raise RuntimeError('Requested matmul precision was not applied')
            if (bool(torch.backends.cuda.matmul.allow_tf32) != (precision == 'tf32') or
                    bool(torch.backends.cudnn.allow_tf32) != (precision == 'tf32')):
                raise RuntimeError('Requested TF32 permissions were not applied')
            receipt.update(applied_policy=precision, autocast_enabled=actual_amp,
                autocast_dtype=str(dtype) if amp else None, matmul_precision=actual_precision,
                autocast_cache_enabled=torch.is_autocast_cache_enabled(),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                weight_dtype='torch.float32', public_output_dtype='torch.float32',
                semantics='TF32 permits eligible kernels; AMP selects eligible operators, not every operator')
            receipt.update(reduction_record)
            yield receipt


def allowed_feature_dtypes(precision: ComputePrecision) -> tuple[torch.dtype, ...]:
    import torch
    if precision == 'amp_bf16':
        return (torch.float32, torch.bfloat16)
    if precision == 'amp_fp16':
        return (torch.float32, torch.float16)
    return (torch.float32,)
