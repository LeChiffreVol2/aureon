"""Measured scene stages and parity helpers, independent of a model runtime."""
from dataclasses import dataclass
import hashlib
import math
from time import perf_counter

import numpy as np

from ..recipe import (INPUT_UNITS, S2_BANDS, NUM_CLASSES, NODATA_CLASS,
                      WINDOW_SIZE, STRIDE, SegmentationResult, normalize_s2,
                      tile_anchors)


@dataclass(frozen=True)
class PreparedTiles:
    values: np.ndarray
    windows: tuple
    height: int
    width: int
    total: int
    skipped: int
    timings: dict


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array).reshape(-1)).cast('B')).hexdigest()


def prepare_tiles(image, valid_mask):
    """Use the unchanged recipe normalization/anchors; retain row-major order."""
    start = perf_counter()
    normalized = normalize_s2(image, valid_mask, bands=S2_BANDS, units=INPUT_UNITS)
    normalized_at = perf_counter()
    height, width = valid_mask.shape
    rows, columns = tile_anchors(height), tile_anchors(width)
    windows = []
    for row in rows:
        for col in columns:
            bottom, right = min(row + WINDOW_SIZE, height), min(col + WINDOW_SIZE, width)
            if valid_mask[row:bottom, col:right].any():
                windows.append((row, col, bottom, right))
    # The unchanged 1024px recipe bound also bounds this allocation (<84 MiB).
    tiles = np.zeros((len(windows), 12, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    for index, (row, col, bottom, right) in enumerate(windows):
        tiles[index, :, :bottom-row, :right-col] = normalized[:, row:bottom, col:right]
    return PreparedTiles(tiles, tuple(windows), height, width,
                         len(rows)*len(columns), len(rows)*len(columns)-len(windows),
                         {'normalization_seconds': normalized_at-start,
                          'tiling_seconds': perf_counter()-normalized_at})


def blend_logits(prepared, logits, valid_mask, *, batch_size=8, predictor_calls=None):
    """Uniform FP32 accumulation in original recipe order, then argmax."""
    expected = (len(prepared.windows), NUM_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
    if (not isinstance(logits, np.ndarray) or logits.dtype != np.float32
            or logits.shape != expected or not np.isfinite(logits).all()):
        raise ValueError('Expected finite FP32 logits for exactly the prepared tiles')
    if valid_mask.dtype != bool or valid_mask.shape != (prepared.height, prepared.width):
        raise ValueError('Validity grid does not match prepared tiles')
    accumulator = np.zeros((NUM_CLASSES, prepared.height, prepared.width), dtype=np.float32)
    coverage = np.zeros(valid_mask.shape, dtype=np.uint16)
    for output, (row, col, bottom, right) in zip(logits, prepared.windows):
        with np.errstate(over='ignore', invalid='ignore'):
            accumulator[:, row:bottom, col:right] += output[:, :bottom-row, :right-col]
        coverage[row:bottom, col:right] += 1
    if not np.isfinite(accumulator).all() or np.any(valid_mask & (coverage == 0)):
        raise ValueError('Blended logits overflowed or left valid pixels uncovered')
    np.divide(accumulator, coverage[None], out=accumulator, where=coverage[None] > 0)
    classes = accumulator.argmax(axis=0).astype(np.uint8)
    classes[~valid_mask] = NODATA_CLASS
    count = len(prepared.windows)
    return SegmentationResult(classes, valid_mask.copy(), None,
        'aureon-landcover-v1', 'draft', prepared.total, count, prepared.skipped,
        batch_size, predictor_calls if predictor_calls is not None else math.ceil(count/batch_size))


def compare_logits(reference, candidate, *, atol=1e-5, rtol=1e-5):
    for value in (atol, rtol):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Parity tolerances must be finite nonnegative numbers')
    if (reference.shape != candidate.shape or reference.dtype != np.float32
            or candidate.dtype != np.float32 or reference.ndim != 4
            or reference.shape[1:] != (NUM_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
            or len(reference) == 0 or not np.isfinite(reference).all()
            or not np.isfinite(candidate).all()):
        raise ValueError('Parity requires nonempty matching finite FP32 tile logits')
    maximum, absolute_sum, close, matching = 0.0, 0.0, True, 0
    for baseline, proposed in zip(reference, candidate):
        baseline64 = baseline.astype(np.float64)
        delta = np.abs(proposed.astype(np.float64)-baseline64)
        maximum = max(maximum, float(delta.max()))
        absolute_sum += float(delta.sum())
        close = close and bool(np.all(delta <= atol + rtol*np.abs(baseline64)))
        matching += int(np.count_nonzero(baseline.argmax(axis=0) == proposed.argmax(axis=0)))
    pixels = len(reference)*WINDOW_SIZE*WINDOW_SIZE
    agreement = matching/pixels
    return {'atol': atol, 'rtol': rtol, 'allclose': close,
            'bitwise_equal': array_hash(reference) == array_hash(candidate),
            'max_abs_error': maximum, 'mean_abs_error': absolute_sum/reference.size,
            'class_agreement': agreement, 'tile_count': len(reference),
            'passed': close and agreement == 1.0}


def cost_report(warm_job_seconds, startup_seconds, validation_seconds, valid_area_km2,
                *, hourly_rate=1.0, currency='THB'):
    values = (warm_job_seconds, startup_seconds, validation_seconds, valid_area_km2, hourly_rate)
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Cost inputs must be finite nonnegative numbers')
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError('Declare the currency for the hourly rate')
    warm = warm_job_seconds*hourly_rate/3600
    startup = startup_seconds*hourly_rate/3600
    cold = (warm_job_seconds+startup_seconds)*hourly_rate/3600
    validation = validation_seconds*hourly_rate/3600
    per_area = warm/valid_area_km2 if valid_area_km2 > 0 else None
    if any(not math.isfinite(value) for value in (warm, startup, cold, validation)
           + (() if per_area is None else (per_area,))):
        raise ValueError('Derived cost overflowed; supply representable rates and durations')
    return {'basis': 'user_selected_linear_whole_machine_wall_time_scenario',
            'currency': currency, 'hourly_rate': hourly_rate, 'actual_bill': False,
            'warm_job_seconds': warm_job_seconds, 'warm_job_cost': warm,
            'cold_start_seconds': startup_seconds,
            'cold_start_cost': startup,
            'cold_job_cost': cold,
            'cold_job_cost_kind': 'Additive warm-job plus separately measured initialization scenario; not an observed cold-request latency',
            'validation_overhead_seconds': validation_seconds,
            'validation_overhead_cost': validation,
            'valid_area_km2': valid_area_km2,
            'warm_cost_per_valid_km2': per_area,
            'exclusions': ['source acquisition', 'network and storage charges',
                           'idle capacity', 'electricity/depreciation unless included in hourly rate',
                           'tax and commercial margin'],
            'scope': 'One local job; not a cloud quote, measured electricity bill or production price'}
