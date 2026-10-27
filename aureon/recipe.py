"""Draft, bounded KAO-TerraMind land-cover recipe with an injected predictor.

This correctness reference is a parity target for future serving adapters,
including TerraTorch/vLLM, and makes no speed or accuracy improvement claim.
This module performs array operations only: it does not load a model, infer
georeferencing, resample bands, or write an export. The caller must supply an
already aligned Sentinel-2 surface-reflectance stack and an explicit validity
mask. Numerical tests with synthetic predictors do not establish model parity
or task accuracy.

The band order and published normalization statistics are from the upstream C
helper, pinned to revision 4e7252d3c83b3e877744da214005184a4133883f:
https://huggingface.co/kaopanboonyuen/KAO-TerraMind-C/tree/4e7252d3c83b3e877744da214005184a4133883f
The tiling and validation implementation below is authored for AUREON.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


RECIPE_ID = "aureon-landcover-v1"
RECIPE_STATUS = "draft"
INPUT_UNITS = "surface_reflectance_x10000"
S2_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)
S2_MEAN = (
    1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
    2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648,
)
S2_STD = (
    2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311,
)
WINDOW_SIZE = 224
STRIDE = 168
NUM_CLASSES = 10
NODATA_CLASS = 255
MAX_SIDE = 1024
MAX_PIXELS = MAX_SIDE * MAX_SIDE
MAX_BATCH_SIZE = 16


class RecipeCancelled(RuntimeError):
    """The caller cancelled the recipe; no partial result is returned."""


@dataclass(frozen=True)
class SegmentationResult:
    """Array results on the caller's original grid, without georeferencing.

    ``max_uncalibrated_score`` is the largest softmax score after blending
    logits. It is not calibrated confidence. Invalid pixels are NaN in that
    array and 255 in ``class_map``. Class IDs preserve the predictor's order;
    the model catalog defines their labels.
    """

    class_map: np.ndarray
    valid_mask: np.ndarray
    max_uncalibrated_score: np.ndarray | None
    recipe_id: str
    recipe_status: str
    tiles_total: int
    tiles_executed: int
    tiles_skipped: int
    batch_size: int
    predictor_calls: int


def _validate_input(
    image: np.ndarray,
    valid_mask: np.ndarray,
    bands: Sequence[str],
    units: str,
) -> tuple[int, int]:
    try:
        declared_bands = tuple(bands)
    except TypeError as exc:
        raise ValueError("bands must declare the exact S2_BANDS order") from exc
    if not all(isinstance(band, str) for band in declared_bands) or declared_bands != S2_BANDS:
        raise ValueError("bands must exactly match the declared S2_BANDS order")
    if not isinstance(units, str) or units != INPUT_UNITS:
        raise ValueError(f"units must be explicitly declared as {INPUT_UNITS!r}")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[0] != 12:
        raise ValueError("image must be a NumPy array with shape [12, H, W]")
    if image.dtype.kind not in "iuf":
        raise ValueError("image must contain real numeric samples")
    height, width = image.shape[1:]
    if not (0 < height <= MAX_SIDE and 0 < width <= MAX_SIDE):
        raise ValueError(f"each image side must be between 1 and {MAX_SIDE} pixels")
    if height * width > MAX_PIXELS:
        raise ValueError(f"image must have at most {MAX_PIXELS} pixels")
    if (
        not isinstance(valid_mask, np.ndarray)
        or valid_mask.dtype != np.dtype(bool)
        or valid_mask.shape != (height, width)
    ):
        raise ValueError("valid_mask must be a boolean NumPy array with shape [H, W]")
    if not np.isfinite(image[:, valid_mask]).all():
        raise ValueError("all bands must be finite at every valid pixel")
    return height, width


def normalize_s2(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bands: Sequence[str],
    units: str,
) -> np.ndarray:
    """Validate and normalize a declared S2 stack to float32 [12, H, W].

    Values use the published bandwise (value - mean) / standard deviation
    transform, without clipping or estimating missing bands. Invalid pixels
    become normalized zero even when their raw values contain NaN or infinity.
    The explicit mask remains the authority for validity; a raw zero is valid
    whenever the caller marks it valid.
    """
    _validate_input(image, valid_mask, bands, units)
    normalized = np.zeros(image.shape, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        np.copyto(normalized, image, where=valid_mask[None, :, :], casting="unsafe")
        normalized -= np.asarray(S2_MEAN, dtype=np.float32)[:, None, None]
        normalized /= np.asarray(S2_STD, dtype=np.float32)[:, None, None]
    normalized[:, ~valid_mask] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("valid input cannot be represented as finite normalized float32")
    return normalized


def tile_anchors(length: int) -> tuple[int, ...]:
    """Return deterministic full-coverage anchors, ending at the far edge."""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
        raise ValueError("tile length must be an integer")
    if not 0 < length <= MAX_SIDE:
        raise ValueError(f"tile length must be between 1 and {MAX_SIDE}")
    if length <= WINDOW_SIZE:
        return (0,)
    edge = int(length) - WINDOW_SIZE
    anchors = list(range(0, edge + 1, STRIDE))
    if anchors[-1] != edge:
        anchors.append(edge)
    return tuple(anchors)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RecipeCancelled("land-cover recipe cancelled; no result was returned")


def _validated_logits(output: np.ndarray, batch_count: int) -> np.ndarray:
    expected_shape = (batch_count, NUM_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
    if not isinstance(output, np.ndarray) or output.shape != expected_shape:
        raise ValueError(f"predictor must return a NumPy array with shape {expected_shape}")
    if output.dtype.kind not in "iuf" or not np.isfinite(output).all():
        raise ValueError("predictor logits must be finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        logits = output.astype(np.float32, copy=False)
    if not np.isfinite(logits).all():
        raise ValueError("predictor logits must be representable as finite float32")
    return logits


def segment_tiled(
    image: np.ndarray,
    valid_mask: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    *,
    bands: Sequence[str],
    units: str,
    batch_size: int = 1,
    include_max_score: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> SegmentationResult:
    """Run the draft recipe with a caller-supplied segmentation predictor.

    Each call receives one float32 batch [N, 12, 224, 224] and must return
    finite logits [N, 10, 224, 224], where N is at most ``batch_size``. Windows
    have stride 168 (25% nominal overlap); the final anchor covers each edge.
    Overlap adds model work and
    must be evaluated against a non-overlapping baseline before promotion.
    Chips smaller than a window receive bottom/right normalized-zero padding.
    Fully invalid windows are
    skipped. Float32 logits are uniformly blended before argmax and optional
    stable softmax, with no resizing, smoothing, ensemble, or score threshold.

    Inputs are bounded to 1024 x 1024 and batches to 16 tiles; default batch
    size 1 is the sequential reference. Batches preserve tile and accumulation
    order. Larger batches permit serving adapters to schedule requests; they
    do not by themselves establish a speedup or model batch parity. The
    caller's predictor has its own memory and runtime limits. Cancellation is
    checked at tile boundaries, around predictor calls, and before returning;
    it cannot interrupt a predictor executing. Exceptions return no result.
    """
    if not callable(predictor):
        raise ValueError("predictor must be callable")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be a callable or None")
    if not isinstance(include_max_score, bool):
        raise ValueError("include_max_score must be boolean")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or not 1 <= batch_size <= MAX_BATCH_SIZE
    ):
        raise ValueError(f"batch_size must be an integer between 1 and {MAX_BATCH_SIZE}")
    batch_size = int(batch_size)
    _check_cancelled(cancelled)
    normalized = normalize_s2(image, valid_mask, bands=bands, units=units)
    height, width = normalized.shape[1:]
    mask = valid_mask.copy()
    rows, columns = tile_anchors(height), tile_anchors(width)
    accumulator = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    executed = 0
    skipped = 0
    predictor_calls = 0
    batch = None
    pending = []

    def flush_batch() -> None:
        nonlocal executed, predictor_calls
        if not pending:
            return
        _check_cancelled(cancelled)
        logits = _validated_logits(predictor(batch[:len(pending)]), len(pending))
        _check_cancelled(cancelled)
        predictor_calls += 1
        for index, (row, column, bottom, right) in enumerate(pending):
            region = (slice(row, bottom), slice(column, right))
            target = accumulator[:, region[0], region[1]]
            with np.errstate(over="ignore", invalid="ignore"):
                target += logits[index, :, : bottom - row, : right - column]
            if not np.isfinite(target).all():
                raise ValueError("float32 logit accumulation overflowed")
            coverage[region] += 1
            executed += 1
        pending.clear()

    for row in rows:
        for column in columns:
            _check_cancelled(cancelled)
            bottom = min(row + WINDOW_SIZE, height)
            right = min(column + WINDOW_SIZE, width)
            region = (slice(row, bottom), slice(column, right))
            if not mask[region].any():
                skipped += 1
                continue
            if batch is None:
                batch = np.zeros((batch_size, 12, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
            tile = batch[len(pending)]
            tile.fill(0.0)
            tile[:, : bottom - row, : right - column] = normalized[:, region[0], region[1]]
            pending.append((row, column, bottom, right))
            if len(pending) == batch_size:
                flush_batch()
    flush_batch()

    np.divide(accumulator, coverage[None, :, :], out=accumulator, where=coverage[None, :, :] > 0)
    class_map = np.argmax(accumulator, axis=0).astype(np.uint8)
    class_map[~mask] = NODATA_CLASS
    score = None
    if include_max_score:
        maximum = np.max(accumulator, axis=0)
        # exp(logit - max) is stable; max(exp) is exactly 1. Extreme finite
        # differences may underflow to zero, as intended for softmax.
        with np.errstate(over="ignore", under="ignore"):
            accumulator -= maximum[None, :, :]
            np.exp(accumulator, out=accumulator)
        denominator = accumulator.sum(axis=0, dtype=np.float32)
        score = np.full((height, width), np.nan, dtype=np.float32)
        np.divide(1.0, denominator, out=score, where=mask)
    _check_cancelled(cancelled)
    return SegmentationResult(
        class_map=class_map,
        valid_mask=mask,
        max_uncalibrated_score=score,
        recipe_id=RECIPE_ID,
        recipe_status=RECIPE_STATUS,
        tiles_total=len(rows) * len(columns),
        tiles_executed=executed,
        tiles_skipped=skipped,
        batch_size=batch_size,
        predictor_calls=predictor_calls,
    )
