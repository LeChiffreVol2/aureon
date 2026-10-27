"""Bounded, explicit geospatial input and categorical output for scene jobs.

No model, network access, band resampling or implicit radiometric correction is
performed here. Rasterio is optional and imported only when a raster is used.
Arrays are limited to the existing recipe's 1024-pixel side bound. A caller's
band/unit/CRS declarations are recorded, not independently certified.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from ..catalog import CLASS_NAMES
from ..recipe import INPUT_UNITS, MAX_SIDE, NODATA_CLASS, NUM_CLASSES, S2_BANDS


@dataclass(frozen=True)
class GeoRasterInput:
    image: np.ndarray
    valid_mask: np.ndarray
    metadata: dict
    input_identity: dict


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Rasterio is required for geospatial scene input/output.") from exc
    return rasterio


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(array):
    description = {"shape": list(array.shape), "dtype": array.dtype.str}
    digest = hashlib.sha256(_canonical(description) + b"\0")
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return {**description, "sha256": digest.hexdigest(),
            "hash_format": "json_header_nul_c_order_bytes_v1"}


def _local_file(path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Input must be an existing local raster file.")
    return resolved


def _supporting_files(dataset, primary):
    """Bind external .msk/.aux files that participate in raster interpretation."""
    paths = sorted({_local_file(path) for path in dataset.files} - {primary})
    if len(paths) > 8 or any(path.parent != primary.parent for path in paths):
        raise ValueError("Raster supporting files must be bounded and beside the source GeoTIFF.")
    return [{"path": str(path), "sha256": _file_hash(path), "bytes": path.stat().st_size}
            for path in paths]


def _verify_supporting_files(dataset, primary, expected):
    if _supporting_files(dataset, primary) != expected:
        raise ValueError("Raster supporting files changed while reading.")


def _grid(dataset, expected_crs, rasterio, *, max_side=MAX_SIDE):
    if dataset.driver != "GTiff":
        raise ValueError("Input must be a GeoTIFF, not a virtual or remote dataset.")
    if not (0 < dataset.height <= max_side and 0 < dataset.width <= max_side):
        raise ValueError(f"Each raster side must be between 1 and {max_side} pixels.")
    if not isinstance(expected_crs, str) or not expected_crs.strip():
        raise ValueError("expected_crs must explicitly declare the intended CRS.")
    expected = rasterio.crs.CRS.from_user_input(expected_crs)
    if dataset.crs is None or dataset.crs != expected:
        raise ValueError("Raster CRS does not match the declared expected_crs.")
    transform = dataset.transform
    values = [float(value) for value in tuple(transform)[:6]]
    if not all(math.isfinite(value) for value in values) or transform.determinant == 0:
        raise ValueError("Raster transform must be finite and invertible.")
    if transform == rasterio.Affine.identity():
        raise ValueError("An identity transform does not declare a usable source grid.")
    return {"height": dataset.height, "width": dataset.width,
            "crs_wkt": dataset.crs.to_wkt(), "crs": dataset.crs.to_string(),
            "transform": values, "bounds": list(dataset.bounds),
            "area_or_point": dataset.tags().get("AREA_OR_POINT", "Area"),
            "pixel_coordinates_preserved": True, "resampling": "none",
            "reprojection": "none"}


def _radiometry(units, declaration):
    if units == INPUT_UNITS:
        if declaration is not None:
            raise ValueError("Model-domain input must not supply an additional radiometry conversion.")
        return {"schema": "aureon.radiometry.v1", "source_units": units,
                "output_units": INPUT_UNITS, "operation": "float32(source_value)",
                "clipping": False, "source_declaration_verified": False}
    if units != "stored_dn":
        raise ValueError(f"units must explicitly be {INPUT_UNITS!r} or 'stored_dn'.")
    if not isinstance(declaration, dict) or declaration.get("schema") != "aureon.radiometry.v1":
        raise ValueError("stored_dn requires an explicit aureon.radiometry.v1 declaration.")
    offsets = declaration.get("offsets")
    scale = declaration.get("reflectance_scale")
    if (not isinstance(offsets, (list, tuple)) or len(offsets) != len(S2_BANDS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in offsets)):
        raise ValueError("Radiometry offsets must contain twelve finite numbers in S2_BANDS order.")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("reflectance_scale must be a finite positive number.")
    provenance = declaration.get("provenance")
    if (not isinstance(provenance, dict)
            or not all(isinstance(provenance.get(key), str) and provenance[key].strip()
                       for key in ("source_product_id", "metadata_sha256", "source_url"))
            or not re.fullmatch(r"[0-9a-f]{64}", provenance["metadata_sha256"])):
        raise ValueError("Radiometry requires product ID, metadata SHA-256 and source URL provenance.")
    return {"schema": "aureon.radiometry.v1", "source_units": "stored_dn",
            "output_units": INPUT_UNITS,
            "operation": "float32((float64(DN) + offset) * reflectance_scale * 10000)",
            "offsets": [float(value) for value in offsets], "reflectance_scale": float(scale),
            "provenance": json.loads(_canonical(provenance)), "clipping": False,
            "source_declaration_verified": False}


def read_s2_geotiff(path, *, bands, units, mask_policy, expected_crs,
                    radiometry=None, valid_mask_path=None):
    """Read an explicitly aligned twelve-band stack without changing its grid.

    ``mask_policy='all_bands_valid'`` intersects all twelve raster band masks.
    An optional one-band 0/1 mask is intersected too, after exact grid checks.
    Source zeros remain valid unless a declared raster/mask marks them invalid.
    Finite values are required on valid pixels. Invalid pixels become zero in
    model-domain values, and the recipe subsequently fills normalized zero.

    For ``units='stored_dn'``, ``radiometry`` must name twelve additive offsets,
    a reflectance scale and source metadata provenance. The explicit conversion
    is (DN + offset) * scale * 10000. GDAL scale/offset tags are never applied
    automatically. Such declarations are traceable, not certified by this IO.
    """
    try:
        declared_bands = tuple(bands) if not isinstance(bands, str) else ()
    except TypeError as exc:
        raise ValueError("bands must explicitly match S2_BANDS in order.") from exc
    if declared_bands != S2_BANDS:
        raise ValueError("bands must explicitly match S2_BANDS in order.")
    if mask_policy != "all_bands_valid":
        raise ValueError("mask_policy must explicitly be 'all_bands_valid'.")
    conversion = _radiometry(units, radiometry)
    source = _local_file(path)
    source_hash = _file_hash(source)
    rio = _rasterio()
    mask_identity = None
    with rio.Env(GDAL_CACHEMAX=64 * 1024 * 1024, GDAL_NUM_THREADS="1"):
        with rio.open(source, "r") as dataset:
            metadata = _grid(dataset, expected_crs, rio)
            supporting_files = _supporting_files(dataset, source)
            if dataset.count != 12 or any(np.dtype(dtype).kind not in "iuf" for dtype in dataset.dtypes):
                raise ValueError("Source must contain exactly twelve real numeric raster bands.")
            descriptions = list(dataset.descriptions)
            if any(value is not None for value in descriptions) and descriptions != list(S2_BANDS):
                raise ValueError("Declared band order conflicts with the raster band descriptions.")
            height, width = dataset.height, dataset.width
            valid = np.ones((height, width), dtype=bool)
            for index in dataset.indexes:
                valid &= dataset.read_masks(index) > 0
            if valid_mask_path is not None:
                mask_path = _local_file(valid_mask_path)
                mask_hash = _file_hash(mask_path)
                with rio.open(mask_path, "r") as mask_dataset:
                    mask_grid = _grid(mask_dataset, expected_crs, rio)
                    mask_supporting_files = _supporting_files(mask_dataset, mask_path)
                    if (mask_dataset.count != 1 or mask_grid != metadata
                            or np.dtype(mask_dataset.dtypes[0]).kind not in "biu"):
                        raise ValueError("Validity mask must be one integer band on the exact source grid.")
                    mask_values = mask_dataset.read(1)
                    if not np.isin(mask_values, (0, 1)).all():
                        raise ValueError("Validity mask must contain only 0 and 1.")
                    valid &= (mask_values == 1) & (mask_dataset.read_masks(1) > 0)
                    _verify_supporting_files(mask_dataset, mask_path, mask_supporting_files)
                if _file_hash(mask_path) != mask_hash:
                    raise ValueError("Validity mask changed while reading.")
                mask_identity = {"path": str(mask_path), "sha256": mask_hash,
                                 "bytes": mask_path.stat().st_size,
                                 "supporting_files": mask_supporting_files,
                                 "policy": "binary_one_and_raster_valid"}
            image = np.empty((12, height, width), dtype=np.float32)
            for index in dataset.indexes:
                raw = dataset.read(index)
                if not np.isfinite(raw[valid]).all():
                    raise ValueError("Source contains nonfinite samples at valid pixels.")
                with np.errstate(over="ignore", invalid="ignore"):
                    if units == "stored_dn":
                        converted = raw.astype(np.float64)
                        converted += conversion["offsets"][index - 1]
                        converted *= conversion["reflectance_scale"]
                        converted *= 10000.0
                        image[index - 1] = converted
                    else:
                        image[index - 1] = raw
                if not np.isfinite(image[index - 1][valid]).all():
                    raise ValueError("Model-domain valid values cannot be represented as finite float32.")
                image[index - 1, ~valid] = 0.0
            raster_metadata = {"dtypes": list(dataset.dtypes), "band_descriptions": descriptions,
                               "scales_not_automatically_applied": list(dataset.scales),
                               "offsets_not_automatically_applied": list(dataset.offsets),
                               "nodata": [str(value) if value is not None else None
                                          for value in dataset.nodatavals],
                               "mask_flags": [[flag.name for flag in row]
                                              for row in dataset.mask_flag_enums]}
            _verify_supporting_files(dataset, source, supporting_files)
    if _file_hash(source) != source_hash:
        raise ValueError("Source GeoTIFF changed while reading.")
    identity = {"schema": "aureon.georaster_input.v1",
                "source": {"path": str(source), "sha256": source_hash,
                           "bytes": source.stat().st_size, "raster_metadata": raster_metadata},
                "source_supporting_files": supporting_files,
                "band_order": list(S2_BANDS), "band_order_authority": "caller_declaration_checked_against_descriptions_when_present",
                "units": INPUT_UNITS, "source_units": units, "expected_crs": expected_crs,
                "radiometry": conversion, "radiometry_sha256": _json_hash(conversion),
                "mask_policy": mask_policy, "additional_valid_mask": mask_identity,
                "valid_pixels": int(np.count_nonzero(valid)), "total_pixels": height * width,
                "grid_sha256": _json_hash(metadata), "grid": metadata,
                "image": _array_identity(image), "valid_mask": _array_identity(valid),
                "source_changed_during_read": False, "resampling_performed": False}
    identity["identity_sha256"] = _json_hash(identity)
    image.flags.writeable = False
    valid.flags.writeable = False
    return GeoRasterInput(image, valid, metadata, identity)


def _validate_output(path, expected, source, *, nodata, cog, rasterio):
    with rasterio.open(path, "r") as dataset:
        grid = _grid(dataset, source.metadata["crs"], rasterio)
        if (grid != source.metadata or dataset.count != 1 or dataset.dtypes != ("uint8",)
                or dataset.nodata != nodata or not np.array_equal(dataset.read(1), expected)
                or not np.array_equal(dataset.read_masks(1) > 0, source.valid_mask)):
            raise ValueError("Reopened raster does not preserve output values, validity or the source grid.")
        layout = dataset.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        if cog and layout != "COG":
            raise ValueError("Requested COG output is missing GDAL's COG layout marker.")
        return {"sha256": _file_hash(path), "bytes": path.stat().st_size,
                "dtype": "uint8", "shape": list(expected.shape), "nodata": nodata,
                "layout": layout or "GTiff", "reopen_verified": True,
                "overviews": dataset.overviews(1)}


def write_segmentation_geotiffs(output_dir, result, source, *, provenance, cog=False):
    """Write an exclusive categorical delivery and verify it by reopening.

    Creates class_map.tif (classes 0..9; invalid 255), valid_mask.tif (0/1;
    nodata 0), and receipt.json. COG output uses nearest-neighbor overviews.
    Failure can leave an incomplete *new* directory but never overwrites an
    existing directory. This function makes no model-quality or speed claim.
    """
    if not isinstance(source, GeoRasterInput) or type(cog) is not bool:
        raise ValueError("source must be GeoRasterInput and cog must be a boolean.")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Nonempty explicit job/model provenance is required.")
    provenance_bytes = _canonical(provenance)
    if len(provenance_bytes) > 256 * 1024:
        raise ValueError("Provenance must fit in 256 KiB; reference larger evidence by hash.")
    if _json_hash(source.metadata) != source.input_identity["grid_sha256"]:
        raise ValueError("Source grid metadata changed after input admission.")
    classes = getattr(result, "class_map", None)
    valid = getattr(result, "valid_mask", None)
    shape = (source.metadata["height"], source.metadata["width"])
    if any(type(length) is not int or not 0 < length <= MAX_SIDE for length in shape):
        raise ValueError(f"Each output side must be between 1 and {MAX_SIDE} pixels.")
    if (not isinstance(classes, np.ndarray) or classes.shape != shape or classes.dtype != np.uint8
            or not isinstance(valid, np.ndarray) or valid.dtype != bool or valid.shape != shape
            or not np.array_equal(valid, source.valid_mask)):
        raise ValueError("Result must have uint8 classes and the unchanged boolean source validity mask.")
    if np.any(classes[valid] >= NUM_CLASSES) or np.any(classes[~valid] != NODATA_CLASS):
        raise ValueError("Class values must be 0..9 on valid pixels and 255 elsewhere.")
    destination = Path(output_dir).absolute()
    if destination.exists():
        raise FileExistsError("Output directory must not already exist.")
    if not destination.parent.is_dir():
        raise ValueError("Output parent directory must already exist.")
    rio = _rasterio()
    from rasterio.shutil import copy as copy_raster
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    file_records = {}
    with rio.Env(GDAL_CACHEMAX=64 * 1024 * 1024, GDAL_NUM_THREADS="1") as env:
        if cog and "COG" not in env.drivers():
            raise RuntimeError("The installed GDAL does not provide the COG driver.")
        destination.mkdir(exist_ok=False)
        for name, values, nodata, description in (
            ("class_map.tif", classes, NODATA_CLASS, "KAO-TerraMind B class index"),
            ("valid_mask.tif", valid.astype(np.uint8), 0, "valid=1; invalid=0"),
        ):
            final_path = destination / name
            staging = destination / ("." + name + ".source.tif") if cog else final_path
            profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
                       "dtype": "uint8", "nodata": nodata, "crs": source.metadata["crs_wkt"],
                       "transform": rio.Affine(*source.metadata["transform"]), "compress": "deflate",
                       "tiled": True, "blockxsize": 256, "blockysize": 256}
            with rio.open(staging, "w", **profile) as dataset:
                dataset.write(values, 1)
                dataset.set_band_description(1, description)
                dataset.update_tags(AUREON_SCHEMA="aureon.georaster_delivery.v1",
                                    AUREON_INPUT_ID=source.input_identity["identity_sha256"],
                                    AUREON_PROVENANCE_SHA256=provenance_hash,
                                    AUREON_CLASS_NAMES=json.dumps(CLASS_NAMES),
                                    AREA_OR_POINT=source.metadata["area_or_point"])
            if cog:
                copy_raster(staging, final_path, driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=256,
                            RESAMPLING="NEAREST", OVERVIEW_RESAMPLING="NEAREST", NUM_THREADS="1")
                staging.unlink()
            file_records[name] = _validate_output(final_path, values, source,
                                                  nodata=nodata, cog=cog, rasterio=rio)
    receipt = {"schema": "aureon.georaster_delivery.v1", "status": "completed",
               "created_utc": datetime.now(timezone.utc).isoformat(), "format": "COG" if cog else "GeoTIFF",
               "source": source.input_identity, "grid": source.metadata,
               "class_names": list(CLASS_NAMES), "nodata_class": NODATA_CLASS,
               "provenance": json.loads(provenance_bytes), "provenance_sha256": provenance_hash,
               "files": file_records, "score_exported": False, "quality_evaluation": "not_evaluated",
               "speedup_measured": False, "reopen_validation": "passed"}
    with (destination / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt
