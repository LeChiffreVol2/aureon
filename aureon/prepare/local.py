"""Explicit local GeoTIFF preparation for the existing streaming scene contract.

Only a bounded output crop is read into arrays, one 256-square block at a time.
Complete local source files (including GDAL sidecars) are hashed before/after.
No remote access, SAFE discovery, reprojection across CRS, or model execution.
Radiometry and band mapping are caller declarations; optional product XML can
corroborate the radiometric numbers, not the provenance of supplied pixels.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

from ..recipe import S2_BANDS, INPUT_UNITS

SPEC_SCHEMA = "aureon.local_scene_spec.v1"
PREPARATION_ID = "aureon-local-sentinel2-v1"
SIDE = 256
MAX_SIDE = 4096
MAX_BLOCK_BYTES = 32 * 1024**2
EXCLUDED_SCL = (0, 1, 3, 8, 9, 10, 11)
XML_IDS = dict(zip(S2_BANDS, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def local_file(value, base):
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value) or str(value).startswith("/vsi"):
        raise ValueError("Supply local files, not URLs or GDAL virtual paths")
    path = Path(value)
    path = (base / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Expected an existing regular local file")
    return path


def read_json(path):
    if not 0 < path.stat().st_size <= 1024**2:
        raise ValueError("JSON must be nonempty and at most 1 MiB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def load_spec(value, base_dir=None):
    base = Path(base_dir or Path.cwd()).resolve(strict=True)
    if isinstance(value, (str, Path)):
        path = local_file(value, base)
        base, value = path.parent, read_json(path)
    if not isinstance(value, dict) or len(canonical(value)) > 1024**2:
        raise ValueError("Expected a bounded local scene spec object")
    spec = json.loads(canonical(value))
    common = {"schema", "kind", "scene_id", "expected_crs", "window", "provenance", "units", "bands"}
    extra = {"raster", "valid_mask", "validity_policy"} if spec.get("kind") == "aligned" else {"scl", "radiometry"}
    if spec.get("schema") != SPEC_SCHEMA or spec.get("kind") not in ("aligned", "l2a") or set(spec) - common - extra:
        raise ValueError("Expected aureon.local_scene_spec.v1, kind aligned/l2a, without unknown fields")
    if not isinstance(spec.get("expected_crs"), str) or not spec["expected_crs"].strip():
        raise ValueError("Declare expected_crs explicitly")
    provenance = spec.get("provenance")
    if (not isinstance(provenance, dict) or len(canonical(provenance)) > 16384
            or not isinstance(provenance.get("source_product_id"), str) or not provenance["source_product_id"].strip()):
        raise ValueError("Provenance must declare source_product_id and fit within 16 KiB")
    identifier = spec.get("scene_id")
    if identifier is not None and (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier)):
        raise ValueError("scene_id must use 1..128 letters, digits, '.', '_' or '-'")
    if spec["kind"] == "aligned":
        if spec.get("bands") != list(S2_BANDS) or spec.get("units") != INPUT_UNITS:
            raise ValueError("Aligned input requires the exact twelve-band order and surface_reflectance_x10000 units")
        if not isinstance(spec.get("validity_policy"), str) or not spec["validity_policy"].strip():
            raise ValueError("Declare the external validity mask's policy")
        files = {key: local_file(spec[key], base) for key in ("raster", "valid_mask")}
    else:
        if spec.get("units") != "stored_dn" or not isinstance(spec.get("bands"), dict) or set(spec["bands"]) != set(S2_BANDS):
            raise ValueError("L2A requires stored_dn units and exactly the twelve band-to-file mappings")
        files = {band: local_file(spec["bands"][band], base) for band in S2_BANDS}
        files["SCL"] = local_file(spec["scl"], base)
        if len(set(files.values())) != 13:
            raise ValueError("Every spectral band and SCL must map to a distinct local file")
    return spec, files, base


def identity(dataset, path):
    sources = {local_file(name, path.parent) for name in dataset.files} | {path}
    if len(sources) > 9 or any(member.parent != path.parent for member in sources):
        raise ValueError("At most eight supporting files beside each GeoTIFF are supported")
    return {"original_path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
            "supporting_files": [{"original_path": str(p), "sha256": file_hash(p), "bytes": p.stat().st_size}
                                 for p in sorted(sources - {path})]}


def grid(dataset, expected):
    import rasterio
    if dataset.driver != "GTiff" or dataset.crs != rasterio.crs.CRS.from_user_input(expected):
        raise ValueError("Local GeoTIFFs must match expected_crs; JP2/SAFE/VRT and cross-CRS reprojection are unsupported")
    if not dataset.crs.is_projected:
        raise ValueError("The current scene engine requires a projected CRS for area/cost accounting")
    t = dataset.transform
    if (min(dataset.width, dataset.height) < 1 or not all(math.isfinite(v) for v in tuple(t)[:6])
            or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0 or t == rasterio.Affine.identity()):
        raise ValueError("Require a finite north-up, nonempty georeferenced grid")
    # GDAL decodes entire blocks even for a tiny crop. Bound that hidden work.
    block_bytes = max(h*w for h, w in dataset.block_shapes) * sum(np.dtype(v).itemsize for v in dataset.dtypes)
    if block_bytes > MAX_BLOCK_BYTES:
        raise ValueError("Source decoder blocks exceed 32 MiB; retile the input before preparation")
    return {"shape": [dataset.height, dataset.width], "crs": str(dataset.crs),
            "transform": list(t)[:6], "decoder_block_bytes_upper_bound": block_bytes}


def crop_window(spec, dataset):
    from rasterio.windows import Window
    values = spec.get("window", [0, 0, dataset.width, dataset.height])
    if (not isinstance(values, list) or len(values) != 4 or any(type(v) is not int for v in values)
            or min(values[:2]) < 0 or not all(1 <= v <= MAX_SIDE for v in values[2:])
            or values[0]+values[2] > dataset.width or values[1]+values[3] > dataset.height):
        raise ValueError("window is [column, row, width, height], inside the source, with each output side 1..4096")
    return Window(*values)


def radiometry(spec, base):
    value = spec.get("radiometry")
    if not isinstance(value, dict) or set(value) - {"quantification_value", "offsets", "product_metadata"}:
        raise ValueError("Declare radiometry quantification_value, offsets, and optionally product_metadata")
    q, offsets = value.get("quantification_value"), value.get("offsets")
    number = lambda x: type(x) in (int, float) and math.isfinite(x)
    if (not number(q) or not 0 < q <= 1e9 or not isinstance(offsets, dict) or set(offsets) != set(S2_BANDS)
            or any(not number(v) or abs(v) > 1e9 for v in offsets.values())):
        raise ValueError("Positive finite quantification and twelve finite per-band offsets are required")
    result = {"quantification_value": q, "offsets": offsets, "values_match_product_xml": False,
              "pixel_product_association_verified": False, "source_units": "stored_dn",
              "formula": "float32(DN + BOA_ADD_OFFSET) * float32(10000 / BOA_QUANTIFICATION_VALUE), before resampling"}
    if value.get("product_metadata") is not None:
        path = local_file(value["product_metadata"], base)
        if not 0 < path.stat().st_size <= 4*1024**2:
            raise ValueError("Product XML must be nonempty and at most 4 MiB")
        data = path.read_bytes()
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ValueError("XML document types and entities are unsupported")
        try:
            root = ET.fromstring(data)
        except ET.ParseError as error:
            raise ValueError("Product metadata is not valid XML") from error
        nodes = lambda name: [n for n in root.iter() if n.tag.split("}")[-1] == name]
        quantities = nodes("BOA_QUANTIFICATION_VALUE")
        parsed = {}
        try:
            xml_quantity = float(quantities[0].text) if len(quantities) == 1 else None
            for node in nodes("BOA_ADD_OFFSET"):
                key = int(node.attrib["band_id"])
                if key in parsed:
                    raise ValueError("Duplicate XML band offsets")
                parsed[key] = float(node.text)
        except (KeyError, TypeError) as error:
            raise ValueError("Product XML requires numeric quantification and band_id/offset values") from error
        if (len(quantities) != 1 or xml_quantity != q
                or any(parsed.get(XML_IDS[b]) != offsets[b] for b in S2_BANDS)):
            raise ValueError("Declared radiometry disagrees with supplied product XML (missing offsets must be explicit)")
        result.update(values_match_product_xml=True, product_metadata={"original_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return result


def windows(height, width):
    from rasterio.windows import Window
    for row in range(0, height, SIDE):
        for col in range(0, width, SIDE):
            yield Window(col, row, min(SIDE, width-col), min(SIDE, height-row))


def source_window(dataset, target, shape):
    from rasterio.windows import Window, from_bounds
    from rasterio.transform import array_bounds
    window = from_bounds(*array_bounds(*shape, target), transform=dataset.transform)
    if (window.col_off < -1e-6 or window.row_off < -1e-6
            or window.col_off+window.width > dataset.width+1e-6
            or window.row_off+window.height > dataset.height+1e-6):
        raise ValueError("Requested B02 crop is not fully covered by every band and SCL")
    left, top = max(0, math.floor(window.col_off)-2), max(0, math.floor(window.row_off)-2)
    right = min(dataset.width, math.ceil(window.col_off+window.width)+2)
    bottom = min(dataset.height, math.ceil(window.row_off+window.height)+2)
    return Window(left, top, right-left, bottom-top)


def aligned_block(datasets, window):
    raster, mask = datasets["raster"], datasets["valid_mask"]
    data, declared = raster.read(window=window), mask.read(1, window=window)
    if not np.isin(declared, (0, 1)).all():
        raise ValueError("External validity mask values must be exactly 0 or 1")
    valid = (declared == 1) & (mask.read_masks(1, window=window) > 0)
    valid &= np.all(raster.read_masks(window=window) > 0, axis=0)
    if not np.isfinite(data[:, valid]).all():
        raise ValueError("All twelve bands must be finite at every valid pixel")
    data[:, ~valid] = 0
    return data, valid, None


def l2a_block(datasets, target, shape, conversion):
    from rasterio.warp import reproject, Resampling
    data = np.empty((12, *shape), np.float32)
    valid = np.ones(shape, bool)
    for index, band in enumerate(S2_BANDS):
        source = datasets[band]
        window = source_window(source, target, shape)
        raw = source.read(1, window=window)
        mask = (source.read_masks(1, window=window) > 0) & (raw != 0)
        corrected = raw.astype(np.float32)
        corrected += np.float32(conversion["offsets"][band])
        corrected *= np.float32(10000 / conversion["quantification_value"])
        corrected[~mask] = np.nan
        kwargs = dict(src_transform=source.window_transform(window), src_crs=source.crs,
            dst_transform=target, dst_crs=source.crs, num_threads=1, warp_mem_limit=16)
        reproject(corrected, data[index], src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, **kwargs)
        coverage = np.zeros(shape, np.float32)
        reproject(mask.astype(np.float32), coverage, resampling=Resampling.bilinear, **kwargs)
        valid &= (coverage >= np.float32(1-1e-6)) & np.isfinite(data[index])
    source = datasets["SCL"]
    window = source_window(source, target, shape)
    raw = source.read(1, window=window)
    if not np.isin(raw, range(12)).all():
        raise ValueError("SCL must contain only classes 0..11")
    raw[source.read_masks(1, window=window) == 0] = 0
    scl = np.zeros(shape, np.uint8)
    reproject(raw, scl, src_transform=source.window_transform(window), src_crs=source.crs,
              dst_transform=target, dst_crs=source.crs, resampling=Resampling.nearest,
              num_threads=1, warp_mem_limit=16)
    valid &= ~np.isin(scl, EXCLUDED_SCL)
    data[:, ~valid] = 0
    return data, valid, scl


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare_scene(value, output, *, base_dir=None):
    import rasterio
    from rasterio.windows import Window
    began = perf_counter()
    spec, files, base = load_spec(value, base_dir)
    output = Path(output).absolute()
    if output.exists() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("Output must be a fresh directory with an existing regular parent")
    output = output.parent.resolve()/output.name
    conversion = radiometry(spec, base) if spec["kind"] == "l2a" else None
    created = False
    try:
        with ExitStack() as stack:
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_NUM_THREADS="1",
                                             GDAL_TIFF_INTERNAL_MASK=True, VSI_CACHE=False))
            datasets = {key: stack.enter_context(rasterio.open(path)) for key, path in files.items()}
            grids = {key: grid(ds, spec["expected_crs"]) for key, ds in datasets.items()}
            source = datasets["raster" if spec["kind"] == "aligned" else "B02"]
            crop = crop_window(spec, source)
            height, width = int(crop.height), int(crop.width)
            target = source.window_transform(crop)
            if spec["kind"] == "aligned":
                mask = datasets["valid_mask"]
                if (source.count != 12 or source.dtypes != ("float32",)*12 or mask.count != 1
                        or np.dtype(mask.dtypes[0]).kind not in "biu" or mask.shape != source.shape
                        or mask.transform != source.transform
                        or (any(source.descriptions) and source.descriptions != S2_BANDS)
                        or source.scales != (1.,)*12 or source.offsets != (0.,)*12 or source.nodata == 0
                        or mask.scales != (1.,) or mask.offsets != (0.,)
                        or source.tags().get("units", INPUT_UNITS) != INPUT_UNITS):
                    raise ValueError("Aligned input requires twelve FP32 bands, matching integer mask/grid, identity TIFF scale/offset and no nodata=0")
            else:
                if (not source.crs.is_projected or source.crs.linear_units != "metre"
                        or source.transform.a != 10 or source.transform.e != -10):
                    raise ValueError("B02 must define a native projected 10 m grid")
                for key, ds in datasets.items():
                    if ds.count != 1 or (ds.dtypes != ("uint16",) if key != "SCL" else np.dtype(ds.dtypes[0]).kind not in "ui"):
                        raise ValueError("L2A spectral inputs must be one uint16 stored-DN band; SCL must be integer")
                    if ds.nodata not in (None, 0):
                        raise ValueError("L2A source nodata must be zero or absent; raw DN zero is always invalid")
                    if ds.transform.a not in (10, 20, 60) or ds.transform.e != -ds.transform.a:
                        raise ValueError("Local L2A bands must use square native 10/20/60 m pixels")
                    delta = ((ds.transform.c-source.transform.c)/10, (ds.transform.f-source.transform.f)/10)
                    if any(abs(v-round(v)) > 1e-6 for v in delta):
                        raise ValueError("Local L2A origins must align to the declared B02 10 m grid")
                    if key != "SCL":
                        expected = (1/conversion["quantification_value"], conversion["offsets"][key]/conversion["quantification_value"])
                        declared = (ds.scales[0], ds.offsets[0])
                        if declared != (1., 0.) and not np.allclose(declared, expected, rtol=1e-12, atol=1e-12):
                            raise ValueError("TIFF radiometric scale/offset conflict with the declaration")
                    elif ds.scales != (1.,) or ds.offsets != (0.,):
                        raise ValueError("SCL must not declare radiometric scaling")
                    source_window(ds, target, (height, width))
            records = {key: identity(ds, files[key]) for key, ds in datasets.items()}
            # Reserve output worst-case plus margin; callers still enforce their host-wide free-space floor.
            if shutil.disk_usage(output.parent).free < height*width*64 + 64*1024**2:
                raise ValueError("Insufficient free space for the bounded prepared output")
            source_identity = {"files": records, "grids": grids, "spec": spec, "radiometry": conversion}
            identity_hash = hashlib.sha256(canonical(source_identity)).hexdigest()
            scene_id = spec.get("scene_id") or "s2-"+identity_hash[:24]
            output.mkdir()
            created = True
            profile = dict(driver="GTiff", width=width, height=height, crs=source.crs, transform=target,
                tiled=True, blockxsize=SIDE, blockysize=SIDE, compress="deflate", NUM_THREADS="1", BIGTIFF="IF_SAFER")
            hashes = [hashlib.sha256(), hashlib.sha256()]
            valid_pixels = 0
            histogram = {str(i): 0 for i in range(12)}
            with ExitStack() as writers:
                image = writers.enter_context(rasterio.open(output/"scene.tif", "w", count=12, dtype="float32", predictor=3, **profile))
                mask = writers.enter_context(rasterio.open(output/"valid_mask.tif", "w", count=1, dtype="uint8", **profile))
                scl_out = (writers.enter_context(rasterio.open(output/"scl.tif", "w", count=1, dtype="uint8", **profile))
                           if conversion else None)
                image.descriptions = S2_BANDS
                image.update_tags(units=INPUT_UNITS, preparation=PREPARATION_ID)
                mask.descriptions = ("valid_mask",)
                for window in windows(height, width):
                    native = Window(crop.col_off+window.col_off, crop.row_off+window.row_off, window.width, window.height)
                    block, valid, scl = (aligned_block(datasets, native) if conversion is None else
                        l2a_block(datasets, source.window_transform(native), (int(window.height), int(window.width)), conversion))
                    image.write(block, window=window)
                    image.write_mask(valid.astype(np.uint8)*255, window=window)
                    mask.write(valid.astype(np.uint8), 1, window=window)
                    hashes[0].update(block.tobytes())
                    hashes[1].update(valid.astype(np.uint8).tobytes())
                    valid_pixels += int(valid.sum())
                    if scl_out is not None:
                        scl_out.write(scl, 1, window=window)
                        for key, count in zip(*np.unique(scl, return_counts=True)):
                            histogram[str(int(key))] += int(count)
            if not valid_pixels:
                raise ValueError("Prepared crop contains no valid pixels under the declared policy")
            actual = [hashlib.sha256(), hashlib.sha256()]
            with rasterio.open(output/"scene.tif") as image, rasterio.open(output/"valid_mask.tif") as mask:
                if image.transform != target or mask.transform != target or image.crs != source.crs or image.descriptions != S2_BANDS:
                    raise ValueError("Prepared output grid/band verification failed")
                for window in windows(height, width):
                    block, validity = image.read(window=window), mask.read(1, window=window)
                    if not np.array_equal(image.dataset_mask(window=window) > 0, validity == 1):
                        raise ValueError("Written internal and external masks differ")
                    actual[0].update(block.tobytes())
                    actual[1].update(validity.tobytes())
            if [v.hexdigest() for v in hashes] != [v.hexdigest() for v in actual]:
                raise ValueError("Prepared pixels changed during writing")
            if records != {key: identity(ds, files[key]) for key, ds in datasets.items()}:
                raise ValueError("Source or supporting files changed during preparation")
            if conversion and conversion.get("product_metadata"):
                meta = conversion["product_metadata"]
                if file_hash(meta["original_path"]) != meta["sha256"]:
                    raise ValueError("Product XML changed during preparation")
            member = lambda name: {"path": name, "sha256": file_hash(output/name), "bytes": (output/name).stat().st_size}
            source_record = {"source_product_id": spec["provenance"]["source_product_id"],
                "caller_provenance_json": canonical(spec["provenance"]).decode(),
                "caller_metadata_independently_verified": False, "band_mapping_independently_verified": False,
                "local_files": records,
                "input_identity_sha256": identity_hash, "acquisition_performed": False}
            preprocessing = {"preparation_id": PREPARATION_ID, "input_kind": spec["kind"],
                "source_window_col_row_width_height": [int(v) for v in (crop.col_off, crop.row_off, crop.width, crop.height)],
                "source_grids": grids, "units": INPUT_UNITS, "reflectance_clipping": False,
                "normalization": "not_applied; unchanged model z-score is applied at inference",
                "invalid_fill": "0; explicit mask determines validity", "radiometry": conversion,
                "spectral_resampling": "bilinear" if conversion else "none",
                "SCL_resampling": "nearest" if conversion else "none", "cross_crs_reprojection": False,
                "SCL_excluded": list(EXCLUDED_SCL) if conversion else None,
                "validity_policy": ("all spectral source masks valid over bilinear support (coverage >= 1-1e-6); SCL exclusions"
                                    if conversion else spec["validity_policy"]),
                "rasterio_version": rasterio.__version__, "gdal_version": rasterio.__gdal_version__,
                "block_side": SIDE, "gdal_cache_bytes": 64*1024**2, "warp_memory_MiB": 16, "threads": 1,
                "quality_validation": "not_evaluated"}
            if conversion:
                preprocessing.update(scl=member("scl.tif"), SCL_histogram=histogram)
            manifest = {"schema": "aureon.sentinel2_scene.v1", "scene_id": scene_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "source": source_record,
                "raster": {**member("scene.tif"), "bands": list(S2_BANDS), "dtype": "float32", "units": INPUT_UNITS,
                           "crs": str(source.crs), "shape": [12, height, width], "transform": list(target)[:6], "nodata": None},
                "valid_mask": {**member("valid_mask.tif"), "dtype": "uint8", "values": [0, 1],
                               "valid_pixels": valid_pixels, "valid_fraction": valid_pixels/(height*width)},
                "preprocessing": preprocessing}
            verification = {"schema": "aureon.local_scene_preparation.v1", "status": "passed", "scene_id": scene_id,
                "source_unchanged": True, "written_values_and_masks_exact": True,
                "ordered_block_sha256": {"scene": actual[0].hexdigest(), "mask": actual[1].hexdigest()},
                "hash_scope": "256-square row-major blocks; each scene block in band-first C order",
                "seconds": perf_counter()-began, "model_executed": False, "network_access": False,
                "preparation_source_sha256": file_hash(__file__)}
            write_json(output/"preparation.json", verification)
            manifest["preparation_receipt"] = member("preparation.json")
            # Publish the admission manifest last. Failed preparation never yields an admissible manifest.
            write_json(output/"manifest.json", manifest)
        return {"status": "prepared", "scene_id": scene_id, "manifest": str(output/"manifest.json"),
                "manifest_sha256": file_hash(output/"manifest.json"), "valid_pixels": valid_pixels,
                "shape": [12, height, width], "model_executed": False, "seconds": perf_counter()-began}
    except BaseException as error:
        if created and not (output/"failure.json").exists():
            write_json(output/"failure.json", {"status": "failed", "error": str(error), "model_executed": False})
        raise


def check_scene(manifest):
    from ..scene_pipeline.stream_io import WindowSource, windows as source_windows
    from ..recipe import normalize_s2
    count = 0
    with WindowSource(manifest) as source:
        if not source.dataset.crs.is_projected:
            raise ValueError("The current scene engine requires a projected CRS for area/cost accounting")
        for window in source_windows(source.height, source.width):
            from ..scene_pipeline.stream_io import rio_window
            valid = source.valid(window)
            normalize_s2(source.dataset.read(window=rio_window(window)), valid, bands=S2_BANDS, units=INPUT_UNITS)
            count += int(valid.sum())
        if count != source.manifest["valid_mask"].get("valid_pixels") or count < 1:
            raise ValueError("Declared valid-pixel count differs or scene has no valid pixels")
        source.verify()
        return {"status": "passed", "scene_id": source.manifest["scene_id"], "manifest_sha256": source.manifest_hash,
                "shape": [12, source.height, source.width], "valid_pixels": count, "model_executed": False,
                "caller_metadata_certified": False}
