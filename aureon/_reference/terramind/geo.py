"""One existing all-valid 224px S2 crop and continuous synthetic COG delivery."""
from pathlib import Path
from .common import INPUT_BANDS, binding, file_hash, read_json, verify_file, write_json


def prepare(source_manifest, output, *, row0=0, col0=0):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    source_manifest = Path(source_manifest).resolve(strict=True)
    source = read_json(source_manifest)
    if source.get("schema") != "aureon.sentinel2_scene.v1": raise ValueError("Expected existing real-scene manifest")
    raster = source["raster"]; mask_record = source["valid_mask"]
    if raster["bands"] != INPUT_BANDS or raster["units"] != "surface_reflectance_x10000":
        raise ValueError("Exact S2L2A bands/units required")
    image_path = verify_file(raster, source_manifest.parent)
    mask_path = verify_file(mask_record, source_manifest.parent)
    if any(type(x) is not int or x < 0 for x in (row0, col0)): raise ValueError("Invalid crop offset")
    window = Window(col0, row0, 224, 224)
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2), rasterio.open(image_path) as ds, rasterio.open(mask_path) as ms:
        if ds.count != 12 or ds.dtypes != ("float32",)*12 or ds.crs is None:
            raise ValueError("Expected georeferenced FP32 twelve-band raster")
        if (ds.width, ds.height, ds.crs, ds.transform) != (ms.width, ms.height, ms.crs, ms.transform):
            raise ValueError("Mask grid mismatch")
        if ds.crs.to_string() != raster["crs"] or list(ds.transform)[:6] != raster["transform"]:
            raise ValueError("Source manifest grid mismatch")
        if row0 + 224 > ds.height or col0 + 224 > ds.width: raise ValueError("Crop outside source")
        if any(v != 1. for v in ds.scales) or any(v != 0. for v in ds.offsets):
            raise ValueError("Unexpected TIFF radiometric metadata")
        if max(h*w for h,w in ds.block_shapes)*12*4 > 32*1024**2:
            raise ValueError("Source blocks exceed finite decoder budget")
        image = ds.read(window=window)
        mask = (ms.read(1, window=window) == 1) & (ds.read_masks(window=window) > 0).all(axis=0)
        if not mask.all() or not np.isfinite(image).all():
            raise ValueError("First native slice requires all-valid finite pixels; no implicit filling")
        grid = {"crs": ds.crs.to_string(), "transform": list(ds.window_transform(window))[:6], "shape": [224,224]}
    verify_file(raster, source_manifest.parent); verify_file(mask_record, source_manifest.parent)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    np.save(output / "image.npy", image, allow_pickle=False)
    np.save(output / "mask.npy", mask, allow_pickle=False)
    payloads = {}
    for name in ("image", "mask"):
        payloads[name] = binding(output / (name + ".npy")); payloads[name]["path"] = name + ".npy"
    manifest = {"schema": "aureon.terramind_native_input.v1", "source_manifest": binding(source_manifest),
                "source_product_id": source["source"]["source_product_id"], "source_raster": binding(image_path),
                "source_mask": binding(mask_path), "window": [row0,col0,row0+224,col0+224],
                "bands": INPUT_BANDS, "units": "surface_reflectance_x10000", "grid": grid,
                "mask_policy": "All source pixels valid; upstream model receives no invalid-pixel fill",
                "preprocessing": "Existing prepared radiometry retained; no resampling/standardization here",
                **payloads}
    write_json(output / "manifest.json", manifest)
    return binding(output / "manifest.json")


def load_input(path):
    import numpy as np
    path = Path(path)
    manifest = read_json(path)
    if manifest["schema"] != "aureon.terramind_native_input.v1" or manifest["bands"] != INPUT_BANDS or manifest["units"] != "surface_reflectance_x10000":
        raise ValueError("Native input identity mismatch")
    image = np.load(verify_file(manifest["image"], path.parent), allow_pickle=False)
    mask = np.load(verify_file(manifest["mask"], path.parent), allow_pickle=False)
    if image.shape != (12,224,224) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("Invalid native image")
    if mask.shape != (224,224) or mask.dtype != np.bool_ or not mask.all():
        raise ValueError("Invalid native mask")
    return image, mask, manifest


def write_cog(output, values, mask, grid, provenance):
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    if values.shape != (2,224,224) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-channel FP32 native output")
    if mask.shape != (224,224) or mask.dtype != np.bool_ or not mask.all():
        raise ValueError("First slice only supports all-valid masks")
    if not provenance.get("synthetic") or provenance.get("sensor_observation") is not False:
        raise ValueError("Synthetic output provenance required")
    if grid.get("shape") != [224,224]: raise ValueError("Output grid shape differs")
    transform = rasterio.Affine(*grid["transform"])
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_TIFF_INTERNAL_MASK=True), MemoryFile() as memory:
        with memory.open(driver="GTiff", width=224, height=224, count=2, dtype="float32",
                         crs=grid["crs"], transform=transform, tiled=True, blockxsize=256, blockysize=256) as ds:
            ds.write(values); ds.write_mask(mask.astype("uint8")*255)
            ds.set_band_description(1,"synthetic_S1GRD_VV"); ds.set_band_description(2,"synthetic_S1GRD_VH")
            ds.update_tags(synthetic="true", sensor_observation="false", units="dB",
                           numeric_domain="upstream_unstandardized_S1GRD", scientific_quality="not_evaluated")
        with memory.open() as ds:
            copy(ds, output, driver="COG", compress="DEFLATE", blocksize=256, overview_count=0)
    # Independent reopen of the encoded file, including continuous values and spatial metadata.
    with rasterio.open(output) as ds:
        passed = (ds.count == 2 and ds.dtypes == ("float32", "float32")
                  and ds.crs == rasterio.crs.CRS.from_user_input(grid["crs"]) and ds.transform == transform
                  and np.array_equal(ds.read(), values) and np.array_equal(ds.dataset_mask()>0, mask)
                  and ds.tags().get("synthetic") == "true" and ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG")
    if not passed: raise RuntimeError("Continuous COG round-trip audit failed")
    return {"file": binding(output), "decoded_values_exact": True, "grid_and_mask_exact": True,
            "cog_layout": True, "bands": ["VV","VH"], "units": "dB",
            "units_source": "https://arxiv.org/html/2504.11171v5#S5.T3",
            "physical_quality": "not_evaluated"}
