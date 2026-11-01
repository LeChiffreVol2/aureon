"""Prepare an explicitly described local raster export; never execute a model.

This file-based example is not a TorchGeo sample/tensor adapter. Run in the
separate AUREON dependency environment after exporting from your own workflow.
"""
import argparse
import json
from pathlib import Path


def prepare_export(*, raster, valid_mask, output, crs, source_product_id,
                   validity_policy, bands, units, normalization):
    # Fail before creating output: preparation must not standardize data a second
    # time. Declarations cannot detect a caller mislabelling normalized pixels.
    if normalization != "none":
        raise ValueError("Export physical reflectance before training/model normalization; normalization must be none")
    from aureon.prepare import prepare_scene
    from aureon.terramind_native.common import INPUT_BANDS
    if list(bands) != INPUT_BANDS or units != "surface_reflectance_x10000":
        raise ValueError("Declare twelve ordered S2L2A bands and surface_reflectance_x10000 units")
    spec = {"schema": "aureon.local_scene_spec.v1", "kind": "aligned",
            "expected_crs": crs, "bands": list(bands), "units": units,
            "raster": str(Path(raster).resolve()), "valid_mask": str(Path(valid_mask).resolve()),
            "provenance": {"source_product_id": source_product_id,
                           "normalization": "none; model owns standardization at inference"},
            "validity_policy": validity_policy}
    return prepare_scene(spec, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("raster", "valid-mask", "output", "crs", "source-product-id", "validity-policy", "units", "normalization"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--bands", nargs=12, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_export(**vars(args))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
