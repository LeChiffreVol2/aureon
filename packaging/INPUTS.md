# User-owned Sentinel-2 preparation

The installed `aureon-prepare` commands never download imagery or load a model.
Use local GeoTIFFs in a projected CRS. SAFE/JP2 discovery, arbitrary reprojection
and automatic radiometric guessing are not supported. Export your own data to
GeoTIFF first, preserving its actual metadata; do not substitute private upstream
example chips. The supplied JSON examples are editable specifications, not data.

## Aligned raster

`aligned-input.json` expects a georeferenced FP32 raster with exactly twelve bands:
`B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`, in surface reflectance ×10000.
Supply a single-band uint8 0/1 validity mask on exactly the same grid, with 1 valid
across all twelve bands, and state the mask policy. Valid zero or negative
reflectance is retained; invalid pixels are controlled by the explicit mask.
Set `expected_crs`, `source_product_id`, acquisition metadata and `scene_id` from
your product. No normalization is required before preparation; the model recipe
owns it. `window` is `[col0, row0, width, height]` on this raster (Rasterio order).

## Separate L2A bands

`l2a-input.json` maps twelve distinct local band GeoTIFFs and an SCL GeoTIFF.
Set `units` to `stored_dn`, actual `quantification_value`, actual per-band offsets
and the corresponding `MTD_MSIL2A.xml` when available. Example -1000 offsets are
not universal defaults. The output radiometry is `(DN + offset) / quantification
* 10000`; product metadata, when supplied, corroborates those values. Sources
must share the declared projected CRS, with supported north-up grid alignment.
The crop is defined on the B02 reference grid. Preparation handles the declared
band resolutions within that CRS. Default SCL rejection classes are
0, 1, 3, 8, 9, 10 and 11. It does not infer scientific validity from filenames.

## Output and limits

Scene preparation writes a prepared manifest, source/raster/mask identities and
bounded model input. Grid sides are at most 4096 pixels. Full source files and
their permitted sidecars are hash checked; moving/altering referenced input files
can invalidate provenance. Preserve originals and the prepared directory.

`aureon-prepare generation` selects one all-valid 224×224 window from the prepared
scene. Its `row0`/`col0` are relative to that scene. It does not fill cloud/nodata
pixels, resample or convert it to a large-scene generative task. The same
preparation functions used by prior numerical evidence remain unchanged.

Use `aureon-prepare check --manifest ...` for scene identity validation. Use
`aureon-engine check --environment generation --config ...` after generating a
TerraMind config. Hash preflight does not replace the runtime's pixel/grid checks.
