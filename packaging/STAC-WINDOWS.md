# STAC assets to bounded TerraMind scene inference

`STACGenerationRaster` connects the explicit offline STAC selection to the
existing `generate_raster_scene` engine/COG path. Open the reader in a context,
pass the caller-owned resident `GenerationEngine`, then close both contexts.
The reader never loads a model. No full-scene array or full-file download fallback
is used. The scene executor retains the independent 224-pixel recipe,
coordinate-derived seeds and row-major FP32 uniform reconstruction.

```python
from aureon.api import GenerationEngine
from aureon.api.spatial import generate_raster_scene
from aureon.sources.stac_raster import STACGenerationRaster

# observation is one result of resolve_stac_item(...), with explicit grid,
# twelve S2L2A bands, native reflectance_x10000 units and a separate 0/1 mask.
with STACGenerationRaster(observation, local_root=asset_root) as source:
    source.inspect_validity()  # optional full-scene, bounded-memory preflight
    with GenerationEngine(model_file=model_file, tokenizer_file=tokenizer_file,
                          work=work, execute_model=True) as engine:
        receipt = generate_raster_scene(engine, source, output,
                    execute_model=True, min_free_bytes=1024**3)
```

`examples/generation/stac_windows.py` provides a local command-line example;
without `--execute-model` it only inspects inputs. The normal CLI/SDK assets and
model eligibility requirements still apply. No weights are downloaded by it.

## Data contract

- Actual GeoTIFF CRS, affine grid, dimensions, band indexes, scales and offsets
  must match the declared selection. It does not silently resample or normalize.
- Float32, uint16, int16 and uint8 image data are accepted; these integer types
  convert exactly to FP32. The twelve-band order is B01,B02,B03,B04,B05,B06,B07,
  B08,B8A,B09,B11,B12. Nonempty TIFF band descriptions/units must agree.
- The separate explicit validity raster is uint8 with 0/1 values. It is combined
  with each raster's own validity and finite-value checks. Clouds need a
  caller-prepared mask; cloud classes are not guessed.
- Default invalid policy rejects a tile. `invalid_policy='skip_invalid_tiles'`
  explicitly skips every affected tile and writes a coverage mask. Coverage is
  the union of generated tiles, not the original source-valid mask.
- Only self-contained GeoTIFFs are admitted. Local sidecars, symlinks/reparse
  points and root escapes are rejected. A local asset root must not be writable
  by an untrusted concurrent process; filesystem race isolation requires a
  stronger OS sandbox. Local files are hashed at open and verified before delivery.
- Output is synthetic S1GRD VV/VH in dB. COP-GEN's S1RTC route has different
  units and semantics and does not use this reader/scene recipe automatically.

## Explicit remote I/O

For unsigned HTTPS assets supply a `RangePolicy` with exact permitted hosts:

```python
from aureon.sources.http_range import RangePolicy

policy = RangePolicy(allowed_hosts=frozenset({'your-approved-host.example'}),
                     max_range_bytes=1024**2, max_total_bytes=32*1024**2,
                     max_requests=512)
with STACGenerationRaster(observation, remote_policy=policy,
        max_remote_total_bytes=64*1024**2) as source:
    tile = source.read(window)
```

The aggregate admission bound is the number of selected remote assets multiplied
by the per-source byte budget. Increase budgets explicitly for larger scenes;
exhaustion fails instead of switching to a full download. The total range cache
defaults to 8 MiB, shared equally among selected assets; GDAL's cache is 64 MiB.
Each decoded source block is conservatively checked against a 32 MiB bound.
Temporary request/cast buffers and reconstruction have additional bounded memory;
these values are not a claim about complete Python/GDAL process RSS.

The controlled reader pins a checked address, verifies TLS hostname, requires
206, exact Content-Range/length, a strong ETag and consistent object size, and
uses If-Match on later reads. It revalidates identity before delivery. Redirects
are disabled by default, with explicit same-origin/allowlisted policies available.
Private destinations are disabled by default. It never hands an asset URL to
GDAL's unrestricted network client. GDAL receives an opaque filename and bounded
seek/read opener; sidecar requests cannot fetch arbitrary resources.

A server ETag is a version validator, **not a cryptographic content hash**. If
STAC selection supplies an expected whole-file SHA-256, use a locally verified
file; remote range mode refuses to silently downgrade that expectation. Provenance
records the selected band/mask mapping, STAC metadata digest, local content hashes
or remote validator scope, byte/request counts and grid. It omits URL paths,
queries and authorization headers. This first STAC convenience reader accepts
unsigned public HTTPS sources; provider-specific credentials are not inferred.

Pass the same cancellation Event to both reader and scene executor. Cancellation
interrupts stalled body reads; DNS and TLS/connect retain their bounded socket
behavior. Close the context on all exits; do not share a reader concurrently
between threads. Scene queues may own multiple prepared tiles but source reads
remain ordered through the scene iterator.

## What has been checked

Real Rasterio CPU fixtures cover local and loopback range windows, exact values
against the previous prepared reader, COG/grid/read-back parity, mask coverage,
identity mutation, budgets, actual metadata conflicts and clean ownership.
The engine in these new tests is a CPU double. They establish I/O correctness,
not a new neural-model parity, speedup, scientific seam-quality or live-provider
qualification. Existing model evidence retains its original artifact scope.

The implementation uses Rasterio's [custom opener](https://rasterio.readthedocs.io/en/stable/topics/vsi.html#python-file-and-filesystem-openers)
and [windowed reads](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html).
A small requested window can still decode complete TIFF blocks, which is why
block-size admission is independent of window size.
