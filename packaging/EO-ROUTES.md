# Explicit conditioning and generated targets

These APIs are included in the selected Best v0.1.0 wheel (`e09e710c`). Any-to-any is the architecture direction: each adapter
must implement and validate a route before the planner admits it. COP-GEN remains
a candidate and is not registered as supported.

`EORequest` separates `ConditioningSpec` observations from `GeneratedTarget`
outputs. Each declares ordered bands, units and its own `EOGrid`; observations
also declare normalization, validity policy, source ID and optional UTC acquisition
time. A generated raster is synthetic, not a sensor observation. Grid correctness
does not establish scientific fidelity or uncertainty calibration.

Run the dependency-light [planning example](../examples/generation/eo_plan.py):

```sh
python examples/generation/eo_plan.py
```

`plan_eo(request)` performs no I/O or hardware probing. `eo_capabilities()` lists
the existing TerraMind tiny/small/base/large generation IDs. Only S2L2A → S1GRD is
implemented: exact twelve raw reflectance-x10000 bands, explicit all-valid mask,
and VV/VH output in upstream unstandardized dB. Input and output must have matching
224×224 grids. Other modalities, model IDs, changed grids, partial bands and
unimplemented recipes raise `UnsupportedRoute`. The planner performs syntactic
EPSG validation; `GenerationInput` validates the CRS through Rasterio on admission.
No reprojection or normalization is inferred.

With an explicitly constructed [GenerationEngine](CONCURRENT-GENERATION.md), use
the same owned input as the existing SDK:

```python
from aureon.api import GenerationInput, generate_eo

data = GenerationInput(
    image, bands=request.conditioning[0].bands,
    units="surface_reflectance_x10000", normalization="none",
    valid_mask=valid_mask, grid=request.conditioning[0].grid.native(),
    source_id=request.conditioning[0].source_id,
)
result = generate_eo(engine, request, data, job_id="first-eo-result")
```

The helper borrows the engine; the caller retains ownership and closes it. It
delegates the existing `GenerationRecipe` and execution profile to `engine.generate`,
preserving busy/closed/error behavior and output arrays. Added provenance is detached.
The request preserves the SDK's `recommended` default; set `reference` explicitly
for the native reference path. Both use the existing FP32 contract. Sampling parameters remain
adapter-specific: no universal diffusion loop, scheduler or conditioning timestep
is imposed. Acquisition time is metadata, not temporal model conditioning.

## Select STAC assets offline

The [STAC Item specification](https://github.com/radiantearth/stac-spec/blob/v1.1.0/item-spec/item-spec.md)
describes assets and acquisition metadata. It does not define a model's input
normalization or infer a validity mask. This resolver supports a bounded subset
of STAC1.0/1.1 single-instant Items, not full schema validation or catalog crawling.

```python
from aureon.api import AssetBand, STACSelection, resolve_stac_item

selection = STACSelection(
    request.conditioning[0],
    tuple(AssetBand("image", i + 1, image_sha256) for i in range(12)),
    AssetBand("valid", 1, mask_sha256),
    "one_is_valid_zero_is_invalid",
)
selected, = resolve_stac_item(
    item, selections=(selection,), base_href="https://example.org/item.json",
)
```

Supply expected content digests when known, an exact acquisition time, ordered asset
band indexes, a distinct binary mask asset, and one explicit Item URL or local
asset root. Declared projection, band, unit and scale/offset conflicts are rejected.
Non-identity scale/offset needs explicit external preprocessing. Missing asset
metadata never becomes a guessed grid, normalization or all-valid mask. Unsupported
interval Items and cloud QA bit fields require an explicit separate adapter.

Hashes may be omitted for range-only COG access; the reader must then establish
a strong ETag/version and length before consistent bounded reads. It must not
claim whole-file cryptographic verification from partial ranges. Both legacy
`raster:bands` and STAC1.1 `bands` raster fields are checked; conflicting or
non-identity representations are rejected. WKT2/PROJJSON declarations require
explicit reconciliation outside this EPSG-only resolver.

Returned objects preserve grids and contain optional expected hashes with
`identity_verified=False`; the resolver reads zero asset bytes. Local containment
is lexical only. A subsequent reader must enforce symlink containment or remote
host/redirect/private-network policy, byte/window limits, range validation,
cancellation and content identity. URLs with credentials/query/fragment are rejected;
credentials belong in a separate transport channel. An HTTPS reference is not
permission to fetch it. Source and generated coverage remain distinct.

## Validation scope

`packaging/ci_eo.py` needs NumPy and Rasterio in an existing environment. It verifies
admission, STAC rejection, ownership and real SDK forwarding with a fake model
session while blocking neural imports and network. The same tests run against a
locally installed wheel and its matching sdist. CPU fixtures do not qualify a new
model route, GPU parity, speed, physical accuracy or universal any-to-any support.
