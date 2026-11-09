# Use AUREON from your Earth observation workflow

**Best v0.1.0 interfaces.** Use one distribution identified by
the [installation guide](../packaging/CANDIDATE-QUICKSTART.md) and its exact
delivery receipt. There is no separate core/feature installation choice.
GitHub Release v0.1.0 provides Best `e09e710c`; original artifacts retain their historical names.

Compatible TerraMind base file, array and HTTP scene routes share the preserved
graph/reuse/RNG primitives. Array inference keeps its owned-memory contract.
Recorded FP32/token/RNG checks and performance apply to the exact artifacts,
input/recipe and hardware identified in the delivery receipt. Other families
keep their own qualification; fused B2 is retired.

AUREON routes supported inference through a resident model and manages request
state, eligible encoder reuse and decoder graphs. Replace the inference entry
point explicitly; importing AUREON does not accelerate an existing upstream
`model.forward` call. The walkthrough below uses **TerraMind 1.0 base, one all-valid
S2L2A224 input, synthetic S1GRD VV/VH output, highest FP32**. Other implemented
families/modes have their [own scope](../packaging/TERRAMIND-FAMILIES.md).
For an existing TorchGeo/TerraTorch module, start with the explicit
[CPU model bridge](../packaging/MODEL-BRIDGE.md) or
[resident borrowed-model contract](../packaging/BORROWED-ADAPTERS.md).
Those adapters do not turn arbitrary training configs or modalities into native
accelerated generation routes.

## Python arrays with a resident model

Use your existing pinned model/tokenizer files and a NumPy array or CPU Torch
tensor you already own. `GenerationEngine` binds the assets lazily, selects the
recommended profile and owns one resident core. No manual configuration file is
needed. The first explicit `generate` validates the environment/assets and loads
the model; constructing or closing an unused engine creates no files or GPU work.

```python
from aureon.api import GenerationInput, GenerationEngine

bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
data = GenerationInput(
    image,                       # float32 (12, 224, 224), physical reflectance x10000
    bands=bands,
    units="surface_reflectance_x10000",
    valid_mask=valid_mask,        # bool or uint8 (224, 224), all pixels explicitly valid
    normalization="none",        # before model/training z-score normalization
    source_id="my-input-identity",
)
with GenerationEngine(
    "terramind-v1-base",
    model_file="/data/models/TerraMind_v1_base.pt",
    tokenizer_file="/data/models/TerraMind_Tokenizer_S1GRD.pt",
    work="/data/work/new-engine",  # new directory under an existing parent
    execute_model=True,
) as engine:
    result = engine.generate(data, seed=17)
    synthetic_sar = result["array"]       # owned CPU float32 (2, 224, 224), VV/VH in dB
    provenance = result["provenance"]
    selection = result["execution_selection"]
    # Further explicit calls reuse this resident model. They are additional jobs.
```

This path creates no input raster, NPY array, token array or output COG. It retains
an owned assets-only config, session identity, loaded-asset identity, JSON job receipts and cleanup
evidence in `work`. Return arrays are independent of the model and of later
results. Constructor and per-call snapshots protect input bytes from caller
mutation. Each call owns RNG/scheduler state and only one call runs at a time.
Close/context exit releases the session's runtime; no background service is needed.

The default is `recommended`; `engine.generate(data, seed=17,
execution_profile="reference")` explicitly selects the native reference for a call.
These are additional model jobs, not automatic comparison calls.

For explicit request records, `GenerationRequest(..., input_data=data,
execution_profile="recommended")` works with `run_model(..., execute_model=True)`;
the in-memory result is under `result["job"]`. Choose either `input_data` or
`input_manifest`. Repeated one-shot calls reload the model; use a session for reuse.
Calls accept the existing exact `recipe` mapping for seed and sampling controls.
All sampling fields remain explicit; this does not authorize hidden step reduction.

Pure array inference does not require CRS or an affine transform. If you need a
COG later, supply `grid={"crs": "EPSG:32647", "transform": [10, 0, 500000, 0, -10,
1500000]}` **using your actual grid** when constructing the input, then call:

```python
from aureon.api import export_generation
receipt = export_generation(result, "/data/results/new-synthetic-s1grd.tif")
```

Export checks that arrays still match their inference identities, requires the
input grid and a new output path, and verifies the encoded values/grid/mask after
writing. It never invents coordinates or executes the model. Preserve the returned
provenance with the exported artifact. A COG is optional delivery, not part of the
model's conditioning for this adapter. Synthetic S1GRD is not observed SAR;
execution fidelity and scientific usefulness are different claims.

## Advanced configuration and the existing session API

Use your existing, separately obtained pinned model/tokenizer files. An
assets-only configuration now omits the input manifest; it hashes assets and
checks the generation dependency profile without loading a model:

```python
from aureon.terramind_native.offline import create_config

create_config(
    model="/data/models/TerraMind_v1_base.pt",
    tokenizer="/data/models/TerraMind_Tokenizer_S1GRD.pt",
    output="/data/config/generation-assets.json",  # new file, existing parent
    seed=17,
)
```

The equivalent CLI is `python -m aureon.api generation-config --model-file PATH
--tokenizer-file PATH --output NEW_JSON`, with no `--manifest`. A prior file-bound
configuration also works with memory input. Asset checks never download weights.

Open that config with `open_model_session("terramind-v1-base", config=...,
work=..., execution_profile="recommended", execute_model=True)`. Its `generate`
method accepts `recipe=recipe` (the complete sampling mapping), rather than the
short facade's `seed=`. File-bound and assets-only configurations both work.
The convenience facade defaults to eight token steps and ten diffusion steps.
Current source also accepts an explicit `GenerationRecipe` for the existing finite
native recipe domain. Expert denoiser controls remain on the lower-level API.
All routes use the same runtime forward.

## Select execution and inspect fallback

| Selection | Behavior |
| --- | --- |
| `execution_profile="recommended"` | Requests exact conditioning reuse and graph denoising for 8 token / 10 diffusion steps. Other admitted recipes use reuse-off/native with a reason. Runtime admission can also fall back. |
| `execution_profile="reference"` | Reuse-off, native denoising, same highest-FP32 sampling and model. |
| Omitted or `None` on the existing session/request API | Existing defaults: reuse-auto, native denoising. Expert `reuse="off"` and `denoiser="graph"` remain available. |

`GenerationEngine` defaults to recommended, requires a named profile at construction,
and uses `None` on a call to inherit it. The table's expert behavior applies to
`open_model_session` / `GenerationRequest`, whose old defaults remain unchanged.

Named profiles own both controls; conflicting expert overrides fail before model
loading. For one resident request, `session.generate(data,
execution_profile="reference")` selects the reference; on that lower-level session,
explicitly passing `execution_profile=None` enables expert controls for that call. Custom decoder
hooks cannot compose with named profiles.

```sh
python -m aureon.api generation-plan --execution-profile recommended
python -m aureon.api generation-plan --execution-profile recommended --token-steps 1
python -m aureon.api generation-plan --execution-profile reference
```

These commands are **recipe plans**, not dependency/input/device preflight.
`plan_generation_execution(recipe=recipe, execution_profile="recommended")`
offers the same CPU-only Python check. Plans report `effective: null` and
`cuda: not_probed`. File generation also accepts `--execution-profile`; the
installed published CLI does not yet have this flag.

Completed job/provenance `execution_selection` reports `requested`, `planned`,
`effective`, `eligibility`, `fallback` and `reasons`. Actual topology/device and
decoder tensor signatures control runtime eligibility. For example, a runtime
that cannot use a graph can report reuse-auto/native rather than graph. A capture,
execution or post-capture signature failure terminates the session; it does not
silently rerun a partly executed request. Eligibility does not establish parity,
performance, or qualification on new hardware. The old measured scope stays tied
to its original artifacts; this changed source needs its own functional acceptance.

## Keep TorchGeo/TerraTorch preparation

You can keep your data loading and preprocessing workflow, but its output must
meet the selected model's inference contract. TorchGeo composes spatial datasets
and can reproject/resample imagery; samples may also contain training targets.
See the [upstream TorchGeo tutorial](https://docs.torchgeo.org/en/latest/tutorials/torchgeo.html).
TerraTorch inference has data-module-specific conventions; see its
[inference guide](https://torchgeo.org/terratorch/1.2.12/guide/inference/).
Those upstream capabilities do not qualify arbitrary inputs for this AUREON adapter.

For one sample before training transforms/model normalization:

```python
data = GenerationInput.from_torchgeo(
    sample, bands=bands, units="surface_reflectance_x10000",
    valid_mask=independently_derived_valid_mask, normalization="none",
    source_id="my-product-and-window",
)
```

Only `sample["image"]` is read. A field named `mask` is not automatically a
validity mask: it may contain class labels. Neither `bounds` nor `crs` is silently
converted into a grid. Pass a real post-transform affine/grid when exporting.
This accepts one sample mapping with explicit semantics, not an arbitrary
DataLoader batch, model object, checkpoint or transform stack.

| Input contract | What the adapter does |
| --- | --- |
| Bands/order | Requires all 12 ordered bands above; rejects missing/reordered lists. No RGB substitution or guessing from shape. |
| Units | Requires unstandardized surface reflectance x10000. Convert stored DN with the product's quantification/offsets first. No implicit clipping, unit scaling or dtype conversion. |
| Normalization owner | `normalization="none"` is required; upstream TerraMind standardizes once. Rejects declared already-normalized data. It cannot detect falsely labelled values. |
| Shape/dtype | One CPU NumPy/strided Torch FP32 `(12,224,224)` array; no implicit batch squeeze or GPU transfer. |
| Validity | Separate bool/uint8 all-one `(224,224)` mask. Rejects invalid pixels/nonfinite image values; no fill or training-label reinterpretation. |
| Geometry | Optional for arrays; CRS plus finite nonsingular six-coefficient affine for georeferenced export. Coefficients are `a,b,c,d,e,f`: `x=a*col+b*row+c`, `y=d*col+e*row+f`, pixel corners. |
| Identity | Immutable snapshot, pixel/mask hashes, bands/units/normalization/grid/source identifier; model/tokenizer/source/dependency/recipe identities join the result provenance. Source identifier is a caller declaration. |
| Acquisition/conditioning | No timestamp/location conditioning is added in this fixed adapter. A future modality requiring it must define/validate it explicitly. |

## File export and separate environments

The published release already accepts local aligned GeoTIFF exports and provides
the local authenticated API. If your research environment cannot satisfy AUREON's
pins, export your unstandardized imagery and a separate validity mask there, then
prepare and infer in AUREON's environment. Do not upgrade a working research
environment just to combine the two stacks.

The [file preparation example](../examples/generation/prepare_export.py) uses the
existing `prepare_scene` path. Supply a 12-band FP32 tiled GeoTIFF with exact band
descriptions, physical reflectance x10000, identity TIFF scale/offset, north-up
projected metre CRS and a matching binary mask. Output extent is bounded to4096
per side; larger input needs an explicit bounded crop via the standard JSON spec.

```sh
python examples/generation/prepare_export.py \
  --raster /data/export/s2.tif --valid-mask /data/export/valid.tif \
  --output /data/work/new-prepared --crs EPSG:32647 \
  --source-product-id MY_PRODUCT --validity-policy 'my nodata and cloud policy' \
  --bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 \
  --units surface_reflectance_x10000 --normalization none
python -m aureon.prepare check --manifest /data/work/new-prepared/manifest.json
python -m aureon.prepare generation --manifest /data/work/new-prepared/manifest.json \
  --output /data/work/new-generation-input --row0 0 --col0 0
```

This creates the supported all-valid 224crop. Invalid pixels fail; choose a valid
crop or stop. The helper rejects declared normalized values before creating output.
For raw L2A bands use the existing [input guide](../packaging/INPUTS.md), which
explicitly applies product offsets/quantification. Training augmentation is not
automatically valid inference preprocessing.

Use the locked Linux/WSL Python3.12 **generation** environment for TerraMind:
torch2.10.0, torchvision0.25.0, numpy2.2.6, TerraTorch1.2.13, rasterio1.5.2,
diffusers0.39.0, einops0.8.2, huggingface-hub0.36.2, timm1.0.30 and
transformers4.57.6. The complete transitive hash lock is
[generation lock](../packaging/lock-generation-linux-py312-cu128.txt); use the matching bootstrap
and installation instructions, not this list as a replacement lock. TorchGeo is
not an extra runtime dependency for the sample mapping adapter.

KAO-TerraMind B's **scene** profile instead pins torch2.14.1,
torchvision0.29.1, numpy2.5.4, TerraTorch1.2.13, TorchGeo0.9.0 and rasterio1.5.2.
The legacy A/C full-reference route uses **scene-legacy**. Keep it separate
from the original scene environment. Exact metadata
checks are available through `python -m aureon.api check --environment generation`;
they do not load a model or qualify CUDA.

For an HTTP boundary follow the [local API guide](../packaging/ASYNC-SERVICE.md):
upload a prepared manifest/payloads, submit explicitly, poll, download and verify.
Run the worker in AUREON's generation environment. A client in your research
environment can call that existing HTTP contract; it need not import TerraTorch
into the worker's process. The published service remains file-to-COG; this memory
slice does not silently add an ndarray HTTP transport or change service contracts.

## Bounded submission and lazy streams — current source

`submit` accepts a validated `GenerationInput` or `GenerationJob` and returns a
`GenerationHandle`. Multiple producer threads may submit; one dispatcher calls
the existing resident session serially. Request RNG, scheduler, precision and
actual execution-selection receipts remain in that session. This is concurrent
submission, with a maximum model batch size of one. It does not implement fused
batching, interleaved token/diffusion steps, Ray Serve or an LLM router.

```python
from aureon.api import GenerationEngine, GenerationJob, GenerationRecipe

def jobs():
    for index, crop in enumerate(my_generation_inputs):  # lazy GenerationInput iterator
        yield GenerationJob(
            crop, recipe=GenerationRecipe(seed=17 + index),
            job_id=f"crop-{index}", attempt_id="1",
        )

with GenerationEngine(
    model_file="/data/models/TerraMind_v1_base.pt",
    tokenizer_file="/data/models/TerraMind_Tokenizer_S1GRD.pt",
    work="/data/work/new-stream", execute_model=True,
    max_pending=2, max_results=2,
) as engine:
    with engine.generate_many(jobs(), max_in_flight=2) as results:
        for result in results:
            consume(result["array"], result["provenance"])
```

The stream yields in input order and reserves credits before advancing its input.
It does not materialize the iterable or start a background DataLoader. The next
CPU input preparation and caller-side result processing/export may overlap the
preceding or next serial model call. Use the stream context manager, or call
`results.close()`, when breaking early. Closing the stream cancels its queued
requests, finishes/discards any running request, releases its completed outputs
and closes its input iterator if that iterator supports close. Unrelated handles
and already returned caller-owned arrays remain untouched.

Plain `GenerationInput` items use common `seed=17` (or an explicit common recipe).
`GenerationJob` carries a separate immutable recipe/seed, profile, job ID and
attempt ID for every item. A job with profile `None` inherits the engine profile.
Shared seed/recipe/profile overrides cannot be combined with jobs. For spatial
workloads, use stable coordinate-derived seeds/identifiers from your declared
tiling policy rather than relying on enumeration order across rearranged tiles.

| Credit / operation | Contract |
| --- | --- |
| `max_pending` (default2) | Reserved, queued and running input slots |
| `max_results` (default2) | Output slots reserved from admission through result transfer/discard |
| `max_input_bytes` | Defaults to `max_pending × 2,458,624`: one FP32 12×224×224 image plus boolean mask per slot |
| `max_output_bytes` | Defaults to `max_results × 451,584`: one FP32 2×224×224 result plus boolean mask per slot |
| Full credits | Nonblocking `GenerationBusy`; consume/discard a result or retry later. A blocked stream with no owned pending item does not advance its source. |
| `handle.result(timeout=None)` | Transfer the result once and release its engine slot; repeated consumption fails. Timeout leaves the request admitted. |
| `handle.cancel()` | True only for work that has not started; `result()` then raises `concurrent.futures.CancelledError`. |
| `handle.discard()` | Cancel queued work, release completed output, or finish and drop a running result. Running credits release only after completion. |
| `engine.close(cancel_pending=False, timeout=None)` | Stop admission and drain accepted work by default; optionally cancel queued work. Completed handles remain readable. Timeout stops admission but never kills an active model call. Call close again after it finishes. |

Byte credits cover admitted array payloads. They do not cap model weights, CUDA
workspace/graphs, JSON metadata, upstream iterator/DataLoader buffers or arrays
the caller retains after consuming results. Reserve model/workspace separately.
Dropping a handle without consuming/discarding it does not free its result credit.
The queue retains no exception traceback that would keep a failed input alive.
Internal execution failure terminates admission and fails pending work without
retrying the model; per-request admission errors do not restart the resident core.
FIFO applies to committed requests; CPU preparation reservations may complete
in a different order across producers.
The close timeout bounds dispatcher draining; the existing native cleanup then
runs synchronously and has no newly introduced timeout or forced interruption.

Synchronous `generate` preserves its existing busy behavior until queued mode is
first used; thereafter it submits and consumes through the same queue. Concurrent
submission cannot activate during an already running direct call. Use
`GenerationBusy`, `GenerationClosed` and `GenerationFailed` from `aureon.api` for
generation errors. They alias the existing TerraMind session types; the older
`Session*` exports belong to the separate KAO-TerraMind session API.

## Explicit upstream bridges and execution capabilities

Use `TorchGeoAdapter` with per-sample callbacks. It neither reinterprets training
labels as validity nor infers a raster transform from a bounding box:

```python
from aureon.api import TorchGeoAdapter

adapter = TorchGeoAdapter(
    bands=bands, units="surface_reflectance_x10000", normalization="none",
    valid_mask=lambda sample: sample["valid_pixels"],
    grid=lambda sample: sample["aureon_grid"],  # actual CRS and six-value affine
    source_id=lambda sample: sample["product_and_window_id"],
)
with engine.generate_many(my_torchgeo_samples, adapter=adapter) as results:
    for result in results:
        consume(result["array"], result["provenance"])
```

Use an open engine for this snippet. Omit `grid` for array-only output. The
adapter accepts one unbatched sample at a time; an already batched tensor,
z-scored imagery or ambiguous radiometry still fails. An explicit custom adapter
may return `GenerationJob` to attach per-sample seeds/profiles/identifiers.

`GenerationRecipe.from_terratorch` maps only the factory settings that AUREON's
existing TerraTorch1.2.13 adapter uses. It does not load or patch a model:

```python
from aureon.api import GenerationRecipe

recipe = GenerationRecipe.from_terratorch(
    dict(modalities=["S2L2A"], output_modalities=["S1GRD"],
         standardize=True, img_size=224, pretrained=False,
         decoding_steps=8, temps=1.0, top_p=0.8, top_k=0, timesteps=10),
    seed=17, model_id="terramind-v1-base", terratorch_version="1.2.13",
)
result = engine.generate(data, recipe=recipe)
```

Unknown settings, unsupported model/version/modalities and automatic pretrained
acquisition are rejected. Actual dependencies and pinned local assets are still
checked on engine open. The native finite recipe admits token steps1or8,
diffusion steps1–50, temperature1, top_p0.8, top_k0 and a nonnegative63-bit seed.
Recommended execution outside8token/10diffusion explicitly falls back to the
existing native/reference mechanisms; the receipt explains the selection.

`engine.capabilities()` exposes prepare/execute/deliver ownership, fixed input
and output semantics, precision, limits and unsupported batching modes.
`generation_compatibility(data, recipe=..., execution_profile=...)` produces a
detached contract key including pinned model/tokenizer, sampling and precision,
plus separate input identity/grid. Equal keys do not mean equal input pixels,
permission to combine requests or completed GPU qualification. Root's spatial
and deterministic placement APIs are separately exposed from `aureon.api`; they
do not turn this serial core into fused batching or model-qualified tiled output.

## Explicit EO routes and offline STAC

Declare observed inputs, generated targets, per-modality grids and an
adapter-specific sampling recipe with the [EO route contract](../packaging/EO-ROUTES.md).
The planner admits the current TerraMind S2L2A → S1GRD route and delegates to the
existing GenerationEngine. The offline STAC resolver requires explicit asset,
band, validity and grid selections; it performs no network or model execution.
Other generation modalities and COP-GEN remain unsupported candidates.

## Additional explicit adapters and raster sources

Use [borrowed models, temporal features and adapter registration](../packaging/BORROWED-ADAPTERS.md) for caller-owned FP32 modules and actual upstream CPU aggregation. Use [bounded STAC windows](../packaging/STAC-WINDOWS.md) to connect explicit assets to the existing scene executor. These interfaces are included in the selected Best wheel; CUDA adapters and live remote providers retain separate qualification.

The [experimental COP-GEN namespace](../aureon/copgen_native/PROFILE.md) provides a Torch2.10
optional extra and a separate legacy Torch2.6 source profile. It does not change the normal TerraMind
install or register a universal EO route. Packaged CPU fixtures do not transfer
the earlier source-bound GPU acceptance to this wheel.
