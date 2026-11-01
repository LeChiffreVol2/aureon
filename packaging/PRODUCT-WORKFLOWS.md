# Complete local workflows

**Runtime selection:** use the [locked integrated Best package](CANDIDATE-QUICKSTART.md), `e09e710c...`, v0.1.0. Original `0fa5ef2b` measurements remain historical evidence. The installed package and current examples do not imply new performance or hosted qualification.

The locked integrated wheel has exact FP32 acceptance for TerraMind base
224x224 and a 448x320 four-tile scene, including local HTTP/Cloud delivery.
Use [Best workflows](BEST-WORKFLOWS.md) for the default base path. Other
family/array interfaces retain their separate evidence; no new GPU runs are
claimed here. The published GitHub release remains unchanged.

## Scenes, tiles and qualification

The library accepts scenes larger than one generation tile. **224x224 is the
native tile size for this TerraMind route, not an application scene limit.**
Scene dimensions need not be multiples of 224: terminal windows anchor to the
scene edge and overlapping output uses the declared FP32 reconstruction recipe.
Each dimension must be at least 224. Width, queue, decode and storage budgets
control admission; increase them explicitly for a qualified workload.

Existing CPU/Rasterio checks include a 1793x2017 scene with 90 windows, odd edge
sizes, overlap, skipped invalid tiles, stable seeds, coverage and resident reuse.
Those checks use a deterministic CPU model double: they prove data/lifecycle
contracts, not neural speed or scientific seam quality. The small 224x224 speed
benchmark is a separate measured workload and does not narrow these interfaces.

## Check prerequisites before allocating a model

In the provisioned generation environment, place the official pinned files in
one directory using their published filenames, then run:

```sh
aureon-engine doctor --model terramind-v1-base \
  --assets-dir /models --verify-assets --work /work/first-job
```

The JSON report separates dependency mismatches, missing assets, wrong file
sizes/checksums and work-volume capacity, with remedies. It never installs,
downloads, creates directories, loads a model or probes CUDA. Without
`--verify-assets`, matching file sizes leave the report `incomplete`; size is not
an identity check. `passed` covers these prerequisites only, not GPU capacity or
the scientific quality of your data. Exit status is 0 only for `passed`.

For TiM, declare the intended target, for example
`--model terramind-v1-small-tim --tim-modality S1GRD`. The doctor checks only the
needed tokenizer files. Backbone with ordinary S2L2A input needs no tokenizer.
KAO-TerraMind profiles remain separate; a checkpoint check does not validate a
prepared bundle or a guarded execution qualification.

## Run a prepared scene through bounded tiled generation

Prepare aligned rasters first: a 12-band FP32 GeoTIFF in the documented S2L2A
band order, surface reflectance ×10000, and a separate uint8 validity raster on
the same grid. Follow [input preparation](INPUTS.md) for radiometric offsets,
bands and masks. A training label map is not a validity mask. The scene wrapper
does not normalize, infer units, reproject or fill cloud pixels.
Inputs must be self-contained GeoTIFFs. External masks, auxiliary metadata,
world files and overview sidecars are rejected, including companions introduced
during a read; otherwise those files could influence pixels without being bound
to the declared source identities. Prepare a self-contained copy explicitly.

```sh
aureon-engine scene-plan --model terramind-v1-base \
  --image /data/scene.tif --valid-mask /data/valid_mask.tif \
  --model-file /models/TerraMind_v1_base.pt \
  --tokenizer-file /models/TerraMind_Tokenizer_S1GRD.pt \
  --work /work/scene-session --output /results/scene-one \
  --stride 224 --max-in-flight 2 --max-buffer-bytes 134217728
```

`scene-plan` reads source rasters and identities, checks every input pixel and
reports planned/eligible/skipped tile counts, reconstruction and queue budgets,
decode limits and output-space admission. It creates no result and loads no model.
Work/output must be separate fresh directories under existing parents. Asset
identities are checked by the existing engine loader when execution is requested;
use `doctor --verify-assets` to inspect them beforehand.

Run the same arguments with `generate-scene` and add `--execute-model`:

```sh
aureon-engine generate-scene --model terramind-v1-base \
  --image /data/scene.tif --valid-mask /data/valid_mask.tif \
  --model-file /models/TerraMind_v1_base.pt \
  --tokenizer-file /models/TerraMind_Tokenizer_S1GRD.pt \
  --work /work/scene-session --output /results/scene-one \
  --stride 224 --max-in-flight 2 --execute-model
```

The default `reject` policy stops on invalid/cloud/nodata/nonfinite input before
model construction. Explicit `--invalid-policy skip_invalid_tiles` omits every
224×224 window containing invalid pixels. It performs no hidden imputation.
Consequently some source-valid pixels can remain uncovered; those pixels have
NaN output and validity mask 0, and provenance distinguishes source validity
from generated coverage. Do not interpret omitted areas as model predictions.

The recipe uses stable coordinate-derived tile seeds, fixed FP32 generation and
bounded stripe reconstruction. Different stride/context/seeds define different
generative recipes. Tiled output is not claimed equivalent to a monolithic scene,
and numerical parity alone does not establish seam or geophysical quality.

Delivery contains COG, provenance, tile receipts and preflight information.
`completion.json` is written only after successful source verification and owned
engine shutdown. Failed/cancelled attempts may retain partial artifacts for
inspection; their presence does not mean delivery completed. SDK cancellation is
cooperative between tiles and is checked after COG export, between verification
blocks and before the completion receipt. It does not preempt an active CUDA
call or GDAL copy. A cancellation observed during export/read-back/owned shutdown
prevents completed delivery; partial COG/provenance files alone are not success.

In Python use `SceneGenerationRequest`, `inspect_generation_scene` and
`run_generation_scene` from `aureon.api`. Budgets bound application buffers, not
total model/driver/allocator memory or caller DataLoader prefetch. See
[parallel execution](CONCURRENT-GENERATION.md) for the separate queue/lane/replica
contracts. `max_in_flight` does not itself mean fused neural batching.

## Use a TorchGeo or TerraTorch DataLoader

`TorchGeoBatchAdapter` accepts actual CPU BCHW FP32 batches and lazily yields
owned `GenerationInput` or `GenerationJob` objects. Supply explicit per-item
validity, grid and source-identity selectors. The optional image selector supports
nested TerraTorch dictionaries; no arbitrary transform or borrowed model is
silently patched. The adapter does not infer CRS, normalization or cloud semantics.

The [runnable integration example](../examples/generation/dataloader.py) shows
Torch default collation, an eight-item/32 MiB image-batch bound and a two-request
engine queue. It exports each result immediately instead of accumulating scenes.
The caller owns DataLoader workers, its prefetch and shutdown. Adapter limits do
not retroactively bound those allocations. Consume/close the result stream when
stopping early so pending engine credits are released.

`seed_policy="same"` preserves an explicitly repeated seed.
`seed_policy="input_identity"` derives a stable seed from each input identity and
base recipe, independent of how samples are grouped into batches. Select and
record the policy; do not compare different seed policies as exact execution
parity. Generation consumes unstandardized reflectance ×10000; Backbone/TiM
preprocessing is different, as documented in [family interfaces](TERRAMIND-FAMILIES.md).

## Operate the local service

Use the existing [authenticated upload/job/download flow](ASYNC-SERVICE.md).
`GET /v1/metrics` is operator-only and reports aggregate queue/lease/worker,
reservation, retention and once-only shadow usage without tenant/job/path/key
labels. Customer credentials cannot access it. See [operator monitoring and
recovery](SERVICE-OPERATIONS.md#operator-monitoring).

Queue/replica CPU and process tests use explicitly simulated workers. Real-model
acceptance and hardware profiles have separate artifact-bound receipts. A
placement estimate is not a reserved GPU or a working cross-model dispatcher.
These additions do not activate hosted access, payments or provider resources.

See [resident scene HTTP, feature schemas and SDK dispatch](RESIDENT-WORKFLOWS.md)
for the current source candidate, including process ownership, finite budgets,
large artifact transport and separate source/installed/hardware qualification.
