# TerraMind 1.0 families and encoder features

AUREON **v0.1.0** distinguishes three TerraMind interfaces. A model size alone
does not identify its task or output contract.

| Mode | Model IDs (`SIZE` = `tiny`, `small`, `base`, `large`) | Output |
| --- | --- | --- |
| Generation | `terramind-v1-SIZE` | Generated modality arrays through the generation API; explicit export can produce georeferenced imagery. |
| Backbone | `terramind-v1-SIZE-backbone` | Native transformer feature layers. No prediction head or geospatial export. |
| Thinking in Modalities (TiM) | `terramind-v1-SIZE-tim` | Native feature layers conditioned on internally generated modality tokens. Those tokens are not a delivered synthetic image. |

`TerraMindEncoder` supplies the Backbone and TiM interfaces. It calls the pinned
TerraTorch 1.2.13 constructors for each size. The tiny/small/base/large encoder
dimensions are 192/384/768/1024, with 12/12/12/24 layers respectively. TiM uses
the corresponding 4/6/12/24-layer decoder internally.

## Local assets and explicit execution

Provide the model checkpoint and any required tokenizer checkpoints as existing
local files. The `aureon.terramind_native.families` registry records the official
repository, immutable revision, filename, byte size and SHA256 for every asset.

The encoder validates those identities before construction and again after
loading. It never acquires assets, resolves a floating revision, or installs a
dependency.

Construction requires `execute_model=True`, but is lazy: entering the context
does not import Torch, read checkpoints, allocate a model, or execute inference.

The first valid `predict` call loads one engine-owned resident model. Later calls
reuse it. `close()` releases that ownership and is idempotent. There is no borrowed
`from_model` bridge; caller-owned models are never modified.

```python
from aureon.api.terramind_encoder import TerraMindEncoder
# image: upstream-standardized float32 pixels, shape (B, 12, 224, 224).
# Prepare these pixels as described under Input preprocessing below.
with TerraMindEncoder(
    "terramind-v1-small-backbone",
    model_file="/models/TerraMind_v1_small.pt",
    modalities=["S2L2A"],
    device="cpu",
    execute_model=True,
) as encoder:
    result = encoder.predict({"S2L2A": image}, seed=17)
    feature_layers = result["features"]
    provenance = result["provenance"]
```

Use `device="cuda"` or `"cuda:N"` explicitly for a provisioned CUDA device.

CPU is the default, not a claim that large checkpoints will fit every machine.

This interface implements FP32 evaluation under inference mode, with autocast
disabled and highest matmul precision. RNG and changed precision policies are
restored after successful and failed calls. A concurrent call or close on the
same encoder fails immediately. A model-load/forward/output-contract failure
makes the encoder unusable until a new instance is constructed.

## TiM and tokenizer requirements

```python
with TerraMindEncoder(
    "terramind-v1-tiny-tim",
    model_file="/models/TerraMind_v1_tiny.pt",
    modalities=["S2L2A"],
    tim_modalities=["S1GRD"],
    tokenizer_files={
        "S1GRD": "/models/TerraMind_Tokenizer_S1GRD.pt",
    },
    tim_decoding_steps=1,
    merge_method="mean",
    execute_model=True,
) as encoder:
    features = encoder.predict({"S2L2A": image}, seed=17)["features"]
```

TiM defaults to `tim_modalities=["LULC"]`. Supported targets are S2L2A, S1GRD,
S1RTC, DEM, LULC and NDVI. Supply exactly the tokenizers for the union of TiM
targets and tokenized input modalities (LULC and NDVI). Missing or extraneous
tokenizer mappings fail before checkpoint construction. A target can overlap an
input only for LULC or NDVI, matching upstream's tokenized-input fallback rule.

Inputs must remain **full-band pixel tensors with the expected upstream preprocessing**, not feature tensors,
precomputed tokenizer codes, selected bands, or generation API output wrappers.

Canonical input names and channel counts are S2L2A (12), S2L1C (13), S2RGB (3),
S1GRD (2), S1RTC (2), DEM (1), NDVI (1), and LULC (class indices). Continuous
inputs are finite FP32 `(B, C, 224, 224)`; LULC is int64 `(B, 224, 224)` with
indices 0–9. Every configured modality is required, with the same positive batch
size. Custom bands/modalities, coordinates, text, other spatial sizes, nodata
masks and missing-input inference are not exposed by this interface.

The wrapper does not infer units or apply normalization, resizing or masking.

Supply values with the numeric preprocessing expected by the pinned upstream
backbone/TiM workflow. Full-band validation cannot establish sensor units or
scientific correctness. Input tensors and the modality mapping are copied before
upstream can mutate them; outputs are detached, independent feature copies on
the selected device.

## Input preprocessing: generation versus encoder features

These interfaces intentionally have different preprocessing ownership:

| Interface | S2L2A values supplied by the caller | Standardization owner |
| --- | --- | --- |
| `GenerationInput` / `GenerationEngine` | Surface reflectance ×10000, before model standardization | Generation model, once |
| `TerraMindEncoder` Backbone/TiM | Pixels standardized with the pinned TerraMind 1.0 pretraining mean/std | Caller, once |

The encoder API's full-band requirement describes bands and image layout;
it does not mean unstandardized sensor values. The pinned upstream encoder
and TiM forward methods do not standardize images. Do not feed generation's
raw input directly to an encoder, and do not feed standardized encoder pixels
to `GenerationInput`.

For an all-valid S2L2A tensor already in FP32 surface reflectance ×10000 and
the documented 12-band order, prepare the encoder input explicitly:

```python
import torch
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std,
)
# raw_s2: CPU torch.float32, (B, 12, 224, 224), reflectance ×10000.
assert raw_s2.dtype == torch.float32
key = "untok_sen2l2a@224"
mean = torch.tensor(v1_pretraining_mean[key], dtype=torch.float32)[None, :, None, None]
std = torch.tensor(v1_pretraining_std[key], dtype=torch.float32)[None, :, None, None]
image = ((raw_s2 - mean) / std).contiguous()
```

This uses the installed TerraTorch 1.2.13 constants instead of duplicating them
in AUREON. Record your raw asset identity, preprocessing recipe and resulting
tensor/NPZ identity with your application. For the `encode` CLI, store `image`
under the `S2L2A` NPZ key. Other modalities require their own documented units
and upstream preprocessing; the S2 example is not a generic normalization rule.

## Feature layout and merge semantics

`result["features"]` preserves the upstream list of per-layer outputs. The last
layer includes upstream's encoder normalization; earlier layers are returned as
upstream produces them. Each tensor has `(B, tokens, channels)` layout.

| `merge_method` | Native result |
| --- | --- |
| `"mean"` (default), `"max"` | Reduce image modalities; 196 patch tokens at 224×224. |
| `"concat"` | Concatenate modality feature channels; all configured inputs are required. |
| `"dict"` | Per-layer dictionaries with upstream modality keys and tensor values. |
| `None` | Preserve the full concatenated token sequence. |

For concat, AUREON constructs and strictly loads the ordinary encoder before
selecting upstream's concat operation. It does not invent the untrained
`missing_mod_token` placeholder, which cannot be reached with the required full
input set. TiM supplies its own target tokens before this merge.

The pinned upstream TiM `"dict"` branch names outputs using the original input
keys; it does not expose all generated modalities as separate dictionary entries.

AUREON preserves that behavior. Use `None` to retain the complete token sequence,
or mean/max/concat to combine the modalities. Features are never labeled as
segmentation predictions, generated SAR, COGs, or georeferenced products.

## Integrity and qualification

The loader does not use upstream's permissive checkpoint filters. Every active
model, TiM sampler and tokenizer state tensor must exist with the exact shape and
dtype, be finite when floating-point, and equal the resident loaded value. The
TiM sampler namespace is bound explicitly to the original full checkpoint keys.

Only identified inactive modality embeddings and the unused backbone decoder
state may remain unused; provenance records those keys and active tensor hashes.

Missing tensors never fall back to random initialization.

The local v0.1.0 wheel `bf2baf7005b4899e42a90c3dff612c3d275d0b87566fd37d7ebb8c4c93777c48` passes real-weight GPU acceptance for
all four sizes in Generation, Backbone and TiM on RTX 3070 8 GiB / WSL2,
Python 3.12 and pinned TerraTorch 1.2.13. For each of the twelve routes, two
resident SDK requests and one cold CLI request exactly match an independently
loaded upstream FP32 reference with identical checkpoint tensors, input and seed.
This is 36 requests plus one tiny-generation HTTP job: upload, submit, status,
verified COG/provenance download and idempotent replay without another inference.

The tested input is one retained all-valid Sentinel-2 L2A 224x224 crop, seed17.
Generation uses 8 token / 10 diffusion steps, temperature1, top-p0.8 and top-k0;
base uses its recommended guarded graph/reuse profile and other sizes use native
execution. Backbone uses S2L2A and mean merge; TiM adds only S1GRD conditioning
with one decoding step and mean merge. Other advertised tokenizer/input choices
remain implementation contracts, not newly GPU-qualified combinations.

Generation arrays/tokens, all feature layers, replay and caller-state checks
pass exact FP32 parity. Nine delivered COGs were reopened and checked for pixels,
grid, validity and synthetic labels. All tested owned workers release CUDA to
zero after output release and teardown. The SDK caller owns returned feature
tensors; keeping those outputs alive intentionally retains their memory.

Generation sessions own highest-FP32 policy for their resident lifetime and
restore the caller policy on close; each request restores RNG. Encoder calls
restore caller RNG/precision on return. Cold dependency imports and file-export
feature ownership are covered by regression tests. This bounded acceptance is
not a throughput comparison, multi-GPU qualification, long-run leak test,
scientific-accuracy evaluation, or arbitrary-scene/modality certification.
The original published archives and their separate evidence are unchanged.

## Generation and local service

Generation supports all four model sizes with the same explicit all-valid,
FP32 S2L2A `(12, 224, 224)` → synthetic S1GRD `(2, 224, 224)` contract. The
S1GRD tokenizer is shared, while each model uses its own pinned checkpoint.

```python
from aureon.api import GenerationEngine
with GenerationEngine(
    "terramind-v1-small",
    model_file="/models/TerraMind_v1_small.pt",
    tokenizer_file="/models/TerraMind_Tokenizer_S1GRD.pt",
    work="/work/new-job",
    execute_model=True,
) as engine:
    result = engine.generate(prepared_input, seed=17)
```

`prepared_input` is a `GenerationInput` with explicit full-band S2L2A semantics.

Use `export_generation` with a valid grid to deliver COG/provenance. Source and
sampling checks remain active. The recommended profile keeps base's eligible
reuse/graph route; tiny/small/large start with native execution. Their explicit
expert optimizations have no inherited base or N-lane GPU qualification.

```bash
aureon-engine models
aureon-engine assets --model terramind-v1-small
aureon-engine generation-plan --model terramind-v1-small
aureon-engine generation-config --model terramind-v1-small --model-file /models/TerraMind_v1_small.pt --tokenizer-file /models/TerraMind_Tokenizer_S1GRD.pt --output /work/assets.json
aureon-engine generate --model terramind-v1-small --config /work/assets.json --manifest /data/prepared-input.json --output /work/new-generation --job-id j1 --attempt-id a1 --execution-profile recommended --execute-model
```

The generation local async service uses the selected model ID and rejects a
configuration bound to another size before opening the model. Omitted denoiser
selection uses graph for base and native for the other sizes. Supply existing
local assets and the locked generation environment; no automatic download occurs.

See [the local service flow](ASYNC-SERVICE.md). The current source candidate also
serves Backbone/TiM through the `features` job and `fp32-features-v1` profile.
[Resident workflows](RESIDENT-WORKFLOWS.md) describes the frozen operator config,
numeric NPZ input, streaming downloads and matching qualification scope.

## Encoder CLI

Use a NumPy `.npz` with canonical modality keys (for example `S2L2A`) and the
full-band batched arrays described above. Pickled/object arrays are rejected.

```bash
aureon-engine encode --model terramind-v1-small-backbone --model-file /models/TerraMind_v1_small.pt --input /data/modalities.npz --output /work/new-features --device cpu --execute-model
aureon-engine encode --model terramind-v1-tiny-tim --model-file /models/TerraMind_v1_tiny.pt --tim-modality S1GRD --tokenizer S1GRD=/models/TerraMind_Tokenizer_S1GRD.pt --input /data/modalities.npz --output /work/new-tim-features --device cpu --execute-model
```

This creates `features.npz` and `provenance.json` in a new output directory.

Feature layer and modality keys are recorded explicitly. The input archive is
limited to 512 MiB uncompressed; that file bound is not a GPU memory guarantee.

The features preserve native semantics and have no automatically assigned CRS,
prediction labels or physical measurement claim.

Upstream references: [TerraMind guide](https://torchgeo.org/terratorch/1.2.13/guide/terramind/)
and [official model repository](https://github.com/IBM/terramind).

