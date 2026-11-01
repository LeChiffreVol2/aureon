# Resident inference and async delivery

**Runtime selection:** use the [locked integrated Best package](CANDIDATE-QUICKSTART.md), `e09e710c...`, v0.1.0. Original `0fa5ef2b` measurements remain historical evidence. The installed package and current examples do not imply new performance or hosted qualification.

For TerraMind base, start with [Best scene and resident workflows](BEST-WORKFLOWS.md).
Use `GenerationSession` / `open_model_session` with `PreparedSceneRequest`
and `run_prepared_scene` for repeated scenes. The original owner thread,
FP32 graph/reuse and lifecycle execute directly. One owner runs one native
tile at a time; queue size is not physical batching.

Backbone/TiM feature APIs and other model families retain their documented
contracts and separate qualification. See [model modes](TERRAMIND-FAMILIES.md).
Keep input grids, bands, units and validity explicit; no silent cloud filling.

**TerraMind HTTP limit:** the locked worker executes TerraMind base only for
`generation` and `generation_scene`. Shared configuration/discovery also lists
other sizes, but those requests are rejected before model execution; listing
is not serving support. It also rejects the Backbone/TiM `features` executor.
Use the separate Backbone/TiM SDK/CLI interfaces for feature extraction.

## Configure a scene service

Save an operator-owned `scene-service.json`:

```json
{
  "schema": "aureon.terramind_service_workflow.v1",
  "model_id": "terramind-v1-base",
  "task": "generation_scene",
  "model_file": "/models/TerraMind_v1_base.pt",
  "tokenizer_files": {"S1GRD": "/models/TerraMind_Tokenizer_S1GRD.pt"},
  "options": {"stride": 224, "invalid_policy": "reject", "execution_profile": "recommended", "max_in_flight": 1},
  "limits": {}
}
```

Defaults are finite: 1 GiB input, 2 GiB total output, 1 GiB per artifact,
8192 pixels per side, 128 MiB reconstruction buffers, 32 MiB decode buffers,
8 GiB scratch, and a 1 GiB free-disk reserve. Retained output reservations are
8 GiB per tenant and 16 GiB across the service. Inspect the normalized workflow
and adjust explicit limits for a qualified deployment; these are application
budgets, not a guarantee that any model fits physical RAM/VRAM.

```sh
aureon-service init --state /srv/aureon/scene-state \
  --credentials /srv/aureon/scene-keys.json --tenant developer \
  --model terramind-v1-base --environment generation \
  --profile fp32-generation-scene-v1 --workflow-config scene-service.json

aureon-service serve --state /srv/aureon/scene-state \
  --work /srv/aureon/scene-worker --execute-model --port 8042 \
  --max-jobs 100 --max-seconds 3600
```

Use a Linux-native work volume for real workers. The config is normalized and
frozen into private state. Customers select no asset path or mutable execution
options; they submit a seed. The model and tokenizer are reused across jobs.
Configuration binding is not a claim that weights have already loaded: inspect
readiness and each job's model provenance.

Create the upload manifest next to the self-contained input GeoTIFFs:

```sh
aureon-engine service-input --task generation_scene \
  --image /data/scene/image.tif --valid-mask /data/scene/mask.tif \
  --output /data/scene/service-input.json
```

This binds existing files with streamed hashes and sizes. It does not copy,
transform or validate pixel semantics. The resulting manifest has this shape:

```json
{
  "schema": "aureon.terramind_scene_input.v1",
  "image": {"path": "image.tif", "sha256": "REPLACE_WITH_FILE_SHA256", "bytes": 123},
  "mask": {"path": "mask.tif", "sha256": "REPLACE_WITH_FILE_SHA256", "bytes": 456},
  "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
  "units": "surface_reflectance_x10000"
}
```

The command supplies actual hashes and sizes; placeholders above only explain
the schema. The service validates the
entire prepared input in a bounded CPU parser before allowing model execution.
Input pixels, masks, grid and file identities are checked without hidden repair.

```sh
aureon-service-client --url http://127.0.0.1:8042 \
  --credentials /srv/aureon/scene-keys.json --tenant developer run \
  --manifest /data/scene/service-input.json --model terramind-v1-base \
  --profile fp32-generation-scene-v1 --task generation_scene --seed 17 \
  --idempotency-key scene-one --recovery-file /results/scene-one-request.jsonl \
  --output /results/download-one
```

The client uploads, polls, submits, polls the job and saves the verified COG,
provenance and receipts. It durably records input/job IDs in the explicit recovery file. See
[download recovery](ASYNC-SERVICE.md#resume-a-download-without-submitting-again). Retry an
uncertain submission using the same input and idempotency key. File presence
alone is not success; check the completed job and verified download receipt.
Large artifacts are copied, hashed, served and persisted in bounded chunks.
The network deadline remains finite; a slow connection may require an explicitly
larger configured deadline and client timeout. Do not disable admission or
storage bounds to accommodate large inputs.

## Feature transport contract (not executable by the selected HTTP worker)

The following schema remains available for transport/adapter development and
CPU simulated tests. The selected worker rejects its executor before model
loading. For actual inference, use `TerraMindEncoder` or the `encode` CLI from
the [family guide](TERRAMIND-FAMILIES.md); do not start a feature HTTP service
expecting this package to execute it.

Use task `features`, profile `fp32-features-v1`, and a corresponding model ID,
such as `terramind-v1-tiny-backbone` or `terramind-v1-tiny-tim`. The same workflow
config schema applies. Backbone with S2L2A uses `tokenizer_files: {}` and
`options: {"modalities": ["S2L2A"]}`. TiM with S1GRD adds its tokenizer and
`"tim_modalities": ["S1GRD"]` to the options. These choices are operator-owned.

Create its manifest with `aureon-engine service-input --task features --inputs
/data/features/inputs.npz --output /data/features/manifest.json`. The feature
manifest uses schema `aureon.terramind_features_input.v1` and an `inputs` member
containing the relative NPZ path, byte size and SHA256.
The NPZ contains native FP32 full-band batches, or int64 LULC indices, with no
pickled objects. Its aggregate uncompressed array bound is 512 MiB; the service
also enforces its configured input and batch limits. Downloads contain
`features.npz`, feature layout/provenance and job receipts. There is no synthetic
COG attached to latent feature extraction.

## Dispatch several exact models

`ModelPool` uses immutable `GenerationRoute` and `EncoderRoute` registrations,
explicit `DeviceSlot` assignments and `PoolResources`. Call `submit(route_name,
input, seed=..., tenant=...)` and consume each returned handle. The
[executable example](../examples/generation/model_pool.py) shows complete setup.
Run process-based code behind Python's `if __name__ == "__main__":` guard.

Each resident session lives in a spawned process to isolate global RNG, precision
and CUDA state. Compatible work reuses it. The scheduler serves admitted tenants
in round-robin order and evicts only idle sessions. Memory is reusable only after
verified close and process exit. Worker death or failed cleanup quarantines its
slot. It does not substitute another model or use an LLM to decide a route.

Provision host/device budgets using the selected model and hardware. Input and
output copies, retained results and declared transient/resident workspaces have
separate credits. Consumed outputs belong to the caller, so accumulating them
outside the pool is the caller's memory responsibility. Cancellation removes
queued jobs; an active CUDA operation is not forcibly interrupted. Always close
the pool and inspect its cleanup receipt. A timeout means draining is incomplete.

This is a same-host execution API. The HTTP workflow above remains one bound
model per state. Multiple physical GPUs, multiple hosts, sustained operational
loads and new precision profiles require their own qualification; they are not
implied by process tests or by the number of registered model IDs.
