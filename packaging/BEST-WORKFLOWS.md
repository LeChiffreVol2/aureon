# TerraMind generation workflow

> **Installed-package scope:** These features ship in the single v0.1.0
> package and call the resident Best directly. No source controller or model IPC
> is required. See [installation](CANDIDATE-QUICKSTART.md); original benchmark
> scope and published alpha artifacts remain unchanged.


One supported default SDK/CLI route uses the preserved numerical core with scene
delivery and bounded repeated work. Version remains **v0.1.0**; publication is
closed. Install the single current generation artifact identified by its delivery
receipt. Historical archives remain evidence, not alternate default installs.

The core comes from wheel `0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97`.
Its source inventory is shipped under `aureon/_reference/terramind/SOURCE.json`.
Eleven Python files are verbatim; the helper file is a verbatim prefix excluding
benchmark functions. The typed facade verifies this inventory once before use.
Per-job execution retains original FP32 sampling, graph/reuse, and file checks.
This source identity is not a new throughput result.

## Install and configure once

Use a fresh Linux/WSL Python 3.12 environment. Install the reviewed **generation**
wheel with its supplied dependencies. [Dependency profiles](LOCKED-INSTALL.md) and
[pinned assets](ASSETS.md) document the requirements. The review tests use existing
dependencies offline; a new developer's normal install may download dependencies.
Neither installation nor the commands below fetch model weights automatically.

```sh
pip install /absolute/reviewed/aureon_inference-0.1.0-py3-none-any.whl
pip check
aureon-engine --help
aureon-engine check --environment generation
```

Prepare your own Sentinel-2 files using the [input contract](INPUTS.md) and
`aureon-prepare scene`. Radiometric offsets, grid alignment, band order and
cloud/invalid policy must be explicit. The scene interface consumes a prepared
twelve-band **FP32 surface_reflectance_x10000** GeoTIFF and a separate uint8 0/1
validity GeoTIFF with the same CRS, transform and dimensions. Each dimension must
be at least 224; it need not be a multiple of 224. There is no silent filling or
resampling. Large compressed strips can exceed the decode budget: prepare tiled
storage explicitly rather than removing that budget.

Create an all-valid 224x224 seed input from that preparation and bind the local
model assets. This seed manifest is required by the preserved config contract;
each scene tile supplies its own manifest during execution.

```sh
aureon-prepare generation --manifest /data/prepared/manifest.json \
  --row0 0 --col0 0 --output /work/seed-input
aureon-engine generation-config \
  --model-file /assets/TerraMind_v1_base.pt \
  --tokenizer-file /assets/TerraMind_Tokenizer_S1GRD.pt \
  --manifest /work/seed-input/manifest.json --output /work/generation.json \
  --token-steps 8 --seed 17
```

Paths shown are placeholders. Create their parents first; outputs must be fresh.
The first window must be valid or select another valid row/column. Asset-only
configs, other TerraMind sizes/modalities and approximate precision are not
accepted by this preserved base S2L2A-to-S1GRD path.

## One crop or a whole scene

```sh
aureon-engine generate --config /work/generation.json \
  --manifest /work/seed-input/manifest.json --work /work/crop-resident \
  --job-id first --execute-model

aureon-engine scene-plan --image /data/prepared/scene.tif \
  --valid-mask /data/prepared/valid_mask.tif --output /results/scene-1 --job-id scene-1

aureon-engine generate-scene --config /work/generation.json --work /work/scene-resident \
  --image /data/prepared/scene.tif --valid-mask /data/prepared/valid_mask.tif \
  --output /results/scene-1 --job-id scene-1 --execute-model
```

Use the actual file paths recorded by your preparation receipt. Scene planning
reads source files but creates no outputs, loads no model and does not query CUDA.
The `generate-scene` command checks the scene and budgets before constructing
the selected resident, then checks mutation again at execution admission. Defaults are 128 MiB reconstruction/adapter buffers, 32 MiB decode
blocks, 512 MiB new workspace, 512 MiB output and 1 GiB free-disk reserve. CLI flags
allow explicit larger budgets. These are host/workspace admission bounds, **not
a total process RAM/VRAM cap**; model, GDAL and allocator memory are separate.

Each eligible tile uses the same native FP32 run. Coordinate-derived seeds and
row-major uniform reconstruction are recorded. `--invalid-policy reject` is the
default. `skip_invalid_tiles` leaves uncovered pixels masked and does not invent
sensor data. `--stride` controls overlap; overlap is not a guarantee of invisible
seams or physical accuracy. S1GRD output is synthetic VV/VH in dB.

Only a verified `completion.json` marks scene delivery. Keep it with the COG,
provenance, mask/coverage and staged native receipts. A partial TIFF or a
`failure.json` is not a completed result.

After restart, verify the retained input/native artifacts and completed delivery:

```sh
aureon-engine verify-scene --output /results/scene-1 --job-id scene-1 --attempt-id 1
```

This is a read-only integrity check. It runs no model and does not retry, remove
or charge for any work. Original source and native receipts must remain available.

## Reuse one resident for bounded repeated jobs

```python
from aureon.api import open_model_session, PreparedSceneRequest, GenerationWorkQueue

requests = [
    PreparedSceneRequest(image="/data/prepared/scene.tif",
                     valid_mask="/data/prepared/valid_mask.tif",
                     output=f"/results/repeat-{i}", job_id=f"repeat-{i}",
                     seed=17 + i, hourly_rate=1., currency="THB")
    for i in range(2)
]
with open_model_session("terramind-v1-base", config="/work/generation.json",
                        work="/work/repeated-resident", execute_model=True) as resident:
    with GenerationWorkQueue(resident, max_jobs=2,
                       max_admitted_bytes=4 * 1024**3,
                       execute_model=True) as queue:
        handles = [queue.submit(request) for request in requests]
        for handle in handles:
            print(handle.result()["cog"])
```

The queue holds descriptors and bounded unconsumed receipts, not whole-scene
arrays. It serializes this resident's tile calls; it is not GPU batching or a
distributed broker. `max_jobs` counts pending, running and unconsumed results.
Consume/discard handles to release those credits. Storage admission stays charged
for the queue lifetime because outputs remain on disk. Starting another queue
does not delete prior files or reserve storage against other processes.

Call `handle.cancel()` to cancel before/after native calls and before delivery.
It cannot interrupt an active CUDA kernel. Queue close drains before returning
the caller's resident; a timeout leaves it owned until a later successful close.
Do not close or directly run a borrowed resident. Recoverable source/output
errors do not imply model failure. A terminal native failure requires closing the
owner and opening a fresh resident. There are no automatic retries or deletions;
a retry uses fresh output/work paths and a new attempt ID, after verifying any
existing completion. This local queue does not resume across a process crash.

## Cost and scale

```sh
aureon-engine cost --resource-seconds 3600 \
  --completed-outputs 100 --hourly-rate 1 --currency THB
aureon-engine capacity --devices /work/budgets.json \
  --per-replica-bytes 6442450944 --reserve-bytes 2147483648 \
  --replicas-per-device 2
```

The cost command yields 1 THB total and 10 THB per 1,000 delivered outputs for
those inputs. Rates are caller assumptions; this is not a provider quote or
payment ledger. Include startup, failed work and delivery in the chosen occupancy
boundary. For several paid resources sum their occupied resource-seconds rather
than using concurrent wall time alone. Supply an interval including resident
startup/close to the cost command; optional scene receipt costs describe their
narrower boundary explicitly.

`budgets.json` is a list such as
`[{"host":"worker-a","device":"0","total_bytes":25769803776,"free_bytes":23622320128}]`.
These numbers are illustrative capacity declarations, not measured hardware.
The planner supports arbitrary positive N and separate host/device placements,
rejects overcommit and emits each fresh worker's `CUDA_VISIBLE_DEVICES` setting.
It starts no processes. Apply visibility **before** a worker imports CUDA; inside
each process the frozen core uses visible `cuda:0`. Peak-per-replica must include
CUDA context, graphs and non-Torch allocations, not just Torch allocated memory.
Physical multi-GPU/multi-host speed and useful concurrent replicas per GPU must
be measured on the target workload. Do not advertise simulated capacity as speedup.

Default execution selects reuse auto and graph; explicit `execution_profile="reference"`
selects native and reuse off. Both keep highest FP32 and TF32/autocast disabled. The facade does not silently
fall back to lower precision. Other existing library APIs retain their separately
documented artifact and qualification scope.

## API scope and compatibility

`GenerationRequest` with file input and `open_model_session("terramind-v1-base", ... )`
default to the same selected core: automatic encoder reuse and graph decoding.
`GenerationSession`, `PreparedSceneRequest`, `GenerationWorkQueue`, and
`run_prepared_scene` are canonical names for those accepted implementations.
The older `Best*` names and `aureon-engine best` commands remain compatibility
aliases; they do not select another package or numerical core.

Use `execution_profile="reference"` (CLI `--execution-profile reference`) for
explicit FP32 off/native diagnostics. Scope remains base, prepared all-valid
S2L2A 224x224 tiles to synthetic S1GRD; the measured recipe has eight token and ten
diffusion steps. No universal performance claim applies to other recipes or
whole-scene preparation/reconstruction.

Array APIs retain their array contract: `GenerationRequest(input_data=...)`
selects that interface automatically; `open_model_session(..., input_kind="array")`
opens it explicitly. `GenerationEngine` and the broader `SceneGenerationRequest`
remain compatibility interfaces. Other TerraMind sizes, Backbone/TiM and
KAO-TerraMind keep their own runtimes and qualification scopes. They are not
aliases of this base-only file session. Base file execution rejects asset-only
configs and unsupported expert pairs with a specific error rather than silently
falling back to another implementation.

## In-memory generation on the shared numerical core

TerraMind base `GenerationEngine.generate` and
`open_model_session(..., input_kind="array")` use the frozen selected RNG,
schedule, encoder-reuse, decoder binding and CUDA graph primitives. The active
memory adapter keeps the selected single-request numerical operation order; its
normalized AST and policy guards are checked against the frozen source by
`packaging/check_memory_core.py`. The file resident and frozen source stay unchanged.
The assets-only loader and memory/lane orchestration are explicit active adapters,
not whole-file copies of the selected implementation.

`GenerationInput` needs no grid for inference. Results own their CPU arrays and
carry semantic provenance. No tensor is silently written to NPY or COG and read
back to invoke a file API. Call `export_generation` explicitly when a real supplied
grid is available; the library does not invent a CRS or transform.

Compatible base array sessions default to the recommended graph/reuse profile;
reference and unsupported-recipe fallback keep their declared FP32 behavior.
Other families retain their own implementation and evidence. Each base result
records `numerical_core`, including the exact frozen manifest, original wheel,
active adapter route and normalized numerical span identity. This source receipt
is not a claim of measured parity or performance for a newly built artifact.

Existing bounded queues and N-lane admission remain in force. Admitted lanes use
independent B1 graph pools, buffers, scheduler state and CUDA streams while sharing
immutable weights and the frozen decoder/graph primitives. This is not a B2 kernel
or a promise of physical overlap/speedup; actual admission and stream receipts
remain the authority. HTTP scene and Cloud consumers can use the same
`GenerationEngine` / `SceneGenerationRequest` APIs without a new model owner.
