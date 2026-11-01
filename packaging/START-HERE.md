# Start with AUREON

Use the [single current distribution](CANDIDATE-QUICKSTART.md), then follow the
[TerraMind scene and resident workflow](BEST-WORKFLOWS.md). It selects the
preserved best FP32 graph/reuse core for the qualified base workflow by default.
No separate feature/core package is required. This Best v0.1.0 delivery is available
on GitHub. Original assets remain preserved; PyPI and hosted inference remain closed.

The commands below cover CPU preparation and retained model-specific workflows.
Their earlier GPU evidence retains its original artifact scope; the latest
acceptance receipt identifies which routes were checked on the current wheel.

## CPU walkthrough, no assets or GPU

Unpack the matching source archive in a separate directory, then run its public
example from outside that directory. Python isolated mode ensures imports come
from the installed package:

```sh
tar -xzf /absolute/aureon_inference-0.1.0.tar.gz
python -I -B aureon_inference-0.1.0/examples/generation/cpu_walkthrough.py --output cpu-demo
```

Expected: `status: passed`, `prepared_shape: [12, 224, 224]`,
`pixels_and_grid_verified: true`, `model_executed: false` and `benchmark_run: false`.
The example creates artificial pixels, prepares input and checks the crop, mask
and geospatial transform. It imports no neural runtime and makes no model-quality
or speed claim. Choose a new output directory for a repeat run. Delete `cpu-demo`
and the temporary environment after testing if you no longer need them.

## Generate from your own arrays

Follow the [GenerationEngine example](../docs/DEVELOPER-INTEGRATION.md) for a
NumPy or CPU tensor input and an array result. Optional COG export requires your
actual grid. The remaining sections describe the existing file-based CLI path.
Model execution needs the pinned assets and compatible NVIDIA GPU/driver; it is
always explicit and is not part of the CPU walkthrough.

## Supply assets

Before preparing a real job, inspect prerequisites without loading a model:

```sh
aureon-engine doctor --model terramind-v1-base --environment generation \
  --assets-dir /absolute/assets --work /absolute/work
```

Read each issue's `code` and `remedy`. A matching file size is not a verified
checkpoint: add `--verify-assets` for a full SHA256 check when assets are ready.
This command performs no download, model execution or CUDA probe. Use
`aureon-engine devices` separately for driver/device inventory.

```sh
aureon-engine assets --model kao-terramind-b
aureon-engine assets --model terramind-v1-base
```

These commands only list official revision-specific URLs, sizes and SHA256s.
Obtain assets separately from [the pinned sources](ASSETS.md) under their terms,
then verify their hashes. No upstream helper or remote Python code needs to be run.

KAO-TerraMind B users can create their own bundle from the published checkpoint:

```sh
mkdir -p "$HOME/aureon-assets"
aureon prepare-vllm --model kao-terramind-b \
  --checkpoint /absolute/assets/terramind_v1_kaoB_best.pt \
  --output "$HOME/aureon-assets/kao-terramind-b"
aureon verify-bundle --bundle "$HOME/aureon-assets/kao-terramind-b"
```

This command creates the local bundle used by AUREON Engine. It verifies the
pinned checkpoint identity before CPU weights-only conversion, using CPU memory
and disk. It does not start inference or download files. Output is not
overwritten; the receipt establishes identity, not model parity.

## Prepare user-owned Sentinel-2 input

Copy `examples/packaged_scene/aligned-input.json` or `examples/packaged_scene/l2a-input.json` from the source archive to
your data directory and fill in actual paths and provenance. See
[the input contract](INPUTS.md) for band order, units, masks and pixel windows.
Example paths, CRS and radiometric offsets are placeholders, not sample data.
Relative paths are resolved beside the spec, not beside the executable.

```sh
mkdir -p "$HOME/aureon-work"
aureon-prepare scene --spec /absolute/my-data/aligned-input.json \
  --output "$HOME/aureon-work/prepared-first"
aureon-prepare check --manifest "$HOME/aureon-work/prepared-first/manifest.json"
```

For TerraMind select an all-valid 224×224 window inside that prepared scene:

```sh
aureon-prepare generation --manifest "$HOME/aureon-work/prepared-first/manifest.json" \
  --row0 0 --col0 0 --output "$HOME/aureon-work/generation-input-first"
```

This CPU command preserves the existing grid/radiometry and never fills invalid
pixels. It returns a binding to `generation-input-first/manifest.json`.

## First TerraMind result

For a whole scene, keep the prepared rasters and inspect admission first:

```sh
aureon-engine scene-plan --model terramind-v1-base \
  --image "$HOME/aureon-work/prepared-first/scene.tif" \
  --valid-mask "$HOME/aureon-work/prepared-first/valid_mask.tif" \
  --output "$HOME/aureon-work/scene-result-first" --job-id scene-first
```

Planning executes no model and creates no result. It checks input/grid/mask
and declared memory/storage budgets. Follow [Best workflows](BEST-WORKFLOWS.md)
to execute a whole scene or retain one resident for repeated jobs. Do not crop
your whole scene to 224 merely to fit the single-tile example below; only the
configuration seed manifest needs a valid native tile.

Use the generation environment. Config and job paths must be new; create parents.

```sh
mkdir -p "$HOME/aureon-config"
aureon-engine generation-config \
  --model-file /absolute/assets/TerraMind_v1_base.pt \
  --tokenizer-file /absolute/assets/TerraMind_Tokenizer_S1GRD.pt \
  --manifest "$HOME/aureon-work/generation-input-first/manifest.json" \
  --output "$HOME/aureon-config/generation.json" --token-steps 8 --seed 17
aureon-engine check --environment generation --config "$HOME/aureon-config/generation.json"
aureon-engine generate --model terramind-v1-base \
  --config "$HOME/aureon-config/generation.json" --output "$HOME/aureon-work/generation-first" \
  --job-id first --attempt-id one --reuse auto --denoiser graph --execute-model
```

Read the returned JSON `job` record for the COG/provenance locations and retain the
`cleanup` receipt. Output is synthetic VV/VH, not observed SAR or a class map.
The recipe is all-valid224, 8 token/10 diffusion steps, highest FP32. Seed/state
isolation, exact eligible encoder reuse and graph guards remain unchanged.
`--execution-profile reference` explicitly selects diagnostic native/off execution. Config creation
binds the actual installed upstream source and dependency versions; regenerate
the config when relocating its environment/assets.

## First KAO-TerraMind B result

Use the same current wheel with its optional `[scene-legacy]` dependencies in a
compatible environment and the bundle prepared above. This full-reference route
does not require a guarded qualification receipt. Its earlier GPU evidence
remains artifact-specific; the current base generation acceptance does not
requalify every KAO-TerraMind route. Do not mix original-release scene pins into
the generation environment. See [dependency contracts](LOCKED-INSTALL.md).

```sh
aureon-engine run --model kao-terramind-b --environment scene-legacy \
  --profile fp32-full-reference-v1 --manifest "$HOME/aureon-work/prepared-first/manifest.json" \
  --bundle "$HOME/aureon-assets/kao-terramind-b" \
  --output "$HOME/aureon-work/b-first" --execute-model
```

The JSON result identifies `class_map`, `valid_mask`, `provenance`, `job_receipt`
and `receipt`. Keep them together. Classes are 0–9, nodata 255; they are not
calibrated probabilities. The facade returns after its owned worker closes.

Continue with [Python/resident use](PUBLIC-API.md) or [authenticated local
serving](ASYNC-SERVICE.md). [Support](SUPPORT.md) explains common input/install
errors. [Model support](MODEL-SUPPORT.md) distinguishes implemented interfaces
from exact-artifact acceptance. New inputs and hardware do not inherit scientific
accuracy or performance claims from previous measured runs.

See [resident scene HTTP, feature schemas and SDK dispatch](RESIDENT-WORKFLOWS.md)
for the current source candidate, including process ownership, finite budgets,
large artifact transport and separate source/installed/hardware qualification.
