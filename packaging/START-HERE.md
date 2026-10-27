# Install AUREON v0.1.0 and produce your first result

Use Linux x86_64 (including WSL2), CPython 3.12 and your own local Sentinel-2 input.
The kit contains no weights, imagery, CUDA driver or installed dependencies.
This guide accompanies public alpha v0.1.0. Obtain the exact kit using the
[download and SHA256 quickstart](../README.md#install). The
[scoped rights disposition is approved](../docs/RIGHTS.md); the frozen kit retains
historical pre-decision review wording, superseded only for pending owner status.
First-result model commands below are explicit actions for the user;
installation does not run them.

## Install

Unpack the release kit and enter its directory. Compare the ZIP SHA256 with the
separately supplied release record, then verify its files. The manifest identifies
the exact wheel/sdist; several internal revisions share version 0.1.0.

```sh
sha256sum --check SHA256SUMS
export KIT="$(pwd)"
export VENV="$HOME/aureon-generation-0.1.0"
export PROFILE=generation
export CUDA=cu128
# For KAO-TerraMind B use another fresh VENV, PROFILE=scene and CUDA=cu130.
python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --require-hashes -r "$KIT/lock-bootstrap-$PROFILE.txt"
"$VENV/bin/python" -m pip install --require-hashes --no-build-isolation \
  -r "$KIT/lock-$PROFILE-linux-py312-$CUDA.txt"
"$VENV/bin/python" -m pip install --no-index --no-deps "$KIT/aureon_inference-0.1.0-py3-none-any.whl"
"$VENV/bin/python" -m pip check
cd /tmp
"$VENV/bin/python" -I -B "$KIT/verify_delivery.py" \
  --wheel "$KIT/aureon_inference-0.1.0-py3-none-any.whl" \
  --manifest "$KIT/ARTIFACTS.json" --environment "$PROFILE"
source "$VENV/bin/activate"
aureon-engine models
aureon-engine check --environment "$PROFILE"
```

For offline installation from a complete local archive cache, add
`--no-index --find-links /absolute/dependency-archives` to both dependency installs.
Keep the pins and bootstrap; some dependencies build from source. See
[dependency installation](LOCKED-INSTALL.md). Do not combine profiles or install
all extras. vLLM is not needed for the normal Engine routes below.

## Supply assets

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

Despite its historical name, `prepare-vllm` creates the local bundle also used
by the Engine; it does not start vLLM, execute inference or download anything.
It checks the pinned size/hash before CPU weights-only deserialization and refuses
to overwrite output. It may use CPU memory and disk for conversion. No operator
must create the bundle for you. The receipt establishes identity, not model parity.

## Prepare user-owned Sentinel-2 input

Copy `examples/aligned-input.json` or `examples/l2a-input.json` from the kit to
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
`--denoiser native` and `--reuse off` remain explicit alternatives. Config creation
binds the actual installed upstream source and dependency versions; regenerate
the config when relocating its environment/assets.

## First KAO-TerraMind B result

Use the scene environment and the bundle you created above. Full-reference is
the independent first-user route; it does not require someone else's guarded receipt.

```sh
aureon-engine run --model kao-terramind-b --environment scene \
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
