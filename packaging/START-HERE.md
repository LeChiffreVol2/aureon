# Install AUREON and generate your first Earth observation result

This guide starts with **TerraMind 1.0 base**: prepared Sentinel-2 L2A → synthetic
S1GRD VV/VH. It takes you from the v0.1.0 release kit to an explicit local model job.
For segmentation, use the separate [KAO-TerraMind B route](#first-kao-terramind-b-result).

## Before you start

| You need | Why |
| --- | --- |
| Linux x86_64 or WSL2, Bash, CPython 3.12 with `venv`/pip and `curl` | These are the platform and shell used by the commands below. Run them inside Linux/WSL, not PowerShell. |
| A compatible NVIDIA GPU and driver for model execution | The recorded configuration is RTX 3070 8 GiB / WSL. Other devices are not covered by that acceptance. Generation uses the CUDA 12.8 dependency profile. |
| Disk space for an environment, two weights files and your data/results | TerraMind weights total 2,665,307,952 bytes (about 2.67 GB); CUDA dependencies add several GiB. |
| Your own supported Sentinel-2 GeoTIFF input | The kit supplies editable JSON templates, not satellite imagery. See the exact input requirements in step 3. |

Check `python3.12 --version` and `nvidia-smi` in your Linux terminal. If Python,
venv support or the driver is missing, provision the host first. The package does
not install a GPU driver. CPU preparation and metadata checks work without model
execution; a passing dependency check does not test CUDA.

**No imagery or weights yet?** After installing, try the optional
[CPU-only walkthrough](../docs/CPU-WALKTHROUGH.md). It prepares artificial pixels
and checks the input pipeline without downloading weights or starting a model.

All paths below are in your Linux home directory. Use fresh output directories;
commands deliberately refuse to overwrite prior results. Keep the original
input and model files in place because manifests bind their identities.

## Install

### 1. Download the exact kit

Use a new download directory. These commands verify both the ZIP and its contents:

```sh
mkdir "$HOME/aureon-download"
cd "$HOME/aureon-download"
curl -fL https://github.com/LeChiffreVol2/aureon/releases/download/v0.1.0/aureon-0.1.0-public-alpha.zip \
  -o aureon-0.1.0-public-alpha.zip
printf '%s  %s\n' '3e844b14ba7a4a8138a3386ab2dd928e426ffcbf018f3183acf91a854297db07' \
  'aureon-0.1.0-public-alpha.zip' | sha256sum --check
python3.12 -m zipfile -e aureon-0.1.0-public-alpha.zip .
cd aureon-0.1.0-public-alpha
sha256sum --check SHA256SUMS
export AUREON_KIT="$(pwd)"
```

If you already verified and extracted this kit, enter its directory and set
`AUREON_KIT` there. Stop on any failed checksum. Use the kit, not GitHub's
**Source code (zip)** download or an unverified package-index install.
The repository guide you are reading includes onboarding improvements made after
the immutable kit was released; the wheel and dependency locks are unchanged.

### 2. Install the generation profile

```sh
set -e  # Stop this Bash session on an error; do not continue a failed install.
export AUREON_ENV="$HOME/aureon-generation-0.1.0"
test ! -e "$AUREON_ENV"
python3.12 -m venv "$AUREON_ENV"
"$AUREON_ENV/bin/python" -m pip install --require-hashes \
  -r "$AUREON_KIT/lock-bootstrap-generation.txt"
"$AUREON_ENV/bin/python" -m pip install --require-hashes --no-build-isolation \
  -r "$AUREON_KIT/lock-generation-linux-py312-cu128.txt"
"$AUREON_ENV/bin/python" -m pip install --no-index --no-deps \
  "$AUREON_KIT/aureon_inference-0.1.0-py3-none-any.whl"
"$AUREON_ENV/bin/python" -m pip check
cd /tmp
"$AUREON_ENV/bin/python" -I -B "$AUREON_KIT/verify_delivery.py" \
  --wheel "$AUREON_KIT/aureon_inference-0.1.0-py3-none-any.whl" \
  --manifest "$AUREON_KIT/ARTIFACTS.json" --environment generation
source "$AUREON_ENV/bin/activate"
aureon-engine models
aureon-engine check --environment generation
```

Use a new environment name if that directory already exists. Do not continue
installing into an existing unrelated environment. A successful `pip check`
reports no broken requirements; the verifier and profile command report
`"status": "passed"`. No step above loads weights or runs inference.
For offline dependency installation, see [locked profiles](LOCKED-INSTALL.md).

In a **new terminal**, reactivate the environment before continuing:

```sh
source "$HOME/aureon-generation-0.1.0/bin/activate"
export AUREON_KIT="$HOME/aureon-download/aureon-0.1.0-public-alpha"
```

## Supply assets

TerraMind requires **both** its base checkpoint and S1GRD tokenizer. Review the
[official pinned sources and model terms](ASSETS.md), then download these exact files:

```sh
mkdir -p "$HOME/aureon-assets"
cd "$HOME/aureon-assets"
curl -fL --retry 3 \
  https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/resolve/fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a/TerraMind_v1_base.pt \
  -o TerraMind_v1_base.pt
curl -fL --retry 3 \
  https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD/resolve/38937fd35012c2dcf81987b400cec725b5eb0466/TerraMind_Tokenizer_S1GRD.pt \
  -o TerraMind_Tokenizer_S1GRD.pt
printf '%s  %s\n' \
  '83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec' 'TerraMind_v1_base.pt' \
  'd83e43ccbdbfb787724d42a4e09dce6abb122e8f286889a379af59ffbe979ed1' 'TerraMind_Tokenizer_S1GRD.pt' \
  | sha256sum --check
```

If you already have matching files, use them at these paths and verify them;
skip the downloads. `aureon-engine assets --model terramind-v1-base` lists identities
and URLs only. It does not download files. TerraMind uses these two files directly
and does **not** need the KAO-TerraMind bundle conversion below.

## Prepare user-owned Sentinel-2 input

### 3. Choose and edit an input specification

| Your data | Kit template | Required preparation |
| --- | --- | --- |
| One aligned twelve-band raster | `examples/aligned-input.json` | FP32 GeoTIFF in reflectance ×10000, plus a uint8 0/1 validity-mask GeoTIFF on the same grid |
| Separate L2A band rasters and SCL | `examples/l2a-input.json` | Twelve band GeoTIFFs and SCL in the declared projected CRS, with actual product radiometric metadata |

Raw SAFE/JP2 folders, RGB previews and arbitrary GeoTIFFs are not accepted directly.
Export supported GeoTIFFs first. The exact band order is
`B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`.
The [input guide](INPUTS.md) explains masks, radiometry and supported grid alignment.

For the aligned route, replace `/absolute/my-data` below with your data directory:

```sh
cp "$AUREON_KIT/examples/aligned-input.json" /absolute/my-data/aligned-input.json
```

Edit that JSON before running: set the actual raster/mask paths, `expected_crs`,
product/acquisition identifiers, mask policy and `window`. Relative paths resolve
beside the JSON file. Do not leave the example CRS, provenance or offsets unchanged
unless they describe your data. `window` is **[column, row, width, height]**; the
512 × 512 template window must fit inside your raster. A 224 × 224 all-valid
window is sufficient for generation. Do not fabricate an all-valid mask for
cloudy or missing pixels.

### 4. Prepare the scene and generation crop

```sh
mkdir -p "$HOME/aureon-work"
aureon-prepare scene --spec /absolute/my-data/aligned-input.json \
  --output "$HOME/aureon-work/prepared-first"
aureon-prepare check --manifest "$HOME/aureon-work/prepared-first/manifest.json"
aureon-prepare generation --manifest "$HOME/aureon-work/prepared-first/manifest.json" \
  --row0 0 --col0 0 --output "$HOME/aureon-work/generation-input-first"
```

For the L2A route, use your edited `l2a-input.json` instead. Preparation is CPU-only.
The last command selects a 224 × 224 crop: `row0`/`col0` are relative to the
**prepared scene**, not the original raster. Choose offsets with valid finite
pixels throughout. A cloudy or out-of-bounds crop fails rather than filling pixels.
Success returns a binding to `generation-input-first/manifest.json`.

## First TerraMind result

### 5. Bind the local files and check the configuration

```sh
mkdir -p "$HOME/aureon-config"
aureon-engine generation-config \
  --model-file "$HOME/aureon-assets/TerraMind_v1_base.pt" \
  --tokenizer-file "$HOME/aureon-assets/TerraMind_Tokenizer_S1GRD.pt" \
  --manifest "$HOME/aureon-work/generation-input-first/manifest.json" \
  --output "$HOME/aureon-config/generation.json" --token-steps 8 --seed 17
aureon-engine check --environment generation \
  --config "$HOME/aureon-config/generation.json"
```

Config creation returns `"status": "created"`; preflight returns `"status": "passed"`
and `"model_loaded": false`. This hashes the local assets and records dependencies;
it does not load the model or probe CUDA. Use a fresh config filename if one exists.
Regenerate the config when relocating the environment/assets or changing its input.

### 6. Execute one generation job

This is the step that loads the model and uses the GPU. If another job owns the
GPU, wait for it to finish. Run **either this CLI job or the Python example** in
[the README](../README.md#a-first-look-at-the-api); each is a real model execution.

```sh
aureon-engine generate --model terramind-v1-base \
  --config "$HOME/aureon-config/generation.json" \
  --output "$HOME/aureon-work/generation-first" \
  --job-id first --attempt-id one --reuse auto --denoiser graph --execute-model \
  > "$HOME/aureon-work/generation-first-result.json"
```

A successful command exits with code 0. If it fails, inspect stderr; an empty
redirected JSON file is not a successful result. `generation-first` must be new.
The recipe uses all-valid224, eight token steps, ten diffusion steps and highest
FP32. Native denoising (`--denoiser native`) and reuse-off (`--reuse off`) remain
explicit alternatives. Do not change precision to work around an environment error.

### 7. Find and inspect the result

```sh
python - <<'PY'
import json
from pathlib import Path
result = json.loads((Path.home() / "aureon-work/generation-first-result.json").read_text())
assert result["job"]["status"] == "completed"
assert result["cleanup"]["passed"]
print("COG:", result["job"]["cog"]["file"]["path"])
print("Provenance:", result["job"]["provenance"]["path"])
print("Raw output:", result["job"]["raw"]["path"])
PY
```

Open the printed COG in your GIS viewer. It has two continuous FP32 bands,
**synthetic S1GRD VV/VH in dB**, and an internal validity mask. Keep the COG,
provenance, raw output and receipts together. It is not a segmentation class map
or an observed SAR measurement. Exact numerical parity is not scientific validation
on your new scene. See [support scope](../docs/MODEL-SUPPORT.md).

## First KAO-TerraMind B result

This is an independent **segmentation** route. Skip it if you only want generation.
Use the same verified kit and input specification, but create a separate scene
environment; do not install its lock into the generation environment.

```sh
set -e  # Stop this Bash session on an error; do not continue a failed install.
export AUREON_ENV="$HOME/aureon-scene-0.1.0"
test ! -e "$AUREON_ENV"
python3.12 -m venv "$AUREON_ENV"
"$AUREON_ENV/bin/python" -m pip install --require-hashes -r "$AUREON_KIT/lock-bootstrap-scene.txt"
"$AUREON_ENV/bin/python" -m pip install --require-hashes --no-build-isolation \
  -r "$AUREON_KIT/lock-scene-linux-py312-cu130.txt"
"$AUREON_ENV/bin/python" -m pip install --no-index --no-deps \
  "$AUREON_KIT/aureon_inference-0.1.0-py3-none-any.whl"
"$AUREON_ENV/bin/python" -m pip check
cd /tmp
"$AUREON_ENV/bin/python" -I -B "$AUREON_KIT/verify_delivery.py" \
  --wheel "$AUREON_KIT/aureon_inference-0.1.0-py3-none-any.whl" \
  --manifest "$AUREON_KIT/ARTIFACTS.json" --environment scene
source "$AUREON_ENV/bin/activate"
aureon-engine check --environment scene
```

Download and hash-check the [pinned KAO-TerraMind B checkpoint](ASSETS.md#kao-terramind-b--checkpoint),
then convert it into a fresh local bundle:

```sh
mkdir -p "$HOME/aureon-assets"
aureon prepare-vllm --model kao-terramind-b \
  --checkpoint /absolute/assets/terramind_v1_kaoB_best.pt \
  --output "$HOME/aureon-assets/kao-terramind-b"
aureon verify-bundle --bundle "$HOME/aureon-assets/kao-terramind-b"
```

This command creates the local bundle used by AUREON Engine. It verifies the
pinned checkpoint identity before CPU weights-only conversion, using CPU memory
and disk. It does not start inference or download files.

Prepare your scene with step 3–4 above (the TerraMind crop is unnecessary here), then:

```sh
aureon-engine run --model kao-terramind-b --environment scene \
  --profile fp32-full-reference-v1 --manifest "$HOME/aureon-work/prepared-first/manifest.json" \
  --bundle "$HOME/aureon-assets/kao-terramind-b" \
  --output "$HOME/aureon-work/b-first" --execute-model
```

Full-reference is the first-user route; it does not require another operator's
guarded qualification. The JSON identifies `class_map`, `valid_mask`, `provenance`,
`job_receipt` and `receipt`. Classes are 0–9, with nodata 255; retain all artifacts.

## Continue

[Python and resident sessions](PUBLIC-API.md) · [Local HTTP service](ASYNC-SERVICE.md) ·
[Troubleshooting](SUPPORT.md) · [Supported models](../docs/MODEL-SUPPORT.md)

The [approved scoped rights statement](../docs/RIGHTS.md) accompanies the release.
Frozen archive review wording is historical; its pending owner-disposition status
is superseded without changing the original archives.
