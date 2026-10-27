# Try the input pipeline without a model

[First generation](../packaging/START-HERE.md) · [Input contract](../packaging/INPUTS.md) · [Documentation](README.md)

Use this optional walkthrough after installing the **generation profile** to
check the installed preparation pipeline before obtaining imagery or weights.
It requires NumPy, Rasterio and the installed AUREON package, but no GPU or model files.

The example creates **artificial pixels**, a corresponding mask and an editable
input specification. It runs the same scene preparation, manifest check and
224 × 224 generation-crop commands used in the real workflow. It then verifies
pixel values, an asymmetric crop, its geospatial transform and validity mask.
Neural-runtime imports are explicitly blocked.

## Run the example

Save [cpu_walkthrough.py](../examples/generation/cpu_walkthrough.py) from this
repository to a local file using GitHub's **Download raw file** button. The example
was added after the immutable v0.1.0 kit; it is not inside that ZIP. You can also
use the file from a clone of this repository. No source install is needed.

Activate the environment from the installation guide, then run the downloaded file:

```sh
source "$HOME/aureon-generation-0.1.0/bin/activate"
python -I -B /absolute/downloads/cpu_walkthrough.py \
  --output "$HOME/aureon-cpu-walkthrough"
```

Replace the script path with where you saved it. The output directory must not
already exist. All files stay there; the example does not download anything.

## Expected result

The command exits with code 0 and prints JSON including:

```json
{
  "status": "passed",
  "input": "artificial CPU fixture, not satellite imagery",
  "prepared_shape": [12, 224, 224],
  "all_pixels_valid": true,
  "pixels_and_grid_verified": true,
  "model_executed": false,
  "benchmark_run": false
}
```

The full result also includes the generated manifest path. Open
`walkthrough-result.json` in your chosen output directory to inspect it later.
The example leaves the artificial source GeoTIFFs, `aligned-input.json`,
`prepared/` and `generation-input/` for inspection. Running again with the same
output path must fail without overwriting them; choose another fresh directory.

This confirms installed CPU preparation behavior. It does not test the GPU,
load TerraMind, generate S1GRD, or demonstrate model accuracy or speed. For your
first meaningful generation, return to the [first-result guide](../packaging/START-HERE.md),
use your actual Sentinel-2 files and metadata, and supply the pinned model assets.
Do not reuse the fixture's all-valid mask policy for a real scene.
