# Explicit COP-GEN FP32 dependency profiles

This experimental adapter is included in the unpublished AUREON v0.1.0 candidate.
It consumes shared EORequest types through its own plan_copgen entry point; it is
not registered as a universal route in plan_eo. The existing published alpha does
not include this candidate. Source-bound GPU acceptance and installed CPU tests
are distinct; the new wheel has not received a new installed GPU acceptance.

## Install the optional Torch2.10 profile

Use the candidate wheel and its matching extracted source archive. From the
extracted source directory, in a dedicated Linux/WSL Python3.12 environment:

```bash
python -m pip install --constraint aureon/copgen_native/requirements-torch210.txt \
  "/absolute/path/to/aureon_inference-0.1.0-py3-none-any.whl[copgen]"
python -m pip check
```

The extra adds ml-collections1.0.0 and lmdb1.6.2. Mandatory TerraMind dependency
pins remain unchanged. The constraints bind the tested direct tuple, including
transitive packages used by this adapter; they are not a full hashed lock.
No weights or upstream code are bundled or acquired by package installation.
Supply separately verified assets and explicitly select
`dependency_profile="torch210"` when constructing CopgenEngine. The required CUDA
build is checked before model ownership. Installation alone does not run a model.

## Profile and execution contracts

Two exact profiles are admitted on Linux/WSL Python 3.12. Select the profile
explicitly; no version auto-detection or fallback changes the recipe. The
constructor retains `dependency_profile="torch260"` as its legacy default.

The newer `dependency_profile="torch210"` uses the normal TerraMind package's
Torch 2.10.0, torchvision 0.25.0, NumPy 2.2.6, Rasterio 1.5.2 and einops 0.8.2.
The tested Torch module is 2.10.0+cu128 (CUDA 12.8). Its other direct dependencies
are ml-collections 1.0.0, lmdb 1.6.2, absl-py 2.5.1, Pillow 13.0.0, tqdm 4.70.1
and PyYAML 6.0.3. `requirements-torch210.txt` supplies exact direct constraints;
`profiles.py` rejects mismatches before any CUDA ownership or cleanup.
Only ml-collections and lmdb were missing from the existing default environment.
The compatibility acceptance installed them in a separate 1.4 MB overlay using
normal pip dependency resolution, with no `--no-deps`, base package replacement
or retained-environment mutation. `pip check` passed for the combined environment.
The candidate `[copgen]` extra contains these two additional requirements:
`ml-collections==1.0.0` and `lmdb==1.6.2`. Use the profile constraints as well;
unconstrained transitive upgrades are not qualified by the acceptance.

The legacy `torch260` profile needs a separate source environment. It is not
compatible with the normal wheel dependency metadata; do not treat a no-deps
wheel install as a clean legacy-profile installation. Its acceptance used Python
3.12.3, Torch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 1.26.4, einops 0.8.1,
ml-collections 1.0.0, Rasterio 1.4.3, lmdb 1.6.2, absl-py 2.1.0, Pillow 11.1.0
and tqdm 4.67.1. `requirements.txt` lists direct dependencies; it is not a hashed
transitive lock. The original source acceptance retains its complete observed environment separately. PyYAML is pinned to 6.0.3. Do not combine this profile with the newer
TerraMind dependency profile or upgrade an existing environment to satisfy both.
Torch's CUDA12.4 wheel index is documented at
https://pytorch.org/get-started/previous-versions/#v260.

The caller must separately provision all source files and the three EMA assets
named by `pins.json`, preserving their directory names and hashes. The source
root must not contain additional Python files. Config execution is limited to
the exact reviewed local config; arbitrary caller configs and remote imports
are not supported. Initialization enforces the selected exact dependency/build tuple, one
visible GPU and an empty process allocator. All model work and close occur on
the loading thread. The caller owns that process; this is not a multi-tenant
sandbox or a borrowed-model interface.

Global CUDA cleanup is allowed only after successful profile, device and empty
allocator admission. Rejected construction leaves the caller's CUDA context
untouched and reports cleanup as not owned/not measured. Ownership is consumed
once on close, including a cleanup failure, so a later close cannot clean a
subsequent owner's context.

The route is raw `surface_reflectance_x10000`, B04/B03/B02/B08, a finite FP32
`(4,192,192)` array and a separate all-valid bool mask, to synthetic S1RTC VV/VH
linear backscatter on the same explicitly supplied grid. The adapter validates
CRS and bounded plain source IDs before model work. There is no implicit band
reorder, crop, resampling, mask inference or reflectance conversion. The caller
must prepare and document those operations. Only the retained Thailand crop was
qualified; physical/scientific quality and other acquisitions are not established.

Example with already prepared arrays and caller-owned paths:

```python
from aureon.api.eo import (
    EOGrid, EORequest, ConditioningSpec, GeneratedTarget, SamplingRecipe,
)
from aureon.copgen_native import CopgenInput, CopgenEngine, export_copgen

grid = EOGrid("EPSG:32647", (10., 0., 632420., 0., -10., 1583670.), (192, 192))
bands = ("B04", "B03", "B02", "B08")
data = CopgenInput(image_fp32, valid_mask=valid_mask_bool, bands=bands,
                   units="surface_reflectance_x10000", grid=grid,
                   source_id="explicit-observation-id")
request = EORequest(
    "copgen-base",
    (ConditioningSpec("S2L2A", bands, "surface_reflectance_x10000", "none",
                      "explicit_all_valid", grid, "explicit-observation-id"),),
    (GeneratedTarget("S1RTC", ("VV", "VH"), "linear_backscatter", grid),),
    SamplingRecipe("copgen-latent-s1rtc-fp32-v1", 17, (("sample_steps", 50),)),
    "reference",
)
with CopgenEngine(source=pinned_source_root, assets=pinned_asset_root,
                 execute_model=True, dependency_profile="torch210") as engine:
    result = engine.generate(request, data)
    receipt = export_copgen(result, new_output_directory)
```

Only 50 native solver evaluations and uint32 seeds are admitted. Both reference
and adapter use two resident VAEs, all twelve joint transformer modalities, the
native stochastic VAE posterior and omitted-modality noise draws. A module-local
Torch view explicitly disables upstream inner autocast. Highest FP32, TF32 off
and math SDPA define this compatibility profile; the upstream log label `flash`
names its SDPA code branch, not the selected PyTorch backend. Request RNG and
precision policy are restored. This is not parity against unmodified autocast
or a full-ten-VAE-resident native constructor.

The result owns CPU arrays and provenance. `export_copgen` writes into a new
directory and independently verifies COG values, grid, mask and tags; it refuses
modified arrays or an existing directory. Mask values describe generated coverage,
not real observation validity. A failed generation/close is terminal. Cleanup
failures remain visible and do not replace a primary model error or strand process
ownership; zero allocator usage is reported only when actually measured.

The original six-call Torch2.6 acceptance and later CPU ownership-review receipts
remain frozen against their own source hashes. A separate two-call Torch2.10
acceptance (native reference then adapter, seed17) qualifies the newer source
snapshot for this route and profile: exact latent/decoded/linear output/mask/RNG,
both COG read-backs and measured CUDA close at zero allocated/reserved bytes.
The evidence remains bound to the reviewed source snapshot and exact dependency tuple.
The new acceptance does not establish parity across Torch versions, multiple
seeds, repeat/reload behavior, physical quality, speed improvement or long-run
leak freedom. CPU injected-failure tests cover ownership and error cleanup;
those failures were not injected on GPU. Both acceptance budgets are closed.
This is internal source qualification, not installed public-wheel acceptance.
