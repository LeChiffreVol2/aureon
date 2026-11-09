# AUREON v0.1.0: one installed Engine

The main v0.1.0 package integrates features directly with the Best runtime
derived from `0fa5ef2b…`. Its original run, owner thread, FP32 sampling, graph/reuse
and cleanup are retained. Scenes, SDK/CLI, local HTTP and private Cloud consume
that installed package; no model subprocess adapter or second wheel is required.

Adding features changes the archive hash. The delivery receipt identifies the
integrated wheel; `0fa5ef2b…` remains the immutable numerical/performance origin,
not a second installation choice. Historical failed builds remain evidence.
Best is available from GitHub Release v0.1.0. Historical artifacts and comparison
numbers are preserved; hosted inference and PyPI remain closed.

The locked local wheel SHA256 is
`e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64`.
Verify this identity before installation. The current [GitHub release](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0)
provides these exact Best bytes. Original alpha assets remain under historical
names; the Git tag and automatic source snapshots retain their original identity.

## Guided setup

From a current public source checkout, one command creates an isolated installation,
checks exact payloads and exercises CPU preprocessing:

```sh
python3.12 tools/install.py --execute --prefix "$HOME/aureon-env"
```

Use `--plan` first for a read-only preview. See [first-use instructions](../docs/FIRST-USE.md)
for offline archives, logs and feedback. Weights and real inference remain explicit.
The tool is a source companion; it is not added to the frozen wheel or source archive.

## Install the reviewed archive

Use a fresh Linux x86_64 / WSL2 Python 3.12 environment. For normal dependency
resolution, install the hash-pinned wheel:

```sh
python -m pip install "https://github.com/LeChiffreVol2/aureon/releases/download/v0.1.0/aureon_inference-0.1.0-py3-none-any.whl#sha256=e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64"
```

For the tested Torch 2.10 / TerraTorch 1.2.13 generation snapshot, download the
wheel and source archive, verify their hashes below, and extract the source.
Use its locks from a new environment:

```sh
tar -xzf aureon_inference-0.1.0.tar.gz
python3.12 -m venv "$HOME/aureon-best-env"
. "$HOME/aureon-best-env/bin/activate"
python -m pip install --require-hashes -r aureon_inference-0.1.0/packaging/lock-bootstrap-generation.txt
python -m pip install --require-hashes --no-build-isolation -r aureon_inference-0.1.0/packaging/lock-generation-linux-py312-cu128.txt
python -m pip install --no-index --no-deps ./aureon_inference-0.1.0-py3-none-any.whl
python -m pip check
aureon-engine check --environment generation
```

Follow the current quickstart for pinned model assets, input preparation and
explicit CLI/SDK/local HTTP execution. Installation does not acquire model
weights or imagery and does not run inference. KAO-TerraMind full-reference
routes use the documented optional scene-legacy dependencies and their own scope.

The original alpha and Best both have version 0.1.0. A version-only upgrade may
retain the old wheel, so use a fresh environment and verify SHA-256. Do not mix
the original scene/cu130 stack with this generation environment. The source
archive is the frozen matching build input; its earlier unpublished-status text
is superseded by this release note only. Rebuilding source creates a distinct
artifact. No developer/private kit is published in this update.

## Direct resident Best

```python
from aureon.api import GenerationSession

with GenerationSession(
    "/absolute/path/config.json", "/absolute/path/fresh-session",
    execute_model=True,
) as session:
    result = session.run(
        "/absolute/path/input-manifest.json", job_id="scene-1", attempt_id="1",
        recipe=dict(token_steps=8, diffusion_steps=10, temperature=1.0,
                    top_p=0.8, top_k=0, seed=17),
    )
    print(result["cog"], result["provenance"])
```

`GenerationSession.run` and its lifecycle are the original implementation,
inherited directly. Recommended construction selects graph/reuse once. The
`reference_fp32` profile is explicit diagnostics, not a second package.

Use [whole scenes](BEST-WORKFLOWS.md) for bounded raster tiling/reconstruction and
[local HTTP](ASYNC-SERVICE.md) for upload, submit, status and verified download.
The same installation supplies all these interfaces. Private Cloud orchestration
imports the installed Engine; provider/customer code is excluded from its wheel.

In-memory/model-family interfaces remain available in the same package with their
declared contracts. Base file-route performance does not qualify other sizes or
memory routes. Exact FP32 output is separate from physical SAR accuracy, seam
quality, sustained load and multi-GPU scaling.
