# AUREON v0.1.0: one installed Engine

The main v0.1.0 package integrates features directly with the Best runtime
derived from `0fa5ef2b…`. Its original run, owner thread, FP32 sampling, graph/reuse
and cleanup are retained. Scenes, SDK/CLI, local HTTP and private Cloud consume
that installed package; no model subprocess adapter or second wheel is required.

Adding features changes the archive hash. The delivery receipt identifies the
integrated wheel; `0fa5ef2b…` remains the immutable numerical/performance origin,
not a second installation choice. Historical failed builds remain evidence.
Published GitHub artifacts and historical comparison numbers remain unchanged.
PyPI, hosted activation, deployment and publication remain closed.

The locked local wheel SHA256 is
`e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64`.
Verify this identity before installation. The published GitHub alpha is a
separate retained release; this local selection does not replace its downloads.

## Install the reviewed archive

Use Linux/WSL, Python 3.12 and the pinned compatible generation environment.
The local delivery receipt and SHA256SUMS identify the one current wheel. In an
already prepared environment without another AUREON build:

```sh
set -eu
cd /absolute/path/to/selected-delivery
sha256sum --check SHA256SUMS
python -I -c "from importlib.metadata import distributions; import sys; sys.exit('Use an environment without another AUREON build') if any((d.metadata.get('Name', '').lower().replace('_', '-') == 'aureon-inference') for d in distributions()) else None"
python -m pip install --no-index --no-deps ./aureon_inference-0.1.0-py3-none-any.whl
cd /absolute/path/outside/source-checkouts
python -I -m aureon.api --help
python -I -m aureon.serving --help
```

The package declares generation dependencies. Use the pinned constraints when
preparing a new environment; no dependency installation or weight download is
implicit in the SDK. No PyPI publication is claimed.

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
