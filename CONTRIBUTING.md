# Contributing to AUREON

Use Python 3.12. For a source change, work in your own branch or fork and keep
runtime changes separate from dependency, numerical-recipe and documentation changes.
Start with the [public API](packaging/PUBLIC-API.md) and [support matrix](docs/MODEL-SUPPORT.md).

Run the standard-library checks from the repository root:

```sh
python3.12 -I -B tools/check_public_source.py
python3.12 -B packaging/ci_cpu.py --output /tmp/aureon-cpu-result.json
```

Use a fresh receipt filename on each CPU test run. These checks do not install
dependencies, load models, download weights, use a GPU or run a benchmark.
Native geospatial tests require the matching locked profile. GPU/model execution
must be explicitly planned and distinguished from CPU-only validation.

Describe the user-visible change, focused tests and limitations in the pull
request. Preserve provenance, masks/grids, FP32 semantics, state/seed isolation,
resource ownership and graph/reuse guards. Do not rewrite previous evidence or
claim numerical/performance gains from synthetic tests. Review the explicit
runtime/support allowlists when adding package files.

Do not commit weights, imagery, credentials, environments, local outputs, caches
or generated build metadata. Avoid private absolute paths in docs and tests.
Use synthetic fixtures with no scientific-quality claims for small CPU tests.

Contributions are reviewed under the accompanying Apache-2.0 terms; identify
third-party sources and preserve their notices. Public issues are appropriate
for sanitized bugs and feature requests. Report security vulnerabilities privately
using [SECURITY.md](SECURITY.md). Contact: **norviccompanylimited@gmail.com**.
