# Contributing to AUREON

Thanks for helping make Earth foundation models easier to use. A useful contribution
can be a clearer example, a reproducible bug report, an installation fix or a
focused improvement to the Engine.

[Documentation](docs/README.md) · [Architecture](docs/ARCHITECTURE.md) · [Open an issue](https://github.com/LeChiffreVol2/aureon/issues/new/choose)

## Choose a starting point

| Contribution | A helpful first step |
| --- | --- |
| Documentation or examples | Explain which step is confusing and show the corrected workflow. |
| Bug report | Include a minimal reproduction, expected behavior and sanitized environment details. |
| Engine change | Describe the user-visible problem and the affected model, profile or public API. |
| New model or execution profile | Discuss the input/output contract, assets and validation scope in an issue first. |

For suspected vulnerabilities, follow [SECURITY.md](SECURITY.md) and report privately.

## Find your way around

| Path | Responsibility |
| --- | --- |
| [`aureon/api/`](aureon/api) | Stable user imports, requests, results, model discovery and session/scene orchestration |
| [`aureon/planning/`](aureon/planning) | Framework-free placement, resource estimates and admission credits |
| [`aureon/integrations/`](aureon/integrations) | Model/task/checkpoint bridges and explicit preprocessing |
| [`aureon/terramind_native/`](aureon/terramind_native) | TerraMind execution, graph/reuse and model-state ownership |
| [`aureon/scaling/`](aureon/scaling) | Replica execution and transport contracts |
| [`aureon/sources/`](aureon/sources) | Local/STAC and controlled HTTP-range raster input |
| [`aureon/prepare/`](aureon/prepare) | Input-preparation command line |
| [`aureon/scene_pipeline/`](aureon/scene_pipeline) | Scene execution and geospatial streaming |
| [`aureon/serving/`](aureon/serving) | Local service and client |
| [`examples/generation/`](examples/generation) | CPU-only generation-input walkthrough |
| [`examples/packaged_scene/`](examples/packaged_scene) | Input specifications and runnable entry points |
| [`packaging/tests/`](packaging/tests) | Package, API and installed-layout contracts |
| [`packaging/`](packaging) | Dependency profiles, installation guides and build allowlists |
| [`docs/`](docs) | Public documentation and release reference |

Start with the [public API](packaging/PUBLIC-API.md) and
[support matrix](docs/MODEL-SUPPORT.md). Internal and underscore-prefixed modules
are implementation details, not a promise of API stability.

## Make and check a change

This repository contains the reusable library, supported local service, examples,
tests and public documentation. Product operations, customer records, private
research material and competitor experiment environments are maintained separately.
Main inference uses PyTorch/TerraTorch; an optional comparator is not a required
runtime dependency. Shared checkpoint compatibility and third-party attribution
remain where the supported workflows need them.

See [library boundaries](packaging/LIBRARY-SCOPE.md) for what belongs in runtime,
source-only tests and separate private operations. Tests of the reusable library
stay public; private experiment drivers and provider/account tooling do not.

Bring changes from experimental work into this repository as focused source,
test and documentation changes. Follow the existing build allowlists instead of
copying an entire development workspace. Record which exact artifact was tested;
acceptance of an experimental wheel does not qualify a differently rebuilt wheel.
Preserve immutable release tags and archives. Sanitized benchmark methods/results
can be reviewed for public documentation separately from private data and tooling.

Work in your own branch or fork with Python 3.12. Keep changes focused so reviewers
can understand their effects on users, dependencies and numerical behavior.
From the repository root, run:

```sh
python3.12 -I -B tools/check_public_source.py
python3.12 -I -B packaging/check_architecture.py
python3.12 -B packaging/ci_cpu.py --output /tmp/aureon-cpu-result.json
```

Use a fresh receipt filename for each CPU test run. These standard-library checks
do not install dependencies, load weights, use a GPU or run a benchmark.
For documentation-only changes, the static check validates local links, anchors,
HTML image references and the source allowlist. Preview your Markdown as well.

If you add a public documentation or artwork file, update `PUBLIC_FILES` in
`tools/check_public_source.py`. Runtime and packaged support files have separate
allowlists in `packaging/runtime-files.json` and `packaging/build_review.py`.

Native geospatial tests require the appropriate locked profile. Model execution
needs explicitly scoped validation; CPU fixtures do not establish numerical
parity, model quality or performance gains.

## Open a pull request

Explain the problem, the resulting behavior and the checks you ran. Include a
small before/after example or screenshot when it makes a documentation change
clearer. State any untested behavior or support limitation.

Preserve grid and mask semantics, provenance, FP32 reference behavior, state/seed
isolation, resource ownership and graph/reuse guards. Identify dependency or
preprocessing changes explicitly. Preserve existing evidence and artifact identities.

Do not commit weights, imagery, credentials, environments, local outputs, caches
or generated build metadata. Use synthetic fixtures for small CPU tests and keep
private absolute paths out of reports and examples.

## Attribution and community

Contributions are reviewed under the accompanying [Apache-2.0 terms](LICENSE).
Identify third-party sources and preserve their notices. Keep discussions focused
on the work, provide reproducible evidence and give other contributors useful context.

Contact: [norviccompanylimited@gmail.com](mailto:norviccompanylimited@gmail.com).

## Typed consumers and process tests

Follow [Python contracts and CPU validation](packaging/TYPING.md) for strict
consumer checks, bounded NumPy contracts and Linux service process tests.

For a new model adapter, use the [resource and serving guide](packaging/SCALABLE-POLICY.md)
and its CPU conformance runner. Declare lifecycle ownership and exact tested
dependency profiles; distinguish executable capability from model/hardware evidence.
