# Selected TerraMind runtime source

These AUREON-owned files are licensed under Apache-2.0, as recorded in the
distribution's LICENSE and NOTICE. TerraMind and TerraTorch are upstream
dependencies with their own attribution and terms; weights are not bundled.
The original `upstream-source.json` records the separately installed source
checked by the loader. It is not an assertion that those sources are bundled.

`SOURCE.json` binds this closure to the exact selected v0.1.0 wheel and matching
source archive. Eleven Python files and the upstream source manifest are
verbatim copies. `harness.py` is only the original prefix containing imports,
`capture_failure` and `trace_native`; its original hash, extracted hash, source
span and helper AST hashes are recorded. The experiment/comparison drivers are
excluded. Historical headers and scope text remain unchanged for source identity.

Relative imports keep the numerical path intact after relocation. Source paths
and the inventory written through `__file__` change. The public session facade
holds a shared process lifecycle lock outside this frozen code. A relocation is
not a new performance measurement or a replacement of the original artifact.

Reproduce the payloads using `packaging/freeze_best_core.py` with the exact wheel
and source archive. Do not edit these numerical files to add features or typing;
new typed delivery code belongs outside this namespace.
