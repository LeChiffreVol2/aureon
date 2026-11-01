# Library source and distribution boundaries

The public repository contains the reusable Engine, Python API, CLI, portable
local service, documentation and contributor tests. Provider deployment,
commercial account/payment operations, customer records, internal experiments,
model assets and research evidence are maintained separately.

The wheel contains only `packaging/runtime-files.json`, generated package
metadata and required license/attribution files. Tests, CPU doubles and build
checks belong in the source distribution and repository; they are not installed
as Engine runtime. They verify API behavior, request state, isolation, recovery,
geospatial delivery and packaging boundaries without requiring private data.

Some reusable service implementation lives under `aureon.cloud` for historical
import compatibility. Only its ten declared local-control/guarded-execution
modules are admitted. This does not include a provider dispatcher, commercial
onboarding, payment handling or private service configuration. Generic local
tenant isolation and usage accounting are legitimate library capabilities.

`packaging/check_library_scope.py` rejects private/test/experiment modules even
when accidentally added to the runtime allowlist. It checks local import
closure, including lazy and type-only imports, without loading model packages.
The build stager applies the path policy; archive checks enforce exact wheel and
sdist contents. The public CI does not depend on a private workspace or historical
competitor Git objects.

The current source removes disconnected prototype modules and the old research
console command `aureon serve`. Use `aureon-service` for local inference serving;
the existing scene API console remains available. TerraMind preparation/check
commands remain, while the private `screen` benchmark driver is not distributed.
Shared checkpoint conversion/validation and required upstream notices remain.

Published v0.1.0 artifacts are unchanged. A source rebuild is a distinct artifact;
CPU tests and unchanged numerical bodies do not confer prior GPU qualification.
See [support](MODEL-SUPPORT.md), [tests](TYPING.md) and [contributing](../CONTRIBUTING.md).
