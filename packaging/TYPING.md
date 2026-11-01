# Python contracts and CPU validation

The current source ships a PEP 561 `py.typed` marker, facade reexports and stubs
for the existing scene/client interfaces. Generation requests, result arrays,
handles, streams, recipes, execution profiles and queue metadata have explicit
types. This is a source addition to version 0.1.0; the original release archives
are unchanged.

`GenerationResult` requires a float32 NumPy `array`, a boolean `valid_mask` and
`provenance`. Receipt fields are optional. Array dimensions, band order, units,
validity and model eligibility remain runtime checks. NumPy/CPU-Torch inputs
and extensible provenance/diagnostic mappings deliberately retain `Any` where
the runtime accepts more than one representation. Types do not certify physical
accuracy, numerical parity or a hardware profile.

Two independent type gates protect the library. `check_types.py` validates
positive and deliberately invalid consumer examples against the shipped facade.
`check_implementation.py` copies every `.py` in the exact runtime allowlist into
fresh scratch and omits all `.pyi` files. Every owned implementation is an
explicit strict-mypy target, including the selected-core facade and integrity
verifier. The twelve frozen numerical files declared in
`aureon/_reference/terramind/SOURCE.json` are the sole source exception: the pinned
manifest, exact file list and every payload hash must verify before exclusion.
The helper prefix is recorded as an excerpt, not a whole-file copy. A missing,
changed or additional Python file fails the check. Receipts distinguish strict
targets from verified frozen files and never claim all shipped bodies are typed.
Each module receives a source hash, signature coverage, explicit `Any` count and
diagnostics. Any remaining diagnostic fails the gate; there are no
`ignore_errors` sections.

Torch and NumPy use their actual installed definitions. Optional untyped
TerraTorch, legacy protocol, Rasterio/Affine and JSON/base64 providers have
explicit external import boundaries. Runtime objects cross those boundaries
through guarded tensor/array types, structural protocols, or documented dynamic
metadata. Serialization metadata and versioned upstream extension points remain
open; this does not turn the numerical implementation into an unchecked module.

The zero-argument upstream query `torch.cuda.is_initialized()` has no return
annotation in the installed Torch provider. The implementation checker records
an exact `untyped_calls_exclude` entry for that function. It does not exclude
`torch.cuda` or any AUREON function, suppress owned-module diagnostics, or alter
runtime code. Its initialization behavior is also exercised by the installed
CPU and real-device lifecycle checks.

Torch **2.10.0** also declares `torch.cuda.streams.Event.record(self, stream=None)`
without annotations; 2.11 annotates its stream argument. For 2.10.0 only (including
local CPU/CUDA build suffixes), the checker records that exact external method in
`untyped_calls_exclude`. Its AUREON call sites still use typed `Event`/`Stream`
values and remain strict targets. No AUREON function, whole provider module or
other diagnostic is excluded. The receipt lists the installed provider version
and precise boundaries; failures also print their full diagnostics in CI logs.

Historical comparison tools and their preservation gates belong to a separate
project. Public CI checks the reusable library and does not fetch historical
competitor Git objects. [Distribution boundaries](LIBRARY-SCOPE.md) prevent
private tools from entering the installed runtime.

The original qualified source hashes remain recorded. The annotation-only
candidate ledger additionally checks executable AST, constants and configuration
identity; it does not qualify a rebuilt wheel on GPU or replace prior evidence.

## Run with existing dependencies

The `tensor` suite uses CPU Torch and Rasterio for scheduling, session ownership
and synthetic COG/grid round trips. Model-provider imports remain blocked.
The upstream scheduler integration is separately available in the pinned
generation environment with `python -B -m unittest discover -s tests -p test_terramind_upstream.py`.
It uses a tiny CPU decoder and no pretrained model or GPU; it is not part of the
dependency-light CI job. Repository tests are source-only, never wheel payloads.

Use Python 3.12 with NumPy 2.2.6, mypy 1.11.2 and Torch 2.11.0 type definitions
already installed. Local CUDA build suffixes are recorded; the checker never
loads a model or probes a GPU. The runners
never install or download dependencies. Every scratch directory and receipt
must be new. Run from the source root, substituting your owned temporary paths:

```sh
python3.12 -I -B packaging/check_types.py --scratch /tmp/aureon-types-new --output /tmp/aureon-types-new.json
python3.12 -I -B packaging/check_implementation.py --scratch /tmp/aureon-implementation-new --output /tmp/aureon-implementation-new.json
python3.12 -I -B packaging/ci_extended.py numpy --scratch /tmp/aureon-numpy-new --output /tmp/aureon-numpy-new.json
python3.12 -I -B packaging/ci_extended.py tensor --scratch /tmp/aureon-tensor-new --output /tmp/aureon-tensor-new.json
python3.12 -I -B packaging/ci_extended.py service --scratch /tmp/aureon-service-new --output /tmp/aureon-service-new.json
```

The service process suite needs Linux and a Linux-local writable scratch
directory. A Windows-mounted filesystem does not meet its ownership checks.
Missing dependencies or unsupported platforms fail explicitly; a skipped test
does not pass these gates. NumPy contracts use CPU doubles with neural imports
blocked. Service tests launch bounded fake workers, not pretrained models.

The type runner checks positive examples and requires every intentionally
invalid example to fail at its expected line and diagnostic code. The package
contracts separately compare stub exports and signatures with the runtime.

## Check an installed wheel

An offline, noneditable wheel installation can be checked using `--runtime`
with its installation prefix and `--source` with the matching extracted sdist.
Run from outside both source and installation directories:

```sh
python3.12 -I -B /source/packaging/check_types.py --source /source --runtime /installed-prefix --scratch /tmp/aureon-installed-types-new --output /tmp/aureon-installed-types-new.json
python3.12 -I -B /source/packaging/check_implementation.py --source /source --runtime /installed-prefix --scratch /tmp/aureon-installed-implementation-new --output /tmp/aureon-installed-implementation-new.json
python3.12 -I -B /source/packaging/ci_extended.py numpy --source /source --runtime /installed-prefix --scratch /tmp/aureon-installed-numpy-new --output /tmp/aureon-installed-numpy-new.json
python3.12 -I -B /source/packaging/ci_extended.py service --source /source --runtime /installed-prefix --scratch /tmp/aureon-installed-service-new --output /tmp/aureon-installed-service-new.json
```

The runtime runner records every imported AUREON module and rejects imports
outside the selected prefix. Test sources and installed runtime are separate.
The type runner selects that same prefix as its module search root.

## CI scope

The hosted Linux workflow runs source checks, standard-library package contracts
and service process contracts. A separate hosted job creates a disposable Python
3.12 environment containing pinned setuptools 84.2.1, mypy 1.11.2, NumPy 2.2.6,
Rasterio 1.5.2 and CPU Torch 2.11.0 for implementation typing and tensor/grid tests.
It checks source consumers, all implementation bodies and NumPy/tensor contracts, builds the wheel and sdist, then
installs only that wheel with `--no-index --no-deps`. Imports, CLI help, types and
CPU/process contracts run against the installed prefix and the extracted sdist
from an unrelated working directory. CPU Torch is an isolated validation dependency, not an added default library
dependency. No model download/execution, GPU use, benchmark or publication is
part of CI. The local runners
continue to require existing dependencies; they never install anything themselves.
