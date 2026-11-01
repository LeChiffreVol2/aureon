# Generation execution and concurrency

Use the existing resident engine with `execution_profile="recommended"` for
eligible encoder reuse and decoder CUDA Graph execution. Use `"reference"`
for explicit reference execution. Keep checkpoint/input/recipe/seed and precision
identities when comparing results.

The unpublished tensor-batching experiment has been removed from the current
source: `NeuralBatchConfig`, `NeuralBatchSession` and the
`GenerationEngine(neural_batch=...)` option are no longer available. Old examples
using that option must omit it; unsupported keywords raise before any model
loading or workspace creation. This change does not rewrite existing wheels.
The earlier directly executable experimental fused-pair session is also retired;
shared pair lifecycle remains private to concrete lane implementations.

Request queues, bounded input/result ownership, configurable independent graph
lanes and replica scheduling remain available. They do not fuse requests into
one neural batch or promise a throughput increase on every GPU. Follow
[concurrent generation](CONCURRENT-GENERATION.md),
[resource policies](SCALABLE-POLICY.md) and
[replica/state/delivery contracts](SCALABLE-RUNTIME.md).

The rejected timestep-scalar cache is not part of this runtime. Recommended
generation, exact FP32 references, RNG restoration and geospatial delivery keep
their existing contracts. Shared native preparation/ownership helpers remain
because the independent lane implementation uses them; their historical file
names are not new execution profiles or performance claims.

Version remains v0.1.0. Current source, immutable candidate artifacts and the
published release have separate identities and qualification scopes. Inspect the
[candidate guide](CANDIDATE-QUICKSTART.md) before choosing an artifact.
