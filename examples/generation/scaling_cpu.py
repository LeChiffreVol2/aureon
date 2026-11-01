"""Offline stage example. Creates no model, downloads, socket or background service."""
from __future__ import annotations
import json
from aureon.scaling import Pipeline, Stage, StageContext


def prepare(value: int, context: StageContext) -> int:
    context.check()
    return value + context.seed


def infer(value: int, context: StageContext) -> int:
    context.check()
    return value * 2


def main() -> None:
    with Pipeline([Stage("prepare", prepare, workers=2), Stage("infer", infer)],
                  max_requests=4, max_live_bytes=4096) as pipeline:
        pending = [pipeline.submit("job-" + str(index), 10, tenant="example",
                                    seed=index, reservation_bytes=1024) for index in range(4)]
        results = [handle.result() for handle in pending]
        assert results == [20, 22, 24, 26]
        print(json.dumps({"results": results, "metrics": pipeline.metrics(),
                          "scope": "CPU control flow, no model or GPU performance claim"}))


if __name__ == "__main__":
    main()
