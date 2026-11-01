"""Two resident TerraMind jobs through common dispatch; explicit offline execution."""
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from aureon.api.best import BestTerraMindSession
import argparse
import json
from aureon.api import open_model_session
from aureon.terramind_native.common import read_json


def main(argv: Sequence[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--denoiser", choices=("native", "graph"))
    parser.add_argument("--execution-profile", choices=("recommended", "reference"))
    parser.add_argument("--execute-model", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_model:
        parser.error("Pass --execute-model explicitly; nothing loaded")
    recipe = read_json(args.config)["recipe"]
    with open_model_session("terramind-v1-base", config=args.config, work=args.work,
                            denoiser=args.denoiser, execution_profile=args.execution_profile,
                            execute_model=True) as session:
        for index in range(2):
            result = cast("BestTerraMindSession", session).run(job_id=f"generation-{index+1}", attempt_id="first",
                                 recipe={**recipe, "seed": recipe["seed"] + index})
            print(json.dumps(result, sort_keys=True))
    print(json.dumps({"cleanup": session.close()}, sort_keys=True))


if __name__ == "__main__":
    main()
