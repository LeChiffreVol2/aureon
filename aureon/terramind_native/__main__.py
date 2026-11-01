from __future__ import annotations
from collections.abc import Sequence
import argparse
import json


def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect local TerraMind inputs and configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-manifest", required=True); prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--row0", type=int, default=0); prepare_parser.add_argument("--col0", type=int, default=0)
    check = commands.add_parser("check"); check.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .geo import prepare
        result = prepare(args.source_manifest,args.output,row0=args.row0,col0=args.col0)
    elif args.command == "check":
        from .loader import verify_config
        config,paths,recipe = verify_config(args.config)
        result = {"bindings_verified":True, "model_constructed":False, "recipe":recipe}
    print(json.dumps(result,indent=2))
    return 1 if result.get("status","").startswith("failed") else 0


if __name__ == "__main__": raise SystemExit(main())
