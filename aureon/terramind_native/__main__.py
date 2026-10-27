import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit native TerraMind finite proof; source-only")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-manifest", required=True); prepare.add_argument("--output", required=True)
    prepare.add_argument("--row0", type=int, default=0); prepare.add_argument("--col0", type=int, default=0)
    check = commands.add_parser("check"); check.add_argument("--config", required=True)
    run = commands.add_parser("screen")
    for name in ("config", "screen-spec", "screen-sha256", "output"): run.add_argument("--"+name, required=True)
    run.add_argument("--execute-model", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .geo import prepare
        result = prepare(args.source_manifest,args.output,row0=args.row0,col0=args.col0)
    elif args.command == "check":
        from .loader import verify_config
        config,paths,recipe = verify_config(args.config)
        result = {"bindings_verified":True, "model_constructed":False, "recipe":recipe}
    else:
        if not args.execute_model: parser.error("Model execution requires --execute-model and the designated GPU-owner process")
        from .harness import run_screen
        result = run_screen(args.config,args.screen_spec,args.screen_sha256,args.output)
    print(json.dumps(result,indent=2))
    return 1 if result.get("status","").startswith("failed") else 0


if __name__ == "__main__": raise SystemExit(main())
