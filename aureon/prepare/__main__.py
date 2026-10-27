"""Prepare local Sentinel-2 GeoTIFFs or check a prepared scene without a GPU."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("scene", help="Prepare an explicit aligned or L2A local JSON spec")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True, help="Fresh directory; parent must exist")
    check = commands.add_parser("check", help="CPU-only full input/hash validation; no model loading")
    check.add_argument("--manifest", required=True)
    generation = commands.add_parser("generation", help="Prepare one all-valid 224px TerraMind crop from a scene manifest; CPU only")
    generation.add_argument("--manifest", required=True)
    generation.add_argument("--output", required=True, help="Fresh crop directory")
    generation.add_argument("--row0", type=int, default=0, help="Top row within the prepared scene")
    generation.add_argument("--col0", type=int, default=0, help="Left column within the prepared scene")
    args = parser.parse_args(argv)
    try:
        if args.command == "generation":
            from ..terramind_native.geo import prepare as prepare_generation
            result = prepare_generation(args.manifest, args.output, row0=args.row0, col0=args.col0)
        else:
            from .local import prepare_scene, check_scene
            result = (prepare_scene(args.spec, args.output) if args.command == "scene"
                      else check_scene(args.manifest))
    except ImportError as error:
        parser.exit(2, f"Input preparation needs the selected scene or generation dependency profile: {error}\n")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Input preparation failed: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
