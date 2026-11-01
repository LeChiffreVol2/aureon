"""Local checkpoint preparation and inspection commands."""

from __future__ import annotations
from collections.abc import Sequence
import argparse
import json
from .catalog import get_catalog, plan_inference


def main(argv: Sequence[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='AUREON — KAO-TerraMind inference engine')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('catalog', help='Print pinned model and recipe metadata')
    plan = commands.add_parser('plan', help='Print preparation requirements without probing a service')
    plan.add_argument('--model', required=True)
    plan.add_argument('--backend', choices=('pytorch',), help='Default: PyTorch for KAO-TerraMind B')
    prepare = commands.add_parser('prepare-bundle', help='Convert a supplied pinned local checkpoint; never downloads')
    prepare.add_argument('--model', required=True)
    prepare.add_argument('--checkpoint', required=True)
    prepare.add_argument('--output', required=True, help='A new directory under an existing parent')
    verify = commands.add_parser('verify-bundle', help='Verify local bundle files without loading a model')
    verify.add_argument('--bundle', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'catalog':
            result = get_catalog()
        elif args.command == 'plan':
            result = plan_inference(args.model, **({'backend': args.backend} if args.backend is not None else {}))
        elif args.command == 'verify-bundle':
            from .runtime import verify_bundle
            result = verify_bundle(args.bundle)
        else:
            from .checkpoint_bundle import prepare_checkpoint_bundle
            model = next((m for m in get_catalog()['models'] if m['id'] == args.model), None)
            if model is None:
                raise ValueError('Select a pinned KAO-TerraMind model ID.')
            result = prepare_checkpoint_bundle(args.checkpoint, args.output, model_id=args.model,
                                        expected_sha256=model['identity']['published_checkpoint_sha256'])
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

