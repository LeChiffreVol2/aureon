"""Explicit AUREON preparation, inference and benchmark commands."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import uuid

from .catalog import get_catalog, plan_inference


def _load_array(path, kind):
    import numpy as np

    source = Path(path).resolve(strict=True)
    if source.suffix.lower() != '.npy' or not source.is_file():
        raise ValueError('Inputs must be regular .npy files; pickle and NPZ are unsupported.')
    if source.stat().st_size > 112 * 1024 * 1024:
        raise ValueError('Input exceeds the bounded array budget.')
    array = np.load(source, mmap_mode='r', allow_pickle=False)
    if not isinstance(array, np.ndarray):
        if hasattr(array, 'close'):
            array.close()
        raise ValueError('NPZ archives are unsupported even when renamed to .npy.')
    if kind == 'image':
        valid = (array.ndim == 3 and array.shape[0] == 12 and
                 all(1 <= size <= 1024 for size in array.shape[1:]) and array.dtype.kind in 'fiu')
    elif kind == 'mask':
        valid = array.ndim == 2 and all(1 <= size <= 1024 for size in array.shape) and array.dtype == np.bool_
    else:
        valid = (array.ndim == 4 and 1 <= array.shape[0] <= 16 and
                 array.shape[1:] == (12, 224, 224) and array.dtype == np.float32)
    if not valid:
        array._mmap.close()
        raise ValueError('Invalid ' + kind + ' shape/dtype; see command help and docs/INFERENCE.md.')
    if kind == 'tiles' and not np.isfinite(array).all():
        array._mmap.close()
        raise ValueError('Benchmark tiles must contain only finite normalized samples.')
    return array


def _benchmark(args):
    from .benchmark import BenchmarkIdentity, benchmark_predictors
    from .runtime import verify_bundle
    from .transport import VLLMPoolingPredictor

    for name, minimum, maximum in (('warmup', 0, 10), ('runs', 1, 30), ('batch_size', 1, 16),
                                  ('reference_batch_size', 1, 16), ('concurrency', 1, 16)):
        if not minimum <= getattr(args, name) <= maximum:
            raise ValueError(f'{name} must be between {minimum} and {maximum}.')
    for name in ('atol', 'rtol'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(name + ' must be finite and nonnegative.')

    if args.output:
        destination = Path(args.output).absolute()
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        destination = Path(__file__).resolve().parent.parent / 'runs' / 'benchmarks' / (stamp + '-' + uuid.uuid4().hex[:8] + '.json')
    if destination.exists() or destination.suffix.lower() != '.json':
        raise ValueError('Benchmark output must be a new .json file.')
    verified = verify_bundle(args.bundle)
    if destination.resolve().is_relative_to(Path(verified['bundle_dir'])):
        raise ValueError('Benchmark output must be outside the model bundle.')
    tiles = _load_array(args.tiles, 'tiles')
    recipe = get_catalog()['recipes'][0]
    identity = BenchmarkIdentity(verified['model_id'],
        verified['model_identity']['published_checkpoint_sha256'], recipe['id'],
        recipe['identity']['configuration_sha256'], args.dtype)
    common = dict(base_url=args.url, dtype=args.dtype, timeout=args.timeout)
    candidate = VLLMPoolingPredictor(verified['model_id'], max_in_flight=args.concurrency, **common)
    sync_reference = None
    if args.reference_backend == 'pytorch':
        from .reference import PyTorchPredictor
        reference = PyTorchPredictor(args.bundle, device=args.reference_device, dtype=args.dtype)
        sync_reference = reference.synchronize
    else:
        reference = VLLMPoolingPredictor(verified['model_id'], max_in_flight=1, **common)
    environment = {'client_os': platform.platform(), 'client_python': platform.python_version(),
                   'server_environment': args.server_label or 'not_recorded',
                   'reference_backend': args.reference_backend, 'candidate_backend': 'vllm',
                   'reference_device': args.reference_device if args.reference_backend == 'pytorch' else 'remote_service_not_attested'}
    for name in ('numpy', 'torch', 'terratorch', 'vllm'):
        try:
            environment['client_' + name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            environment['client_' + name] = 'not_installed'
    result = benchmark_predictors(reference, candidate, tiles,
        reference_identity=identity, candidate_identity=identity,
        reference_batch_size=args.reference_batch_size, candidate_batch_size=args.batch_size,
        warmup=args.warmup, runs=args.runs, atol=args.atol, rtol=args.rtol,
        min_class_agreement=1.0, synchronize_reference=sync_reference, environment=environment)
    after = verify_bundle(args.bundle)
    if (after['receipt_sha256'] != verified['receipt_sha256'] or after['files_sha256'] != verified['files_sha256']):
        raise ValueError('Bundle changed during benchmark; no receipt was written.')
    result.update({'created_utc': datetime.now(timezone.utc).isoformat(),
        'comparison_kind': 'pytorch_to_vllm' if args.reference_backend == 'pytorch' else 'vllm_request_scheduling',
        'server_checkpoint_attestation': 'not_available', 'bundle_receipt_sha256': verified['receipt_sha256'],
        'bundle_files_sha256': verified['files_sha256'],
        'reference_max_in_flight': 1 if args.reference_backend == 'vllm' else None,
        'candidate_max_in_flight': args.concurrency,
        'production_approved': False, 'ground_truth_accuracy_evaluated': False,
        'receipt_path': str(destination)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='AUREON — KAO-TerraMind inference engine')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('catalog', help='Print pinned model and recipe metadata')
    plan = commands.add_parser('plan', help='Print preparation requirements without probing a service')
    plan.add_argument('--model', required=True)
    plan.add_argument('--backend', choices=('vllm', 'pytorch'), help='Default: PyTorch for KAO-TerraMind B')
    model_service = commands.add_parser('serve-model', help='Explicitly start the KAO-TerraMind B HTTP model service in WSL')
    model_service.add_argument('--backend', choices=('pytorch', 'vllm'), default='pytorch')
    model_service.add_argument('--port', type=int, default=8002)
    model_service.add_argument('--io-mode', choices=('staged', 'pinned'),
                               help='Explicit HTTP buffer profile; pinned is experimental until benchmarked')
    model_service.add_argument('--output', required=True, help='Fresh directory directly under runs/services')
    model_service.add_argument('--thp-policy', choices=('inherit', 'disable'),
                               help='Override the service process THP policy')
    console = commands.add_parser('serve', help='Open the local inspection console; does not serve a model')
    console.add_argument('--port', type=int, default=4324)
    prepare = commands.add_parser('prepare-vllm', help='Convert a supplied pinned local checkpoint; never downloads')
    prepare.add_argument('--model', required=True)
    prepare.add_argument('--checkpoint', required=True)
    prepare.add_argument('--output', required=True, help='A new directory under an existing parent')
    verify = commands.add_parser('verify-bundle', help='Verify local bundle files without loading a model')
    verify.add_argument('--bundle', required=True)
    infer = commands.add_parser('infer', help='Run a scene through an already running local vLLM service')
    infer.add_argument('--input', required=True, help='Raw surface_reflectance_x10000 .npy [12,H,W], each side <=1024')
    infer.add_argument('--mask', required=True, help='Boolean .npy [H,W]; true means valid across all 12 bands')
    infer.add_argument('--output', required=True, help='New delivery directory under an existing parent')
    infer.add_argument('--no-score', action='store_true')
    infer.add_argument('--reuse-connections', action='store_true',
                       help='Reuse bounded HTTP connections across scene tiles; close them when the scene ends')
    infer.add_argument('--json-backend', choices=('stdlib', 'orjson'), default='stdlib')
    infer.add_argument('--response-format', choices=('base64', 'bytes_only'), default='base64')
    infer.add_argument('--input-format', choices=('torch', 'npy'), default='torch')
    infer.add_argument('--base64-backend', choices=('stdlib', 'pybase64'), default='stdlib')
    infer.add_argument('--transport', choices=('json', 'binary', 'binary_group', 'binary_pack', 'shared_memory'), default='json')
    bench = commands.add_parser('benchmark', help='Measure real predictors on identical normalized tiles; save a receipt')
    bench.add_argument('--tiles', required=True, help='Normalized float32 .npy [N,12,224,224], 1<=N<=16')
    bench.add_argument('--output', help='New JSON file; default runs/benchmarks/<timestamp-id>.json')
    bench.add_argument('--reference-backend', choices=('vllm', 'pytorch'), default='vllm')
    bench.add_argument('--reference-device', default='cuda', help='Explicit PyTorch reference device, e.g. cuda:0 or cpu')
    bench.add_argument('--reference-batch-size', type=int, default=1)
    bench.add_argument('--server-label', help='Caller-recorded server hardware/runtime identity; not an attestation')
    bench.add_argument('--warmup', type=int, default=1)
    bench.add_argument('--runs', type=int, default=5)
    bench.add_argument('--atol', type=float, default=1e-5)
    bench.add_argument('--rtol', type=float, default=1e-5)
    for command in (infer, bench):
        command.add_argument('--bundle', required=True)
        command.add_argument('--url', default='http://127.0.0.1:8000')
        command.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
        command.add_argument('--batch-size', type=int, default=1 if command is infer else 4)
        command.add_argument('--concurrency', type=int, default=1 if command is infer else 4)
        command.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args(argv)
    try:
        if args.command == 'catalog':
            result = get_catalog()
        elif args.command == 'plan':
            result = plan_inference(args.model, **({'backend': args.backend} if args.backend is not None else {}))
        elif args.command == 'serve-model':
            from .queued_service import main as service_main
            service_args = ['--backend', args.backend, '--port', str(args.port), '--output', args.output]
            if args.thp_policy is not None:
                service_args.extend(['--thp-policy', args.thp_policy])
            if args.io_mode is not None:
                service_args.extend(['--io-mode', args.io_mode])
            return service_main(service_args)
        elif args.command == 'serve':
            from .server import main as console_main
            return console_main(['--port', str(args.port)])
        elif args.command == 'verify-bundle':
            from .runtime import verify_bundle
            result = verify_bundle(args.bundle)
        elif args.command == 'infer':
            from .runtime import run_scene
            result = run_scene(args.bundle, _load_array(args.input, 'image'), _load_array(args.mask, 'mask'),
                output_dir=args.output, dtype=args.dtype, batch_size=args.batch_size,
                max_in_flight=args.concurrency, base_url=args.url, timeout=args.timeout,
                include_max_score=not args.no_score, reuse_connections=args.reuse_connections,
                json_backend=args.json_backend, response_format=args.response_format,
                input_format=args.input_format, base64_backend=args.base64_backend, transport=args.transport)
        elif args.command == 'benchmark':
            result = _benchmark(args)
        else:
            from .vllm_bundle import prepare_vllm_bundle
            model = next((m for m in get_catalog()['models'] if m['id'] == args.model), None)
            if model is None:
                raise ValueError('Select a pinned KAO-TerraMind model ID.')
            result = prepare_vllm_bundle(args.checkpoint, args.output, model_id=args.model,
                                        expected_sha256=model['identity']['published_checkpoint_sha256'])
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

