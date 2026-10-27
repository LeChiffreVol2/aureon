"""Run one supplied prepared scene, then independently audit it (opt-in only)."""
import argparse
import json
import math
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--streaming', action='store_true', help='Bounded 4096-side path with full FP32 reference')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--hourly-rate', type=float, required=True,
                        help='User-declared accounting scenario, not a measured tariff')
    parser.add_argument('--currency', required=True)
    args = parser.parse_args(argv)
    if not args.execute_model:
        parser.error('Explicit --execute-model is required; no model or output was created')
    if sys.platform != 'linux':
        parser.error('Model execution requires the supported Linux/WSL environment')
    output, audit = args.output.absolute(), args.audit_output.absolute()
    if (output.exists() or output.is_symlink() or audit.exists() or audit.is_symlink()
            or not output.parent.is_dir() or not audit.parent.is_dir()
            or output == audit or output in audit.parents):
        parser.error('Use separate fresh job and audit paths under existing parents')
    if not math.isfinite(args.hourly_rate) or args.hourly_rate < 0:
        parser.error('Hourly rate must be finite and nonnegative')
    # Import only after explicit execution and output checks. The installed
    # package resolves its own source files; the caller's cwd is irrelevant.
    if args.streaming:
        from aureon.scene_cli import run_streaming_scene
        from aureon.scene_api.audit import audit_job
        run_streaming_scene(args.manifest, args.bundle, output, execute_model=True,
                            hourly_rate=args.hourly_rate, currency=args.currency)
        job_folder, artifacts = output/'job', output/'job/delivery/artifacts'
        receipt = json.loads((job_folder/'job.json').read_text())
    else:
        from aureon.scene_pipeline.job import run_job
        from aureon.scene_pipeline.audit import audit_job
        receipt = run_job(args.manifest.resolve(strict=True), args.bundle.resolve(strict=True),
                          output, execute_model=True, hourly_rate=args.hourly_rate, currency=args.currency)
        job_folder, artifacts = output, output/'artifacts'
    report = audit_job(job_folder, audit)
    summary = {'status': report['status'], 'job_receipt': str(job_folder / 'job.json'),
               'audit': str(audit), 'class_map': str(artifacts/'class_map.tif'),
               'valid_mask': str(artifacts/'valid_mask.tif'),
               'parity': receipt['parity'], 'task_accuracy_evaluated': False,
               'scope': 'One offline FP32 job and CPU audit; no speedup comparison'}
    print(json.dumps(summary, allow_nan=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
