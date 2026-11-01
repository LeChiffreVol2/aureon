"""Selected-core generation and explicit cost/capacity planning commands."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from time import perf_counter

from ..planning.best_runtime import BestDeviceBudget, best_usage_cost, plan_best_replicas


def main(argv: Sequence[str] | None = None, *, prog: str = "aureon-engine best") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capacity = sub.add_parser("capacity", help="Plan N independent processes per GPU; no launch or device discovery")
    capacity.add_argument("--devices", type=Path, required=True, help="JSON list of declared device/host/memory budgets")
    capacity.add_argument("--per-replica-bytes", type=int, required=True, help="Peak including model, graph, CUDA context and other allocations")
    capacity.add_argument("--reserve-bytes", type=int, required=True)
    capacity.add_argument("--replicas-per-device", type=int, default=1)
    capacity.add_argument("--profile", choices=("best_fp32", "reference_fp32"), default="best_fp32")
    cost = sub.add_parser("cost", help="Normalize supplied resource occupancy; no billing or performance prediction")
    cost.add_argument("--resource-seconds", type=float, required=True)
    cost.add_argument("--completed-outputs", type=int, required=True)
    cost.add_argument("--hourly-rate", type=float, required=True)
    cost.add_argument("--currency", default="THB")
    verify = sub.add_parser("verify-scene", help="Verify a retained completed scene after restart; no model or retry")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--job-id")
    verify.add_argument("--attempt-id")
    generate = sub.add_parser("generate", help="One explicit job through the preserved best numerical core")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--work", type=Path, required=True)
    generate.add_argument("--job-id", default="generation")
    generate.add_argument("--attempt-id", default="1")
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--profile", choices=("best_fp32", "reference_fp32"), default="best_fp32")
    generate.add_argument("--hourly-rate", type=float, default=1., help="Assumed THB/resource-hour, including cold setup/cleanup; not a provider price")
    generate.add_argument("--execute-model", action="store_true")
    for name in ("scene-plan", "scene"):
        scene = sub.add_parser(name, help="Inspect a prepared scene" if name == "scene-plan" else "Deliver a prepared scene with the best resident")
        scene.add_argument("--image", type=Path, required=True)
        scene.add_argument("--valid-mask", type=Path, required=True)
        scene.add_argument("--output", type=Path, required=True)
        scene.add_argument("--job-id", required=True)
        scene.add_argument("--attempt-id", default="1")
        scene.add_argument("--seed", type=int, default=17)
        scene.add_argument("--stride", type=int, default=224)
        scene.add_argument("--invalid-policy", choices=("reject", "skip_invalid_tiles"), default="reject")
        scene.add_argument("--max-buffer-bytes", type=int, default=128*1024**2)
        scene.add_argument("--max-decode-bytes", type=int, default=32*1024**2)
        scene.add_argument("--max-workspace-bytes", type=int, default=512*1024**2)
        scene.add_argument("--max-output-bytes", type=int, default=512*1024**2)
        scene.add_argument("--min-free-bytes", type=int, default=1024**3)
        if name == "scene":
            scene.add_argument("--config", type=Path, required=True)
            scene.add_argument("--work", type=Path, required=True)
            scene.add_argument("--profile", choices=("best_fp32", "reference_fp32"), default="best_fp32")
            scene.add_argument("--hourly-rate", type=float, default=1., help="Assumed THB/resource-hour, not a provider price")
            scene.add_argument("--execute-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "capacity":
            raw = json.loads(args.devices.read_text(encoding="utf-8"))
            if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
                raise ValueError("devices must contain a JSON list of budget objects")
            result = plan_best_replicas([BestDeviceBudget(**row) for row in raw],
                per_replica_bytes=args.per_replica_bytes, reserve_per_device_bytes=args.reserve_bytes,
                replicas_per_device=args.replicas_per_device, profile=args.profile).to_dict()
        elif args.command == "cost":
            result = best_usage_cost(resource_seconds=args.resource_seconds,
                completed_outputs=args.completed_outputs, hourly_rate=args.hourly_rate, currency=args.currency)
        elif args.command == "verify-scene":
            from .best_workflows import verify_best_scene
            result = verify_best_scene(args.output, job_id=args.job_id, attempt_id=args.attempt_id)
        elif args.command in ("scene", "scene-plan"):
            if args.command == "scene" and args.execute_model is not True:
                parser.error("Pass --execute-model explicitly; no model was loaded")
            from .best_workflows import BestSceneRequest, inspect_best_scene, run_best_scene
            request = BestSceneRequest(image=args.image, valid_mask=args.valid_mask, output=args.output,
                job_id=args.job_id, attempt_id=args.attempt_id, seed=args.seed, stride=args.stride,
                invalid_policy=args.invalid_policy, max_buffer_bytes=args.max_buffer_bytes,
                max_decode_bytes=args.max_decode_bytes, max_workspace_bytes=args.max_workspace_bytes,
                max_output_bytes=args.max_output_bytes, min_free_bytes=args.min_free_bytes,
                hourly_rate=args.hourly_rate if args.command == "scene" else None,
                currency="THB" if args.command == "scene" else None)
            # Inspect local data and budgets before allocating the resident.
            result = inspect_best_scene(request)
            if args.command == "scene":
                from .best import BestTerraMindSession
                started = perf_counter()
                with BestTerraMindSession(args.config, args.work, execute_model=True,
                                          profile=args.profile) as session:
                    result = run_best_scene(session, request, execute_model=True)
                result["command_cost"] = best_usage_cost(resource_seconds=perf_counter()-started,
                    completed_outputs=1, hourly_rate=args.hourly_rate)
                result["command_cost_boundary"] = "resident construction through scene delivery and close; excludes preliminary scene-plan; includes internal admission"
        else:
            if args.execute_model is not True:
                parser.error("Pass --execute-model explicitly; no model was loaded")
            # Validate cost assumptions before any model or filesystem execution.
            best_usage_cost(resource_seconds=0., completed_outputs=0, hourly_rate=args.hourly_rate)
            from .best import BestTerraMindSession
            started = perf_counter()
            with BestTerraMindSession(args.config, args.work, execute_model=True,
                                      profile=args.profile) as session:
                result = dict(session.run(args.manifest, job_id=args.job_id,
                    attempt_id=args.attempt_id, recipe={"token_steps":8, "diffusion_steps":10,
                    "temperature":1., "top_p":.8, "top_k":0, "seed":args.seed}))
            result["command_cost"] = best_usage_cost(resource_seconds=perf_counter()-started,
                completed_outputs=1, hourly_rate=args.hourly_rate)
            result["command_cost_boundary"] = "one resident construction through completed delivery and close; cold single job"
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"Best runtime request failed: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
