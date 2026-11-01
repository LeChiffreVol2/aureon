"""Explicit, versioned local KAO-TerraMind Engine commands."""
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast
import argparse
from dataclasses import asdict
import json
from zipfile import BadZipFile
from .profiles import PROFILES as ENVIRONMENTS, inspect_profile
from .v1 import MODELS, PROFILES, SceneRequest
from ..terramind_native.families import GENERATION_MODELS, BACKBONE_MODELS, TIM_MODELS
from .models import GENERATION_MODEL, GenerationRequest, list_models, run_model, open_model_session

if TYPE_CHECKING:
    from .best import BestTerraMindSession


def _base_scene(args: argparse.Namespace) -> dict[str, object]:
    """Default scene delivery on the selected resident; no second numerical engine."""
    from .best_workflows import BestSceneRequest, inspect_best_scene, run_best_scene
    if args.model_file is not None or args.tokenizer_file is not None:
        raise ValueError("Base scene execution uses --config from generation-config with a seed manifest; model-file/tokenizer-file remain for other family compatibility")
    if args.max_in_flight != 1:
        raise ValueError("The selected base scene route requires --max-in-flight 1")
    if args.command == "generate-scene":
        if args.execute_model is not True:
            raise ValueError("Pass --execute-model explicitly; nothing loaded")
        if args.config is None or args.work is None:
            raise ValueError("Base scene execution requires --config and --work")
    request = BestSceneRequest(image=args.image, valid_mask=args.valid_mask, output=args.output,
                              job_id=args.job_id, attempt_id=args.attempt_id, seed=args.seed,
                              stride=args.stride, invalid_policy=args.invalid_policy,
                              max_buffer_bytes=args.max_buffer_bytes, max_decode_bytes=args.max_decode_bytes,
                              max_workspace_bytes=args.max_workspace_bytes,
                              max_output_bytes=512*1024**2 if args.max_output_bytes is None else args.max_output_bytes,
                              min_free_bytes=args.min_free_bytes)
    if args.command == "scene-plan":
        return inspect_best_scene(request)
    inspect_best_scene(request)  # Preflight before resident construction; execution rechecks mutation.
    with cast("BestTerraMindSession", open_model_session(GENERATION_MODEL, config=args.config, work=args.work,
                 execution_profile=args.execution_profile, execute_model=True)) as session:
        return run_best_scene(session, request, execute_model=True)


def main(argv: Sequence[str] | None=None) -> int:
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "best":
        from .best_cli import main as best_main
        return best_main(arguments[1:])
    if arguments and arguments[0] in ("cost", "capacity", "verify-scene"):
        from .best_cli import main as planning_main
        return planning_main(arguments, prog="aureon-engine")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("best", help="Compatibility aliases for the default selected-core commands")
    sub.add_parser("cost", help="Normalize measured resource occupancy")
    sub.add_parser("capacity", help="Plan independent resident processes from explicit budgets")
    sub.add_parser("verify-scene", help="Verify retained completed scene artifacts without a model")
    sub.add_parser("profiles",help="Show pins and profiles; no model imports")
    sub.add_parser("models",help="List both model families, tasks and support limits; no model imports")
    sub.add_parser("devices", help="Explicit NVIDIA driver inventory by physical UUID; no Torch import or model execution")
    prepared = sub.add_parser("service-input", help="Bind existing prepared files into a fresh upload manifest; no model, transform or copy")
    prepared.add_argument("--task", choices=("generation_scene", "features"), required=True)
    prepared.add_argument("--image")
    prepared.add_argument("--valid-mask")
    prepared.add_argument("--inputs", help="Numeric modality-keyed NPZ for a features job")
    prepared.add_argument("--output", required=True, help="Fresh manifest path in a parent containing all input members")
    assets = sub.add_parser("assets", help="List official pinned sources/checksums; never downloads weights")
    assets.add_argument("--model", choices=(*MODELS, *GENERATION_MODELS, *BACKBONE_MODELS, *TIM_MODELS))
    check = sub.add_parser("check",help="Check dependency metadata without importing a model")
    check.add_argument("--environment",choices=ENVIRONMENTS,required=True)
    check.add_argument("--config",help="Offline TerraMind asset/source hashes; requires generation environment")
    doctor = sub.add_parser("doctor", help="Actionable offline dependency/asset checks; no model or CUDA calls")
    doctor.add_argument("--model", choices=(*MODELS, *GENERATION_MODELS, *BACKBONE_MODELS, *TIM_MODELS), required=True)
    doctor.add_argument("--environment", choices=ENVIRONMENTS)
    doctor.add_argument("--assets-dir")
    doctor.add_argument("--verify-assets", action="store_true")
    doctor.add_argument("--modality", action="append")
    doctor.add_argument("--tim-modality", action="append")
    doctor.add_argument("--work")
    doctor.add_argument("--min-free-bytes", type=int, default=0)
    plan = sub.add_parser("generation-plan", help="CPU-only execution selection; no assets, model or CUDA probe")
    plan.add_argument("--model", choices=GENERATION_MODELS, default=GENERATION_MODEL)
    plan.add_argument("--execution-profile", choices=("recommended", "reference"), default="recommended")
    plan.add_argument("--token-steps", type=int, choices=(1,8), default=8)
    plan.add_argument("--diffusion-steps", type=int, choices=range(1,51), default=10)
    config = sub.add_parser("generation-config",help="Bind existing local pinned assets; no download or model load")
    for name in ("model-file", "tokenizer-file", "output"):
        config.add_argument("--"+name, required=True)
    config.add_argument("--model", choices=GENERATION_MODELS, default=GENERATION_MODEL)
    config.add_argument("--manifest", help="Prepared input; omit for an assets-only in-memory session config")
    config.add_argument("--seed",type=int,default=17)
    config.add_argument("--token-steps",type=int,choices=(1,8),default=8)
    generate = sub.add_parser("generate",help="Explicit TerraMind S2L2A to synthetic S1GRD job and owned close")
    generate.add_argument("--model",choices=GENERATION_MODELS,default=GENERATION_MODEL)
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", "--work", dest="output", required=True)
    generate.add_argument("--job-id", default="generation")
    generate.add_argument("--attempt-id", default="1")
    generate.add_argument("--seed", type=int, help="Use the 8-token/10-diffusion recipe with this seed; omit to retain the configured recipe")
    generate.add_argument("--manifest",help="Optional prepared all-valid 224px input override")
    generate.add_argument("--environment",choices=("generation",),default="generation")
    generate.add_argument("--denoiser",choices=("native","graph"),default=None)
    generate.add_argument("--reuse",choices=("auto","off"),default=None)
    generate.add_argument("--execution-profile",choices=("recommended","reference"),
                          help="Base file default is recommended auto/graph; reference selects diagnostic off/native")
    generate.add_argument("--execute-model",action="store_true")
    encode = sub.add_parser("encode", help="TerraMind Backbone/TiM features from a modality-keyed NPZ")
    encode.add_argument("--model", choices=(*BACKBONE_MODELS, *TIM_MODELS), required=True)
    for name in ("model-file", "input", "output"):
        encode.add_argument("--" + name, required=True)
    encode.add_argument("--tokenizer", action="append", default=[], metavar="MODALITY=FILE")
    encode.add_argument("--tim-modality", action="append")
    encode.add_argument("--device", default="cpu")
    encode.add_argument("--merge-method", choices=("mean", "max", "concat", "dict"), default="mean")
    encode.add_argument("--seed", type=int, default=17)
    encode.add_argument("--execute-model", action="store_true")
    for name in ("scene-plan", "generate-scene"):
        scene = sub.add_parser(name, help="Inspect or execute bounded tiled TerraMind generation from prepared rasters")
        scene.add_argument("--model", choices=GENERATION_MODELS, default=GENERATION_MODEL)
        for field in ("image", "valid-mask", "output"):
            scene.add_argument("--" + field, required=True)
        scene.add_argument("--config", help="Base seed-manifest config from generation-config")
        scene.add_argument("--work", help="Fresh resident work directory; required for execution")
        scene.add_argument("--model-file", help="Other-family compatibility route")
        scene.add_argument("--tokenizer-file", help="Other-family compatibility route")
        scene.add_argument("--job-id", default="scene")
        scene.add_argument("--attempt-id", default="1")
        scene.add_argument("--max-workspace-bytes", type=int, default=512*1024**2, help="Base selected-core staging bound")
        scene.add_argument("--seed", type=int, default=17)
        scene.add_argument("--stride", type=int, default=224)
        scene.add_argument("--invalid-policy", choices=("reject", "skip_invalid_tiles"), default="reject")
        scene.add_argument("--execution-profile", choices=("recommended", "reference"), default="recommended")
        scene.add_argument("--max-in-flight", type=int, default=1)
        scene.add_argument("--max-buffer-bytes", type=int, default=128*1024**2)
        scene.add_argument("--max-decode-bytes", type=int, default=32*1024**2)
        scene.add_argument("--max-output-bytes", type=int, help="Base default 512 MiB; other-family compatibility default 2 GiB")
        scene.add_argument("--min-free-bytes", type=int, default=1024**3)
        if name == "generate-scene":
            scene.add_argument("--execute-model", action="store_true")
    run = sub.add_parser("run",help="One explicit scene job and owned shutdown")
    for name in ("manifest","bundle","output"):
        run.add_argument("--"+name,required=True)
    run.add_argument("--model",choices=MODELS,required=True)
    run.add_argument("--profile",choices=PROFILES,required=True)
    run.add_argument("--environment",choices=("scene","scene-legacy"),default="scene")
    run.add_argument("--qualification")
    run.add_argument("--execute-model",action="store_true")
    args = parser.parse_args(argv)
    if args.command == "devices":
        from .gpu_devices import probe_nvidia_gpus
        import subprocess
        try:
            result = {"schema": "aureon.nvidia_inventory.v1", **asdict(probe_nvidia_gpus()),
                      "scope": "physical NVML inventory, not CUDA process ordinals or a reservation",
                      "cuda_context_created": False, "model_executed": False,
                      "performance_qualified": False}
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            parser.exit(2, f"GPU inventory failed: {error}\n")
    elif args.command == "service-input":
        from ..serving.prepared_input import prepare_service_input
        try:
            result = prepare_service_input(args.output, task=args.task, image=args.image,
                valid_mask=args.valid_mask, inputs=args.inputs)
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(2, f"Service input preparation failed: {error}\n")
    elif args.command == "doctor":
        from .diagnostics import diagnose_runtime
        try:
            diagnosis = diagnose_runtime(args.model, assets_dir=args.assets_dir, environment=args.environment,
                verify_assets=args.verify_assets, modalities=args.modality or ("S2L2A",),
                tim_modalities=args.tim_modality, work=args.work, min_free_bytes=args.min_free_bytes)
        except (ValueError, OSError) as error:
            parser.exit(2, f"Engine diagnostics failed: {error}\n")
        print(json.dumps(diagnosis, sort_keys=True, allow_nan=False))
        return 0 if diagnosis["status"] == "passed" else 1
    elif args.command == "generation-plan":
        from .models import plan_generation_execution
        result = plan_generation_execution(execution_profile=args.execution_profile, model_id=args.model,
                    recipe={"token_steps": args.token_steps, "diffusion_steps": args.diffusion_steps,
                            "temperature": 1., "top_p": .8, "top_k": 0, "seed": 17})
    elif args.command in ("scene-plan", "generate-scene"):
        try:
            if args.model == GENERATION_MODEL:
                result = _base_scene(args)
            else:
                if args.config is not None:
                    raise ValueError("Other-family scene execution uses model-file/tokenizer-file, not the base config route")
                if any(value is None for value in (args.model_file, args.tokenizer_file, args.work)):
                    raise ValueError("Other-family scene execution requires model-file, tokenizer-file and work")
                from .scene_generation import SceneGenerationRequest, inspect_generation_scene, run_generation_scene
                scene_request = SceneGenerationRequest(
                    image=args.image, valid_mask=args.valid_mask, model_file=args.model_file,
                    tokenizer_file=args.tokenizer_file, work=args.work, output=args.output,
                    model=args.model, seed=args.seed, stride=args.stride, invalid_policy=args.invalid_policy,
                    execution_profile=args.execution_profile, max_in_flight=args.max_in_flight,
                    max_buffer_bytes=args.max_buffer_bytes, max_decode_bytes=args.max_decode_bytes,
                    max_output_bytes=2*1024**3 if args.max_output_bytes is None else args.max_output_bytes,
                    min_free_bytes=args.min_free_bytes)
                result = (inspect_generation_scene(scene_request) if args.command == "scene-plan"
                          else run_generation_scene(scene_request, execute_model=args.execute_model))
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(2, f"Engine scene generation failed: {error}\n")
    elif args.command == "assets":
        from .assets import list_assets
        result = {"schema": "aureon.asset_sources.v1", "assets": list_assets(args.model),
                  "downloaded": False, "model_imported": False}
    elif args.command == "models":
        result = {"schema":"aureon.model_registry.v1", "models":list_models(), "model_imported":False}
    elif args.command == "profiles":
        result = {"environments":ENVIRONMENTS,"execution_profiles":PROFILES,
                  "rule":"Choose one environment. Guarded profiles require qualified KAO-TerraMind B on the scene stack."}
    elif args.command == "check":
        try:
            if args.config:
                if args.environment != "generation":
                    raise ValueError("--config is only supported with --environment generation")
                from ..terramind_native.offline import inspect_config
                result = inspect_config(args.config)
            else:
                result = inspect_profile(args.environment)
        except (ValueError,OSError,RuntimeError,KeyError) as error:
            parser.exit(2,f"Engine preflight failed: {error}\n")
    elif args.command == "generation-config":
        try:
            from ..terramind_native.offline import create_config
            result = create_config(model=args.model_file,tokenizer=args.tokenizer_file,model_id=args.model,
                                   input_manifest=args.manifest,output=args.output,
                                   seed=args.seed,token_steps=args.token_steps)
        except (ValueError,OSError,RuntimeError,KeyError) as error:
            parser.exit(2,f"Engine config failed: {error}\n")
    elif args.command == "encode":
        try:
            from .encoder_io import encode_file
            tokenizer_files = {}
            for item in args.tokenizer:
                modality, separator, path = item.partition("=")
                if not separator or not modality or not path or modality in tokenizer_files:
                    raise ValueError("Use unique --tokenizer MODALITY=FILE entries")
                tokenizer_files[modality] = path
            result = encode_file(model_id=args.model, model_file=args.model_file, input_file=args.input,
                                 output=args.output, tokenizer_files=tokenizer_files,
                                 tim_modalities=args.tim_modality, device=args.device,
                                 merge_method=args.merge_method, seed=args.seed, execute_model=args.execute_model)
        except (ValueError, OSError, RuntimeError, KeyError, BadZipFile) as error:
            parser.exit(2, f"Engine encoder failed: {error}\n")
    elif args.command == "generate":
        try:
            request: GenerationRequest | SceneRequest = GenerationRequest(args.config,args.output,args.job_id,args.attempt_id,
                                        model_id=args.model,environment=args.environment,
                                        input_manifest=args.manifest,reuse=args.reuse,denoiser=args.denoiser,
                                        execution_profile=args.execution_profile,
                                        recipe=None if args.seed is None else {"token_steps":8, "diffusion_steps":10,
                                            "temperature":1., "top_p":.8, "top_k":0, "seed":args.seed})
            result = run_model(cast(GenerationRequest, request),execute_model=args.execute_model)
        except (ValueError,OSError,RuntimeError,KeyError) as error:
            parser.exit(2,f"Engine generation failed: {error}\n")
    else:
        try:
            request = SceneRequest(args.manifest,args.bundle,args.output,args.model,args.profile,args.environment,args.qualification)
            result = asdict(run_model(request,execute_model=args.execute_model))
        except (ValueError,OSError,RuntimeError) as error:
            parser.exit(2,f"Engine request failed: {error}\n")
    print(json.dumps(result,default=str,sort_keys=True,allow_nan=False))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
