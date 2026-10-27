"""Explicit, versioned local KAO-TerraMind Engine commands."""
import argparse
from dataclasses import asdict
import json
from .profiles import PROFILES as ENVIRONMENTS, inspect_profile
from .v1 import MODELS, PROFILES, SceneRequest, run_scene
from .models import GENERATION_MODEL, GenerationRequest, list_models, run_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("profiles",help="Show pins and profiles; no model imports")
    sub.add_parser("models",help="List both model families, tasks and support limits; no model imports")
    assets = sub.add_parser("assets", help="List official pinned sources/checksums; never downloads weights")
    assets.add_argument("--model", choices=(*MODELS, GENERATION_MODEL))
    check = sub.add_parser("check",help="Check dependency metadata without importing a model")
    check.add_argument("--environment",choices=ENVIRONMENTS,required=True)
    check.add_argument("--config",help="Offline TerraMind asset/source hashes; requires generation environment")
    config = sub.add_parser("generation-config",help="Bind existing local pinned assets; no download or model load")
    for name in ("model-file", "tokenizer-file", "manifest", "output"):
        config.add_argument("--"+name, required=True)
    config.add_argument("--seed",type=int,default=17)
    config.add_argument("--token-steps",type=int,choices=(1,8),default=8)
    generate = sub.add_parser("generate",help="Explicit TerraMind S2L2A to synthetic S1GRD job and owned close")
    generate.add_argument("--model",choices=(GENERATION_MODEL,),required=True)
    for name in ("config", "output", "job-id", "attempt-id"):
        generate.add_argument("--"+name, required=True)
    generate.add_argument("--manifest",help="Optional prepared all-valid 224px input override")
    generate.add_argument("--environment",choices=("generation",),default="generation")
    generate.add_argument("--denoiser",choices=("native","graph"),default="native")
    generate.add_argument("--reuse",choices=("auto","off"),default="auto")
    generate.add_argument("--execute-model",action="store_true")
    run = sub.add_parser("run",help="One explicit scene job and owned shutdown")
    for name in ("manifest","bundle","output"):
        run.add_argument("--"+name,required=True)
    run.add_argument("--model",choices=MODELS,required=True)
    run.add_argument("--profile",choices=PROFILES,required=True)
    run.add_argument("--environment",choices=("scene","scene-legacy"),default="scene")
    run.add_argument("--qualification")
    run.add_argument("--execute-model",action="store_true")
    args = parser.parse_args(argv)
    if args.command == "assets":
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
            result = create_config(model=args.model_file,tokenizer=args.tokenizer_file,
                                   input_manifest=args.manifest,output=args.output,
                                   seed=args.seed,token_steps=args.token_steps)
        except (ValueError,OSError,RuntimeError,KeyError) as error:
            parser.exit(2,f"Engine config failed: {error}\n")
    elif args.command == "generate":
        try:
            request = GenerationRequest(args.config,args.output,args.job_id,args.attempt_id,
                                        model_id=args.model,environment=args.environment,
                                        input_manifest=args.manifest,reuse=args.reuse,denoiser=args.denoiser)
            result = run_model(request,execute_model=args.execute_model)
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
