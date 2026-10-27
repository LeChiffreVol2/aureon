"""Runnable installed SDK example; assets and model execution are explicit."""
import argparse
from dataclasses import asdict
import json
from aureon.api import SceneRequest,run_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","bundle","output","model","profile"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--environment",default="scene")
    parser.add_argument("--execute-model",action="store_true")
    args = parser.parse_args(argv)
    request = SceneRequest(args.manifest,args.bundle,args.output,args.model,args.profile,args.environment,args.qualification)
    print(json.dumps(asdict(run_scene(request,execute_model=args.execute_model)),default=str))


if __name__ == "__main__":
    main()
