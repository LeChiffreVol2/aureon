"""Run explicit prepared scenes through one public resident EngineSession."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resident KAO-TerraMind B EngineSession example")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--profile", required=True, choices=("fp32-full-reference-v1", "fp32-stream-guarded-v3", "fp32-guarded-v2"))
    parser.add_argument("--qualification")
    parser.add_argument("--execute-model", action="store_true")
    args = parser.parse_args(argv)
    from aureon.api import EngineSession
    with EngineSession(args.bundle, args.work, profile_id=args.profile,
                       qualification=args.qualification, execute_model=args.execute_model) as engine:
        for manifest in args.manifest:
            result = engine.run(manifest)
            print(json.dumps({"receipt": str(result.receipt), "numerical_status": result.numerical_status}), flush=True)
        print(json.dumps(engine.metadata()), flush=True)
    print(json.dumps(engine.close()), flush=True)


if __name__ == "__main__":
    main()
