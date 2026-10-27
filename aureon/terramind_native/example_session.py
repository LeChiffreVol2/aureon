"""Explicit two-job resident example. --help never imports the model runtime."""
import argparse
import json

from .common import read_json
from .session import TerraMindSession


def main():
    parser = argparse.ArgumentParser(description="Run two serial synthetic S1GRD jobs with one TerraMind model load")
    parser.add_argument("--config", required=True, help="Verified local native adapter configuration")
    parser.add_argument("--work", required=True, help="Fresh session directory, existing parent")
    parser.add_argument("--input-manifest", help="Optional prepared all-valid 224x224 S2L2A manifest")
    parser.add_argument("--token-steps", type=int, choices=(1, 8), default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--denoiser", choices=("native", "graph"), default="native",
                        help="Opt-in fixed-profile FP32 denoiser graph; native preserves the ordinary path")
    parser.add_argument("--execute-model", action="store_true")
    args = parser.parse_args()
    if not args.execute_model:
        parser.error("Model execution requires --execute-model; nothing was loaded")
    recipe = {**read_json(args.config)["recipe"], "token_steps": args.token_steps, "seed": args.seed}
    with TerraMindSession(args.config, args.work, execute_model=True, reuse="auto", denoiser=args.denoiser) as session:
        for index in range(2):
            record = session.run(args.input_manifest, job_id="example-%d" % (index+1), attempt_id="first",
                                 recipe={**recipe, "seed": args.seed+index})
            print(json.dumps({"job_id": record["job_id"], "cog": record["cog"]["file"],
                              "provenance": record["provenance"], "reuse_decision": record["reuse_decision"],
                              "denoiser_mode": record["denoiser_mode"], "decoder": record["decoder"],
                              "model_load_count": record["model_load_count"]}))
    print(json.dumps({"cleanup": session.close()}))


if __name__ == "__main__": main()
