# Errors and support

Contact **norviccompanylimited@gmail.com** for installation/use questions.
For private security reports use the same owner-confirmed address as described
in [SECURITY.md](SECURITY.md). No response-time or remediation SLA is promised.

| Symptom | Action |
| --- | --- |
| `python3.12`, `venv` or `curl` missing | Run the guide in Linux/WSL Bash and provision the missing host prerequisite before starting installation |
| Lock file, `ARTIFACTS.json` or examples missing | Extract the installation kit from the release assets, not GitHub Source code (zip); set `AUREON_KIT` to its extracted directory |
| `aureon-engine: command not found` | Activate the intended venv in this terminal; use its `bin/python -m pip show aureon-inference` to confirm installation |
| Have SAFE/JP2, RGB or no imagery | Export the required GeoTIFF inputs and metadata first, or try the [CPU walkthrough](../docs/CPU-WALKTHROUGH.md); it is not a model prediction |
| Missing dependency / ProfileError | Run `pip check` and `aureon-engine check --environment ...` in the selected Python3.12 Linux venv; reinstall the matching hash lock in a fresh environment |
| Wrong wheel or file hash | Compare with the release manifest; do not replace a checksum with the observed wrong value |
| Checkpoint mismatch | Use the exact revision and filename from `aureon-engine assets`; renamed or fine-tuned weights are a different identity |
| Empty result JSON after CLI execution | Check the command exit code and stderr; shell redirection can create an empty file even when a job fails |
| Existing output path | Choose a new config, bundle, work or download path; preserve existing results |
| Wrong bands/units/grid | Correct the declared source mapping/radiometry and matching mask; do not force the manifest or label DN as reflectance |
| Invalid TerraMind crop | Select an all-valid224 window; cloud/nodata filling and arbitrary shapes are unsupported |
| Missing execution opt-in | Supply `--execute-model` only when intending to run inference |
| Guarded qualification mismatch | Use a matching artifact/runtime receipt or the explicit full-reference first-user route |
| CUDA driver/memory error | Check the device/driver and other workloads; installation checks do not qualify hardware; never silently lower precision |
| HTTP timeout after submission | Preserve input/job IDs and idempotency key; use status/resume/download, not a new logical submission |

Send the exact wheel hash, selected profile, Python/OS/driver versions, `pip check`,
sanitized command/error and receipt IDs. Keep keys, imagery and personal paths out
of public reports. Source, examples and third-party notices ship in the matching
sdist. This alpha demonstrates specialized local workflows, not every hardware
combination or a hosted service. Follow the installed `--help` for command syntax.
