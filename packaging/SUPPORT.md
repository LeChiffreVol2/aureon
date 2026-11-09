# Errors and support

**Runtime selection:** use the [locked integrated Best package](CANDIDATE-QUICKSTART.md), `e09e710c...`, v0.1.0. Original `0fa5ef2b` measurements remain historical evidence. The installed package and current examples do not imply new performance or hosted qualification.

Contact **norviccompanylimited@gmail.com** for installation/use questions.
For private security reports use the same owner-confirmed address as described
in [SECURITY.md](SECURITY.md). No response-time or remediation SLA is promised.

| Symptom | Action |
| --- | --- |
| Which release file should I install? | Use Best `e09e710c` from GitHub and verify the [current hashes](CANDIDATE-QUICKSTART.md). Files prefixed `historical-` preserve earlier deliveries |
| Scene output has visible tile seams despite exact parity | The accepted candidate preserves native tile behavior. Exact reconstruction/COG checks do not establish seamless imagery or physical SAR realism |
| Candidate scenes pass, but throughput differs from the historical chart | Scene acceptance is not performance qualification. The reference chart belongs to wheel `0fa5ef2b…`; current-wheel competitor speedup and general non-regression are not established |
| Version 0.1.0 is installed but a current API is missing | Verify the exact wheel hash/profile using [CANDIDATE-QUICKSTART.md](CANDIDATE-QUICKSTART.md); the version string alone cannot distinguish Best from the original alpha |
| Existing TorchGeo/TerraTorch conflicts with native dependency pins | Keep the existing environment separate. The selected best runtime uses the retained generation environment; use the generation pins that match the locked integrated wheel |
| New input inspector is not in the frozen sdist | Use the current checkout companion example with the locked installed wheel. The immutable source archive does not acquire newer examples; do not overlay Engine source |
| Old source example passes `neural_batch` | The unpublished tensor-batch option was retired. Omit it and use recommended/reference execution with bounded queue or lane controls; frozen wheels keep their original contents |
| Installer stopped | Inspect `install-result.json` and `setup-logs/` in the selected prefix; correct the cause and choose a fresh prefix. Existing environments and results are retained |
| Missing dependency / ProfileError | Run `pip check` and `aureon-engine check --environment ...` in the selected Python3.12 Linux venv; reinstall the matching hash lock in a fresh environment |
| Wrong wheel or file hash | Compare with the release manifest; do not replace a checksum with the observed wrong value |
| Checkpoint mismatch | Use the exact revision and filename from `aureon-engine assets`; renamed or fine-tuned weights are a different identity |
| Existing output path | Choose a new config, bundle, work or download path; preserve existing results |
| Wrong bands/units/grid | Correct the declared source mapping/radiometry and matching mask; do not force the manifest or label DN as reflectance |
| Invalid TerraMind crop | Single-tile generation needs an all-valid 224 window. For larger scenes use BEST-WORKFLOWS.md; reject invalid tiles or explicitly skip them with coverage reported. No cloud filling |
| Missing execution opt-in | Supply `--execute-model` only when intending to run inference |
| Guarded qualification mismatch | Use a matching artifact/runtime receipt or the explicit full-reference first-user route |
| CUDA driver/memory error | Check the device/driver and other workloads; installation checks do not qualify hardware; never silently lower precision |
| HTTP timeout after submission | Preserve input/job IDs and idempotency key; use status/resume/download, not a new logical submission |

Send the exact wheel hash, selected profile, Python/OS/driver versions, `pip check`,
sanitized command/error and receipt IDs. Keep keys, imagery and personal paths out
of public reports. Source, examples and third-party notices ship in the matching
sdist. This alpha demonstrates specialized local workflows, not every hardware
combination or a hosted service. Follow the installed `--help` for command syntax.
