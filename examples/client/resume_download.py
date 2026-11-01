"""Resume a recorded local-service download without uploading or submitting a job.

Run with the installed Engine using Python -I. The JSONL recovery file comes
from aureon-service-client run --recovery-file. No credentials belong in it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol


@dataclass(frozen=True)
class RecoveryRequest:
    job_id: str
    model: str
    profile: str
    task: str
    manifest_sha256: str


class RecoveryError(ValueError):
    """Safe, actionable local validation message, never remote response text."""


class DownloadClient(Protocol):
    def json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]: ...
    def wait_job(self, identifier: str, timeout: float = 600) -> dict[str, Any]: ...
    def download_result(self, identifier: str, output: Path, *, expected_manifest_sha256: str,
                        model: str, task: str) -> dict[str, Any]: ...


def bounded_json(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RecoveryError("Recovery/credential file exceeds its byte limit")
    return data


def read_recovery(path: Path) -> RecoveryRequest:
    rows = [json.loads(line) for line in bounded_json(path, 256 * 1024).splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RecoveryError("Expected a JSONL recovery record from aureon-service-client")
    requests = [row for row in rows if row.get("event") == "request"]
    jobs = [row for row in rows if row.get("event") == "job_submitted"]
    if len(requests) != 1 or len(jobs) != 1:
        raise RecoveryError("Need one recorded request and job_submitted event. Retain this file and ask the operator to resolve an uncertain submission; do not submit a new key")
    request, job = requests[0], jobs[0]
    required = ("model", "profile", "task", "manifest_sha256")
    if any(not isinstance(request.get(key), str) or not request[key] for key in required):
        raise RecoveryError("Recovery request is missing model/profile/task/input identity")
    if not re.fullmatch(r"[0-9a-f]{64}", request["manifest_sha256"]):
        raise RecoveryError("Recovery input SHA256 is invalid")
    if not isinstance(job.get("job_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", job["job_id"]):
        raise RecoveryError("Recovery job ID is invalid")
    if not isinstance(job.get("input_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", job["input_id"]):
        raise RecoveryError("Recovery input ID is invalid")
    if request["task"] not in {"scene", "generation", "generation_scene", "features"}:
        raise RecoveryError("Unsupported recovery task")
    for row in rows:
        if "manifest_sha256" in row and row["manifest_sha256"] != request["manifest_sha256"]:
            raise RecoveryError("Recovery log contains conflicting input identities")
        if "input_id" in row and row["input_id"] != job.get("input_id"):
            raise RecoveryError("Recovery log contains conflicting upload identities")
    return RecoveryRequest(job["job_id"], request["model"], request["profile"],
                           request["task"], request["manifest_sha256"])


def resume_download(client: DownloadClient, request: RecoveryRequest, output: Path,
                    *, wait_seconds: float = 0) -> dict[str, Any]:
    if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 3600:
        raise RecoveryError("wait-seconds must be finite and between 0 and 3600")
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RecoveryError("Use a fresh download directory with an existing parent; retain partial downloads")

    def validate(status: dict[str, Any]) -> None:
        expected = {"id": request.job_id, "model_id": request.model,
                    "profile_id": request.profile, "task": request.task}
        if any(status.get(key) != value for key, value in expected.items()):
            raise RecoveryError("Server job identity does not match the recovery record")

    status = client.json("GET", "/v1/jobs/" + request.job_id)
    validate(status)
    if status.get("state") in {"queued", "running"} and wait_seconds:
        status = client.wait_job(request.job_id, timeout=wait_seconds)
        validate(status)
    if status.get("state") != "completed":
        raise RecoveryError("Job is not completed. Keep the original job ID/key; no new job was submitted")
    return client.download_result(request.job_id, output,
        expected_manifest_sha256=request.manifest_sha256, model=request.model, task=request.task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Original trusted service URL")
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--recovery-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        request = read_recovery(args.recovery_file)
        credentials = json.loads(bounded_json(args.credentials, 64 * 1024))
        if not isinstance(credentials, dict):
            raise RecoveryError("Invalid credential file")
        token = credentials.get("key") if credentials.get("tenant") == args.tenant else credentials.get(args.tenant)
        if not isinstance(token, str) or not token:
            raise RecoveryError("Credential file has no key for the selected tenant")
        from aureon.serving.client import Client
        result = resume_download(Client(args.url, token=token), request, args.output,
                                 wait_seconds=args.wait_seconds)
    except Exception as error:
        # Network/parser errors may contain server data or credentials. Keep
        # actionable locally generated validation messages; redact all others.
        if isinstance(error, RecoveryError):
            message = str(error)
        else:
            message = "Recovery failed (" + type(error).__name__ + "); retain the same job ID/key and any partial output"
        parser.exit(2, message + ". No upload, submission or cancellation was requested.\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
