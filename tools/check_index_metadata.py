"""Read-only check of the frozen wheel's package-index description; no upload."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from email import policy
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def description_issues(description: str) -> list[str]:
    issues = []
    if re.search(r"\*\*Runtime selection:\*\*[^`\n]*`0fa5ef2b", description):
        issues.append("embedded_readme_selects_historical_runtime")
    if "unpromoted development, not the recommended execution artifact" in description:
        issues.append("embedded_readme_has_superseded_delivery_status")
    return issues


def inspect(wheel: Path) -> dict:
    selected = json.loads((ROOT / "release/v0.1.0.json").read_text(encoding="utf-8"))["artifacts"]["wheel"]
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != selected["sha256"]:
        raise ValueError("Supply the exact selected wheel")
    with zipfile.ZipFile(wheel) as archive:
        message = BytesParser(policy=policy.default).parsebytes(
            archive.read("aureon_inference-0.1.0.dist-info/METADATA"))
    description = message.get_payload(decode=True).decode("utf-8")
    issues = description_issues(description)
    if message["Name"] != "aureon-inference" or message["Version"] != "0.1.0":
        issues.append("distribution_name_or_version_differs")
    return {"wheel_sha256": digest, "runtime_identity_verified": True,
            "index_description_ready": not issues, "issues": issues,
            "scope": "Embedded package-index metadata only; functional/GPU receipts remain unchanged",
            "artifact_modified": False, "published": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.wheel)
    print(json.dumps(result, indent=2))
    return 0 if result["index_description_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
