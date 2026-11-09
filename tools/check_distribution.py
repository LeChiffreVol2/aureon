"""Check public installation and evidence roles without network or model imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ("README.md", "docs/README.md", "docs/DEVELOPER-INTEGRATION.md",
          "packaging/README.md", "packaging/CANDIDATE-QUICKSTART.md", "packaging/SUPPORT.md",
          "packaging/EO-ROUTES.md", "packaging/BORROWED-ADAPTERS.md",
          "packaging/SERVICE-OPERATIONS.md", "packaging/REPLICA-SERVICE.md",
          "packaging/CONCURRENT-GENERATION.md")
STALE = (r"local candidate is unpublished", r"These are unpublished source additions",
         r"Working source, unreleased", r"Integrated source, unreleased",
         r"current unreleased v0\.1\.0 source", r"published GitHub alpha\s+`a11b67e8.*?` is unchanged",
         r"\*\*Runtime selection:\*\*[^`\n]+`0fa5ef2b")


def inspect(root: Path = ROOT) -> dict:
    selected = json.loads((root / "packaging/review/SELECTED-ENGINE.json").read_text(encoding="utf-8"))
    release = json.loads((root / "release/v0.1.0.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "release/quality-v0.1.0.json").read_text(encoding="utf-8"))
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if not (selected["version"] == release["version"] == quality["version"] == project["version"]):
        raise ValueError("Version roles differ")
    if not (selected["wheel_sha256"] == release["artifacts"]["wheel"]["sha256"] == quality["selected_wheel_sha256"]):
        raise ValueError("Selected/download/quality wheel identities differ")
    if quality["performance_wheel_sha256"] != selected["performance_origin_wheel_sha256"]:
        raise ValueError("Historical performance identity differs")
    if quality["performance_applies_to_selected_wheel"] is not False:
        raise ValueError("A new measured cohort is required for a selected-wheel performance claim")
    for field in ("pypi_open", "hosted_api_open", "external_user_study_completed"):
        if quality[field] is not False:
            raise ValueError("Review evidence and distribution status before changing " + field)
    dimensions = [row["dimension"] for row in quality["evidence"]]
    required = {"functional_fidelity", "competitor_performance", "current_wheel_performance",
                "installation", "usability", "scientific_quality", "sustained_service",
                "physical_scaling", "precision_portability"}
    if set(dimensions) != required or len(dimensions) != len(required):
        raise ValueError("Incomplete or duplicate evidence dimensions")
    states = {"qualified_within_scope", "historical_measured", "team_validated", "not_measured", "not_established"}
    if any(row["status"] not in states or not row["scope"].strip() for row in quality["evidence"]):
        raise ValueError("Every evidence row needs a status and scope")
    for name in GUIDES:
        content = (root / name).read_text(encoding="utf-8")
        if any(re.search(pattern, content, re.I) for pattern in STALE):
            raise ValueError("Stale distribution guidance: " + name)
    readme = (root / "README.md").read_text(encoding="utf-8")
    artifact = release["artifacts"]["wheel"]
    expected = artifact["url"] + "#sha256=" + artifact["sha256"]
    if expected not in readme or "tools/install.py --execute" not in readme:
        raise ValueError("README must expose the current verified download and guided setup")
    if any(re.fullmatch(r"\s*(?:python -m )?pip install aureon(?:-inference)?\s*", line)
           for name in GUIDES for line in (root / name).read_text(encoding="utf-8").splitlines()):
        raise ValueError("Do not present an unopened PyPI package as installable")
    return {"passed": True, "guides_checked": len(GUIDES), "evidence_dimensions": len(dimensions),
            "selected_wheel_sha256": selected["wheel_sha256"], "network_used": False,
            "model_calls": 0, "scope": "Distribution consistency, not usability or GPU qualification"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
