"""Build/checker bridge: validate pinned bytes before allowing a frozen exception."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def verify_frozen_runtime(root: Path, names: list[str]) -> dict[str, object]:
    prefix = "aureon/_reference/terramind/"
    selected = [name for name in names if name.startswith(prefix)]
    if not selected and not (root / prefix).exists():
        return {"frozen_python_paths": [], "verified": False}
    path = Path(__file__).resolve().parents[1] / "aureon/_reference/integrity.py"
    spec = importlib.util.spec_from_file_location("_aureon_frozen_integrity_check", path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing frozen integrity verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        fingerprint = module.verify_core(root)
        required = {prefix + name for name in module.FILENAMES} | {prefix + "SOURCE.json", prefix + "NOTICE.md"}
        if not required.issubset(names):
            raise ValueError("Frozen closure incomplete in runtime allowlist")
        return {"verified": True, "frozen_python_paths": list(fingerprint.frozen_python_paths),
                **fingerprint.metadata()}
    finally:
        sys.modules.pop(spec.name, None)
