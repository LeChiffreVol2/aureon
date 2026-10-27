"""Local immutable binding helpers. No model imports or network access."""
import hashlib
import json
from pathlib import Path
import re

INPUT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MODEL_SHA = "83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec"
TOKENIZER_SHA = "d83e43ccbdbfb787724d42a4e09dce6abb122e8f286889a379af59ffbe979ed1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 2 * 1024**2:
        raise ValueError("JSON must be nonempty and at most 2 MiB")
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError("Duplicate JSON key")
            result[k] = v
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_file(item, base):
    if not isinstance(item, dict) or not re.fullmatch("[0-9a-f]{64}", item.get("sha256", "")):
        raise ValueError("A local file and its SHA-256 are required")
    name = item.get("path", "")
    if not isinstance(name, str) or not name or "://" in name or name.startswith("/vsi"):
        raise ValueError("Only explicit local files are admitted")
    path = (Path(base) / name).resolve(strict=True)
    if not path.is_file() or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ValueError("File type/size mismatch: " + str(path))
    if file_hash(path) != item["sha256"]: raise ValueError("File hash mismatch: " + str(path))
    return path


def binding(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": file_hash(path), "bytes": path.stat().st_size}


def validate_recipe(recipe):
    if set(recipe) != {"token_steps", "diffusion_steps", "temperature", "top_p", "top_k", "seed"}:
        raise ValueError("Supply the exact explicit native sampling recipe")
    if type(recipe["token_steps"]) is not int or recipe["token_steps"] not in (1, 8):
        raise ValueError("This finite slice admits 1 or 8 token steps only")
    if type(recipe["diffusion_steps"]) is not int or not 1 <= recipe["diffusion_steps"] <= 50:
        raise ValueError("Diffusion steps must be 1..50")
    if recipe["temperature"] != 1.0 or recipe["top_p"] != 0.8 or type(recipe["top_k"]) is not int or recipe["top_k"] != 0:
        raise ValueError("Only explicit upstream temperature=1, top_p=.8, top_k=0 is supported")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**63:
        raise ValueError("Explicit nonnegative 63-bit seed required")
    return dict(recipe)
