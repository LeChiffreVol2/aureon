"""Verify reviewed source candidates without replacing qualified hashes.

The original hashes identify prior qualified artifacts. Candidate source may
change type declarations only, and requires its exact hash plus an executable
AST/configuration digest anchored to the original source. This is a build-time
source review, never fresh model qualification or acceptance of new constants.
The separately recorded adapter-removal transition changes public behavior and
is explicitly returned as an unqualified candidate, not an annotation change.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def executable_digest(data: bytes) -> str:
    class Annotations(ast.NodeTransformer):
        class_depth = 0
        function_depth = 0

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
            if node.module in {"typing", "__future__", "value_types", "types"}:
                return None
            return node

        def visit_If(self, node: ast.If) -> ast.AST | None:
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                return None
            return self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            self.class_depth += 1
            result = self.generic_visit(node)
            self.class_depth -= 1
            return result

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.returns = None
            node.type_comment = None
            self.function_depth += 1
            result = self.generic_visit(node)
            self.function_depth -= 1
            return result

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.annotation = None
            node.type_comment = None
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
            if self.class_depth and not self.function_depth:
                # Dataclass field names, order and defaults remain executable.
                node.annotation = ast.Constant(value="type annotation")
                return self.generic_visit(node)
            if node.value is None:
                return None
            return self.visit(ast.Assign(targets=[node.target], value=node.value))

        def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "cast" and len(node.value.args) == 2
                    and isinstance(node.value.args[1], ast.Name)
                    and node.targets[0].id == node.value.args[1].id):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if isinstance(node.func, ast.Name) and node.func.id == "cast" and len(node.args) == 2:
                return self.visit(node.args[1])
            return self.generic_visit(node)

    tree = Annotations().visit(ast.parse(data))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def verify_source(root: Path, name: str, data: bytes, qualified_sha256: str) -> dict[str, object]:
    actual = hashlib.sha256(data).hexdigest()
    if actual == qualified_sha256:
        return {"identity": "original_qualified_bytes", "original_sha256": qualified_sha256,
                "candidate_sha256": actual, "new_gpu_qualification": False}
    refactor_path = root / "packaging/review/ADAPTER-REMOVAL.json"
    if refactor_path.is_file():
        refactor = json.loads(refactor_path.read_text(encoding="utf-8"))
        row = refactor.get("files", {}).get(name)
        if row is not None:
            if (refactor.get("schema") != "aureon.adapter_removal.v1"
                    or refactor.get("new_gpu_qualification") is not False
                    or row.get("original_sha256") != qualified_sha256
                    or row.get("candidate_sha256") != actual
                    or row.get("executable_ast_sha256") != executable_digest(data)):
                raise ValueError("Unreviewed adapter-removal source: " + name)
            return {"identity": "behavior_changed_unqualified_candidate", **row,
                    "new_gpu_qualification": False}
    ledger = json.loads((root / "packaging/review/ANNOTATION-TRANSITION.json").read_text(encoding="utf-8"))
    row = ledger["files"].get(name)
    if (ledger.get("schema") != "aureon.annotation_source_transition.v1" or not row
            or row["original_sha256"] != qualified_sha256 or row["candidate_sha256"] != actual
            or row["executable_ast_sha256"] != executable_digest(data)):
        raise ValueError("Unreviewed qualified-source transition: " + name)
    return {"identity": "annotation_only_candidate", **row, "new_gpu_qualification": False}
