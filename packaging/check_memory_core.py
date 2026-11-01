"""Audit the active memory numerical span against the immutable selected source.

Only annotations, six explicitly bound provider names and the documented token
delivery observer differ. This is operation identity, not measured GPU parity.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ALIASES = {"schedule": "schedule_compatibility", "rng_context": "rng_capsule",
           "reuse_context": "exact_conditioning_reuse", "decision_function": "_decision",
           "graph_factory": "DenoiserGraph", "graph_eligibility": "eligibility"}


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Name(id=ALIASES.get(node.id, node.id), ctx=node.ctx)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            if len(node.args) != 2 or not isinstance(node.args[0], ast.Constant):
                raise ValueError("Unexpected executable cast boundary")
            return self.visit(node.args[1])
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None
        return ast.Assign(targets=[self.visit(node.target)], value=self.visit(node.value))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        allowed = ((node.level, node.module, tuple(a.name for a in node.names)) in
                   ((0, "contextlib", ("AbstractContextManager",)),
                    (1, "denoiser_graph", ("DenoiserGraph", "eligibility"))))
        if not allowed:
            raise ValueError("Unexpected import inside numerical span")
        return None


def method(tree: ast.Module, name: str) -> ast.FunctionDef:
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_NativeRuntime")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def check_provider_bindings(body: list[ast.stmt], stop: int) -> None:
    providers = {"schedule": "schedule", "rng_context": "rng", "reuse_context": "reuse",
                 "decision_function": "decision", "graph_factory": "graph_factory",
                 "graph_eligibility": "graph_eligibility"}
    expected = {"core": "core = self.numerical_core"}
    expected.update({alias: f"{alias} = core.{field} if core is not None else {ALIASES[alias]}"
                     for alias, field in providers.items()})
    for name, expression in expected.items():
        bindings = [node for node in body if assigned(node, name)]
        before_span = [node for node in body[:stop] if assigned(node, name)]
        if (len(bindings) != 1 or len(before_span) != 1
                or ast.dump(bindings[0]) != ast.dump(ast.parse(expression).body[0])):
            raise ValueError("Selected provider binding differs: " + name)


def normalized_span(tree: ast.Module, *, memory: bool) -> str:
    body = method(tree, "run").body
    start = next(i for i, n in enumerate(body) if assigned(n, "t"))
    if memory:
        check_provider_bindings(body, start)
    stop = next(i for i, n in enumerate(body) if assigned(n, "decoding_steps_restored")) + 1
    module = Normalize().visit(ast.Module(body=deepcopy(body[start:stop]), type_ignores=[]))
    expected = ('trace_memory(self.model) if delivery == "memory" else '
                'trace_native(self.model, output, detailed=capture_logits)' if memory else
                'trace_native(self.model, output, detailed=capture_logits)')
    expected_node = ast.parse(expected, mode="eval").body
    replaced = 0
    for node in ast.walk(module):
        if isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name) and node.optional_vars.id == "trace":
            if ast.dump(node.context_expr) != ast.dump(expected_node):
                raise ValueError("Token delivery observer differs from declared boundary")
            node.context_expr = ast.Name(id="DECLARED_TOKEN_DELIVERY_OBSERVER", ctx=ast.Load())
            replaced += 1
    if replaced != 1:
        raise ValueError("Expected exactly one token observation context")
    return ast.dump(module, include_attributes=False)


def inspect(root: Path) -> dict[str, object]:
    original_path = root / "aureon/_reference/terramind/session.py"
    active_path = root / "aureon/terramind_native/session.py"
    original, active = ast.parse(original_path.read_bytes()), ast.parse(active_path.read_bytes())
    frozen, adapted = normalized_span(original, memory=False), normalized_span(active, memory=True)
    if frozen != adapted:
        raise ValueError("Active numerical span differs from selected source")
    guards = ("_policy", "_assert_policy", "_versions", "_sync", "_memory")
    for name in guards:
        a = Normalize().visit(ast.Module(body=deepcopy(method(original, name).body), type_ignores=[]))
        b = Normalize().visit(ast.Module(body=deepcopy(method(active, name).body), type_ignores=[]))
        if ast.dump(a) != ast.dump(b):
            raise ValueError("Policy/identity guard differs: " + name)
    digest = hashlib.sha256(frozen.encode()).hexdigest()
    bindings = ast.parse((root / "aureon/terramind_native/selected_core.py").read_bytes())
    recorded = next(n.value.value for n in bindings.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "NUMERICAL_SPAN_SHA256" for t in n.targets)
                    and isinstance(n.value, ast.Constant))
    if recorded != digest:
        raise ValueError("Recorded numerical span identity differs")
    return {"passed": True, "numerical_span_sha256": digest, "policy_guards": list(guards),
            "frozen_session_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "active_session_sha256": hashlib.sha256(active_path.read_bytes()).hexdigest(),
            "normalization": "typing casts/annotations, six verified selected-provider bindings, exact declared token delivery observer",
            "delivery_difference": "Memory tokens are hashed in RAM; original tokens are persisted as NPY. Neither observer changes the sampler return.",
            "scope": "Single-request operation identity; lane orchestration and active asset loader are explicit adapters, not whole-function equality.",
            "model_calls": 0, "performance_qualification": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.source)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
