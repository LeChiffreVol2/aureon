"""Operation identity and negative mutation check; no runtime/model imports."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("memory_audit", ROOT / "packaging/check_memory_core.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class MemoryIdentityTests(unittest.TestCase):
    def test_numerical_span_and_policy_guards_match_selected_source(self):
        self.assertTrue(AUDIT.inspect(ROOT)["passed"])

    def test_a_changed_numerical_operation_is_not_normalized_away(self):
        tree = ast.parse((ROOT / "aureon/terramind_native/session.py").read_bytes())
        changed = deepcopy(tree)
        for node in ast.walk(AUDIT.method(changed, "run")):
            if isinstance(node, ast.keyword) and node.arg == "standardize":
                node.value = ast.Constant(value=False)
        self.assertNotEqual(AUDIT.normalized_span(tree, memory=True), AUDIT.normalized_span(changed, memory=True))

    def test_observer_replacement_is_exact_not_arbitrary_suppression(self):
        tree = ast.parse((ROOT / "aureon/terramind_native/session.py").read_bytes())
        for node in ast.walk(AUDIT.method(tree, "run")):
            if isinstance(node, ast.Name) and node.id == "trace_memory":
                node.id = "some_other_forward"
        with self.assertRaisesRegex(ValueError, "observer"):
            AUDIT.normalized_span(tree, memory=True)

    def test_changed_provider_binding_is_rejected(self):
        tree = ast.parse((ROOT / "aureon/terramind_native/session.py").read_bytes())
        for node in AUDIT.method(tree, "run").body:
            if AUDIT.assigned(node, "rng_context"):
                node.value = ast.Name(id="rng_capsule", ctx=ast.Load())
        with self.assertRaisesRegex(ValueError, "provider binding"):
            AUDIT.normalized_span(tree, memory=True)


if __name__ == "__main__":
    unittest.main()
