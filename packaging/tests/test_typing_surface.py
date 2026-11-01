"""Keep shipped stubs aligned with callable shape and facade exports."""
import ast
import inspect
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def definitions(tree):
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}


def shape(function):
    args = function.args
    return ([n.arg for n in args.posonlyargs], [n.arg for n in args.args],
            [n.arg for n in args.kwonlyargs], args.vararg.arg if args.vararg else None,
            args.kwarg.arg if args.kwarg else None, len(args.defaults),
            [n is not None for n in args.kw_defaults])


class TypingSurfaceTests(unittest.TestCase):
    def test_facade_stub_reexports_exact_runtime_symbols(self):
        def exports(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            bindings = {}
            lazy = next((node for node in tree.body if isinstance(node, ast.AnnAssign)
                         and isinstance(node.target, ast.Name) and node.target.id == "_EXPORTS"), None)
            if lazy is not None:
                return {(len(module) - len(module.lstrip(".")), module.lstrip("."), attribute, name)
                        for name, (module, attribute) in ast.literal_eval(lazy.value).items()}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        bindings[alias.asname or alias.name] = (node.level, node.module, alias.name)
                elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name)
                      and node.value.id in bindings):
                    # Public class/function aliases in .pyi are assignments so
                    # strict consumers can use the class alias as a type too.
                    bindings[node.targets[0].id] = bindings[node.value.id]
            return {(*binding, name) for name, binding in bindings.items() if not name.startswith("_")}
        self.assertEqual(exports(ROOT / "aureon/api/__init__.py"),
                         exports(ROOT / "aureon/api/__init__.pyi"))
        self.assertTrue((ROOT / "aureon/py.typed").is_file())

    def test_declared_stub_methods_match_runtime_call_shape(self):
        for stub in (ROOT / "aureon").rglob("*.pyi"):
            runtime = stub.with_suffix(".py")
            self.assertTrue(runtime.is_file(), str(stub))
            declared = definitions(ast.parse(stub.read_text(encoding="utf-8")))
            actual = definitions(ast.parse(runtime.read_text(encoding="utf-8")))
            for name, node in declared.items():
                if stub.name == "best.pyi" and name == "BestSessionBusy":
                    from aureon.api.best import BestSessionBusy
                    from aureon._reference.terramind.session import SessionBusy
                    self.assertIs(BestSessionBusy, SessionBusy)
                    continue
                self.assertIn(name, actual, str(stub))
                members = definitions(node) if isinstance(node, ast.ClassDef) else {name: node}
                real = definitions(actual[name]) if isinstance(node, ast.ClassDef) else actual
                for method, signature in members.items():
                    # Literal dispatch overloads and **kwargs forwarding intentionally
                    # narrow the public interface; consumer fixtures cover them.
                    if method in {"open_model_session", "from_token_file"}:
                        continue
                    with self.subTest(stub=str(stub), member=name + "." + method):
                        if stub.name == "best.pyi" and method not in real:
                            from aureon.api.best import BestTerraMindSession
                            from aureon._reference.terramind.session import TerraMindSession
                            function = getattr(BestTerraMindSession, method)
                            self.assertIs(function, getattr(TerraMindSession, method))
                            parameters = inspect.signature(function).parameters
                            args = signature.args
                            declared_names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
                            self.assertTrue(declared_names.issubset(parameters))
                            for parameter in parameters.values():
                                if parameter.name not in declared_names:
                                    self.assertIsNot(parameter.default, inspect.Parameter.empty)
                            continue
                        self.assertIn(method, real)
                        self.assertEqual(shape(signature), shape(real[method]))


if __name__ == "__main__":
    unittest.main()
