"""Standard-library source, documentation and release-identity checks. No network."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = {
    'README.md', 'LICENSE', 'NOTICE', 'SECURITY.md', 'CONTRIBUTING.md', '.gitignore', '.gitattributes',
    'docs/README.md', 'docs/MODEL-SUPPORT.md', 'docs/RIGHTS.md', 'docs/releases/v0.1.0.md',
    'release/v0.1.0.json', 'tools/check_public_source.py',
    '.github/ISSUE_TEMPLATE/bug_report.md', '.github/ISSUE_TEMPLATE/feature_request.md',
    '.github/ISSUE_TEMPLATE/config.yml', '.github/pull_request_template.md',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-release-runtime', action='store_true',
                        help='Also require runtime/lock bytes to match the immutable v0.1.0 release')
    parser.add_argument('--report', type=Path, help='Optional new JSON receipt outside the repository')
    args = parser.parse_args()
    if args.report and (args.report.exists() or args.report.resolve().is_relative_to(ROOT)):
        parser.error('Report must be a new path outside the repository')
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    assert project['name'] == 'aureon-inference' and project['version'] == '0.1.0'
    tree = ast.parse((ROOT / 'packaging/build_review.py').read_text(encoding='utf-8'))
    fixed = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'FIXED' for t in node.targets))
    runtime = json.loads((ROOT / 'packaging/runtime-files.json').read_text(encoding='utf-8'))['files']
    allowed = set(fixed) | set(runtime) | set(project['license-files']) | PUBLIC_FILES
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(ROOT).parts}
    assert set(actual) == allowed, {'unexpected': sorted(set(actual) - allowed), 'missing': sorted(allowed - set(actual))}
    forbidden = {'runs', 'build', 'dist', '.venv', '__pycache__', 'weights', 'models', 'inputs', 'node_modules'}
    link_count, python_count, doc_count = 0, 0, 0
    for name, path in actual.items():
        assert not forbidden.intersection(path.relative_to(ROOT).parts), name
        assert not any(part.endswith('.egg-info') for part in path.parts), name
        assert not path.is_symlink(), name
        assert path.suffix.lower() not in {'.whl', '.pt', '.pth', '.safetensors', '.tif', '.tiff', '.npy', '.npz', '.sqlite', '.db'}, name
        text = path.read_text(encoding='utf-8')
        # Real credential-shaped strings and personal checkout paths are excluded.
        assert not re.search(r'(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)', text), name
        if path.suffix == '.py':
            ast.parse(text, filename=name)
            python_count += 1
        if path.suffix == '.md' or path.name == 'NOTICE':
            doc_count += 1
            assert not re.search(r'(?:C:[/\\]Users[/\\]|/mnt/[a-z]/Users/|/root/|codex://|plugin://|(?:\.\./)*runs/)', text), name
            for href in re.findall(r'\]\(([^)]+)\)', text):
                href = href.strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href):
                    assert href.startswith(('https://', 'http://', 'mailto:')), (name, href)
                    continue
                relative, _, fragment = href.partition('#')
                target = (path.parent / unquote(relative)).resolve() if relative else path
                assert target.is_relative_to(ROOT) and target.exists(), (name, href)
                if fragment and target.is_file() and target.suffix == '.md':
                    headings = re.findall(r'^#{1,6}\s+(.+?)\s*#*$', target.read_text(encoding='utf-8'), re.M)
                    anchors = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
                    assert unquote(fragment) in anchors, (name, href)
                link_count += 1
    manifest = json.loads((ROOT / 'release/v0.1.0.json').read_text(encoding='utf-8'))
    assert set(runtime) == set(manifest['runtime_payloads'])
    for name, expected in manifest['dependency_locks'].items():
        if args.verify_release_runtime:
            assert digest(ROOT / name) == expected, name
    if args.verify_release_runtime:
        for name, expected in manifest['runtime_payloads'].items():
            assert digest(ROOT / name) == expected, name
    assert (ROOT / 'LICENSE').read_bytes() == (ROOT / 'packaging/review/LICENSE').read_bytes()
    attrs = (ROOT / '.gitattributes').read_text()
    assert 'aureon/** -text' in attrs and 'packaging/lock-*.txt -text' in attrs
    result = {'passed': True, 'files': len(actual), 'python_files_parsed': python_count,
              'public_documents_checked': doc_count, 'local_links_and_anchors_checked': link_count,
              'runtime_payloads': len(runtime), 'release_runtime_verified': args.verify_release_runtime,
              'network_used': False, 'model_executed': False, 'benchmark_run': False}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
