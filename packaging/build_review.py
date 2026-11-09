"""Build an allowlisted local review candidate. No installation or publication."""

from __future__ import annotations



import argparse

import ast

import fnmatch

from datetime import datetime, timezone

import hashlib

import importlib.metadata

import importlib.util

import json

import os

import posixpath

from pathlib import Path, PurePosixPath

import re

import subprocess

import sys

import tarfile

import zipfile

from urllib.parse import unquote





ROOT = Path(__file__).resolve().parents[1]

VERSION = "0.1.0"

EPOCH = str(int(datetime(2026, 10, 25, tzinfo=timezone.utc).timestamp()))

# Guarded v2/v3 bind these original source bytes. Their existing docstrings and

# provenance links remain; full attribution/proposed comments are shipped in

# NOTICE and SOURCE-NOTICE-OVERLAYS.json. Do not rewrite qualification hashes.

QUALIFIED_NOTICE_SOURCES = {

    "aureon/catalog.py": "5e82eff095948ba83b6ea40145bac84ece30e788e80bcdd01c9f2a9394908708",

    "aureon/recipe.py": "f1e9ac9e5eb4b15c6548509ccf4a6ed80e2ffa0213653b21beed58361fd27e1d",

    "aureon/checkpoint_bundle.py": "4b7d799f35ef1ce3d19c69bc61d9ccc0cae1023c0b688af1fb06f1592c07dd28",

}

FIXED = ['examples/client/resume_download.py', 'tests/test_resume_download.py', 'tests/test_service_wakeup.py', 'tests/test_service_best_routes.py', 'tests/fixtures/qualified_denoiser_lanes_v1.py', 'packaging/check_memory_core.py', 'packaging/tests/test_memory_core_identity.py', 'tests/test_selected_memory_core.py', 'tests/test_single_engine_default.py', 'tests/test_best_runtime_planning.py', 'tests/test_best_cli.py', 'packaging/BEST-WORKFLOWS.md', 'packaging/freeze_best_core.py', 'packaging/verify_frozen_core.py', 'packaging/tests/test_best_core.py', 'tests/test_best_session.py', 'packaging/check_architecture.py', 'packaging/tests/test_architecture.py', 'CONTRIBUTING.md', 'LICENSE', 'NOTICE', 'README.md', 'SECURITY.md', 'docs/ARCHITECTURE.md', 'docs/CPU-WALKTHROUGH.md', 'docs/DEVELOPER-INTEGRATION.md', 'docs/MODEL-SUPPORT.md', 'docs/README.md', 'docs/RIGHTS.md', 'docs/assets/aureon-banner.svg', 'docs/releases/v0.1.0.md', 'examples/generation/borrowed_temporal.py', 'examples/generation/cpu_walkthrough.py', 'examples/generation/dataloader.py', 'examples/generation/eo_plan.py', 'examples/generation/inspect_generation_input.py', 'examples/generation/model_bridge.py', 'examples/generation/model_pool.py', 'examples/generation/prepare_export.py', 'examples/generation/scaling_cpu.py', 'examples/generation/stac_windows.py', 'examples/generation/terratorch_task.py', 'examples/packaged_scene/aligned-input.json', 'examples/packaged_scene/api_job.py', 'examples/packaged_scene/l2a-input.json', 'examples/packaged_scene/offline_job.py', 'packaging/ASSETS.md', 'packaging/ASYNC-SERVICE.md', 'packaging/BORROWED-ADAPTERS.md', 'packaging/CANDIDATE-QUICKSTART.md', 'packaging/CONCURRENT-GENERATION.md', 'packaging/CONTRIBUTING.md', 'packaging/EO-ROUTES.md', 'packaging/GPU-SCALING.md', 'packaging/INPUTS.md', 'packaging/LIBRARY-SCOPE.md', 'packaging/LOCKED-INSTALL.md', 'packaging/MODEL-BRIDGE.md', 'packaging/MODEL-SUPPORT.md', 'packaging/NEURAL-BATCHING.md', 'packaging/PRECISION.md', 'packaging/PRODUCT-WORKFLOWS.md', 'packaging/PUBLIC-API.md', 'packaging/README.md', 'packaging/REPLICA-SERVICE.md', 'packaging/RESIDENT-WORKFLOWS.md', 'packaging/SCALABLE-POLICY.md', 'packaging/SCALABLE-RUNTIME.md', 'packaging/SECURITY.md', 'packaging/SERVICE-OPERATIONS.md', 'packaging/STAC-WINDOWS.md', 'packaging/START-HERE.md', 'packaging/SUPPORT.md', 'packaging/TERRAMIND-FAMILIES.md', 'packaging/TERRATORCH-TASK.md', 'packaging/TYPING.md', 'packaging/UPSTREAM-TASK.md', 'packaging/build_backend.py', 'packaging/build_review.py', 'packaging/check_implementation.py', 'packaging/check_installed.py', 'packaging/check_library_scope.py', 'packaging/check_types.py', 'packaging/ci_copgen.py', 'packaging/ci_cpu.py', 'packaging/ci_eo.py', 'packaging/ci_extended.py', 'packaging/ci_scalability.py', 'packaging/ci_scaling.py', 'packaging/ci_sources.py', 'packaging/ci_terratorch.py', 'packaging/constraints-linux-py312-cu128.txt', 'packaging/constraints-linux-py312-cu130.txt', 'packaging/core_profile.py', 'packaging/dependency-inventory.json', 'packaging/lock-bootstrap-generation.txt', 'packaging/lock-bootstrap-scene.txt', 'packaging/lock-generation-linux-py312-cu128.txt', 'packaging/lock-linux-py312-cu130.txt', 'packaging/lock-scene-linux-py312-cu130.txt', 'packaging/requirements-build.txt', 'packaging/requirements-generation-linux-cu128.txt', 'packaging/requirements-scene-linux-cu128.txt', 'packaging/requirements-scene-linux-cu130.txt', 'packaging/review/ADAPTER-REMOVAL.json', 'packaging/review/ANNOTATION-TRANSITION.json', 'packaging/review/LICENSE', 'packaging/review/NOTICE', 'packaging/review/REVIEW-STATUS.md', 'packaging/review/SOURCE-NOTICE-OVERLAYS.json', 'packaging/review/THIRD-PARTY-NOTICES.md', 'packaging/runtime-files.json', 'packaging/source_identity.py', 'packaging/tests/test_backend.py', 'packaging/tests/test_build_review.py', 'packaging/tests/test_checkpoint_bundle.py', 'packaging/tests/test_core_profile.py', 'packaging/tests/test_default_distribution.py', 'packaging/tests/test_examples.py', 'packaging/tests/test_installed_alpha.py', 'packaging/tests/test_installed_layout.py', 'packaging/tests/test_library_scope.py', 'packaging/tests/test_library_surface.py', 'packaging/tests/test_manifest_digest.py', 'packaging/tests/test_memory_input.py', 'packaging/tests/test_model_support.py', 'packaging/tests/test_public_api.py', 'packaging/tests/test_replica_metadata.py', 'packaging/tests/test_service_contracts.py', 'packaging/tests/test_service_lifecycle.py', 'packaging/tests/test_service_metadata.py', 'packaging/tests/test_service_package.py', 'packaging/tests/test_session.py', 'packaging/tests/test_supervisor_identity.py', 'packaging/tests/test_terramind_registry.py', 'packaging/tests/test_terramind_service_binding.py', 'packaging/tests/test_typing_surface.py', 'packaging/typecheck/negative.py', 'packaging/typecheck/positive.py', 'packaging/verify_delivery.py', 'pyproject.toml', 'release/v0.1.0.json', 'tests/test_adapter_registry.py', 'tests/test_borrowed_model.py', 'tests/test_copgen_native.py', 'tests/test_encoder_precision.py', 'tests/test_encoder_resident_files.py', 'tests/test_eo.py', 'tests/test_generation_batch.py', 'tests/test_generation_concurrency.py', 'tests/test_generation_dataloader.py', 'tests/test_generation_engine.py', 'tests/test_generation_execution_selection.py', 'tests/test_generation_generic_engine.py', 'tests/test_generation_generic_lanes.py', 'tests/test_generation_lanes.py', 'tests/test_generation_memory_delivery.py', 'tests/test_generation_pair_queue.py', 'tests/test_generation_parallel_engine.py', 'tests/test_gpu_devices.py', 'tests/test_http_range_source.py', 'tests/test_library_contracts.py', 'tests/test_model_bridge.py', 'tests/test_model_pool.py', 'tests/test_pool_scaling.py', 'tests/test_replica_capacity.py', 'tests/test_replica_operations.py', 'tests/test_replica_service.py', 'tests/test_resource_policy.py', 'tests/test_runtime_diagnostics.py', 'tests/test_scalable_pool.py', 'tests/test_scaling_runtime.py', 'tests/test_scene_generation.py', 'tests/test_service_input_builder.py', 'tests/test_service_metrics.py', 'tests/test_service_workflow_inputs.py', 'tests/test_service_workflows.py', 'tests/test_serving_validation.py', 'tests/test_stac.py', 'tests/test_stac_raster.py', 'tests/test_temporal_bridge.py', 'tests/test_terramind_encoder.py', 'tests/test_terramind_encoder_io.py', 'tests/test_terramind_family_generation.py', 'tests/test_terramind_session.py', 'tests/test_terramind_graph_runtime.py', 'tests/test_terramind_denoiser_graph.py', 'tests/test_terramind_native.py', 'tests/test_terramind_upstream.py', 'tests/test_terratorch_task.py', 'tests/test_upstream_task.py', 'tests/test_best_workflows.py', 'tools/check_selected_engine.py', 'tools/test_selected_engine.py', 'packaging/review/SELECTED-ENGINE.json', 'docs/EVALUATION.md', 'docs/FIRST-USE.md', 'docs/PUBLISHING.md', 'release/quality-v0.1.0.json', 'tools/install.py', 'tools/check_distribution.py', 'tools/check_index_metadata.py', 'tools/test_distribution.py']

FORBIDDEN_PARTS = {".git", ".archive", ".venv", "models", "inputs", "runs", "__pycache__", "node_modules"}





def sha(data):

    return hashlib.sha256(data).hexdigest()





def safe_name(name):

    if not isinstance(name, str) or not name or any(ord(c) < 32 or ord(c) == 127 for c in name):

        raise ValueError("Invalid candidate path")

    path = PurePosixPath(name)

    if "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts or str(path) != name.rstrip("/"):

        raise ValueError("Invalid relative candidate path")

    return path





def member(root, name):

    path = safe_name(name)

    resolved = root.joinpath(*path.parts)

    if any(p in FORBIDDEN_PARTS for p in path.parts) or any(part.startswith(".env") for part in path.parts):

        raise ValueError("Forbidden candidate member: " + name)

    if any(p.is_symlink() for p in [resolved, *resolved.parents] if p != root.parent):

        raise ValueError("Symlink candidate members are unsupported: " + name)

    if not resolved.is_file() or not resolved.resolve().is_relative_to(root.resolve()):

        raise ValueError("Missing/nonlocal candidate member: " + name)

    return resolved





def inspect_text(name, data):

    text = data.decode("utf-8")

    # A release inspection guard, not a complete secret or ownership audit.

    if re.search(r"(?:C:[/\\]Users[/\\]|/mnt/c/Users/[A-Za-z0-9]|/root/aureon-runtime/(?:venv|review)-)", text, re.I):

        raise ValueError("Personal runtime path in staged member: " + name)

    if re.search(r"(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", text):

        raise ValueError("Credential-like content in candidate member: " + name)





def verify_qualified_source(name, data):

    spec = importlib.util.spec_from_file_location("aureon_source_identity", ROOT / "packaging/source_identity.py")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module.verify_source(ROOT, name, data, QUALIFIED_NOTICE_SOURCES[name])





def validate_source_links(selected):

    """Require local Markdown/HTML links to resolve inside the staged source."""

    names = {name for _, name, _, _ in selected}

    for _, name, _, data in selected:

        if not name.endswith('.md') and PurePosixPath(name).name != 'NOTICE':

            continue

        text = data.decode('utf-8')

        links = re.findall(r'\]\(([^)]+)\)', text)

        links += re.findall(r'(?:href|src)=["\']([^"\']+)["\']', text)

        for href in links:

            href = href.strip('<>').partition('#')[0]

            if not href or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href):

                continue

            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(href)))

            if target.startswith('../') or target.startswith('/'):

                raise ValueError('Source documentation link escapes distribution: ' + name + ': ' + href)

            if target != '.' and target not in names and not any(n.startswith(target.rstrip('/') + '/') for n in names):

                raise ValueError('Missing source documentation target: ' + name + ': ' + href)





def stage(output, *, profile=None):

    if output.exists() or not output.parent.is_dir():

        raise ValueError("Output must be new with an existing parent")

    runtime = json.loads((ROOT / "packaging/runtime-files.json").read_text(encoding="utf-8"))["files"]

    if len(runtime) != len(set(runtime)) or any(not n.startswith("aureon/") for n in runtime):

        raise ValueError("Invalid runtime allowlist")

    policy_spec = importlib.util.spec_from_file_location("aureon_library_scope", Path(__file__).with_name("check_library_scope.py"))

    policy = importlib.util.module_from_spec(policy_spec)

    policy_spec.loader.exec_module(policy)

    policy.validate_runtime_names(runtime)

    frozen_spec = importlib.util.spec_from_file_location("aureon_frozen_build", Path(__file__).with_name("verify_frozen_core.py"))

    frozen_checker = importlib.util.module_from_spec(frozen_spec)

    frozen_spec.loader.exec_module(frozen_checker)

    frozen_checker.verify_frozen_runtime(ROOT, runtime)

    import tomllib

    project = tomllib.loads(member(ROOT, "pyproject.toml").read_text(encoding="utf-8"))["project"]

    project_text=member(ROOT, 'pyproject.toml').read_text(encoding='utf-8')

    recorded_profile=tomllib.loads(project_text).get('tool',{}).get('aureon',{}).get('distribution-profile','generation')

    profile=recorded_profile if profile is None else profile

    if profile not in ('generation','core'):

        raise ValueError('Choose the generation or core distribution profile')

    if recorded_profile=='core' and profile=='generation':

        raise ValueError('Build the default generation profile from the original source checkout')

    if profile=='core':

        spec=importlib.util.spec_from_file_location('aureon_core_profile', Path(__file__).with_name('core_profile.py'))

        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

        project_text=core.apply_core_profile(project_text)

    extra = [p for p in project["license-files"] if p not in FIXED]

    if any(any(c in p for c in "*?[") for p in project["license-files"]):

        raise ValueError("Legal file paths must be explicit")

    if not extra:

        raise ValueError("Third-party license inventory is required")

    names = list(FIXED) + runtime + extra

    if len(names) != len(set(names)):

        raise ValueError("Duplicate candidate paths")

    overlays = json.loads(member(ROOT, "packaging/review/SOURCE-NOTICE-OVERLAYS.json").read_text(encoding="utf-8"))["overlays"]

    for name, prefix in overlays.items():

        if name.startswith("aureon/_reference/terramind/"):

            raise ValueError("Frozen core cannot receive notice overlays")

        if name not in runtime or not name.endswith(".py") or not isinstance(prefix, str):

            raise ValueError("Overlay must target an allowlisted Python file")

        if not prefix.endswith("\n") or any(line.strip() and not line.lstrip().startswith("#") for line in prefix.splitlines()):

            raise ValueError("Only comment/blank notice prefixes are permitted")

    selected = []

    source_identities = {}

    for name in names:

        original = member(ROOT, name).read_bytes()

        prefix = overlays.get(name, "").encode("utf-8")

        if name in QUALIFIED_NOTICE_SOURCES:

            source_identities[name] = verify_qualified_source(name, original)

            prefix = b""

        # Idempotent when the builder is run from its own staged source tree.

        data = original if prefix and original.startswith(prefix) else prefix + original

        if name=='pyproject.toml' and profile=='core':

            data=project_text.encode('utf-8')

        if prefix and ast.dump(ast.parse(original)) != ast.dump(ast.parse(data)):

            raise ValueError("Notice overlay changed Python syntax tree: " + name)

        inspect_text(name, data)

        selected.append((name, name, original, data))

    validate_source_links(selected)

    output.mkdir()

    source = output / "source"

    source.mkdir()

    entries = []

    for origin, name, original, data in selected:

        target = source / name

        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("xb") as stream:

            stream.write(data)

        entries.append({"origin": origin, "path": name, "original_sha256": sha(original),

                        "staged_sha256": sha(data), "bytes": len(data),

                        "notice_prefix_added": name.endswith('.py') and original != data,

                        "distribution_profile_applied": name=='pyproject.toml' and original != data,

                        "qualified_notice_source_preserved": name in QUALIFIED_NOTICE_SOURCES and sha(original) == QUALIFIED_NOTICE_SOURCES[name],

                        "qualified_source_identity": source_identities.get(name)})

    inventory = {"schema": "aureon.review_source_inventory.v1", "version": VERSION, "distribution_profile":profile,

                 "review_only": True, "published": False, "files": entries}

    (source / "SOURCE-INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Include exactly staged review material in the sdist. Runtime package data

    # remains explicitly selected in pyproject, with no environment/model files.

    manifest = "\n".join("include " + e["path"] for e in entries) + "\ninclude SOURCE-INVENTORY.json\ninclude MANIFEST.in\n"

    (source / "MANIFEST.in").write_text(manifest, encoding="utf-8", newline="\n")

    return source, inventory





def audit_distributions(source, dist, inventory, *, selected=None):

    runtime = {e["path"] for e in inventory["files"] if e["path"].startswith("aureon/")}

    source_paths = {e["path"] for e in inventory["files"]} | {"SOURCE-INVENTORY.json", "MANIFEST.in", "PKG-INFO"}

    staged = {e["path"]: e["staged_sha256"] for e in inventory["files"]}

    generated = {

        "SOURCE-INVENTORY.json": (json.dumps(inventory, indent=2) + "\n").encode(),

        "MANIFEST.in": ("\n".join("include " + e["path"] for e in inventory["files"]) + "\ninclude SOURCE-INVENTORY.json\ninclude MANIFEST.in\n").encode(),

    }

    import tomllib

    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    legal = {n for n in staged if any(fnmatch.fnmatchcase(n, p) for p in project["license-files"])}

    dist_info = "aureon_inference-" + VERSION + ".dist-info"

    sdist_root = "aureon_inference-" + VERSION

    reports = []

    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))

    if selected is None and (len(wheels) != 1 or len(sdists) != 1):

        raise ValueError("Expected one wheel and one sdist")

    for artifact in selected if selected is not None else wheels + sdists:

        if artifact.suffix == ".whl":

            with zipfile.ZipFile(artifact) as archive:

                # ZipInfo may normalize Windows separators or truncate NULs.

                # Validate the original member spelling before normalized names.

                for info in archive.infolist():

                    safe_name(info.orig_filename)

                names = archive.namelist()

                normalized = [str(safe_name(n)) for n in names]

                if len(normalized) != len(set(normalized)):

                    raise ValueError("Duplicate wheel members")

                contents = {name: archive.read(name) for name in names if not name.endswith("/")}

            present = {n for n in contents if n.startswith("aureon/")}

            if present != runtime:

                raise ValueError("Wheel runtime differs from allowlist")

            for name, data in contents.items():

                if not name.startswith("aureon/") and not name.startswith(dist_info + "/"):

                    raise ValueError("Unexpected wheel content: " + name)

                if name in staged and sha(data) != staged[name]:

                    raise ValueError("Wheel bytes differ from staged source")

            license_prefix = dist_info + "/licenses/"

            expected_wheel = runtime | {license_prefix+n for n in legal} | {

                dist_info+"/"+n for n in ("METADATA","WHEEL","entry_points.txt","top_level.txt","RECORD")}

            if set(contents) != expected_wheel:

                raise ValueError("Wheel member set differs from exact distribution allowlist")

            wheel_legal = {n[len(license_prefix):] for n in contents if n.startswith(license_prefix)}

            if wheel_legal != legal:

                raise ValueError("Wheel legal inventory differs from staged declaration")

            for name in legal:

                if sha(contents[license_prefix + name]) != staged[name]:

                    raise ValueError("Wheel legal bytes differ from staged source")

            meta = next(data.decode() for n, data in contents.items() if n.endswith(".dist-info/METADATA"))

            from email.parser import Parser

            metadata=Parser().parsestr(meta)

            expected_requirements=list(project.get('dependencies',[]))

            for extra, requirements in project.get('optional-dependencies',{}).items():

                expected_requirements += [requirement+'; extra == "'+extra+'"' for requirement in requirements]

            normalize=lambda value:re.sub(r'\s+', '', value).replace("'", '"')

            if {normalize(v) for v in metadata.get_all('Requires-Dist',[])} != {normalize(v) for v in expected_requirements}:

                raise ValueError('Wheel dependency metadata differs from staged distribution profile')

            if "Classifier: Development Status :: 3 - Alpha" not in meta or "License-Expression: Apache-2.0" not in meta:

                raise ValueError("Review/license metadata missing")

            for required in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "REVIEW-STATUS.md", "SOURCE-NOTICE-OVERLAYS.json"):

                if not any(n.endswith("/licenses/packaging/review/" + required) for n in contents):

                    raise ValueError("Wheel license material missing: " + required)

        else:

            with tarfile.open(artifact, "r:gz") as archive:

                members = archive.getmembers()

                normalized = [str(safe_name(m.name)) for m in members]

                if len(normalized) != len(set(normalized)) or any(PurePosixPath(n).parts[0] != sdist_root for n in normalized):

                    raise ValueError("Duplicate or unexpected sdist roots/members")

                if any(m.issym() or m.islnk() or not (m.isdir() or m.isfile()) for m in members):

                    raise ValueError("Unexpected sdist member type")

                contents = {str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0])): archive.extractfile(m).read()

                            for m in members if m.isfile()}

            for name, data in contents.items():

                generated_metadata = {"setup.cfg"} | {"aureon_inference.egg-info/"+n for n in (

                    "PKG-INFO","SOURCES.txt","dependency_links.txt","entry_points.txt","requires.txt","top_level.txt")}

                if name not in source_paths | generated_metadata:

                    raise ValueError("Unexpected sdist content: " + name)

                if name in staged and sha(data) != staged[name]:

                    raise ValueError("Sdist bytes differ from staged source")

            if not set(staged).issubset(contents):

                raise ValueError("Sdist is missing staged members")

            if any(contents.get(name) != data for name, data in generated.items()):

                raise ValueError("Sdist generated inventory/manifest differs from staged declaration")

        for name, data in contents.items():

            if any(p in FORBIDDEN_PARTS for p in PurePosixPath(name).parts) or name.endswith((".pt", ".bin", ".npy", ".tif", ".png", ".pyc")):

                raise ValueError("Forbidden archive member: " + name)

            inspect_text(name, data)

        reports.append({"file": artifact.name, "bytes": artifact.stat().st_size,

                        "sha256": sha(artifact.read_bytes()), "member_count": len(contents)})

    return reports





def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--profile", choices=('generation','core'))

    args = parser.parse_args(argv)

    if importlib.metadata.version("setuptools") != "84.2.1":

        parser.error("Use a separate build environment with setuptools==84.2.1; do not modify an existing runtime")

    output = args.output.absolute()

    source, inventory = stage(output, profile=args.profile)

    dist = output / "dist"

    dist.mkdir()

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH=EPOCH)

    with (output / "build.log").open("xb") as log:

        # Each PEP 517 hook gets a fresh process: setuptools commands retain

        # distribution options when unrelated hooks share one interpreter.

        for hook in ("build_sdist", "build_wheel"):

            code = "from setuptools import build_meta; import sys; getattr(build_meta,sys.argv[1])(sys.argv[2])"

            subprocess.run([sys.executable, "-B", "-c", code, hook, str(dist)], cwd=source, env=env,

                           stdout=log, stderr=subprocess.STDOUT, check=True)

    report = {"schema": "aureon.review_build.v1", "status": "passed", "published": False,

              "distribution_profile":inventory['distribution_profile'],

              "version": VERSION, "setuptools": importlib.metadata.version("setuptools"),

              "source_files": len(inventory["files"]), "artifacts": audit_distributions(source, dist, inventory),

              "scope": "Allowlist, identity and archive content checks; installation/model acceptance is recorded separately"}

    with (output / "build-report.json").open("x", encoding="utf-8") as stream:

        json.dump(report, stream, indent=2)

        stream.write("\n")

    print(json.dumps(report))

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

