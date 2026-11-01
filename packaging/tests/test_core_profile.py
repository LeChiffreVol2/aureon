"""Core is an explicit alternative build, never a silent change of native pins."""
import importlib.util
from pathlib import Path
import tomllib
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('core_profile_test',ROOT/'packaging/core_profile.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


class CoreProfileTests(unittest.TestCase):
    def test_stage_binds_core_metadata_and_preserves_source(self):
        spec=importlib.util.spec_from_file_location('core_builder_test',ROOT/'packaging/build_review.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        original=(ROOT/'pyproject.toml').read_bytes()
        with tempfile.TemporaryDirectory() as scratch:
            source,inventory=builder.stage(Path(scratch)/'core',profile='core')
            self.assertEqual(inventory['distribution_profile'],'core')
            self.assertEqual(tomllib.loads((source/'pyproject.toml').read_text())['project']['dependencies'],[])
            row=next(r for r in inventory['files'] if r['path']=='pyproject.toml')
            self.assertFalse(row['notice_prefix_added'])
            self.assertEqual(row['staged_sha256'],builder.sha((source/'pyproject.toml').read_bytes()))
            with patch.object(builder,'ROOT',source):
                rebuilt,repeated=builder.stage(Path(scratch)/'restaged')
                self.assertEqual(repeated['distribution_profile'],'core')
                self.assertEqual((source/'pyproject.toml').read_bytes(),(rebuilt/'pyproject.toml').read_bytes())
                with self.assertRaises(ValueError):builder.stage(Path(scratch)/'wrong',profile='generation')
        self.assertEqual((ROOT/'pyproject.toml').read_bytes(),original)

    def test_native_dependency_table_is_preserved_in_generation_extra(self):
        original=(ROOT/'pyproject.toml').read_text()
        before=tomllib.loads(original)
        result=core.apply_core_profile(original);after=tomllib.loads(result)
        self.assertEqual(after['project']['dependencies'],[])
        original_pins=before['project']['dependencies'] or before['project']['optional-dependencies']['generation']
        self.assertEqual(after['project']['optional-dependencies']['generation'],original_pins)
        self.assertEqual(after['project']['optional-dependencies']['borrowed-torch211'],['numpy==2.2.6','torch==2.11.0'])
        self.assertNotIn('torchvision',repr(after['project']['optional-dependencies']['borrowed-torch211']))
        self.assertEqual((before['project']['name'],before['project']['version']),
                         (after['project']['name'],after['project']['version']))
        self.assertEqual(core.apply_core_profile(result),result)
        self.assertEqual((ROOT/'pyproject.toml').read_text(),original)

    def test_only_explicit_profile_overrides_are_accepted(self):
        self.assertIsNone(core.select_build_profile(None))
        self.assertEqual(core.select_build_profile({'aureon-profile':'core'}),'core')
        for settings in ({'aureon-profile':'anything'}, {'arbitrary':'core'},
                         {'aureon-profile':'core','other':'option'}, {'aureon-profile':['core']}):
            with self.assertRaises(ValueError):core.select_build_profile(settings)


if __name__=='__main__':unittest.main()
