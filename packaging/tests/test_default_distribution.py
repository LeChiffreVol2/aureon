"""Default-install contract and migration diagnostics; no model execution."""
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class DefaultDistribution(unittest.TestCase):
    def test_generation_is_default_and_alias_does_not_add_conflicts(self):
        configuration = tomllib.loads((ROOT/'pyproject.toml').read_text())
        project = configuration['project']
        pins = [line for line in (ROOT/'packaging/requirements-generation-linux-cu128.txt').read_text().splitlines()
                if line and not line.startswith('#')]
        self.assertEqual(project['version'], '0.1.0')
        core = configuration.get('tool',{}).get('aureon',{}).get('distribution-profile') == 'core'
        if core:
            self.assertEqual(project['dependencies'], [])
            self.assertEqual(set(project['optional-dependencies']['generation']), set(pins))
            self.assertEqual(project['optional-dependencies']['geo'], ['numpy==2.2.6','rasterio==1.5.2'])
        else:
            self.assertEqual(set(project['dependencies']), set(pins))
            self.assertEqual(project['optional-dependencies']['generation'], [])
            self.assertEqual(project['optional-dependencies']['geo'], [])
        self.assertNotIn('scene', project['optional-dependencies'])
        self.assertNotIn('acquire', project['optional-dependencies'])
        self.assertNotIn('vllm', project['optional-dependencies'])
        self.assertEqual(set(project['optional-dependencies']['scene-legacy']),
                         set(pins+['torchgeo==0.9.0']) if core else {'torchgeo==0.9.0'})

    def test_profile_remedies_do_not_recommend_removed_extra(self):
        from aureon.api.profiles import inspect_profile
        with patch('aureon.api.profiles.metadata.version', return_value='missing'):
            scene = inspect_profile('scene')
            generation = inspect_profile('generation')
        self.assertNotIn('aureon-inference[scene]', scene['remedy'])
        self.assertIn('original GitHub v0.1.0', scene['remedy'])
        self.assertIn('cannot satisfy', scene['remedy'])
        self.assertIn('install aureon-inference from', generation['remedy'])
        self.assertEqual(scene['status'], 'failed')

    def test_pypi_description_links_are_absolute(self):
        import re
        project = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
        for target in re.findall(r'\]\(([^)]+)\)', (ROOT/project['readme']).read_text(encoding='utf-8')):
            self.assertTrue(target.startswith(('https://', 'mailto:')), target)
        self.assertGreaterEqual(len(project['urls']), 4)


if __name__ == '__main__': unittest.main()
