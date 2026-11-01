"""Public service factory routing, without loading any model."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from aureon.serving.backend import Backend
from aureon.serving.control import route
from aureon.cloud.control import file_hash


class StopBeforeModel(RuntimeError):
    pass


class BestServiceRouteTests(unittest.TestCase):
    def test_tile_recommended_base_and_explicit_family_routes(self):
        for model,denoiser,expected in (
            ('terramind-v1-base','graph',{'execution_profile':'recommended'}),
            ('terramind-v1-base','native',{'execution_profile':'reference'}),
        ):
            with self.subTest(model=model,denoiser=denoiser):
                parent=Path(os.environ.get('AUREON_SERVICE_TEST_ROOT',tempfile.gettempdir()))
                parent.mkdir(parents=True,exist_ok=True)
                work=Path(tempfile.mkdtemp(prefix='best-route-',dir=parent))
                config=work/'config.json';config.write_text('{}')
                manifest=work/'manifest.json';manifest.write_text('{}')
                backend=Backend.__new__(Backend)
                backend.owner=threading.get_ident();backend._closed=False;backend.workflow=None
                backend.work=work;backend.config=config;backend.config_hash=file_hash(config);backend.session=None
                backend.route=route(model,'generation','fp32-generation-v1',denoiser=denoiser)
                with patch('aureon.api.open_model_session',side_effect=StopBeforeModel) as factory:
                    with self.assertRaises(StopBeforeModel):
                        backend.run({'job_id':'job','attempt':1,'manifest':str(manifest),'token':'test',
                                     'request':{'parameters':{'seed':17}}})
                args,kwargs=factory.call_args
                self.assertEqual(args,(model,));self.assertTrue(kwargs['execute_model'])
                self.assertEqual({k:kwargs[k] for k in ('execution_profile','reuse','denoiser') if k in kwargs},expected)
                from aureon.api.models import _generation_dispatch
                dispatch=_generation_dispatch(model,'file',kwargs.get('execution_profile'),kwargs.get('reuse'),kwargs.get('denoiser'))
                if model=='terramind-v1-base':
                    self.assertEqual(dispatch.runtime,'selected_core')
                    self.assertEqual(dispatch.profile,'best_fp32' if denoiser=='graph' else 'reference_fp32')


if __name__=='__main__':unittest.main()
