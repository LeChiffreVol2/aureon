"""Installed-entrypoint and service dependency boundaries, without model execution."""
import subprocess
import sys
import unittest


class ServicePackage(unittest.TestCase):
    def test_service_and_client_help_without_optional_runtime(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torchvision','terratorch','numpy','rasterio','vllm','diffusers'):
   raise AssertionError('Unexpected runtime import: '+fullname)
sys.meta_path.insert(0,Block())
from aureon.serving.__main__ import main as server
from aureon.serving.client import main as client
for main,args in [(server,['--help']),(server,['serve','--help']),(server,['worker','--help']),
                  (server,['init','--help']),(client,['--help'])]:
 try: main(args)
 except SystemExit as exc: assert exc.code==0
'''
        run = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_route_rejects_wrong_model_environment_before_model_import(self):
        from aureon.cloud.control import Error
        from aureon.serving.control import route
        for args in [('kao-terramind-a','scene','fp32-full-reference-v1'),
                     ('terramind-v1-base','scene','fp32-generation-v1'),
                     ('terramind-v1-base','generation','fp16-generation')]:
            with self.assertRaises((Error,ValueError)):
                route(*args)
        value=route('terramind-v1-base','generation','fp32-generation-v1',denoiser='native')
        self.assertEqual(value['denoiser'],'native')
        self.assertNotIn('torch',sys.modules)


if __name__ == '__main__': unittest.main()
