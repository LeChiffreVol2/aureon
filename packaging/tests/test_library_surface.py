"""Standard-library public import and stream error contracts; no model imports."""
from pathlib import Path
import subprocess
import sys
import unittest

from aureon.api.generation_queue import GenerationStream


class LibrarySurface(unittest.TestCase):
    def test_exports_and_unused_engine_are_lazy(self):
        code = '''import importlib.abc,sys,tempfile
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','rasterio','terratorch','diffusers','transformers'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from aureon import api
assert len(api.__all__)==len(set(api.__all__))
assert all(hasattr(api,name) for name in api.__all__)
assert {'GenerationEngine','GenerationInput','GenerationJob','GenerationStream','plan_placement'} <= set(api.__all__)
with tempfile.TemporaryDirectory() as folder:
 work=Path(folder)/'unused'
 with api.GenerationEngine(model_file='missing',tokenizer_file='missing',work=work,execute_model=True,parallelism=10**100) as engine:
  assert engine.capabilities()['parallel']['planned']==4
 assert not work.exists()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_producer_error_survives_iterator_cleanup(self):
        class Handle:
            discarded = False
            def discard(self): self.discarded = True
        handle = Handle()
        class Queue:
            def reserve(self): return handle
        class Engine:
            def _submission_queue(self): return Queue()
        class Source:
            closed = 0
            def __iter__(self): return self
            def __next__(self): raise ValueError('bad source item')
            def close(self):
                self.closed += 1
                raise RuntimeError('cleanup failed')
        source = Source()
        stream = GenerationStream(Engine(), source, max_in_flight=1, adapter=None, options={})
        with self.assertRaisesRegex(ValueError, 'bad source item') as failure:
            next(stream)
        self.assertTrue(handle.discarded)
        self.assertIn('cleanup failed', failure.exception.__notes__[0])
        stream.close()
        self.assertEqual(source.closed, 1)


if __name__ == '__main__':
    unittest.main()
