"""Resident file workflows: actual CPU arrays and files, no model weights."""
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest
import weakref
import zipfile
from unittest.mock import patch

import numpy as np
import torch

from aureon.api.encoder_io import encode_file_with_engine, inspect_encoder_input, _export_features
from aureon.api.terramind_encoder import TerraMindEncoder


class ResidentEncoderFiles(unittest.TestCase):
    def engine(self):
        return TerraMindEncoder('terramind-v1-tiny-backbone', model_file='unused.pt',
                                modalities=['S2L2A'], execute_model=True)

    def source(self, root, value=None):
        path = root/'input.npz'
        np.savez(path, S2L2A=np.zeros((1,12,224,224), dtype=np.float32) if value is None else value)
        return path

    def test_two_exports_reuse_caller_engine_and_preserve_numeric_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.source(root)
            engine = self.engine()
            calls = []
            def predict(inputs, *, seed):
                calls.append(seed)
                np.testing.assert_array_equal(inputs['S2L2A'].numpy(), np.zeros((1,12,224,224),dtype=np.float32))
                return {'features':[torch.arange(12,dtype=torch.float32).reshape(1,3,4)],
                        'provenance':{'synthetic_test':True, 'seed':seed}}
            with patch.object(engine, 'predict', side_effect=predict), patch.object(engine, 'close') as close:
                for i in range(2):
                    result = encode_file_with_engine(engine, input_file=source, output=root/f'out{i}', seed=17+i,
                                                     execute_model=True)
                    self.assertEqual(result['engine_ownership'], 'caller')
                    with np.load(root/f'out{i}'/'features.npz',allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['layer_000_merged'],np.arange(12,dtype=np.float32).reshape(1,3,4))
                    self.assertEqual(json.loads((root/f'out{i}'/'provenance.json').read_text())['engine_ownership'],'caller')
                close.assert_not_called()
            self.assertEqual(calls,[17,18]); self.assertEqual(engine.state,'unloaded')
            engine.close()

    def test_invalid_data_and_modality_fail_before_predict(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); engine=self.engine()
            value=np.zeros((1,12,224,224),dtype=np.float32); value[0,0,0,0]=np.nan
            source=self.source(root,value)
            with patch.object(engine,'predict') as predict:
                with self.assertRaisesRegex(ValueError,'finite'):
                    encode_file_with_engine(engine,input_file=source,output=root/'bad',execute_model=True)
                np.savez(source,DEM=np.zeros((1,1,224,224),dtype=np.float32))
                with self.assertRaisesRegex(ValueError,'modalities'):
                    encode_file_with_engine(engine,input_file=source,output=root/'wrong',execute_model=True)
                predict.assert_not_called()
            self.assertFalse((root/'bad').exists()); self.assertFalse((root/'wrong').exists())
            engine.close()

    def test_output_budget_and_failure_do_not_close_caller_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=self.source(root); engine=self.engine()
            with patch.object(engine,'predict',return_value={'features':[torch.zeros(1,3,4)],'provenance':{}}), \
                    patch.object(engine,'close') as close:
                with self.assertRaisesRegex(ValueError,'budget'):
                    encode_file_with_engine(engine,input_file=source,output=root/'small',execute_model=True,max_output_bytes=1024)
                self.assertFalse((root/'small').exists()); close.assert_not_called()
            with patch.object(engine,'predict',side_effect=RuntimeError('synthetic failure')),patch.object(engine,'close') as close:
                with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                    encode_file_with_engine(engine,input_file=source,output=root/'failed',execute_model=True)
                close.assert_not_called()
            engine.close()

    def test_read_only_inspection_does_not_import_torch_or_mutate_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=self.source(root)
            before=source.read_bytes()
            import aureon
            package=Path(aureon.__file__).resolve().parents[1]
            code="""import sys,importlib.abc,json
sys.path.insert(0,sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','terratorch','diffusers'}: raise AssertionError(name)
sys.meta_path.insert(0,Guard())
from aureon.api.encoder_io import inspect_encoder_input
print(json.dumps(inspect_encoder_input(sys.argv[2],modalities=['S2L2A'])))
assert 'torch' not in sys.modules
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(package),str(source)],
                                  capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual(report['batch'],1); self.assertEqual(report['array_bytes'],12*224*224*4)
            self.assertFalse(report['model_imported']); self.assertEqual(before,source.read_bytes())
            self.assertEqual(len(list(root.iterdir())),1)

    def test_configuration_is_detached_and_binds_creation_directory(self):
        engine=self.engine(); original=engine.configuration()
        self.assertTrue(Path(original['model_file']).is_absolute())
        changed=engine.configuration(); changed['options']['modalities']=('DEM',)
        changed['tokenizer_files']['made-up']='wrong'
        self.assertEqual(engine.configuration(),original)
        engine.close()

    def test_opt_in_and_closed_state_are_rejected_before_reading_files(self):
        from aureon.api import ExecutionRequired
        engine=self.engine()
        with self.assertRaises(ExecutionRequired):
            encode_file_with_engine(engine,input_file='absent',output='absent-output')
        engine.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):
            encode_file_with_engine(engine,input_file='absent',output='absent-output',execute_model=True)

    def test_bad_payload_releases_arrays_even_when_exception_is_retained(self):
        for problem in ('nonfinite','crc'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'input.npz'
                second=np.zeros((1,1,224,224),np.float32)
                if problem=='nonfinite':second[0,0,0,0]=np.nan
                np.savez(source,DEM=np.ones_like(second),NDVI=second)
                if problem=='crc':
                    with zipfile.ZipFile(source) as archive:
                        item=archive.getinfo('NDVI.npy')
                        with source.open('r+b') as stream:
                            stream.seek(item.header_offset+26)
                            name_size,extra_size=struct.unpack('<HH',stream.read(4))
                            location=item.header_offset+30+name_size+extra_size+item.compress_size-1
                            stream.seek(location);old=stream.read(1)
                            stream.seek(location);stream.write(bytes([old[0]^1]))
                refs=[];allocate=np.empty
                def capture(*args,**kwargs):
                    value=allocate(*args,**kwargs);refs.append(weakref.ref(value));return value
                caught=None
                with patch('numpy.empty',new=capture):
                    try:inspect_encoder_input(source)
                    except Exception as error:caught=error
                self.assertIsNotNone(caught);self.assertTrue(refs)
                self.assertIsNotNone(caught.__traceback__)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs),'Failed NPZ retained decoded arrays')
                self.assertIsInstance(caught,ValueError if problem=='nonfinite' else zipfile.BadZipFile)

    def test_export_faults_release_all_feature_arrays_with_live_exception(self):
        for problem in ('key','copy','mkdir'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);destination=root/'output'
                first=torch.ones((1,3,4));second=torch.ones((1,3,4))
                tensors=[weakref.ref(first),weakref.ref(second)]
                result={'features':[first,{'bad/key' if problem=='key' else 'S2L2A':second}],'provenance':{}}
                del first,second
                arrays=[];numpy=torch.Tensor.numpy;cpu=torch.Tensor.cpu;mkdir=Path.mkdir;copies=0
                def as_numpy(value,*args,**kwargs):
                    array=numpy(value,*args,**kwargs);arrays.append(weakref.ref(array));return array
                def copy(value,*args,**kwargs):
                    nonlocal copies
                    copies+=1
                    if problem=='copy' and copies==2:raise OSError('injected CPU copy failure')
                    return cpu(value,*args,**kwargs)
                def make(path,*args,**kwargs):
                    if problem=='mkdir' and path==destination:raise OSError('injected mkdir failure')
                    return mkdir(path,*args,**kwargs)
                caught=None
                with patch.object(torch.Tensor,'numpy',new=as_numpy),patch.object(torch.Tensor,'cpu',new=copy), \
                        patch.object(Path,'mkdir',new=make):
                    try:_export_features(result,model_id='fixture',identity={},destination=destination,
                        max_output_bytes=2*1024**2,ownership='caller')
                    except Exception as error:caught=error
                self.assertIsNotNone(caught);self.assertTrue(arrays)
                self.assertIsNotNone(caught.__traceback__)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in arrays),'Failed export retained CPU arrays')
                self.assertTrue(all(ref() is None for ref in tensors),'Failed export retained feature tensors')
                self.assertEqual(result['features'],[])
                self.assertFalse(destination.exists())

    def test_model_failure_releases_wrapper_inputs_with_live_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.source(root);engine=self.engine();refs=[]
            def fail(inputs,**kwargs):
                tensor=inputs['S2L2A'];refs.append(weakref.ref(tensor))
                raise RuntimeError('injected model failure')
            caught=None
            with patch.object(engine,'predict',new=fail),patch.object(engine,'close') as close:
                try:encode_file_with_engine(engine,input_file=source,output=root/'out',execute_model=True)
                except Exception as error:caught=error
                close.assert_not_called()
            self.assertIsNotNone(caught);self.assertIsNotNone(caught.__traceback__)
            gc.collect();self.assertTrue(all(ref() is None for ref in refs))
            engine.close()

    def test_tensor_conversion_failure_releases_partial_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.npz'
            np.savez(source,DEM=np.zeros((1,1,224,224),np.float32),NDVI=np.zeros((1,1,224,224),np.float32))
            engine=TerraMindEncoder('terramind-v1-tiny-backbone',model_file='unused.pt',
                modalities=['DEM','NDVI'],tokenizer_files={'NDVI':'unused-tokenizer.pt'},execute_model=True)
            refs=[];convert=torch.from_numpy;calls=0;caught=None
            def fail_second(array):
                nonlocal calls
                calls+=1
                refs.append(weakref.ref(array))
                if calls==2:raise RuntimeError('injected tensor conversion failure')
                tensor=convert(array);refs.append(weakref.ref(tensor));return tensor
            with patch('torch.from_numpy',new=fail_second),patch.object(engine,'predict') as predict:
                try:encode_file_with_engine(engine,input_file=source,output=root/'out',execute_model=True)
                except Exception as error:caught=error
                predict.assert_not_called()
            self.assertIsNotNone(caught);self.assertIsNotNone(caught.__traceback__)
            gc.collect();self.assertTrue(all(ref() is None for ref in refs))
            engine.close()

    def test_central_directory_is_bounded_before_zipfile_constructor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'base.npz';np.savez(source,DEM=np.zeros((1,1,224,224),np.float32))
            original=source.read_bytes();offset=original.rfind(b'PK\x05\x06')
            cases={}
            changed=bytearray(original);struct.pack_into('<HH',changed,offset+8,65000,65000);cases['count']=changed
            changed=bytearray(original);struct.pack_into('<I',changed,offset+12,65537);cases['directory']=changed
            changed=bytearray(original);struct.pack_into('<H',changed,offset+4,1);cases['multidisk']=changed
            changed=bytearray(original);struct.pack_into('<II',changed,offset+12,0xffffffff,0xffffffff);cases['zip64_end']=changed
            # EOCD can lie about a much larger actual record count. This small
            # fixture proves the parser bounds actual records before ZipFile.
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as archive:
                for i in range(9):archive.writestr(f'entry{i}.npy',b'')
            changed=bytearray(buffer.getvalue());position=changed.rfind(b'PK\x05\x06')
            struct.pack_into('<HH',changed,position+8,1,1);cases['lying_count']=changed
            for name,value in cases.items():
                with self.subTest(name=name):
                    path=root/(name+'.npz');path.write_bytes(value)
                    with patch('aureon.api.encoder_io.zipfile.ZipFile') as constructor:
                        with self.assertRaises(ValueError):inspect_encoder_input(path)
                        constructor.assert_not_called()

    def test_chunked_reads_bound_requests_and_preserve_fortran_c_contiguity(self):
        from aureon.api.encoder_io import _read_arrays
        image=np.asfortranarray(np.arange(2*12*224*224,dtype=np.float32).reshape(2,12,224,224))
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.npz';np.savez_compressed(source,S2L2A=image)
            sizes=[];readinto=zipfile.ZipExtFile.readinto
            def capture(stream,buffer):
                sizes.append(len(buffer));return readinto(stream,buffer)
            # Small chunk forces multiple F-order slabs/rows without a large
            # adversarial allocation and tests exact decoding of compressed data.
            with patch('aureon.api.encoder_io._READ_CHUNK',1024),patch.object(zipfile.ZipExtFile,'readinto',new=capture):
                arrays,_=_read_arrays(source)
            self.assertTrue(sizes);self.assertLessEqual(max(sizes),1024)
            self.assertTrue(arrays['S2L2A'].flags.c_contiguous)
            np.testing.assert_array_equal(arrays['S2L2A'],image)


if __name__=='__main__': unittest.main()
