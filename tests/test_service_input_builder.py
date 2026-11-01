"""File-identity-only manifest building; stdlib, no raster/model execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from aureon.serving.prepared_input import prepare_service_input


class ServiceInputBuilderTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ.get('AUREON_SERVICE_TEST_ROOT',tempfile.gettempdir()))
        self.root=parent/('input-builder-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        (self.root/'nested').mkdir();self.image=self.root/'nested'/'image.tif';self.mask=self.root/'mask.tif';self.inputs=self.root/'inputs.npz'
        self.image.write_bytes(b'identity-only raster fixture');self.mask.write_bytes(b'identity-only mask fixture');self.inputs.write_bytes(b'identity-only feature fixture')
        self.output=self.root/'manifest.json'

    def test_scene_manifest_has_exact_relative_bindings_and_semantics(self):
        result=prepare_service_input(self.output,task='generation_scene',image=self.image,valid_mask=self.mask)
        value=json.loads(self.output.read_bytes())
        self.assertEqual(value['schema'],'aureon.terramind_scene_input.v1')
        self.assertEqual(value['image'],{'path':'nested/image.tif','bytes':self.image.stat().st_size,'sha256':hashlib.sha256(self.image.read_bytes()).hexdigest()})
        self.assertEqual(value['mask']['path'],'mask.tif');self.assertEqual(len(value['bands']),12)
        self.assertEqual(value['units'],'surface_reflectance_x10000');self.assertFalse(result['model_executed']);self.assertFalse(result['files_copied'])
        self.assertEqual(result['pixel_validation'],'deferred_to_service_parser')
        self.assertEqual(result['manifest']['sha256'],hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_features_hash_streams_bounded_chunks_without_copy(self):
        with self.inputs.open('wb') as stream:
            for _ in range(3):stream.write(b'a'*1024**2)
        original=os.fdopen;reads=[]
        class Reader:
            def __init__(inner,wrapped):inner.wrapped=wrapped
            def __enter__(inner):return inner
            def __exit__(inner,*args):return inner.wrapped.__exit__(*args)
            def fileno(inner):return inner.wrapped.fileno()
            def read(inner,size=-1):
                reads.append(size);self.assertTrue(0<size<=1024**2);return inner.wrapped.read(size)
        def wrap(fd,mode,*args,**kwargs):
            stream=original(fd,mode,*args,**kwargs);return Reader(stream) if mode=='rb' else stream
        with patch('aureon.serving.prepared_input.os.fdopen',side_effect=wrap):
            result=prepare_service_input(self.output,task='features',inputs=self.inputs)
        value=json.loads(self.output.read_bytes());self.assertEqual(set(value),{'schema','inputs'})
        self.assertEqual(result['members']['inputs']['bytes'],3*1024**2);self.assertGreaterEqual(len(reads),4)
        self.assertEqual(set(self.root.iterdir()),{self.root/'nested',self.mask,self.inputs,self.output})

    def test_invalid_argument_sets_extensions_and_traversal_create_nothing(self):
        outsider=self.root.parent/(uuid.uuid4().hex+'.npz');outsider.write_bytes(b'outside')
        cases=[{'task':'features','inputs':outsider},{'task':'features','inputs':self.image},
            {'task':'features','inputs':self.inputs,'image':self.image},{'task':'generation_scene','image':self.image},
            {'task':'generation_scene','image':self.image,'valid_mask':self.image},{'task':'unsupported','inputs':self.inputs}]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError,FileNotFoundError)):prepare_service_input(self.output,**arguments)
                self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):prepare_service_input(self.root/'missing'/'manifest.json',task='features',inputs=self.inputs)

    def test_symlink_file_and_parent_are_rejected(self):
        link=self.root/'alias.npz';linked=self.root/'linked'
        try:link.symlink_to(self.inputs);linked.symlink_to(self.root/'nested',target_is_directory=True)
        except OSError:self.skipTest('Local symlink creation unavailable')
        for source in (link,linked/'image.tif'):
            with self.assertRaises(ValueError):prepare_service_input(self.output,task='features',inputs=source)
        self.assertFalse(self.output.exists())

    def test_mutation_during_hash_and_before_publish_create_no_manifest(self):
        import aureon.serving.prepared_input as builder
        original=builder._hash_source
        def mutate(path,parent,limit):
            result=original(path,parent,limit)
            if path==self.mask:self.image.write_bytes(b'changed after image hash')
            return result
        with patch.object(builder,'_hash_source',side_effect=mutate):
            with self.assertRaises(ValueError):prepare_service_input(self.output,task='generation_scene',image=self.image,valid_mask=self.mask)
        self.assertFalse(self.output.exists())
        real_hash=hashlib.sha256;changed=False
        class Hasher:
            def __init__(inner):inner.value=real_hash()
            def update(inner,data):
                nonlocal changed
                inner.value.update(data)
                if not changed:
                    changed=True
                    with self.inputs.open('ab') as stream:stream.write(b'mutation')
            def hexdigest(inner):return inner.value.hexdigest()
        with patch('aureon.serving.prepared_input.hashlib.sha256',side_effect=Hasher):
            with self.assertRaises(ValueError):prepare_service_input(self.output,task='features',inputs=self.inputs)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b'preserve existing user file')
        with self.assertRaises(ValueError):prepare_service_input(self.output,task='features',inputs=self.inputs)
        self.assertEqual(self.output.read_bytes(),b'preserve existing user file')


if __name__=='__main__':unittest.main()
