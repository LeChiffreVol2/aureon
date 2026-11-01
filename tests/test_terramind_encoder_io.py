"""Feature-file delivery contracts using small CPU outputs, not model weights."""
from pathlib import Path
import gc
import io
import json
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from aureon.api.encoder_io import encode_file


class EncoderFiles(unittest.TestCase):
    @staticmethod
    def header(shape, *, descriptor='<f4', order=False, payload=b''):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'descr': descriptor, 'fortran_order': order, 'shape': shape,
        })
        return stream.getvalue() + payload

    def test_hostile_headers_fail_before_array_tensor_or_model_construction(self):
        cases = {
            'huge': ('S2L2A.npy', self.header((1_000_000, 12, 224, 224))),
            'overflow': ('S2L2A.npy', self.header((2**100, 12, 224, 224))),
            'rank': ('S2L2A.npy', self.header((12, 224, 224))),
            'channels': ('S2L2A.npy', self.header((1, 13, 224, 224))),
            'spatial': ('S2L2A.npy', self.header((1, 12, 223, 224))),
            'zero_batch': ('S2L2A.npy', self.header((0, 12, 224, 224))),
            'negative_batch': ('S2L2A.npy', self.header((-1, 12, 224, 224))),
            'boolean_batch': ('S2L2A.npy', self.header((True, 12, 224, 224))),
            'list_shape': ('S2L2A.npy', self.header([1, 12, 224, 224])),
            'object': ('S2L2A.npy', self.header((1, 12, 224, 224), descriptor='|O')),
            'float64': ('S2L2A.npy', self.header((1, 12, 224, 224), descriptor='<f8')),
            'non_native': ('S2L2A.npy', self.header((1, 12, 224, 224), descriptor='>f4')),
            'structured': ('S2L2A.npy', self.header((1, 12, 224, 224), descriptor=[('x', '<f4')])),
            'bad_order': ('S2L2A.npy', self.header((1, 12, 224, 224), order=1)),
            'unknown_modality': ('NOT_A_MODALITY.npy', self.header((1, 12, 224, 224))),
            'float_lulc': ('LULC.npy', self.header((1, 224, 224))),
            'truncated_payload': ('DEM.npy', self.header((1, 1, 224, 224), payload=b'\0'*8)),
            'extra_payload': ('DEM.npy', self.header((1, 1, 224, 224), payload=b'\0'*(4*224*224+1))),
            'bad_magic': ('DEM.npy', b'not-a-numpy-array'),
            'bad_version': ('DEM.npy', b'\x93NUMPY\x04\x00'),
            'truncated_header': ('DEM.npy', b'\x93NUMPY\x01\x00\x64\x00{}'),
            'excess_header': ('DEM.npy', b'\x93NUMPY\x02\x00' + (10_001).to_bytes(4, 'little')),
            'malformed_header': ('DEM.npy', b'\x93NUMPY\x01\x00\x02\x00!!'),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, (member, value) in cases.items():
                with self.subTest(name=name):
                    source = root/(name+'.npz')
                    with zipfile.ZipFile(source, 'w') as archive:
                        archive.writestr(member, value)
                    with patch('numpy.load') as unsafe_loader, patch('numpy.frombuffer') as allocation, \
                            patch('torch.from_numpy') as tensor, \
                            patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
                        with self.assertRaises(ValueError):
                            encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                                        input_file=source, output=root/(name+'-out'), execute_model=True)
                        unsafe_loader.assert_not_called()
                        allocation.assert_not_called()
                        tensor.assert_not_called()
                        factory.assert_not_called()
                    self.assertFalse((root/(name+'-out')).exists())

    def test_all_headers_and_batch_agreement_checked_before_loading_any_member(self):
        valid = io.BytesIO()
        np.save(valid, np.zeros((1, 1, 224, 224), dtype=np.float32), allow_pickle=False)
        cases = (
            self.header((2**80, 1, 224, 224)),
            self.header((2, 1, 224, 224), payload=b'\0'*(2*4*224*224)),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, second in enumerate(cases):
                source = root/f'case-{index}.npz'
                with zipfile.ZipFile(source, 'w') as archive:
                    archive.writestr('DEM.npy', valid.getvalue())
                    archive.writestr('NDVI.npy', second)
                with patch('numpy.frombuffer') as allocation, patch('torch.from_numpy') as tensor, \
                        patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
                    with self.assertRaises(ValueError):
                        encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                                    input_file=source, output=root/f'out-{index}', execute_model=True)
                    allocation.assert_not_called()
                    tensor.assert_not_called()
                    factory.assert_not_called()

    def test_valid_compressed_fortran_and_int64_inputs_preserve_exact_values(self):
        image = np.asfortranarray(np.arange(12*224*224, dtype=np.float32).reshape(1, 12, 224, 224))
        classes = (np.arange(224*224, dtype=np.int64) % 10).reshape(1, 224, 224)
        feature = torch.zeros(1, 3, 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'inputs.npz'
            np.savez_compressed(source, S2L2A=image, LULC=classes)
            with patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
                predict = factory.return_value.__enter__.return_value.predict
                actual = {}
                def capture(inputs, **kwargs):
                    actual.update(inputs)
                    return {'features': [feature], 'provenance': {}}
                predict.side_effect = capture
                encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                            input_file=source, output=root/'out', tokenizer_files={'LULC': 't'}, execute_model=True)
                self.assertEqual(actual['S2L2A'].dtype, torch.float32)
                self.assertEqual(actual['LULC'].dtype, torch.int64)
                np.testing.assert_array_equal(actual['S2L2A'].numpy(), image)
                np.testing.assert_array_equal(actual['LULC'].numpy(), classes)
                self.assertTrue(actual['S2L2A'].is_contiguous())
                self.assertTrue(actual['LULC'].is_contiguous())

    def test_npy_versions_one_two_three_are_decoded_without_unsafe_loader(self):
        image = np.arange(224*224, dtype=np.float32).reshape(1, 1, 224, 224)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for version in ((1, 0), (2, 0), (3, 0)):
                with self.subTest(version=version):
                    array = io.BytesIO()
                    np.lib.format.write_array(array, image, version=version, allow_pickle=False)
                    source = root/f'input-{version[0]}.npz'
                    with zipfile.ZipFile(source, 'w') as archive:
                        archive.writestr('DEM.npy', array.getvalue())
                    with patch('numpy.load') as unsafe_loader, \
                            patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
                        predict = factory.return_value.__enter__.return_value.predict
                        actual = {}
                        def capture(inputs, **kwargs):
                            actual.update(inputs)
                            return {'features': [torch.zeros(1, 3, 4)], 'provenance': {}}
                        predict.side_effect = capture
                        encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                                    input_file=source, output=root/f'out-{version[0]}', execute_model=True)
                        np.testing.assert_array_equal(actual['DEM'].numpy(), image)
                        unsafe_loader.assert_not_called()

    def test_retained_export_frame_releases_owned_feature_tensors(self):
        refs = []
        frames = []
        def predict(*args, **kwargs):
            features = [torch.ones(1, 3, 4), torch.ones(1, 3, 4)]
            refs.extend(weakref.ref(value) for value in features)
            return {'features': features, 'provenance': {'synthetic_test': True}}
        def retain(frame, event, arg):
            if event == 'return' and frame.f_code is encode_file.__code__:
                frames.append(frame)
        with tempfile.TemporaryDirectory() as temp, patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
            root = Path(temp); source = root/'input.npz'
            np.savez(source, S2L2A=np.zeros((1, 12, 224, 224), dtype=np.float32))
            factory.return_value.__enter__.return_value.predict.side_effect = predict
            previous = sys.getprofile()
            try:
                sys.setprofile(retain)
                encode_file(model_id='terramind-v1-tiny-backbone', model_file='weights', input_file=source,
                            output=root/'out', execute_model=True)
            finally:
                sys.setprofile(previous)
            gc.collect()
            self.assertEqual(len(frames), 1)
            self.assertTrue(all(ref() is None for ref in refs), 'Completed export frame retained request feature tensors')
            with np.load(root/'out/features.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['layer_000_merged'], np.ones((1, 3, 4), dtype=np.float32))
            frames.clear()

    def test_native_feature_layers_export_without_raster_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'inputs.npz'
            np.savez(source, S2L2A=np.zeros((1, 12, 224, 224), dtype=np.float32))
            feature = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            with patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
                factory.return_value.__enter__.return_value.predict.return_value = {
                    'features': [feature, {'S2L2A': feature + 1}],
                    'provenance': {'model_id': 'terramind-v1-tiny-tim', 'synthetic_test': True},
                }
                result = encode_file(model_id='terramind-v1-tiny-tim', model_file='weights',
                                     input_file=source, output=root/'out',
                                     tokenizer_files={'LULC': 'tokenizer'}, tim_modalities=['LULC'],
                                     execute_model=True)
                self.assertEqual(factory.call_args.args, ('terramind-v1-tiny-tim',))
                self.assertEqual(factory.call_args.kwargs['modalities'], ('S2L2A',))
            self.assertEqual(result['status'], 'completed')
            with np.load(root/'out/features.npz', allow_pickle=False) as exported:
                np.testing.assert_array_equal(exported['layer_000_merged'], feature.numpy())
                np.testing.assert_array_equal(exported['layer_001_S2L2A'], feature.numpy()+1)
            provenance = json.loads((root/'out/provenance.json').read_text())
            self.assertIn('no task head or COG/grid claim', provenance['output_semantics'])
            self.assertEqual(len(provenance['layout']), 2)
            self.assertNotIn('crs', provenance)

    def test_existing_output_and_unsafe_npz_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as temp, patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
            root = Path(temp)
            with self.assertRaises(ValueError):
                encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                            input_file='missing', output=root, execute_model=True)
            source = root/'unsafe.npz'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('../S2L2A.npy', b'invalid')
            with self.assertRaises(ValueError):
                encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                            input_file=source, output=root/'out', execute_model=True)
            factory.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_object_arrays_never_unpickle(self):
        with tempfile.TemporaryDirectory() as temp, patch('aureon.api.terramind_encoder.TerraMindEncoder') as factory:
            root = Path(temp)
            np.savez(root/'objects.npz', S2L2A=np.array([{'not': 'tensor'}], dtype=object))
            with self.assertRaises(ValueError):
                encode_file(model_id='terramind-v1-tiny-backbone', model_file='m',
                            input_file=root/'objects.npz', output=root/'out', execute_model=True)
            factory.assert_not_called()
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
