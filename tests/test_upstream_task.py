import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from aureon.api import UpstreamTaskBridge, PreprocessingRecipe, TerraTorchSegmentation, TerraTorchTaskConfig
from aureon.terramind_native.encoder_loader import fp32_policy


def upstream_segmentation(args):
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.models.smp_model_factory import SMPModelWrapper
    return SemanticSegmentationTask(model_args={'num_classes':args['outputs']},
        model=SMPModelWrapper(nn.Conv2d(args['inputs'],args['outputs'],1)),ignore_index=-1,plot_on_val=False)


def upstream_regression(args):
    from terratorch.tasks import PixelwiseRegressionTask
    from terratorch.models.smp_model_factory import SMPModelWrapper
    return PixelwiseRegressionTask(model_args={'num_classes':args['outputs']},
        model=SMPModelWrapper(nn.Conv2d(args['inputs'],args['outputs'],1),squeeze_single_class=True),plot_on_val=False)


class BadGeometry(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode
    def forward(self,x):
        y=x if self.mode!='spatial' else x[...,::2,::2]
        y=y[:1] if self.mode=='batch' else y
        return SimpleNamespace(output=y,auxiliary_heads={'extra':y} if self.mode=='aux' else None)


class FlexibleTaskTests(unittest.TestCase):
    def setUp(self):
        self.recipe=PreprocessingRecipe(('a','b'),'dn','standardized',scale=.5,mean=(1.,2.),std=(2.,4.))

    def test_actual_nonfixture_segmentation_and_regression_lightning_parity(self):
        for kind,constructor,channels in [('segmentation',upstream_segmentation,3),('regression',upstream_regression,1)]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                config={'inputs':2,'outputs':channels,'pretrained':False}
                model=constructor(config).eval();path=Path(folder)/'task.ckpt'
                torch.save({'state_dict':model.state_dict(),'hyper_parameters':{'never':'executed'}},path)
                rng=torch.get_rng_state().clone()
                with UpstreamTaskBridge(constructor,config,kind=kind,channels=channels,preprocessing=self.recipe,
                        checkpoint=path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()) as bridge:
                    x=torch.arange(2*8*8,dtype=torch.float32).reshape(1,2,8,8)
                    prepared=self.recipe.apply(x,bands=('b','a'),units='dn')
                    with fp32_policy(),torch.no_grad(),torch.autocast('cpu',enabled=False):reference=model(prepared.tensor).output
                    actual=bridge.predict(x,bands=('b','a'),units='dn')
                    self.assertTrue(torch.equal(reference,actual.values))
                    if kind=='segmentation':self.assertTrue(torch.equal(reference.argmax(1),actual.labels))
                    else:self.assertIsNone(actual.labels)
                    self.assertEqual(actual.kind,kind);json.dumps(bridge.metadata())
                    self.assertEqual(set(bridge.metadata()['checkpoint']['tensor_identity']),set(model.state_dict()))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_token_features_raw_checkpoint_and_ownership(self):
        constructor=lambda args:nn.Sequential(nn.Flatten(2),Transpose())
        model=constructor({}).eval()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'features.pt';torch.save(model.state_dict(),path)
            with UpstreamTaskBridge(constructor,{},kind='features',channels=2,output_container='tensor',
                    preprocessing=self.recipe,checkpoint=path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_format='state_dict') as bridge:
                x=torch.ones(1,2,4,4);before=x.clone()
                expected=model(self.recipe.apply(x,bands=('a','b'),units='dn').tensor)
                result=bridge.predict(x,bands=('a','b'),units='dn')
                self.assertTrue(torch.equal(result.values,expected));result.values.zero_()
                self.assertTrue(torch.equal(x,before));self.assertIsNone(result.labels)

    def test_geometry_and_aux_errors_are_terminal_in_both_bridges(self):
        config=TerraTorchTaskConfig(('a','b'),2)
        for mode in ('batch','spatial','aux'):
            for fixed in (True,False):
                with self.subTest(mode=mode,fixed=fixed):
                    bridge=(TerraTorchSegmentation(BadGeometry(mode).eval(),config,self.recipe) if fixed else
                        UpstreamTaskBridge(lambda args:BadGeometry(mode),{},kind='segmentation',channels=2,preprocessing=self.recipe))
                    with self.assertRaises(ValueError):bridge.predict(torch.ones(2,2,16,16),bands=('a','b'),units='dn')
                    self.assertEqual(bridge.metadata()['state'],'failed')
                    with self.assertRaises(RuntimeError):bridge.predict(torch.ones(2,2,16,16),bands=('a','b'),units='dn')

    def test_config_and_checkpoint_rejected_before_constructor(self):
        for config in ({'pretrained':True},{'nested':{'weights':'remote'}},{'class_path':'os.system'},{1:'bad'},
                       {'nested':({'class_path':'never.execute','pretrained':True},)},
                       {'nested':({'weights':'remote'},)}):
            with patch(__name__+'.upstream_segmentation') as constructor:
                with self.assertRaises(ValueError):UpstreamTaskBridge(constructor,config,kind='segmentation',channels=2,preprocessing=self.recipe)
                constructor.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.pt';torch.save({'unexpected':torch.ones(1)},path)
            with self.assertRaisesRegex(ValueError,'keys'):
                UpstreamTaskBridge(lambda args:nn.Conv2d(2,2,1),{},kind='segmentation',channels=2,preprocessing=self.recipe,
                    checkpoint=path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_format='state_dict')

    def test_retryable_seed_and_invalid_preprocessing(self):
        bridge=UpstreamTaskBridge(lambda args:nn.Conv2d(2,2,1),{},kind='segmentation',channels=2,
                                  preprocessing=self.recipe,output_container='tensor')
        for kwargs in ({'seed':True},{'units':'unknown'}):
            options={'bands':('a','b'),'units':'dn',**kwargs}
            with self.assertRaises(ValueError):bridge.predict(torch.ones(1,2,8,8),**options)
            self.assertEqual(bridge.state,'ready')
        bridge.close();self.assertEqual(bridge.state,'closed')


class Transpose(nn.Module):
    def forward(self,x):return x.transpose(1,2)


if __name__=='__main__':unittest.main()
