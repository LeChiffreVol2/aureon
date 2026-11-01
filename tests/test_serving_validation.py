"""Tiny CPU fixtures distinguish serving ownership from audit value checks."""
import unittest
from unittest.mock import patch
import torch
from torch import nn

from aureon.api.borrowed_model import BorrowedModelBridge, BorrowedValidationPolicy, BorrowedMutation
from aureon.api.adapter_conformance import check_adapter_conformance


def forward(model, values):return model(values['image'])
def output(value):return {'features':value}
def serving():return BorrowedValidationPolicy('serving',True,'fixture-linear-v1')


class ServingTests(unittest.TestCase):
    def bridge(self,model,**kwargs):
        return BorrowedModelBridge(model,forward=forward,output=output,**kwargs)

    def test_serving_requires_ownership_and_contract(self):
        for kwargs in ({},{'exclusive_ownership':True},{'contract_id':'x'}):
            with self.assertRaises(ValueError):BorrowedValidationPolicy('serving',**kwargs)
        with self.assertRaises(ValueError):BorrowedValidationPolicy('unchecked')

    def test_no_weight_clones_or_value_scans_on_serving_requests(self):
        model=nn.Linear(200,100).eval();x=torch.ones(1,200)
        expected=model(x).detach()
        with self.bridge(model,validation=serving()) as bridge:
            self.assertEqual(bridge.metadata()['baseline_bytes'],0)
            self.assertEqual(bridge.metadata()['weight_value_scans_per_request'],0)
            original=torch.isfinite
            sizes=[]
            def finite(value):sizes.append(value.numel());return original(value)
            with patch('torch.isfinite',side_effect=finite),patch('torch.equal',side_effect=AssertionError('weight scan')):
                a=bridge.predict({'image':x});b=bridge.predict({'image':x})
            self.assertEqual(sizes,[200,100,200,100])
            self.assertTrue(torch.equal(a['features'],expected));self.assertTrue(torch.equal(b['features'],expected))
            self.assertIs(bridge._model,model)

    def test_versioned_mutation_and_replacement_fail_terminally(self):
        for action in ('inplace','replace','train'):
            model=nn.Linear(2,2).eval()
            with self.bridge(model,validation=serving()) as bridge:
                with torch.no_grad():
                    if action=='inplace':model.weight.add_(1)
                    elif action=='replace':model.weight=nn.Parameter(model.weight.clone())
                    else:model.train()
                with self.assertRaises((BorrowedMutation,ValueError)):bridge.predict({'image':torch.ones(1,2)})
                self.assertEqual(bridge.state,'failed')

    def test_unsafe_data_writes_are_explicitly_not_detected_in_serving(self):
        model=nn.Linear(2,2).eval()
        with self.bridge(model,validation=serving()) as bridge:
            model.weight.data.add_(1)
            bridge.predict({'image':torch.ones(1,2)})
            self.assertFalse(bridge.metadata()['unsafe_tensor_writes_detected'])
        with self.bridge(model) as audit:
            self.assertGreater(audit.metadata()['baseline_bytes'],0)
            model.weight.data.add_(1)
            with self.assertRaises(BorrowedMutation):audit.predict({'image':torch.ones(1,2)})

    def test_serving_load_scan_rejects_nonfinite_state(self):
        model=nn.Linear(2,2).eval();model.weight.data.fill_(float('nan'))
        with self.assertRaises(ValueError):self.bridge(model,validation=serving())

    def test_conformance_closes_owner_and_detects_bad_ownership(self):
        owners=[]
        def factory():
            bridge=self.bridge(nn.Identity().eval(),validation=serving());owners.append(bridge);return bridge
        fixture={'image':torch.ones(1,2)}
        result=check_adapter_conformance(factory,fixture,{'features':torch.ones(1,2)})
        self.assertTrue(result.exact_fixture_parity);self.assertEqual(owners[0].state,'closed')
        with self.assertRaises(AssertionError):check_adapter_conformance(factory,fixture,{'features':torch.zeros(1,2)})
        self.assertEqual(owners[1].state,'closed')
        class Alias:
            closed=False
            def predict(self,inputs,**kwargs):return {'features':inputs['image']}
            def close(self):self.closed=True
        alias=Alias()
        with self.assertRaisesRegex(AssertionError,'aliases'):check_adapter_conformance(lambda:alias,fixture,{'features':torch.ones(1,2)})
        self.assertTrue(alias.closed)

    def test_conformance_preserves_primary_error_when_close_fails(self):
        class Broken:
            def predict(self,inputs,**kwargs):raise ValueError('forward failed')
            def close(self):raise OSError('cleanup failed')
        with self.assertRaisesRegex(ValueError,'forward failed') as caught:
            check_adapter_conformance(Broken,{'image':torch.ones(1,2)},{'features':torch.ones(1,2)})
        self.assertIn('cleanup failed',caught.exception.__notes__[0])


if __name__=='__main__':unittest.main()
