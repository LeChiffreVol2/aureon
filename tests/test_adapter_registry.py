"""Declared eligibility, real factory dispatch, and no implicit runtime discovery."""
from dataclasses import replace
import unittest
from unittest.mock import Mock
import torch
from torch import nn
from aureon.api.adapter_registry import AdapterRegistry, AdapterSpec, AdapterRequest, DependencyPin
from aureon.api.borrowed_model import BorrowedModelBridge, DeviceTensorPolicy


class RegistryTests(unittest.TestCase):
    def spec(self):
        return AdapterSpec('local-test-v1', ('tensor',), ('features',), ('cpu', 'cuda'),
                           (DependencyPin('torch', ('2.10.0', '2.11.0')),), max_batch_size=2)

    def test_registered_factory_builds_real_borrowed_bridge_and_registry_does_not_own_it(self):
        registry = AdapterRegistry()
        model = nn.Linear(2, 3).eval()
        calls = []
        def factory(request, module):
            calls.append(request)
            return BorrowedModelBridge(module, forward=lambda m, x: m(x['image']), output=lambda x: {'features': x},
                                       policy=DeviceTensorPolicy(request.device, request.allow_unqualified_cuda))
        registry.register(self.spec(), factory)
        request = AdapterRequest('tensor', 'features', graph=True, graph_fallback='eager')
        plan = registry.plan('local-test-v1', request)
        self.assertTrue(plan.admitted)
        self.assertFalse(plan.resolved.graph)
        with registry.create('local-test-v1', request, model) as bridge:
            registry.unregister('local-test-v1')
            actual = bridge.predict({'image': torch.ones(1, 2)})['features']
            self.assertTrue(torch.equal(actual, model(torch.ones(1, 2))))
        self.assertFalse(calls[0].graph)
        self.assertEqual(registry.specifications(), ())

    def test_unsupported_dependencies_precision_batch_and_graph_fail_before_factory(self):
        registry = AdapterRegistry()
        factory = Mock(side_effect=AssertionError('Factory must not execute'))
        registry.register(self.spec(), factory)
        base = AdapterRequest('tensor', 'features')
        for request in (replace(base, precision='float16'), replace(base, batch_size=3),
                        replace(base, graph=True), replace(base, input_kind='temporal'),
                        replace(base, device='cuda:0')):
            self.assertFalse(registry.plan('local-test-v1', request).admitted)
            with self.assertRaises(ValueError): registry.create('local-test-v1', request, None)
        bad = replace(self.spec(), adapter_id='missing-dependency',
                      dependencies=(DependencyPin('aureon-nonexistent-distribution-fixture', ('1.0.0',)),))
        registry.register(bad, factory)
        with self.assertRaises(ValueError): registry.create(bad.adapter_id, base, None)
        factory.assert_not_called()

    def test_capabilities_are_detached_and_registration_is_explicit(self):
        registry = AdapterRegistry()
        registry.register(self.spec(), lambda request, options: options)
        with self.assertRaises(ValueError): registry.register(self.spec(), lambda request, options: options)
        with self.assertRaises(KeyError): registry.plan('unknown', AdapterRequest('tensor', 'features'))
        value = registry.metadata()
        value['adapters'][0]['devices'] = ['imaginary']
        self.assertEqual(registry.specifications()[0].devices, ('cpu', 'cuda'))
        result = registry.plan('local-test-v1', AdapterRequest('tensor', 'features', device='cuda:0', allow_unqualified_cuda=True))
        self.assertTrue(result.admitted)
        self.assertIn('cuda_execution_unqualified_and_not_probed', result.notices)
        self.assertEqual(result.qualification, 'declarations_not_execution_qualification')


if __name__ == '__main__': unittest.main()
