"""Actual CPU tensors/modules; CUDA planning only, never CUDA execution."""
import random
import threading
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from aureon.api.borrowed_model import BorrowedModelBridge, BorrowedMutation, DeviceTensorPolicy
from aureon.api.model_bridge import BridgeCancelled
from aureon.terramind_native.rng import rng_capsule


def forward(model, inputs): return model(inputs['image'])
def output(value): return {'features': value}


class NoCopy(nn.Linear):
    def __deepcopy__(self, memo): raise AssertionError('Model deepcopy is forbidden')
    def close(self): raise AssertionError('Caller close must not be invoked')


class BorrowedTests(unittest.TestCase):
    def test_no_grad_inside_outer_inference_mode_and_other_default_device(self):
        model = nn.Linear(2, 2).eval()
        values = torch.ones(1, 2)
        seen = []
        def inspect(instance, inputs):
            seen.append(torch.is_grad_enabled())
            return instance(inputs['image'])
        for inference in (False, True):
            original = torch.get_default_device()
            try:
                torch.set_default_device('meta')
                with torch.inference_mode(inference):
                    before = torch.is_grad_enabled()
                    with BorrowedModelBridge(model, forward=inspect, output=output) as bridge:
                        result = bridge.predict({'image': values})['features']
                        self.assertFalse(result.is_inference())
                        self.assertFalse(result.requires_grad)
                        self.assertEqual(str(result.device), 'cpu')
                        self.assertEqual(str(torch.get_default_device()), 'meta')
                    self.assertEqual(torch.is_grad_enabled(), before)
            finally:
                torch.set_default_device(original)
        self.assertEqual(seen, [False, False])

    def test_actual_instance_reuse_and_input_output_storage_ownership(self):
        model = NoCopy(3, 2).eval()
        original = model.weight.detach().clone()
        calls = []
        def use(instance, inputs):
            calls.append(id(instance))
            return instance(inputs['image'])
        x = torch.randn(2, 3)
        expected = model(x).detach()
        with BorrowedModelBridge(model, forward=use, output=output) as bridge:
            first = bridge.predict({'image': x})['features']
            self.assertTrue(torch.equal(first, expected))
            first.add_(10)
            second = bridge.predict({'image': x})['features']
            self.assertTrue(torch.equal(second, expected))
            self.assertFalse(second.is_inference())
            self.assertFalse(second.requires_grad)
            self.assertEqual(calls, [id(model), id(model)])
            self.assertEqual(bridge.metadata()['module_copies_per_request'], 0)
        self.assertTrue(torch.equal(model.weight, original))
        self.assertFalse(model.training)
        self.assertTrue(model.weight.requires_grad)
        with self.assertRaises(RuntimeError): bridge.predict({'image': x})

    def test_input_alias_output_and_inplace_input_do_not_mutate_caller(self):
        model = nn.Identity().eval()
        def use(instance, inputs):
            inputs['image'].add_(3)
            return inputs['image']
        with BorrowedModelBridge(model, forward=use, output=output) as bridge:
            x = torch.ones(2, 3)
            actual = bridge.predict({'image': x})['features']
            self.assertTrue(torch.equal(x, torch.ones_like(x)))
            self.assertTrue(torch.equal(actual, x + 3))
            actual.zero_()
            self.assertTrue(bool(x.all()))

    def test_registered_values_gradients_metadata_and_external_change_are_terminal(self):
        for mode in ('buffer', 'parameter_data', 'attribute', 'training', 'gradient', 'signed_zero', 'bool_type', 'dtype_type'):
            with self.subTest(mode=mode):
                model = nn.Linear(2, 2).eval()
                model.register_buffer('count', torch.zeros(1))
                model.options = {'nested': [1]}
                model.flag = True
                model.policy_dtype = torch.float32
                def mutate(instance, inputs):
                    if mode == 'buffer': instance.count.add_(1)
                    elif mode == 'parameter_data': instance.weight.data.add_(1)
                    elif mode == 'attribute': instance.options['nested'].append(2)
                    elif mode == 'training': instance.training = True
                    elif mode == 'gradient': instance.weight.grad = torch.ones_like(instance.weight)
                    elif mode == 'bool_type': instance.flag = 1
                    elif mode == 'dtype_type': instance.policy_dtype = 'torch.float32'
                    else: instance.count.data.mul_(-1)
                    return inputs['image']
                with BorrowedModelBridge(model, forward=mutate, output=output) as bridge:
                    with self.assertRaises((BorrowedMutation, ValueError)):
                        bridge.predict({'image': torch.ones(1, 2)})
                    self.assertEqual(bridge.state, 'failed')
                    with self.assertRaises(RuntimeError): bridge.predict({'image': torch.ones(1, 2)})
                if mode == 'buffer': self.assertEqual(model.count.item(), 1)  # No silent rollback.
        model = nn.Linear(2, 2).eval()
        with BorrowedModelBridge(model, forward=forward, output=output) as bridge:
            with torch.no_grad(): model.weight.add_(1)
            with self.assertRaises(BorrowedMutation): bridge.predict({'image': torch.ones(1, 2)})

    def test_rng_and_policy_restored_after_success_and_failure(self):
        model = nn.Identity().eval()
        for fail in (False, True):
            numpy_before = np.random.get_state()
            python_before = random.getstate()
            torch_before = torch.get_rng_state().clone()
            policy = (torch.get_default_dtype(), torch.get_float32_matmul_precision(),
                      torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.enabled,
                      torch.backends.cudnn.deterministic, torch.get_default_device(), torch.is_grad_enabled())
            def use(instance, inputs):
                random.random(); np.random.rand(); torch.rand(3)
                torch.set_default_dtype(torch.float64)
                torch.set_float32_matmul_precision('medium')
                torch.use_deterministic_algorithms(not policy[2])
                torch.backends.cudnn.enabled = not policy[3]
                torch.backends.cudnn.deterministic = not policy[4]
                torch.set_default_device('meta')
                if fail: raise RuntimeError('injected forward failure')
                return inputs['image']
            with BorrowedModelBridge(model, forward=use, output=output) as bridge:
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'injected'): bridge.predict({'image': torch.ones(2, 3)})
                    self.assertEqual(bridge.state, 'failed')
                else: bridge.predict({'image': torch.ones(2, 3)})
            self.assertEqual(python_before, random.getstate())
            after = np.random.get_state()
            self.assertTrue(np.array_equal(numpy_before[1], after[1]))
            self.assertEqual(numpy_before[2:], after[2:])
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
            self.assertEqual(policy, (torch.get_default_dtype(), torch.get_float32_matmul_precision(),
                torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.enabled,
                torch.backends.cudnn.deterministic, torch.get_default_device(), torch.is_grad_enabled()))

    def test_admission_and_global_busy_leave_healthy_bridge_untouched(self):
        model = nn.Identity().eval()
        with BorrowedModelBridge(model, forward=forward, output=output) as bridge:
            for value in (torch.ones(2, dtype=torch.float64), torch.tensor([float('nan')]), torch.ones(2, device='meta')):
                with self.assertRaises(ValueError): bridge.predict({'image': value})
                self.assertEqual(bridge.state, 'ready')
            with self.assertRaises(ValueError): bridge.predict({'image': torch.ones(2)}, seed=True)
            with rng_capsule(5):
                with self.assertRaises(RuntimeError): bridge.predict({'image': torch.ones(2)})
            self.assertEqual(bridge.state, 'ready')

    def test_cancellation_failure_and_busy_close_contracts(self):
        event = threading.Event(); event.set()
        model = nn.Identity().eval()
        def use(instance, values): event.set(); return values['image']
        with BorrowedModelBridge(model, forward=use, output=output) as bridge:
            with self.assertRaises(BridgeCancelled): bridge.predict({'image': torch.ones(2)}, cancel=event)
            self.assertEqual(bridge.state, 'ready')
            event.clear()
            with self.assertRaises(BridgeCancelled): bridge.predict({'image': torch.ones(2)}, cancel=event)
            self.assertEqual(bridge.state, 'ready')
            bridge._lock.acquire()
            try:
                with self.assertRaises(RuntimeError): bridge.close()
                with self.assertRaises(RuntimeError): bridge.predict({'image': torch.ones(2)})
            finally: bridge._lock.release()
        def callback_cancel(model, values): raise BridgeCancelled('adapter failed mid-forward')
        with BorrowedModelBridge(model, forward=callback_cancel, output=output) as bridge:
            with self.assertRaises(BridgeCancelled): bridge.predict({'image': torch.ones(2)})
            self.assertEqual(bridge.state, 'failed')

    def test_cuda_requires_explicit_policy_and_context_without_probing_in_cpu_test(self):
        model = nn.Identity().eval()
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU probe')), \
                patch.object(torch.cuda, 'device_count', side_effect=AssertionError('GPU probe')):
            self.assertFalse(DeviceTensorPolicy('cuda:0').eligibility()['admitted'])
            with self.assertRaises(ValueError): BorrowedModelBridge(model, forward=forward, output=output, policy=DeviceTensorPolicy('cuda:0'))
            self.assertFalse(torch.cuda.is_initialized())
            with self.assertRaisesRegex(ValueError, 'initialized'):
                BorrowedModelBridge(model, forward=forward, output=output, policy=DeviceTensorPolicy('cuda:0', True))
        with self.assertRaises(ValueError): DeviceTensorPolicy('cuda')
        with self.assertRaises(ValueError): BorrowedModelBridge(model, forward=forward, output=output, stream=object())

    def test_unsupported_state_is_rejected_without_copy_move_or_eval(self):
        with self.assertRaises(ValueError): BorrowedModelBridge(nn.Linear(2, 2), forward=forward, output=output)
        model = nn.Linear(2, 2).eval()
        with self.assertRaises(ValueError): BorrowedModelBridge(model, forward=forward, output=output, policy=DeviceTensorPolicy(max_state_bytes=1))
        model.hidden_tensor = torch.ones(2)
        with self.assertRaises(ValueError): BorrowedModelBridge(model, forward=forward, output=output)


if __name__ == '__main__': unittest.main()
