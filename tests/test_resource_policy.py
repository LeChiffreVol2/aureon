"""Resource decisions with synthetic capability snapshots, no GPU probes."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import unittest

from aureon.api.resource_policy import DeviceCapability, TensorLayout, WorkloadMemory, MemoryCredits, MemoryBudget
from aureon.api.parallel_policy import ParallelPolicy, GIB
from aureon.api.contracts import INPUT_BYTES, OUTPUT_BYTES
from aureon.api.compatibility import inspect_backend_compatibility


class ResourceTests(unittest.TestCase):
    def limits(self, n=32):
        return dict(max_pending=n,max_results=n,max_input_bytes=n*INPUT_BYTES,max_output_bytes=n*OUTPUT_BYTES)

    def test_shape_dtype_batch_weights_and_graph_lifetimes_change_estimate(self):
        base=WorkloadMemory(weights_bytes=123)
        one=base.estimate();two=replace(base,batch_size=2).estimate()
        self.assertEqual(two.weights_bytes,123)
        for key in ('input_bytes','activation_bytes','state_bytes','graph_bytes','pinned_bytes','output_bytes'):
            self.assertEqual(getattr(two,key),2*getattr(one,key))
        half=replace(base,inputs=TensorLayout((12,224,224),'float16')).estimate()
        self.assertEqual(half.activation_bytes,one.activation_bytes//2)
        self.assertGreater(replace(base,inputs=TensorLayout((12,448,448))).estimate().device_bytes,one.device_bytes)
        self.assertEqual(replace(base,graph_copies=0).estimate().graph_bytes,0)
        self.assertEqual(one.device_bytes,one.weights_bytes+one.workspace_bytes+one.reserve_bytes)

    def test_auto_adapts_to_resources_without_gpu_name_or_evidence(self):
        policy=ParallelPolicy('auto',workspace_bytes=24*GIB,pinned_bytes=GIB,output_bytes=GIB)
        small=DeviceCapability('cuda',8*GIB,6*GIB,(8,6),graph=True)
        large=DeviceCapability('cuda',80*GIB,60*GIB,(9,0),graph=True)
        a=policy.plan(self.limits(),None,capability=small)
        b=policy.plan(self.limits(),{'lanes':2},capability=large)
        self.assertGreater(b['planned'],a['planned'])
        self.assertEqual(b,policy.plan(self.limits(),None,capability=large))
        self.assertEqual(b['qualification'],'runtime_admission_not_checked')

    def test_reference_reason_and_exhaustion_are_explicit(self):
        policy=ParallelPolicy('auto')
        cpu=DeviceCapability('cpu',8*GIB,8*GIB)
        self.assertIn('serial_reference',policy.plan(self.limits(),None,capability=cpu)['fallback_reason'])
        self.assertEqual(policy.plan(self.limits(),None,capability=cpu)['planned'],1)
        with self.assertRaisesRegex(ValueError,'fallback is disabled'):
            ParallelPolicy('auto',fallback='error').plan(self.limits(),None,capability=cpu)
        with self.assertRaises(MemoryError):
            policy.plan(self.limits(),None,capability=DeviceCapability('cuda',GIB,1,(8,0),graph=True))
        self.assertIsNotNone(DeviceCapability('cuda',GIB,GIB,(8,0),graph=True,visible_devices=2).lane_rejection())

    def test_host_output_limits_are_independent(self):
        estimate=WorkloadMemory().estimate()
        for policy in (ParallelPolicy('auto',pinned_bytes=estimate.pinned_bytes),
                       ParallelPolicy('auto',output_bytes=estimate.output_bytes)):
            self.assertEqual(policy.plan(self.limits(),None)['planned'],1)
        with self.assertRaises(MemoryError):ParallelPolicy(workspace_bytes=1).plan(self.limits(),None)
        bound=ParallelPolicy(output_bytes=OUTPUT_BYTES).bound_queue_limits(self.limits())
        self.assertEqual(bound['max_output_bytes'],OUTPUT_BYTES)
        self.assertEqual(self.limits()['max_output_bytes'],32*OUTPUT_BYTES)
        with self.assertRaises(MemoryError):ParallelPolicy(output_bytes=1).bound_queue_limits(self.limits())
        with self.assertRaises(MemoryError):ParallelPolicy(workload=WorkloadMemory(weights_bytes=4*GIB)).plan(self.limits(),None)

    def test_invalid_counts_shapes_and_capabilities_fail_closed(self):
        for shape in ((),(True,2),(0,2),(-1,2)):
            with self.assertRaises(ValueError):TensorLayout(shape)
        for kwargs in ({'batch_size':True},{'graph_copies':-1},{'activation_multiplier':0}):
            with self.assertRaises(ValueError):WorkloadMemory(**kwargs)
        with self.assertRaises(ValueError):DeviceCapability('cuda',10,11)
        with self.assertRaises(ValueError):MemoryCredits(device=True)

    def test_atomic_leases_retained_until_explicit_release(self):
        budget=MemoryBudget(MemoryCredits(device=10,pinned=4,output=4))
        graph=budget.reserve(MemoryCredits(device=6))
        def attempt(_):
            try:return budget.reserve(MemoryCredits(device=4,pinned=4,output=4))
            except MemoryError:return None
        with ThreadPoolExecutor(max_workers=8) as pool:tokens=list(pool.map(attempt,range(16)))
        admitted=[t for t in tokens if t is not None]
        self.assertEqual(len(admitted),1)
        with self.assertRaises(RuntimeError):budget.close()
        budget.release(admitted[0]);self.assertEqual(budget.used.device,6)
        with self.assertRaises(ValueError):budget.release(admitted[0])
        budget.release(graph);budget.close()
        with self.assertRaises(RuntimeError):budget.reserve(MemoryCredits())
        metadata=MemoryBudget(MemoryCredits(),max_leases=1)
        metadata.reserve(MemoryCredits())
        with self.assertRaises(MemoryError):metadata.reserve(MemoryCredits())

    def test_dependency_compatibility_is_not_reproducibility_or_qualification(self):
        good=inspect_backend_compatibility('borrowed-cpu',versions={'torch':'2.11.0+cu126','numpy':'2.2.6'},
                                           python=(3,12),platform='win32')
        self.assertTrue(good.dependency_compatible)
        self.assertFalse(good.execution_qualified);self.assertFalse(good.gpu_probed)
        self.assertIsNone(good.reproducibility_lock)
        for version in ('2.12.0','2.11.1','2.11.0rc1'):
            self.assertFalse(inspect_backend_compatibility('borrowed-cpu',versions={'torch':version,'numpy':'2.2.6'},
                python=(3,12),platform='linux').dependency_compatible)
        native=inspect_backend_compatibility('terramind-generation',versions={},python=(3,12),platform='linux')
        self.assertFalse(native.dependency_compatible);self.assertTrue(native.reproducibility_lock)


if __name__=='__main__':unittest.main()
