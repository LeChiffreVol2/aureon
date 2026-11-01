"""No hardware/runtime imports: capacity math is separate from qualification."""
import math
import os
import unittest

from aureon.planning.best_runtime import BestDeviceBudget, best_usage_cost, plan_best_replicas

GIB = 1024**3


class BestRuntimePlanningTests(unittest.TestCase):
    def test_configurable_replica_count_across_devices_and_hosts(self):
        devices = [BestDeviceBudget("GPU-test", 80*GIB, 72*GIB, host=host)
                   for host in ("worker-a", "worker-b")]
        before = dict(os.environ)
        plan = plan_best_replicas(devices, per_replica_bytes=4*GIB,
                                  reserve_per_device_bytes=8*GIB, replicas_per_device=12)
        self.assertEqual(plan.replicas, 24)
        self.assertEqual(plan.placements[0].capacity, 16)
        self.assertEqual(plan.placements[0].unassigned_bytes, 16*GIB)
        self.assertEqual(plan.placements[0].worker_environment(), {"CUDA_VISIBLE_DEVICES":"GPU-test"})
        self.assertEqual(os.environ, before)
        self.assertEqual(plan.to_dict()["processes_launched"], 0)
        self.assertIn("not_measured", plan.qualification)

    def test_memory_overcommit_is_rejected_not_silently_reduced(self):
        device = BestDeviceBudget("0", 8*GIB, 7*GIB)
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan_best_replicas([device], per_replica_bytes=3*GIB,
                              reserve_per_device_bytes=GIB, replicas_per_device=3)
        plan = plan_best_replicas([device], per_replica_bytes=3*GIB,
                                  reserve_per_device_bytes=GIB, replicas_per_device=2)
        self.assertEqual(plan.placements[0].unassigned_bytes, 0)

    def test_duplicate_device_is_not_counted_twice(self):
        device = BestDeviceBudget("GPU-one", 80*GIB, 80*GIB)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            plan_best_replicas([device, device], per_replica_bytes=GIB, reserve_per_device_bytes=0)

    def test_unsupported_precision_and_graph_do_not_fallback(self):
        device = BestDeviceBudget("MIG-GPU-ab/1/2", 8*GIB, 8*GIB, cuda_graph=False)
        with self.assertRaisesRegex(ValueError, "Graph"):
            plan_best_replicas([device], per_replica_bytes=GIB, reserve_per_device_bytes=0)
        plan = plan_best_replicas([device], per_replica_bytes=GIB, reserve_per_device_bytes=0,
                                  profile="reference_fp32")
        self.assertEqual(plan.profile, "reference_fp32")
        with self.assertRaisesRegex(ValueError, "only"):
            plan_best_replicas([device], per_replica_bytes=GIB, reserve_per_device_bytes=0,
                              profile="amp_fp16")

    def test_invalid_counts_and_device_labels(self):
        for token in ("0,1", "-1", "GPU-one\n", "", "$(command)"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                BestDeviceBudget(token, GIB, GIB)
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_best_replicas([BestDeviceBudget("0", GIB, GIB)],
                                  per_replica_bytes=1, reserve_per_device_bytes=0, replicas_per_device=value)

    def test_cost_includes_failed_occupancy_but_divides_by_delivered_outputs(self):
        result = best_usage_cost(resource_seconds=1800, completed_outputs=2, hourly_rate=1)
        self.assertEqual(result["total"], .5)
        self.assertEqual(result["per_completed_output"], .25)
        self.assertEqual(result["per_1000_completed_outputs"], 250)
        self.assertFalse(result["billable"])

    def test_empty_completion_keeps_real_occupancy_without_division_by_zero(self):
        result = best_usage_cost(resource_seconds=3600, completed_outputs=0, hourly_rate=1)
        self.assertEqual(result["total"], 1)
        self.assertIsNone(result["per_completed_output"])
        self.assertIsNone(result["per_1000_completed_outputs"])

    def test_invalid_cost_inputs(self):
        for value in (True, -1, math.inf, math.nan, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                best_usage_cost(resource_seconds=value, completed_outputs=1, hourly_rate=1)
        with self.assertRaises(ValueError):
            best_usage_cost(resource_seconds=1e308, completed_outputs=1, hourly_rate=1e308)


if __name__ == "__main__":
    unittest.main()
