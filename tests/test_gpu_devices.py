"""Physical identity, headroom and stale-observation checks without CUDA."""
from dataclasses import asdict, replace
import subprocess
import sys
import unittest
from unittest.mock import patch

from aureon.api.gpu_devices import NvidiaDevice, NvidiaInventory, cuda_device_slots, probe_nvidia_gpus

UUID = 'GPU-7217aea9-6bd9-42ee-5a0d-ed78b3a3f974'
GIB = 1024**3


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.device = NvidiaDevice(UUID, 'NVIDIA GeForce RTX 3070', (8, 6), 8*GIB, 6*GIB)
        self.inventory = NvidiaInventory((self.device,), 10.)

    def test_explicit_uuid_binding_and_exact_headroom(self):
        slots = cuda_device_slots(self.inventory, allowed_uuids=[UUID], memory_bytes=5*GIB,
                                  reserve_bytes=GIB, max_residents=7, now=12.)
        self.assertEqual(slots[0].cuda_visible_device, UUID)
        self.assertEqual(slots[0].max_residents, 7)
        self.assertEqual(slots[0].memory_bytes, 5*GIB)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            cuda_device_slots(self.inventory, allowed_uuids=[UUID], memory_bytes=5*GIB+1, reserve_bytes=GIB, now=12.)

    def test_reject_aliases_missing_gpus_and_stale_or_future_snapshots(self):
        options = dict(memory_bytes=GIB, reserve_bytes=GIB, now=12.)
        for selectors in ([UUID, UUID.upper()], ['0'], [], UUID):
            with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                cuda_device_slots(self.inventory, allowed_uuids=selectors, **options)
        for now in (9., 16., float('nan'), True):
            with self.subTest(now=now), self.assertRaises(ValueError):
                cuda_device_slots(self.inventory, allowed_uuids=[UUID], memory_bytes=GIB, reserve_bytes=0, now=now)
        with self.assertRaises(ValueError): NvidiaInventory((self.device, replace(self.device, uuid=UUID.upper())), 10.)

    def test_reject_unknown_capacity_and_mutable_or_nonsense_values(self):
        for changes in ({'free_bytes':9*GIB}, {'free_bytes':True}, {'total_bytes':0},
                        {'uuid':'MIG-missing'}, {'compute_capability':[8,6]}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(self.device, **changes)
        self.assertEqual(asdict(self.inventory)['devices'][0]['compute_capability'], (8,6))

    def test_driver_probe_uses_no_shell_and_no_cuda_runtime(self):
        response = subprocess.CompletedProcess([], 0, UUID+', NVIDIA GeForce RTX 3070, 8192, 6144, 8.6\n', '')
        with patch('aureon.api.gpu_devices.shutil.which', return_value='/usr/bin/nvidia-smi'), \
             patch('aureon.api.gpu_devices.subprocess.run', return_value=response) as run:
            result = probe_nvidia_gpus()
            self.assertEqual(result.devices[0], self.device)
            self.assertEqual(run.call_args.args[0][0], '/usr/bin/nvidia-smi')
            self.assertNotIn('shell', run.call_args.kwargs)
            self.assertEqual(run.call_args.kwargs['timeout'], 10.)
        child = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import aureon.api.gpu_devices; assert not any(x in sys.modules for x in ('torch','numpy','terratorch'))"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_unsupported_driver_and_missing_program_fail_explicitly(self):
        for output in ('uuid, gpu, [N/A], 2, 8.6', UUID+', gpu, 8192, 2, [N/A]', 'a,b,c', 'x'*1048577):
            response = subprocess.CompletedProcess([], 0, output, '')
            with patch('aureon.api.gpu_devices.shutil.which', return_value='nvidia-smi'), \
                 patch('aureon.api.gpu_devices.subprocess.run', return_value=response), self.assertRaises(ValueError):
                probe_nvidia_gpus()
        with patch('aureon.api.gpu_devices.shutil.which', return_value=None), self.assertRaises(RuntimeError):
            probe_nvidia_gpus()
        for timeout in (0, -1, 61, float('inf'), True):
            with self.assertRaises(ValueError): probe_nvidia_gpus(timeout_seconds=timeout)


if __name__ == '__main__': unittest.main()
