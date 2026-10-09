import copy
import math
import os
import unittest
from sbercluster.resources import (limit_breaches, memory_pressure_percent,
                                   WindowsJob, verify_worker_job)


class ResourceLimits(unittest.TestCase):
    def setUp(self):
        self.sample = {'cpu_percent': 10., 'ram_percent': 60.,
                       'disk_active_percent': {'0 C:': 10., '1 B:': 2.},
                       'gpus': [{'gpu_percent': 5., 'vram_percent': 20.}]}

    def test_each_disk_is_checked_separately(self):
        self.sample['disk_active_percent']['1 B:'] = 71.
        self.assertEqual(limit_breaches(self.sample, ceiling=70), {'disk:1 B:': 71.})

    def test_exact_ceiling_and_invalid_measurement(self):
        self.sample['cpu_percent'] = 70.
        self.assertEqual(limit_breaches(self.sample, ceiling=70), {})
        self.sample['ram_percent'] = math.nan
        self.assertIn('ram', limit_breaches(self.sample, ceiling=70))

    def test_vram_limit_is_separate_from_gpu_usage(self):
        self.sample['gpus'][0]['vram_percent'] = 90.
        self.assertEqual(limit_breaches(self.sample, ceiling=70), {'vram:0': 90.})

    def test_worker_requires_named_supervisor(self):
        with self.assertRaises(PermissionError):
            verify_worker_job(None, 20, 1024**3)

    def test_memory_pressure_uses_available_memory(self):
        # Reclaimable cache is included in OS-reported available memory.
        self.assertEqual(memory_pressure_percent(1000, 400), 60.)
        with self.assertRaises(ValueError):
            memory_pressure_percent(1000, 1001)

    def test_custom_ceiling_applies_to_every_measured_component(self):
        self.sample['cpu_percent'] = 81.
        self.assertEqual(limit_breaches(self.sample, ceiling=80), {'cpu': 81.})

    @unittest.skipUnless(os.name == 'nt', 'Windows job object')
    def test_native_job_limits_can_be_set_and_queried(self):
        job = WindowsJob(cpu_percent=20, memory_bytes=1024**3)
        try:
            self.assertTrue(job.settings['cpu_limit_verified'])
            self.assertTrue(job.settings['memory_limit_verified'])
        finally:
            job.close()


if __name__ == '__main__':
    unittest.main()
