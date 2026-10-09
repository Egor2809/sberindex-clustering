import unittest
from scripts.run_bounded import await_calm


class Clock:
    def __init__(self): self.value = 0.
    def sleep(self, seconds): self.value += seconds
    def now(self): return self.value


class Monitor:
    def __init__(self, disks, clock, delay=0):
        self.disks=iter(disks);self.clock=clock;self.delay=delay
    def sample(self):
        self.clock.value += self.delay
        return {'cpu_percent':5.,'ram_percent':40.,'disk_active_percent':{'C:':next(self.disks)},
                'gpus':[{'gpu_percent':0.,'vram_percent':10.}]}


class PausedLoadMonitor(unittest.TestCase):
    def test_three_consecutive_readings_reset_after_new_load(self):
        clock=Clock();samples=[]
        result,last=await_calm(Monitor([71,64,65,66,64,63,65],clock),10,0,0,samples,sample_interval=1,recovery_ceiling=65,maximum_sample_gap=1.5,sleep=clock.sleep,now=clock.now)
        self.assertEqual((result,last),('ready',7.))
        self.assertEqual(len(samples),7)
        self.assertTrue(all(s['phase']=='paused' for s in samples))

    def test_monitor_delay_cannot_resume_worker(self):
        clock=Clock();samples=[]
        result,_=await_calm(Monitor([10],clock,delay=.6),10,0,0,samples,sample_interval=1,recovery_ceiling=65,maximum_sample_gap=1.5,sleep=clock.sleep,now=clock.now)
        self.assertEqual(result,'monitor_delay')

    def test_bad_reading_and_timeout_cannot_resume_worker(self):
        clock=Clock();samples=[]
        result,_=await_calm(Monitor([64,float('nan'),64,64],clock),4,0,0,samples,sample_interval=1,recovery_ceiling=65,maximum_sample_gap=1.5,sleep=clock.sleep,now=clock.now)
        self.assertEqual(result,'timeout')


if __name__=='__main__': unittest.main()
