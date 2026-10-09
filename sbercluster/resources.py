"""Windows process limits and sampled system load. Disk load is monitored, not capped."""
from __future__ import annotations
import ctypes as ct
from ctypes import wintypes as wt
import math
import os
from pathlib import Path
import uuid
import psutil


class IOCounts(ct.Structure):
    _fields_ = [(n, ct.c_ulonglong) for n in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class BasicLimits(ct.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ct.c_longlong), ('PerJobUserTimeLimit', ct.c_longlong),
        ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ct.c_size_t),
        ('MaximumWorkingSetSize', ct.c_size_t), ('ActiveProcessLimit', wt.DWORD),
        ('Affinity', ct.c_size_t), ('PriorityClass', wt.DWORD), ('SchedulingClass', wt.DWORD)]


class ExtendedLimits(ct.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IOCounts),
        ('ProcessMemoryLimit', ct.c_size_t), ('JobMemoryLimit', ct.c_size_t),
        ('PeakProcessMemoryUsed', ct.c_size_t), ('PeakJobMemoryUsed', ct.c_size_t)]


class CPULimit(ct.Structure):
    _fields_ = [('ControlFlags', wt.DWORD), ('CpuRate', wt.DWORD)]


class WindowsJob:
    def __init__(self, cpu_percent, memory_bytes):
        if not 0 < cpu_percent <= 100 or not 0 < memory_bytes <= psutil.virtual_memory().total:
            raise ValueError('Invalid resource limit')
        self.k = ct.WinDLL('kernel32', use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
        self.k.CreateJobObjectW.restype = wt.HANDLE
        self.k.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
        self.k.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
        self.k.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self.k.CloseHandle.argtypes = [wt.HANDLE]
        self.name = 'Local\\SberCluster-' + uuid.uuid4().hex
        self.handle = self.k.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        try:
            limits = ExtendedLimits()
            # Kill all assigned processes on handle closure; bound aggregate private committed memory.
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200 | 0x100
            limits.ProcessMemoryLimit = limits.JobMemoryLimit = memory_bytes
            cpu = CPULimit(0x1 | 0x4, round(cpu_percent * 100))
            for kind, value in [(9, limits), (15, cpu)]:
                if not self.k.SetInformationJobObject(self.handle, kind, ct.byref(value), ct.sizeof(value)):
                    raise ct.WinError(ct.get_last_error())
            actual = CPULimit()
            if not self.k.QueryInformationJobObject(self.handle, 15, ct.byref(actual), ct.sizeof(actual), None):
                raise ct.WinError(ct.get_last_error())
            if actual.CpuRate != cpu.CpuRate or actual.ControlFlags != cpu.ControlFlags:
                raise RuntimeError('CPU hard cap verification failed')
            actual_memory = ExtendedLimits()
            if not self.k.QueryInformationJobObject(self.handle, 9, ct.byref(actual_memory), ct.sizeof(actual_memory), None):
                raise ct.WinError(ct.get_last_error())
            if (actual_memory.JobMemoryLimit != memory_bytes or actual_memory.ProcessMemoryLimit != memory_bytes
                    or actual_memory.BasicLimitInformation.LimitFlags & 0x2300 != 0x2300):
                raise RuntimeError('Memory hard cap verification failed')
            self.settings = {'cpu_hard_cap_percent': cpu_percent,
                             'job_private_commit_limit_bytes': memory_bytes,
                             'kill_on_job_close': True, 'cpu_limit_verified': True, 'memory_limit_verified': True}
        except BaseException:
            self.close()
            raise

    def assign(self, process):
        if not self.k.AssignProcessToJobObject(self.handle, wt.HANDLE(int(process._handle))):
            raise ct.WinError(ct.get_last_error())

    def close(self):
        if getattr(self, 'handle', None):
            self.k.CloseHandle(self.handle)
            self.handle = None


def verify_worker_job(name, expected_cpu_percent, expected_memory_bytes):
    """Refuse the internal worker entry point unless attached to the named supervisor job."""
    if not name or not name.startswith('Local\\SberCluster-'):
        raise PermissionError('Missing resource supervisor job')
    k = ct.WinDLL('kernel32', use_last_error=True)
    k.OpenJobObjectW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
    k.OpenJobObjectW.restype = wt.HANDLE
    k.GetCurrentProcess.restype = wt.HANDLE
    k.IsProcessInJob.argtypes = [wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)]
    k.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
    k.CloseHandle.argtypes = [wt.HANDLE]
    handle = k.OpenJobObjectW(0x4, False, name)
    if not handle:
        raise PermissionError('Resource supervisor job unavailable')
    try:
        inside = wt.BOOL()
        if not k.IsProcessInJob(k.GetCurrentProcess(), handle, ct.byref(inside)) or not inside.value:
            raise PermissionError('Worker is not attached to the resource supervisor job')
        cpu, memory = CPULimit(), ExtendedLimits()
        for kind, value in [(15, cpu), (9, memory)]:
            if not k.QueryInformationJobObject(handle, kind, ct.byref(value), ct.sizeof(value), None):
                raise ct.WinError(ct.get_last_error())
        if (cpu.ControlFlags != 5 or cpu.CpuRate != round(expected_cpu_percent * 100)
                or memory.JobMemoryLimit != expected_memory_bytes
                or memory.ProcessMemoryLimit != expected_memory_bytes
                or memory.BasicLimitInformation.LimitFlags & 0x2300 != 0x2300):
            raise PermissionError('Worker resource limits differ from the required profile')
    finally:
        k.CloseHandle(handle)


def resume_initial_thread(process):
    """Resume a CREATE_SUSPENDED process after its job and affinity have been installed."""
    threads = psutil.Process(process.pid).threads()
    if len(threads) != 1:
        raise RuntimeError('Expected one thread in the new suspended worker')
    k = ct.WinDLL('kernel32', use_last_error=True)
    k.OpenThread.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.OpenThread.restype = wt.HANDLE
    k.ResumeThread.argtypes = [wt.HANDLE]
    k.ResumeThread.restype = wt.DWORD
    k.CloseHandle.argtypes = [wt.HANDLE]
    handle = k.OpenThread(0x2, False, threads[0].id)
    if not handle:
        raise ct.WinError(ct.get_last_error())
    try:
        previous = k.ResumeThread(handle)
        if previous != 1:
            raise RuntimeError(f'Unexpected initial suspend count: {previous}')
    finally:
        k.CloseHandle(handle)


class CounterValue(ct.Structure):
    _fields_ = [('status', wt.DWORD), ('value', ct.c_double)]


class CounterItem(ct.Structure):
    _fields_ = [('name', wt.LPWSTR), ('value', CounterValue)]


class NVMLMemory(ct.Structure):
    _fields_ = [('total', ct.c_ulonglong), ('free', ct.c_ulonglong), ('used', ct.c_ulonglong)]


class NVMLUtilization(ct.Structure):
    _fields_ = [('gpu', ct.c_uint), ('memory', ct.c_uint)]


class NvidiaLoad:
    """Persistent read-only NVML queries avoid spawning nvidia-smi per sample.

    https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
    Unsupported or missing readings raise: no missing measurement becomes zero.
    """
    def __init__(self):
        library = Path(os.environ['SystemRoot']) / 'System32' / 'nvml.dll'
        self.api = ct.CDLL(str(library))
        self.initialized = False
        signatures = {
            'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetCount_v2': [ct.POINTER(ct.c_uint)],
            'nvmlDeviceGetHandleByIndex_v2': [ct.c_uint, ct.POINTER(ct.c_void_p)],
            'nvmlDeviceGetMemoryInfo': [ct.c_void_p, ct.POINTER(NVMLMemory)],
            'nvmlDeviceGetUtilizationRates': [ct.c_void_p, ct.POINTER(NVMLUtilization)],
        }
        for name, args in signatures.items():
            fn = getattr(self.api, name); fn.argtypes = args; fn.restype = ct.c_int
        self.check(self.api.nvmlInit_v2()); self.initialized = True
        try:
            count = ct.c_uint(); self.check(self.api.nvmlDeviceGetCount_v2(ct.byref(count)))
            if count.value == 0: raise RuntimeError('No NVIDIA device available for monitoring')
            self.devices = []
            for index in range(count.value):
                handle = ct.c_void_p()
                self.check(self.api.nvmlDeviceGetHandleByIndex_v2(index, ct.byref(handle)))
                self.devices.append(handle)
        except BaseException:
            self.close(); raise

    @staticmethod
    def check(code):
        if code: raise RuntimeError(f'NVML measurement failed: {code}')

    def sample(self):
        values = []
        for handle in self.devices:
            memory, utilization = NVMLMemory(), NVMLUtilization()
            self.check(self.api.nvmlDeviceGetMemoryInfo(handle, ct.byref(memory)))
            self.check(self.api.nvmlDeviceGetUtilizationRates(handle, ct.byref(utilization)))
            if not 0 <= memory.used <= memory.total or memory.total == 0 or utilization.gpu > 100:
                raise RuntimeError('Invalid NVML measurement')
            values.append({'gpu_percent': float(utilization.gpu),
                           'vram_percent': memory.used / memory.total * 100})
        return values

    def close(self):
        if self.initialized:
            self.api.nvmlShutdown(); self.initialized = False


class SystemLoad:
    def __init__(self):
        self.gpu = NvidiaLoad()
        self.p = ct.WinDLL('pdh')
        self.p.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
        self.p.PdhAddEnglishCounterW.argtypes = [wt.HANDLE, wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
        self.p.PdhCollectQueryData.argtypes = [wt.HANDLE]
        self.p.PdhGetFormattedCounterArrayW.argtypes = [wt.HANDLE, wt.DWORD, ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD), ct.c_void_p]
        self.p.PdhCloseQuery.argtypes = [wt.HANDLE]
        self.query, self.disk = wt.HANDLE(), wt.HANDLE()
        self.check(self.p.PdhOpenQueryW(None, 0, ct.byref(self.query)))
        try:
            self.check(self.p.PdhAddEnglishCounterW(self.query, r'\PhysicalDisk(*)\% Idle Time', 0, ct.byref(self.disk)))
            self.check(self.p.PdhCollectQueryData(self.query))
            psutil.cpu_percent()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'Windows performance counter error: {code & 0xffffffff:#x}')

    def sample(self):
        self.check(self.p.PdhCollectQueryData(self.query))
        size, count = wt.DWORD(), wt.DWORD()
        code = self.p.PdhGetFormattedCounterArrayW(self.disk, 0x200, ct.byref(size), ct.byref(count), None)
        if code & 0xffffffff != 0x800007d2:
            raise RuntimeError(f'Disk counter size unavailable: {code & 0xffffffff:#x}')
        buf = ct.create_string_buffer(size.value)
        self.check(self.p.PdhGetFormattedCounterArrayW(self.disk, 0x200, ct.byref(size), ct.byref(count), buf))
        values = ct.cast(buf, ct.POINTER(CounterItem))
        disks = {}
        for index in range(count.value):
            item = values[index]
            if item.name == '_Total':
                continue
            if item.value.status not in (0, 1) or not math.isfinite(item.value.value):
                raise RuntimeError('Disk activity reading unavailable')
            disks[item.name] = max(0., min(100., 100 - item.value.value))
        if not disks:
            raise RuntimeError('No physical disk counters available')
        gpus = self.gpu.sample()
        memory = psutil.virtual_memory()
        # available includes reclaimable standby/cache memory. Counting all cached
        # pages as pressure would stop useful work while Windows can still reclaim them.
        ram_percent = memory_pressure_percent(memory.total, memory.available)
        return {'cpu_percent': psutil.cpu_percent(), 'ram_percent': ram_percent,
                'ram_available_bytes': memory.available,
                'ram_measurement': 'total_minus_available_includes_reclaimable_cache',
                'disk_active_percent': disks, 'gpus': gpus}

    def close(self):
        if getattr(self, 'gpu', None):
            self.gpu.close()
        if self.query:
            self.p.PdhCloseQuery(self.query)
            self.query = None


def limit_breaches(sample, ceiling):
    measured = {'cpu': sample['cpu_percent'], 'ram': sample['ram_percent']}
    measured.update({f'disk:{name}': value for name, value in sample['disk_active_percent'].items()})
    for i, gpu in enumerate(sample['gpus']):
        measured.update({f'gpu:{i}': gpu['gpu_percent'], f'vram:{i}': gpu['vram_percent']})
    return {name: value for name, value in measured.items() if not math.isfinite(value) or value > ceiling}


def memory_pressure_percent(total, available):
    """Measure physical-memory pressure excluding available/reclaimable pages."""
    if not 0 <= available <= total or total <= 0:
        raise ValueError('Invalid total/available memory measurement')
    return (total - available) / total * 100
