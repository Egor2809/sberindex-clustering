"""Optional Windows supervisor with configurable Job Object and sampled load limits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(config):
    # Nothing beyond the standard library is imported until the parent has installed the limits.
    if sys.stdin.readline().strip() != 'run':
        raise RuntimeError('No start signal from resource supervisor')
    from sbercluster.resources import verify_worker_job
    verify_worker_job(os.environ.get('SBERCLUSTER_JOB_NAME'),
                      float(os.environ['SBERCLUSTER_JOB_CPU_PERCENT']),
                      int(os.environ['SBERCLUSTER_JOB_MEMORY_BYTES']))
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if cfg.get('execution', {}).get('task') == 'contest_v3':
        from sbercluster.contest import run
        run(cfg, ROOT)
    elif cfg.get('execution', {}).get('task') == 'research_v2':
        from sbercluster.research import run
        run(cfg, ROOT)
    elif cfg.get('execution', {}).get('task') == 'round2_v5':
        from sbercluster.round2 import run
        run(cfg, ROOT)
    elif cfg.get('execution', {}).get('task') == 'followup_v4':
        from sbercluster.followup import run
        run(cfg, ROOT)
    elif cfg.get('execution', {}).get('task') == 'external_validation_v4':
        import runpy
        settings = cfg['external_validation']
        script = ROOT / 'scripts/external_validation_v4.py'
        if not script.is_file():
            raise FileNotFoundError('external_validation_v4 implementation is unavailable')
        output = (ROOT / settings['output']).resolve()
        if not output.is_relative_to(ROOT.resolve()):
            raise ValueError('external_validation output must stay inside the repository')
        bootstrap = int(settings['bootstrap'])
        if bootstrap < 20:
            raise ValueError('external_validation bootstrap must be at least 20')
        sys.argv = [str(script), '--output', str(output), '--bootstrap', str(bootstrap)]
        runpy.run_path(str(script), run_name='__main__')
    else:
        from sbercluster.cli import main
        main(['run', '--root', str(ROOT), '--config', config, '--execute-clustering'])



def await_calm(monitor, deadline, last_sample, started, samples, *, sample_interval,
               recovery_ceiling, maximum_sample_gap, sleep=time.sleep, now=time.monotonic):
    """Keep an already suspended worker stopped until three safe observations."""
    from sbercluster.resources import limit_breaches
    consecutive = 0
    while now() < deadline:
        sleep(sample_interval)
        sample = monitor.sample()
        measured = now()
        gap = measured - last_sample
        last_sample = measured
        sample.update(seconds=measured-started, phase='paused', sample_gap_seconds=gap)
        samples.append(sample)
        if gap > maximum_sample_gap:
            return 'monitor_delay', last_sample
        consecutive = consecutive + 1 if not limit_breaches(sample, ceiling=recovery_ceiling) else 0
        if consecutive >= 3:
            return 'ready', last_sample
    return 'timeout', last_sample


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ('cpu_hard_cap_percent', 'memory_limit_bytes', 'system_monitor_ceiling_percent',
                'recovery_ceiling_percent', 'sample_interval_seconds', 'threads',
                'active_seconds', 'pause_seconds')
    missing = [name for name in required if name not in profile]
    if missing:
        raise ValueError('Missing supervisor profile fields: ' + ', '.join(missing))
    numeric = [profile[name] for name in required if name != 'threads']
    if (not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric)
            or not 0 < profile['cpu_hard_cap_percent'] <= 100
            or profile['memory_limit_bytes'] <= 0
            or not 0 < profile['recovery_ceiling_percent'] <= profile['system_monitor_ceiling_percent'] <= 100
            or profile['sample_interval_seconds'] <= 0 or profile['active_seconds'] <= 0
            or profile['pause_seconds'] < 0 or not isinstance(profile['threads'], int)
            or profile['threads'] < 1):
        raise ValueError('Invalid supervisor profile')
    return profile


def supervise(config, profile_path, timeout, wait_for_headroom=0, pause_for_headroom=0):
    import psutil
    from sbercluster.resources import SystemLoad, WindowsJob, limit_breaches, resume_initial_thread
    if os.name != 'nt':
        raise RuntimeError('This supervisor requires Windows')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be a finite positive number of seconds')
    if not math.isfinite(wait_for_headroom) or wait_for_headroom < 0:
        raise ValueError('Headroom wait must be finite and nonnegative')
    if not math.isfinite(pause_for_headroom) or pause_for_headroom < 0:
        raise ValueError('Pause timeout must be finite and nonnegative')
    config_path = (ROOT / config).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['execution'].get('allow_clustering') is not True:
        raise PermissionError('The selected experiment configuration is disabled')
    profile_path = (ROOT / profile_path).resolve()
    profile = load_profile(profile_path)
    ceiling = profile['system_monitor_ceiling_percent']
    recovery_ceiling = profile['recovery_ceiling_percent']
    sample_interval = profile['sample_interval_seconds']
    maximum_sample_gap = max(1.5, sample_interval * 1.5)
    log_dir = ROOT / 'artifacts/resource_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_dir.mkdir(parents=True)
    report = {'status': 'preflight', 'config': cfg, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'supervisor_profile': profile,
              'supervisor_profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
              'monitor_ceiling_percent': ceiling, 'target_monitor_interval_seconds': sample_interval,
              'gpu_used_by_training': False, 'disk_limit_type': 'sampled stop; no instantaneous hard cap',
              'timeout_seconds': timeout, 'samples': [],
              'pause_for_headroom_seconds': pause_for_headroom, 'pause_events': [],
              'worker_duty_cycle': {'active_seconds': profile['active_seconds'],
                                    'pause_seconds': profile['pause_seconds']}}
    child = job = monitor = None
    started = time.monotonic()
    try:
        monitor = SystemLoad()
        time.sleep(sample_interval)
        baseline = monitor.sample()
        report['samples'].append({'seconds': time.monotonic() - started, 'phase': 'preflight', **baseline})
        wait_deadline = time.monotonic() + wait_for_headroom
        while limit_breaches(baseline, ceiling=ceiling) and time.monotonic() < wait_deadline:
            time.sleep(sample_interval)
            baseline = monitor.sample()
            report['samples'].append({'seconds': time.monotonic() - started, 'phase': 'preflight', **baseline})
        if limit_breaches(baseline, ceiling=ceiling):
            report.update(status='blocked_by_existing_load', breaches=limit_breaches(baseline, ceiling=ceiling))
            return 2
        report['preflight_wait_seconds'] = time.monotonic() - started
        job = WindowsJob(cpu_percent=profile['cpu_hard_cap_percent'],
                         memory_bytes=profile['memory_limit_bytes'])
        report.update(job.settings)
        threads = str(profile['threads'])
        env = {**os.environ, 'OMP_NUM_THREADS': threads, 'OPENBLAS_NUM_THREADS': threads,
               'MKL_NUM_THREADS': threads, 'NUMEXPR_NUM_THREADS': threads, 'POLARS_MAX_THREADS': threads,
               'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONUNBUFFERED': '1', 'SBERCLUSTER_JOB_NAME': job.name,
               'SBERCLUSTER_JOB_CPU_PERCENT': str(profile['cpu_hard_cap_percent']),
               'SBERCLUSTER_JOB_MEMORY_BYTES': str(profile['memory_limit_bytes'])}
        with (log_dir / 'worker.log').open('w', encoding='utf-8') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--worker', '--config', str(config_path)], cwd=ROOT,
                                     env=env, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x4)
            job.assign(child)
            process = psutil.Process(child.pid)
            affinity = process.cpu_affinity()[-profile['threads']:]
            process.cpu_affinity(affinity)
            report['cpu_affinity'] = affinity
            report['worker_pid'] = child.pid
            process.cpu_percent()
            resume_initial_thread(child)
            child.stdin.write(b'run\n')
            child.stdin.close()
            report['status'] = 'running'
            print(json.dumps({'status': 'running', 'log_dir': str(log_dir), **job.settings}), flush=True)
            last_sample = time.monotonic()
            next_sample = last_sample + sample_interval
            while True:
                # Short compute bursts also spread synchronous import/read activity on HDDs.
                # This is pacing, not a hard disk-I/O cap; measured breaches still stop the job.
                time.sleep(profile['active_seconds'])
                paused = []
                if child.poll() is None:
                    try:
                        process.suspend()
                        paused.append(process)
                        for member in process.children(recursive=True):
                            try:
                                member.suspend()
                                paused.append(member)
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                time.sleep(profile['pause_seconds'])
                if time.monotonic() < next_sample and child.poll() is None:
                    for member in reversed(paused):
                        try:
                            member.resume()
                        except psutil.NoSuchProcess:
                            pass
                    continue
                sample = monitor.sample()
                now = time.monotonic()
                sample['seconds'] = now - started
                sample['sample_gap_seconds'] = now - last_sample
                last_sample = now
                next_sample += sample_interval
                try:
                    members = [process] + process.children(recursive=True)
                    sample['worker_tree_rss_bytes'] = sum(member.memory_info().rss for member in members)
                except psutil.NoSuchProcess:
                    pass
                report['samples'].append(sample)
                if sample['sample_gap_seconds'] > maximum_sample_gap:
                    report['status'] = 'stopped_monitor_delay'
                    job.close()
                    child.wait(timeout=10)
                    return 6
                breaches = limit_breaches(sample, ceiling=ceiling)
                if breaches:
                    # The worker tree is already suspended before this sample.
                    # Keep it stopped until three conservative readings permit resumption.
                    event = {'paused_at_seconds': now - started, 'breaches': breaches}
                    report['pause_events'].append(event)
                    deadline = min(now + pause_for_headroom, started + timeout)
                    outcome, last_sample = await_calm(
                        monitor, deadline, last_sample, started, report['samples'],
                        sample_interval=sample_interval, recovery_ceiling=recovery_ceiling,
                        maximum_sample_gap=maximum_sample_gap) if pause_for_headroom > 0 else ('timeout', last_sample)
                    if outcome == 'monitor_delay':
                        report['status'] = 'stopped_monitor_delay'
                        job.close(); child.wait(timeout=10)
                        return 6
                    if outcome != 'ready':
                        event['last_observed_headroom_breaches'] = limit_breaches(
                            report['samples'][-1], ceiling=recovery_ceiling)
                        report.update(status='stopped_resource_limit', breaches=breaches)
                        job.close()
                        child.wait(timeout=10)
                        return 3
                    event['resumed_at_seconds'] = last_sample - started
                    next_sample = last_sample + sample_interval
                if child.poll() is not None:
                    report.update(status='completed' if child.returncode == 0 else 'worker_failed',
                                  worker_exit_code=child.returncode)
                    return 0 if child.returncode == 0 else 4
                if time.monotonic() - started > timeout:
                    report['status'] = 'stopped_timeout'
                    job.close()
                    child.wait(timeout=10)
                    return 5
                for member in reversed(paused):
                    try:
                        member.resume()
                    except psutil.NoSuchProcess:
                        pass
    except BaseException as exc:
        report.update(status='supervisor_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if job:
            job.close()
        if child and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if monitor:
            monitor.close()
        report['elapsed_seconds'] = time.monotonic() - started
        (log_dir / 'resources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        # A killed worker cannot update its own status file. Reconcile only this run's trusted path.
        if report['status'] != 'completed' and (log_dir / 'worker.log').exists():
            for line in (log_dir / 'worker.log').read_text('utf-8').splitlines():
                try:
                    run_info = json.loads(line)
                    if not isinstance(run_info, dict):
                        continue
                    run_path = Path(run_info.get('research_run', '')).resolve()
                    if run_path.is_relative_to((ROOT / 'runs').resolve()) and (run_path / 'status.json').exists():
                        state = json.loads((run_path / 'status.json').read_text('utf-8'))
                        if state.get('status') == 'running':
                            state.update(status='interrupted', supervisor_status=report['status'], resource_run=log_dir.name)
                            (run_path / 'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
                except (ValueError, TypeError):
                    continue

        print(json.dumps({'status': report['status'], 'report': str(log_dir / 'resources.json')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--profile', help='JSON supervisor profile (local profiles may live in .local/)')
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--wait-for-headroom', type=float, default=0, help='Seconds to wait before starting when existing load is too high')
    p.add_argument('--pause-for-headroom', type=float, default=0,
                   help='Keep worker suspended after a breach; resume at the profile recovery ceiling')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.config)
    else:
        if not args.profile:
            p.error('--profile is required for the supervisor')
        sys.exit(supervise(args.config, args.profile, args.timeout,
                           args.wait_for_headroom, args.pause_for_headroom))
