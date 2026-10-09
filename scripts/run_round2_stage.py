"""Run a missing stage, or verify and reuse its completed export/raw result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
try:
    from .export_round2 import required_evidence
except ImportError:
    from export_round2 import required_evidence

ROOT = Path(__file__).resolve().parents[1]
PATH_SETTINGS = {'output', 'source_dir', 'baseline_dir', 'joint_dir', 'ward_archive'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_configuration(prior, cfg):
    for key in ('schema_version', 'data_contract', 'features', 'graph', 'clustering', 'validation', 'seed'):
        if prior['config'].get(key) != cfg.get(key):
            raise ValueError('Scientific configuration differs: ' + key)
    if prior['config'].get('execution', {}).get('task') != cfg.get('execution', {}).get('task'):
        raise ValueError('Execution task differs')
    clean = lambda value: {k: v for k, v in value.items() if k not in PATH_SETTINGS}
    if clean(prior['config']['followup']) != clean(cfg['followup']):
        raise ValueError('Stage configuration differs')


def verify_completed(root, directory, cfg, *, published):
    read = lambda name: json.loads((directory / name).read_text('utf-8'))
    if published:
        manifest = read('manifest.json')
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'}
        if set(manifest['files_sha256']) != actual:
            raise ValueError('Published manifest does not cover every result file')
        if not {'status.json', 'summary.json', 'provenance.json', 'graphs.json'} <= actual:
            raise ValueError('Published evidence is incomplete')
        for name, h in manifest['files_sha256'].items():
            if digest(directory / name) != h:
                raise ValueError('Published result differs: ' + name)
    state, summary, prior = read('status.json'), read('summary.json'), read('provenance.json')
    if (state.get('status') != 'completed' or summary.get('status') != 'completed'
            or state.get('stage') != cfg['followup']['stage'] or summary.get('stage') != state['stage']):
        raise ValueError('Completed output with matching stage required')
    verify_configuration(prior, cfg)
    expected=required_evidence(cfg['followup'])
    if published:
        expected={name+'.gz' if name=='labels.json' or name.startswith('temporal_omega') else
                  name+'.gz' if name.endswith('/trace.json') and not (directory/name).exists() else name
                  for name in expected}
    missing=sorted(name for name in expected if not (directory/name).is_file())
    if missing:raise ValueError('Completed stage evidence is incomplete: '+', '.join(missing[:5]))
    panel = root / 'data/processed/panel.csv'
    panel_manifest = json.loads((panel.parent / 'manifest.json').read_text('utf-8'))
    if digest(panel) != panel_manifest['panel_sha256'] or prior.get('panel_sha256') != panel_manifest['panel_sha256']:
        raise ValueError('Prepared input differs from completed stage')
    sources = manifest['source_objects'] if published else prior['sources']
    if not sources or sources != prior['sources']:
        raise ValueError('Scientific source manifest differs from provenance')
    object_root = directory.parent / 'source_objects' if published else directory / 'source_objects'
    for name, h in sources.items():
        if digest(object_root / (h + '.py')) != h or digest(root / name) != h:
            raise ValueError('Scientific implementation differs: ' + name)
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args(argv)
    cfg = json.loads((ROOT / a.config).read_text('utf-8'))
    if cfg.get('execution', {}).get('allow_clustering') is not True:
        raise PermissionError('The selected stage configuration is disabled')
    if cfg['execution'].get('task')!='round2_v5':
        raise ValueError('run_round2_stage.py accepts only execution.task=round2_v5')
    run = ROOT / cfg['followup']['output']
    out = ROOT / 'reports/round2-2026-09-24' / run.name
    if out.exists():
        state = verify_completed(ROOT, out, cfg, published=True)
        print(json.dumps({'status': 'reused_verified_completed_stage', 'stage': state['stage'],
                          'path': str(out), 'new_fits': 0}), flush=True)
        return 0
    if run.exists():
        state = verify_completed(ROOT, run, cfg, published=False)
        subprocess.run([sys.executable, str(ROOT / 'scripts/export_round2.py'), '--run', str(run)], cwd=ROOT, check=True)
        print(json.dumps({'status': 'exported_verified_completed_stage', 'stage': state['stage'],
                          'path': str(out), 'new_fits': 0}), flush=True)
        return 0
    subprocess.run([sys.executable, str(ROOT / 'scripts/run_contest.py'), '--config', a.config,
                    '--execute-clustering'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_round2.py'), '--run', str(run)], cwd=ROOT, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
