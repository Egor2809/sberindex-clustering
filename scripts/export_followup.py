"""Export completed exploratory runs as compact, auditable research artifacts.

Only named scientific artifacts and source modules are eligible. Runtime logs,
resource policies and supervisor implementations are intentionally outside this
export format. Every run gets a new directory; existing exports are immutable.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

SCIENTIFIC_MODULES = {
    'sbercluster/followup.py', 'sbercluster/joint.py', 'sbercluster/features.py',
    'sbercluster/graph.py', 'sbercluster/metrics.py', 'sbercluster/models.py',
    'sbercluster/research.py', 'sbercluster/contest_graphs.py',
    'sbercluster/dynamics.py', 'sbercluster/io.py', 'sbercluster/input_contracts.py',
}
PROVENANCE_FIELDS = {
    'created_utc', 'scope', 'git_commit', 'git_worktree_modified',
    'panel_sha256', 'scaler', 'environment', 'ids', 'frozen_prototypes_sha256',
    'road_sha256', 'road_date', 'baseline_labels_sha256',
    'baseline_provenance_sha256', 'graph_scope', 'objective',
    'attribute_graph_weight', 'transport_graph_sha256', 'joint_labels_sha256',
    'joint_provenance_sha256', 'ward_archive_sha256',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def write_gzip_json(path, value):
    content = (json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                          allow_nan=False)+'\n').encode('utf-8')
    # Blank filename and zero mtime produce identical bytes across exports.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write(content)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row
                               if not isinstance(row[k], (list, dict))))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def public_provenance(source):
    """Allowlisted scientific settings, without machine paths/runtime policies."""
    result = {k: source[k] for k in PROVENANCE_FIELDS if k in source}
    cfg = source['config']
    result['config'] = {k: cfg[k] for k in (
        'schema_version', 'seed', 'data_contract', 'features', 'graph',
        'clustering', 'validation') if k in cfg}
    result['config']['followup'] = {k: v for k, v in cfg['followup'].items()
                                    if k not in {'output', 'source_dir', 'baseline_dir', 'joint_dir', 'ward_archive'}}
    result['config']['execution'] = {
        'allow_clustering': cfg.get('execution', {}).get('allow_clustering', False),
        'task': cfg.get('execution', {}).get('task', 'followup_v4'),
    }
    result['implementation'] = {k: v for k, v in source.get('implementation', {}).items()
                                if k in SCIENTIFIC_MODULES}
    return result


def export_run(run, output_root):
    run = Path(run).resolve()
    output = Path(output_root).resolve()/run.name
    status = json.loads((run/'status.json').read_text('utf-8'))
    if status.get('status') != 'completed':
        raise ValueError(f'{run.name}: only completed runs may be exported')
    if output == run or run in output.parents:
        raise ValueError('Export must be outside its source run')
    if output.exists():
        raise FileExistsError(f'Export already exists: {output}')
    source = json.loads((run/'provenance.json').read_text('utf-8'))
    # Validate expected stage products before creating a destination.
    stage = status['stage']
    if stage not in {'small_k', 'joint', 'joint_stability', 'temporal'}:
        raise ValueError('Unsupported scientific stage')
    temporal_files = sorted(run.glob('temporal_omega*.json'))
    required = ['status.json','summary.json','provenance.json','graphs.json']
    required += ['temporal_summary.json'] if stage == 'temporal' else ['metrics.json','labels.json','network_metrics.json']
    for name in required:
        if not (run/name).is_file():
            raise FileNotFoundError(f'{run.name}: missing {name}')
    if stage == 'temporal' and len(temporal_files) != status['temporal_variants']:
        raise ValueError('Temporal variant count differs from completed status')
    for relative, expected in source.get('implementation', {}).items():
        snapshot = run/'implementation_snapshot'/relative
        if relative in SCIENTIFIC_MODULES and snapshot.is_file() and digest(snapshot) != expected:
            raise ValueError(f'Scientific snapshot hash mismatch: {relative}')
    output.mkdir(parents=True, exist_ok=False)
    origin_files = {}

    def read_json(name):
        path = run/name
        value = json.loads(path.read_text('utf-8'))
        origin_files[name] = digest(path)
        return value

    provenance = public_provenance(source)
    provenance['source_provenance_sha256'] = digest(run/'provenance.json')
    provenance['source_run_name'] = run.name
    write_json(output/'provenance.json', provenance)
    origin_files['provenance.json'] = digest(run/'provenance.json')
    for name in ['status.json','summary.json','graphs.json']:
        write_json(output/name, read_json(name))
    if stage == 'temporal':
        rows = read_json('temporal_summary.json')
        write_json(output/'temporal_summary.json', rows)
        write_csv(output/'temporal_summary.csv', rows)
        monthly = []
        for path in temporal_files:
            value = read_json(path.name)
            write_gzip_json(output/(path.name+'.gz'), value)
            monthly.extend({'omega_relative':value['summary']['omega_relative'], **row}
                           for row in value['monthly_metrics'])
        write_csv(output/'monthly_metrics.csv', monthly)
    else:
        rows = read_json('metrics.json')
        write_json(output/'metrics.json', rows)
        write_csv(output/'metrics.csv', rows)
        write_gzip_json(output/'labels.json.gz', read_json('labels.json'))
        write_json(output/'network_metrics.json', read_json('network_metrics.json'))
        for path in sorted(run.glob('joint_region_shuffled_*_permutation.json')):
            write_gzip_json(output/(path.name+'.gz'), read_json(path.name))
    for relative in sorted(SCIENTIFIC_MODULES):
        snapshot = run/'implementation_snapshot'/relative
        if relative in provenance['implementation'] and snapshot.is_file():
            target = output/'implementation_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot, target)
            origin_files['implementation_snapshot/'+relative] = digest(snapshot)
    write_json(output/'manifest.json', {
        'schema_version':1, 'exported_utc':datetime.now(timezone.utc).isoformat(),
        'source_run_name':run.name, 'stage':stage,
        'scope':'Exploratory follow-up; 2024 already inspected; no new confirmatory holdout',
        'source_files_sha256':origin_files,
        'files_sha256':{p.relative_to(output).as_posix():digest(p)
                        for p in sorted(output.rglob('*')) if p.is_file()},
        'exporter_sha256':digest(Path(__file__)),
        'compressed_json':'gzip; UTF-8; deterministic gzip metadata; no label subsampling',
    })
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True,
                        help='Completed run directory; repeat for multiple stages')
    parser.add_argument('--output', default='reports/review-2026-09-24',
                        help='Destination parent; per-run subdirectories must not exist')
    args=parser.parse_args()
    for run in args.run:
        print(json.dumps({'export':str(export_run(run,args.output))}))


if __name__ == '__main__':
    main()
