"""Export completed comparisons with shared content-addressed scientific sources."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,tempfile
try:
    from .export_followup import public_provenance, write_gzip_json, write_json, digest
except ImportError:
    from export_followup import public_provenance, write_gzip_json, write_json, digest


def required_evidence(settings):
    stage=settings['stage'];required={'status.json','summary.json','provenance.json','graphs.json'}
    if stage in ('region_control','dmon'):
        required|={'metrics.json','labels.json','external_comparison.json'}
        if stage=='dmon':
            models={f'dmon_k{k}_seed{seed}' for k in settings['ks'] for seed in settings['seeds']}
            if not models: raise ValueError('Registered DMoN models required')
            required|={model+'/'+name for model in models for name in (
                'best_model.npz','trace.json','info.json','labels.npy','assignments.npy')}
    elif stage=='temporal_grid':
        required|={'temporal_summary.json','temporal_stability.json','transition_events.csv'}
        variants={f'temporal_omega{omega:g}_seed{seed}.json' for omega in settings['omegas'] for seed in settings['seeds']}
        if not variants: raise ValueError('Registered temporal variants required')
        required|=variants
    else: raise ValueError('Unknown round2 stage')
    return required

def export(run, output, compact_traces=False):
    run,output=Path(run),Path(output).resolve();out=output/run.name
    if out.exists(): raise FileExistsError(out)
    output.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+run.name+'-',dir=output))
    try:
        _export(run,output,staging,compact_traces)
        staging.rename(out)
    finally:
        if staging.exists():
            if staging.resolve().parent!=output: raise ValueError('Export staging escaped output directory')
            shutil.rmtree(staging)
    print(str(out),flush=True)
    return out


def _export(run,output,out,compact_traces):
    state=json.loads((run/'status.json').read_text('utf-8'))
    if state.get('status')!='completed': raise ValueError('Only completed experiments may be exported')
    source=json.loads((run/'provenance.json').read_text('utf-8'))
    summary=json.loads((run/'summary.json').read_text('utf-8'))
    if (summary.get('status')!='completed' or summary.get('stage')!=state.get('stage')
            or source['config']['followup'].get('stage')!=state.get('stage')):
        raise ValueError('Completed stage, summary and provenance must agree')
    required=required_evidence(source['config']['followup'])
    missing=sorted(name for name in required if not (run/name).is_file())
    if missing:
        raise ValueError('Completed stage evidence is incomplete: '+', '.join(missing[:5]))
    if not source['sources']: raise ValueError('Scientific source objects required')
    objects=output/'source_objects';objects.mkdir(parents=True,exist_ok=True)
    for rel,h in source['sources'].items():
        origin=run/'source_objects'/f'{h}.py'
        if digest(origin)!=h: raise ValueError('Source object fingerprint differs')
        target=objects/origin.name
        if target.exists() and digest(target)!=h: raise ValueError('Existing source object differs')
        if not target.exists():shutil.copyfile(origin,target)
    provenance=public_provenance(source)
    for key in ['date_utc','packages','sources','input_mode','published_joint_manifest_sha256','published_baseline_manifest_sha256']:
        if key in source: provenance[key]=source[key]
    provenance['source_provenance_sha256']=digest(run/'provenance.json')
    provenance['source_object_root']='../source_objects'
    write_json(out/'provenance.json',provenance)
    allow={'status.json','summary.json','graphs.json','metrics.json','metrics.csv','labels.json',
        'regional_effect.json','external_comparison.json','temporal_summary.json',
        'temporal_stability.json','transition_events.csv'}
    origin_hashes={}
    for path in sorted(run.iterdir()):
        if path.is_file() and (path.name in allow or path.name.startswith('temporal_omega')):
            origin_hashes[path.name]=digest(path)
            if path.name=='labels.json' or path.name.startswith('temporal_omega'):
                write_gzip_json(out/(path.name+'.gz'),json.loads(path.read_text('utf-8')))
            elif compact_traces and path.name=='metrics.json':
                rows=json.loads(path.read_text('utf-8'))
                for row in rows:
                    if 'trace' in row.get('fit',{}):
                        row['fit'].pop('trace')
                        row['fit']['trace_file']=row['id']+'/trace.json.gz'
                write_json(out/path.name,rows)
            else:shutil.copyfile(path,out/path.name)
        elif path.is_dir() and path.name.startswith('dmon_k'):
            target=out/path.name;target.mkdir()
            for name in ['best_model.npz','trace.json','info.json','labels.npy','assignments.npy','epoch1000_labels.npy']:
                origin=path/name
                if not origin.exists():continue
                origin_hashes[path.name+'/'+name]=digest(origin)
                if compact_traces and name=='trace.json':
                    write_gzip_json(target/'trace.json.gz',json.loads(origin.read_text('utf-8')))
                elif compact_traces and name=='info.json':
                    info=json.loads(origin.read_text('utf-8'));info.pop('trace',None);info['trace_file']='trace.json.gz';write_json(target/name,info)
                else:shutil.copyfile(origin,target/name)
    files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'manifest.json',{'schema_version':1,'stage':state['stage'],
        'source_files_sha256':origin_hashes,'files_sha256':files,'source_objects':source['sources'],
        'source_object_root':'../source_objects','scope':'Exploratory; no untouched 2024 holdout'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--compact-traces',action='store_true');p.add_argument('--output',default='reports/round2-2026-09-24');a=p.parse_args();export(a.run,a.output,a.compact_traces)
