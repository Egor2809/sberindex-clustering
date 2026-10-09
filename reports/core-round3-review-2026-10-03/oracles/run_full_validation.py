"""One final pinned full-suite/compile/source-stability check after code freeze."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS']: os.environ[name]='1'
if os.name=='nt':
 import psutil
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def hashes():
 return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['sbercluster','scripts','tests'] for p in sorted((ROOT/directory).rglob('*')) if p.is_file() and p.suffix in ['.py','.cjs']}
started=time.perf_counter();before=hashes()
(OUT/'full-suite-running.json').write_text(json.dumps({'pid':os.getpid(),'source_sha256_before':before,'status':'running'},indent=2)+'\n',encoding='utf-8')
commands=[('tests',[sys.executable,'-m','unittest','discover','-s','tests','-v']),('compile',[sys.executable,'-m','compileall','-q','sbercluster','scripts'])]
results={}
for name,command in commands:
 with (OUT/f'full-{name}.log').open('w',encoding='utf-8') as stream:
  process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
 results[name]={'exit_code':process.returncode,'command':command[1:],'log':f'full-{name}.log'}
log=(OUT/'full-tests.log').read_text('utf-8');match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',log)
assert match,'Incomplete unittest result'
results['tests'].update({'test_count':int(match[1]),'seconds':float(match[2]),'skips':len(re.findall(r'^.+\.\.\. skipped',log,re.M))})
after=hashes();changed=[name for name in before if before[name]!=after.get(name)]
dependency_changes=subprocess.run(['git','diff','--name-only','HEAD','--','requirements*','pyproject.toml','Pipfile*'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
report={'status':'PASS' if all(r['exit_code']==0 for r in results.values()) and not changed and not dependency_changes else 'FAIL',
 'scope':'Full local pinned Windows suite and compilation; scientific findings have independent scoped acceptance',
 'results':results,'source_unchanged':not changed,'changed_sources':changed,'source_sha256_before':before,'source_sha256_after':after,
 'python':platform.python_version(),'platform':platform.platform(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch']},
 'dependency_specifications_changed':dependency_changes,'pip_check':'Not repeated: no dependency or environment changes after previously verified pinned pip check',
 'elapsed_seconds':time.perf_counter()-started,'oracle_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'full-suite.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'tests':results['tests'],'compile_exit':results['compile']['exit_code'],'source_unchanged':not changed,'seconds':report['elapsed_seconds']}))
if report['status']!='PASS':raise SystemExit(1)
