"""Explicit contest runner with an optional Windows resource supervisor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/contest_pilot.json')
    p.add_argument('--execute-clustering',action='store_true')
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--supervisor',metavar='PROFILE',help='Use run_bounded.py with this JSON profile (Windows only)')
    args=p.parse_args(argv)
    if not args.execute_clustering:p.error('--execute-clustering is required')
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    execution=cfg.get('execution',{})
    if execution.get('allow_clustering') is not True:
        raise PermissionError('The selected contest configuration is disabled')
    threads=execution.get('threads',1)
    if type(threads) is not int or threads<1:
        raise ValueError('execution.threads must be a positive integer')
    task=execution.get('task')
    if task not in ('contest_v3','round2_v5','followup_v4'):
        raise ValueError('run_contest.py accepts execution.task=contest_v3, followup_v4 or round2_v5')
    if args.supervisor:
        if os.name!='nt':p.error('--supervisor is available only on Windows')
        return subprocess.call([sys.executable,str(ROOT/'scripts/run_bounded.py'),
            '--config',args.config,'--profile',args.supervisor,'--timeout',str(args.timeout),
            '--wait-for-headroom','30','--pause-for-headroom','60'],cwd=ROOT)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):
        os.environ[name]=str(threads)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    print('Portable runner: use OS/container resource limits. Windows whole-system monitoring is not available.',flush=True)
    if task=='contest_v3':
        from sbercluster.contest import run
    elif task=='round2_v5':
        from sbercluster.round2 import run
    elif task=='followup_v4':
        from sbercluster.followup import run
    run(cfg,ROOT)
    return 0


if __name__=='__main__':
    sys.exit(main())
