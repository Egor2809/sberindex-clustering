"""Compare rerun preparation reports with the published ones, ignoring timings, code hashes and CSV-only fields."""
import json
from pathlib import Path
import sys
from compare_edge_results import close

FILES=['data_audit.json','source_alignment.json','graph_preparation.json']
IGNORED={'seconds','source_hashes','csv_export_diagnostics'}


def load(path):
    return {k:v for k,v in json.loads(path.read_text('utf-8')).items() if k not in IGNORED}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    differences=[]
    for name in FILES:
        published,repeated=load(root/'reports'/name),load(root/'artifacts/preparation'/name)
        if repeated.get('status')=='reference_CSV_not_supplied':
            published={k:published.get(k) for k in repeated if k!='status'}
            repeated.pop('status')
        differences+=close(published,repeated,1e-9,name)
    print('\n'.join(differences[:20]) or 'Preparation reports match the published files')
    sys.exit(1 if differences else 0)
