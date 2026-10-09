"""Reproduce all published frozen assignments without fitting or recalibration."""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.predict_frozen import predict_file
from sbercluster.io import read_verified_artifact, sha256, write_json


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Verification output must be new: ' + str(output))
    archive = root / 'reports/experiments/2026-09-23-v2'
    model = archive / 'validation/frozen_prototypes.json'
    model_hash = hashlib.sha256(read_verified_artifact(archive, 'validation/frozen_prototypes.json')).hexdigest()
    reference_raw = read_verified_artifact(archive, 'validation/reference_assignments.csv')
    monthly_raw = read_verified_artifact(archive, 'validation/monthly_assignments.csv')
    reference = pd.read_csv(BytesIO(reference_raw)).set_index('entity_id').sort_index()
    expected_monthly = pd.read_csv(BytesIO(monthly_raw)).set_index(['entity_id', 'period']).sort_index()
    panel = root / 'data/processed/panel.csv'
    manifest = json.loads((panel.parent / 'manifest.json').read_bytes())
    if sha256(panel) != manifest['panel_sha256']:
        raise ValueError('Prepared panel fingerprint differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        for aggregation in ['monthly', 'annual']:
            predict_file(model, panel, stage / aggregation, aggregation, model_hash)
            provenance = json.loads((stage / aggregation / 'provenance.json').read_bytes())
            if provenance['input_sha256'] != manifest['panel_sha256']:
                raise ValueError('Inference consumed a different prepared panel')
        monthly = pd.read_csv(stage / 'monthly/assignments.csv').rename(columns={'period_start': 'period'})
        monthly = monthly.set_index(['entity_id', 'period']).sort_index()
        if not monthly.index.equals(expected_monthly.index):
            raise ValueError('Monthly assignment keys differ')
        np.testing.assert_array_equal(monthly.cluster, expected_monthly.cluster)
        annual = pd.read_csv(stage / 'annual/assignments.csv')
        counts = {}
        for year, column in [('2023', 'cluster'), ('2024', 'cluster_2024')]:
            result = annual[annual.period_start.str.startswith(year)].set_index('entity_id').sort_index()
            if not result.index.equals(reference.index):
                raise ValueError('Annual assignment keys differ: ' + year)
            np.testing.assert_array_equal(result.cluster, reference[column])
            counts[year] = len(result)
        # The old archive used the same dimensionless distance-gap definition.
        margin_error = float(np.max(np.abs(monthly.relative_distance_margin - expected_monthly.prototype_margin)))
        if margin_error > 1e-12:
            raise ValueError('Published margins differ beyond floating point tolerance')
        summary = {'status': 'passed', 'fitting_executed': False,
            'monthly_labels_exact': len(monthly), 'annual_labels_exact': counts,
            'max_monthly_margin_error': margin_error, 'panel_sha256': manifest['panel_sha256'],
            'model_sha256': model_hash,
            'reference_sha256': hashlib.sha256(reference_raw).hexdigest(),
            'monthly_reference_sha256': hashlib.sha256(monthly_raw).hexdigest()}
        write_json(stage / 'verification.json', summary)
        write_json(stage / 'manifest.json', {'files_sha256': {p.relative_to(stage).as_posix(): sha256(p)
                                                              for p in stage.rglob('*') if p.is_file()}})
        stage.rename(output)
    finally:
        if stage.exists():
            if stage.resolve().parent != output.parent:
                raise ValueError('Verification staging escaped output parent')
            shutil.rmtree(stage)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(ROOT, args.output), indent=2))


if __name__ == '__main__':
    main()
