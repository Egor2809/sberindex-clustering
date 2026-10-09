"""Verified frozen seasonal/reliability annual inference; no recalibration."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sbercluster.io import write_json
from sbercluster.temporal_profiles import predict_temporal_panel


def predict_file(model_path, panel_path, output, year, expected_sha256):
    model_path, panel_path, output = Path(model_path), Path(panel_path), Path(output)
    model_bytes = model_path.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    if model_hash != expected_sha256:
        raise ValueError('Temporal model hash differs from required SHA256')
    if output.exists():
        raise FileExistsError('Use new prediction output directory')
    artifact = json.loads(model_bytes)
    panel_bytes = panel_path.read_bytes()
    input_hash = hashlib.sha256(panel_bytes).hexdigest()
    panel = pd.read_csv(BytesIO(panel_bytes), dtype={'entity_id': str, 'period': str, 'territory_id': str})
    predictions = predict_temporal_panel(panel, artifact, year)
    output.mkdir(parents=True)
    predictions.to_csv(output / 'assignments.csv', index=False)
    write_json(output / 'provenance.json', {'model_sha256': model_hash, 'input_sha256': input_hash,
        'year': year, 'fitting_executed': False, 'recalibration_executed': False,
        'scope': artifact['prediction_scope'], 'territory_boundary_continuity_verified': False,
        'new_entity_count': int(predictions.new_entity.sum()),
        'distance_extrapolation_count': int(predictions.distance_extrapolation.sum())})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    rows = predict_file(args.model, args.panel, args.output, args.year, args.expected_sha256)
    print(json.dumps({'rows': len(rows), 'scope': 'frozen_inductive_descriptive_annual_inference'}))


if __name__ == '__main__':
    main()
