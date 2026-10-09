"""Predict annual consumer-profile labels with a checked complete frontier rule."""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster.io import sha256, write_json
from sbercluster.selection_artifacts import load_artifact, predict_annual_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        model = load_artifact(args.model, args.expected_sha256)
        panel_bytes = Path(args.panel).read_bytes()
        panel_digest = hashlib.sha256(panel_bytes).hexdigest()
        panel = pd.read_csv(BytesIO(panel_bytes), dtype={'entity_id': str, 'period': str})
        labels = predict_annual_panel(panel, model, args.year)
        labels.to_csv(output / 'labels.csv', index=False)
        write_json(output / 'provenance.json', {'model_kind': model['model_kind'],
                   'model_sha256': args.expected_sha256.lower(), 'panel_sha256': panel_digest, 'year': args.year,
                   'labels_sha256': sha256(output / 'labels.csv'), 'n': len(labels),
                   'aggregation': model['aggregation'], 'prediction_scope': model['prediction_scope'],
                   'boundary_stability': 'not inferred from ID persistence'})
        write_json(output / 'status.json', {'status': 'completed', 'n': len(labels)})
        print(json.dumps({'status': 'completed', 'n': len(labels), 'output': str(output)}))
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                           else 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
