"""Apply a frozen prototype JSON to monthly estimates, without any model fitting."""
import argparse
import csv
import hashlib
import importlib.metadata
import platform
from io import BytesIO, StringIO
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

import pandas as pd
from sbercluster.inference import FrozenProfileModel
from sbercluster.io import sha256, write_json, read_verified_artifact


def predict_file(model_path, input_path, output, aggregation='monthly', expected_model_sha256=None):
    model_path, input_path, output = Path(model_path), Path(input_path), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Inference output must be new: ' + str(output))
    published = ROOT / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json'
    if model_path.resolve() == published.resolve():
        trusted = read_verified_artifact(published.parent.parent, 'validation/frozen_prototypes.json')
        archived_sha256 = hashlib.sha256(trusted).hexdigest()
        if expected_model_sha256 is not None and expected_model_sha256 != archived_sha256:
            raise ValueError('Expected fingerprint conflicts with the published model manifest')
        expected_model_sha256 = archived_sha256
    model = FrozenProfileModel.load(model_path, expected_model_sha256)
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != model.model_sha256:
        raise ValueError('Model changed during loading')
    input_bytes = input_path.read_bytes()
    before = hashlib.sha256(input_bytes).hexdigest()
    header = next(csv.reader(StringIO(input_bytes.decode('utf-8-sig'))), [])
    if not header or len(set(header)) != len(header):
        raise ValueError('Missing or duplicate input column names')
    frame = pd.read_csv(BytesIO(input_bytes), dtype={'entity_id': str, 'territory_id': str, 'period': str})
    source_names = ['sbercluster/inference.py', 'sbercluster/features.py', 'sbercluster/input_contracts.py',
                    'sbercluster/io.py', 'scripts/predict_frozen.py']
    source_bytes = {name: (ROOT / name).read_bytes() for name in source_names}
    predictions = model.predict(frame, aggregation)
    if sha256(input_path) != before or sha256(model_path) != model.model_sha256:
        raise ValueError('Input or model changed during inference')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        predictions.to_csv(staging / 'assignments.csv', index=False, lineterminator='\n')
        (staging / 'model.json').write_bytes(model_bytes)
        source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
        for name, raw in source_bytes.items():
            if sha256(ROOT / name) != source_hashes[name]:
                raise ValueError('Inference source changed during execution')
            target = staging / 'source_objects' / (source_hashes[name] + '.py')
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw)
        write_json(staging / 'provenance.json', {
            'schema_version': 1, 'input_sha256': before, 'model_sha256': model.model_sha256,
            'aggregation': aggregation, 'input_rows': len(frame), 'output_rows': len(predictions),
            'scaler': model.scaler, 'fitting_executed': False,
            'expected_model_sha256': expected_model_sha256,
            'environment': {'python': platform.python_version(),
                'packages': {name: importlib.metadata.version(name)
                             for name in ['numpy', 'pandas', 'scipy']}},
            'sources': source_hashes,
            'scope': 'Fixed consumer-profile coordinates; descriptive assignment, not verified economic transition.',
            'margin': 'Relative gap between nearest and second-nearest centroid distances; not a probability.',
            'identity': 'Source territory key consistency checked; new IDs allowed and flagged; boundaries not independently verified.'})
        write_json(staging / 'manifest.json', {'files_sha256': {p.relative_to(staging).as_posix(): sha256(p)
                                                              for p in staging.rglob('*') if p.is_file()}})
        staging.rename(output)
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent:
                raise ValueError('Staging directory escaped output parent')
            shutil.rmtree(staging)
    return {'output': str(output), 'rows': len(predictions), 'fitting_executed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--aggregation', choices=['monthly', 'annual'], default='monthly')
    parser.add_argument('--expected-model-sha256')
    args = parser.parse_args(argv)
    print(json.dumps(predict_file(args.model, args.input, args.output, args.aggregation, args.expected_model_sha256)))


if __name__ == '__main__':
    main()
