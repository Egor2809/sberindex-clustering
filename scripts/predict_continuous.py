"""Apply a byte-pinned continuous atlas to 2023/2024 expenses, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[name] = '1'

from sbercluster.continuous_atlas import CODECS, encode_panel, load_model, read_panel_csv
from sbercluster.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('sbercluster/continuous_atlas.py', 'sbercluster/continuous_geometry.py',
           'sbercluster/features.py', 'sbercluster/selection_frontier.py',
           'sbercluster/io.py', 'scripts/predict_continuous.py')


def predict_file(model_path, panel_path, output, year, codec,
                 expected_model_sha256, expected_panel_sha256):
    """Validate and compute first, then atomically publish a new output directory."""
    model_path, panel_path = Path(model_path), Path(panel_path)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output must not exist: ' + str(output))
    source_bytes = {name: (ROOT / name).read_bytes() for name in SOURCES}
    source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
    model = load_model(model_path, expected_model_sha256)
    panel = read_panel_csv(panel_path, expected_panel_sha256)
    frame = encode_panel(panel, model, year, codec)
    if sha256(model_path) != expected_model_sha256 or sha256(panel_path) != expected_panel_sha256:
        raise ValueError('Input or model changed during inference')
    if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError('Inference source changed during execution')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        csv_path = staging / 'atlas.csv'
        frame.to_csv(csv_path, index=False, lineterminator='\n')
        metadata = {
            'status': 'EXECUTED', 'operation': 'inference_only_no_fit',
            'year': year, 'codec': codec, 'rows': len(frame),
            'model_sha256': expected_model_sha256, 'panel_sha256': expected_panel_sha256,
            'csv_sha256': sha256(csv_path), 'legacy_labels_not_replaced': True,
            'territory_transition_status': 'unresolved', 'no_outcome_inputs': True,
            'fitting_executed': False, 'source_code_sha256': source_hashes,
            'environment': {'python': platform.python_version(), 'packages': {
                name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}},
            'weight_semantics': 'convex_coordinates_not_probabilities',
            'boundary_radius_semantics': 'standardized_feature_distance_not_error_probability',
        }
        write_json(staging / 'metadata.json', metadata)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Output must not exist: ' + str(output))
        staging.rename(output)
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent.resolve():
                raise ValueError('Staging directory escaped output parent')
            shutil.rmtree(staging)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--expected-panel-sha256', required=True)
    parser.add_argument('--year', type=int, choices=(2023, 2024), required=True)
    parser.add_argument('--codec', choices=CODECS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = predict_file(args.model, args.panel, args.output, args.year, args.codec,
                            args.expected_model_sha256, args.expected_panel_sha256)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
