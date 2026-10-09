"""Check integrated continuous inference against unchanged original annual rules.

Uses an explicitly pinned prepared expense panel; never trains or reads outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--expected-panel-sha256', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must not exist')
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = {'status': 'FAIL', 'fitting_executed': False, 'outcome_values_read': False}
    code = 1
    try:
        import numpy as np
        import pandas as pd
        from sbercluster.continuous_atlas import load_model, read_panel_csv
        from sbercluster.selection_artifacts import load_artifact, predict_annual_panel
        from scripts.verify_technical_core import run_cli
        model_path, panel_path = args.model.resolve(), args.panel.resolve()
        load_model(model_path, args.expected_model_sha256)
        panel = read_panel_csv(panel_path, args.expected_panel_sha256)
        pins = json.loads((root/'reports/model-frontier-2026-10-03/models/verification.json').read_bytes())
        source_paths = [root/'sbercluster/continuous_atlas.py', root/'sbercluster/continuous_geometry.py',
                        root/'scripts/predict_continuous.py', Path(__file__), model_path, panel_path]
        before = {str(p.relative_to(root)) if root in p.parents else p.name: digest(p) for p in source_paths}
        calls, checks = [], []
        maximum_error, minimum_weight, total_labels = 0., 1., 0
        with tempfile.TemporaryDirectory(prefix='sbercluster-continuous-') as temporary:
            for year, codec in ((2023, 'legacy_centers'), (2024, 'observed_core95')):
                output = Path(temporary)/str(year)
                arguments = ['scripts.predict_continuous', '--model', str(model_path),
                             '--expected-model-sha256', args.expected_model_sha256,
                             '--panel', str(panel_path), '--expected-panel-sha256', args.expected_panel_sha256,
                             '--year', str(year), '--codec', codec, '--output', str(output)]
                calls.append(run_cli(root, arguments))
                frame = pd.read_csv(output/'atlas.csv').set_index('entity_id')
                if not frame.index.is_unique or len(frame) != panel.loc[panel.period.str.startswith(str(year)+'-'), 'entity_id'].nunique():
                    raise ValueError('Prediction membership differs from input')
                x = frame[[f'x{i}' for i in range(5)]].to_numpy()
                reconstructed = frame[[f'reconstructed{i}' for i in range(5)]].to_numpy()
                residual = frame[[f'residual{i}' for i in range(5)]].to_numpy()
                weights = frame[[f'weight{i}' for i in range(4)]].to_numpy()
                error = float(np.max(np.abs(x-reconstructed-residual)))
                if (not np.isfinite(x).all() or not np.isfinite(reconstructed).all()
                        or not np.isfinite(residual).all() or not np.isfinite(weights).all()
                        or error > 1e-12 or weights.min() < -1e-12
                        or np.max(np.abs(weights.sum(axis=1)-1)) > 1e-12):
                    raise ValueError('Continuous decomposition or convex weights differ')
                maximum_error = max(maximum_error, error)
                minimum_weight = min(minimum_weight, float(weights.min()))
                for name in ('historical_kmeans4', 'sw_constrained_huber75'):
                    pin = next(item for item in pins['models'] if item['id'] == name)
                    saved = root/'reports/model-frontier-2026-10-03/models'/pin['artifact']
                    expected = predict_annual_panel(panel, load_artifact(saved, pin['sha256']), year).set_index('entity_id')
                    if set(expected.index) != set(frame.index):
                        raise ValueError('Legacy membership differs')
                    mismatches = int(np.sum(frame[name].to_numpy() != expected.loc[frame.index, 'cluster'].to_numpy()))
                    if mismatches:
                        raise ValueError('Legacy label mismatch: '+name)
                    checks.append({'year': year, 'model': name, 'labels': len(frame), 'mismatches': mismatches})
                    total_labels += len(frame)
        after = {str(p.relative_to(root)) if root in p.parents else p.name: digest(p) for p in source_paths}
        if before != after:
            raise ValueError('Consumed source/model/panel bytes changed during check')
        report.update(status='PASS', label_checks=checks, exact_legacy_labels=total_labels,
                      maximum_reconstruction_error=maximum_error, minimum_weight=minimum_weight,
                      source_sha256=before, cli_calls=calls, source_snapshot_stable=True)
        code = 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
        report['error'] = type(error).__name__+': '+str(error)
    report['elapsed_seconds'] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'exact_legacy_labels': report.get('exact_legacy_labels'),
                      'error': report.get('error'), 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
