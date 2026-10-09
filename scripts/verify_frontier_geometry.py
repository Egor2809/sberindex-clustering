"""Independently reconcile denominator perturbation and raw-level geometry."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(evidence, panel_path):
    names = ['original_features.npy', 'labels.json', 'provenance.json', 'comparisons.json']
    captured = {name: (evidence / name).read_bytes() for name in names}
    x = np.load(BytesIO(captured['original_features.npy']), allow_pickle=False)
    labels_doc = json.loads(captured['labels.json'])
    ids = labels_doc['ids']
    labels = np.asarray(labels_doc['labels']['historical_kmeans4'])
    scaler = json.loads(captured['provenance.json'])['scaler']
    scale, center = np.asarray(scaler['ratio_iqr']), np.asarray(scaler['ratio_center'])
    published = next(row for row in json.loads(captured['comparisons.json'])
                     if row['id'] == 'historical_kmeans4')
    if x.shape != (len(ids), 5) or len(set(ids)) != len(ids) or not np.isfinite(x).all():
        raise ValueError('Expected unique aligned finite annual profiles')
    a = (1 / scale) / np.linalg.norm(1 / scale)
    b = scale / np.linalg.norm(scale)
    deviation = x - x.mean(axis=0)
    total = float(np.square(deviation).sum())
    between = axis_between = 0.0
    for label in np.unique(labels):
        selected = x[labels == label]
        delta = selected.mean(axis=0) - x.mean(axis=0)
        between += len(selected) * float(delta @ delta)
        axis_between += len(selected) * float(delta @ a) ** 2
    axis_total = float(np.square(deviation @ a).sum())
    n, k = len(x), len(np.unique(labels))
    ch = (between / (k - 1)) / ((total - between) / (n - k))
    fraction_from_ch = ch * (k - 1) / (n - k + ch * (k - 1))
    eta_direct = axis_between / axis_total
    eta_factored = (axis_between / between) / (axis_total / total) * fraction_from_ch

    def eta(values):
        denominator = float(np.square(values - values.mean()).sum())
        return sum(np.sum(labels == c) * float(values[labels == c].mean() - values.mean()) ** 2
                   for c in np.unique(labels)) / denominator

    panel_bytes = panel_path.read_bytes()
    panel = pd.read_csv(BytesIO(panel_bytes))
    annual = panel[panel.period.str.startswith('2023-')].sort_values(['entity_id', 'period'])
    # Column names and ordering are a public feature contract, not fitted here.
    from sbercluster.io import CATEGORIES, TOTAL
    raw_mean_after_median, raw_median_after_mean = [], []
    groups = {entity: frame for entity, frame in annual.groupby('entity_id', sort=False)}
    expected_months = [f'2023-{month:02d}-01' for month in range(1, 13)]
    for entity in ids:
        frame = groups[entity]
        if frame.period.tolist() != expected_months:
            raise ValueError('Expected complete aligned 2023 months')
        ratio = np.log(frame[CATEGORIES].to_numpy(float) / frame[TOTAL].to_numpy(float)[:, None])
        raw_mean_after_median.append(np.median(ratio, axis=0).mean())
        raw_median_after_mean.append(np.median(ratio.mean(axis=1)))
    raw_mean_after_median = np.asarray(raw_mean_after_median)
    raw_median_after_mean = np.asarray(raw_median_after_mean)
    reconstructed = center.mean() + x @ scale / np.sqrt(5)
    np.testing.assert_allclose(ch, published['common_original_geometry']['CH'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eta_direct, eta_factored, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, raw_mean_after_median, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eta(raw_median_after_mean), published['raw_common_log_ratio_eta_squared'], rtol=1e-10, atol=1e-10)
    return {'status': 'passed', 'n': n, 'k': k, 'CH_independent': ch,
            'total_ss': total, 'between_ss': between, 'axis_total_ss': axis_total,
            'axis_between_ss': axis_between, 'between_total_fraction': between / total,
            'axis_eta_squared_direct': eta_direct, 'axis_eta_squared_factored': eta_factored,
            'raw_mean_direction_eta_squared': eta(x @ b),
            'raw_median_monthly_mean_eta_squared': eta(raw_median_after_mean),
            'direction_cosine_inverse_iqr_vs_iqr': float(a @ b),
            'angle_degrees': float(np.degrees(np.arccos(a @ b))),
            'same_order_reconstruction_max_error': float(np.max(np.abs(reconstructed - raw_mean_after_median))),
            'different_aggregation_orders_max_difference': float(np.max(np.abs(raw_mean_after_median - raw_median_after_mean))),
            'interpretation': 'Perturbing total follows inverse-IQR direction; raw mean log-ratio has IQR coefficients. Different directions and aggregation orders are not interchangeable. No denominator-insensitivity claim for the baseline.',
            'input_sha256': {**{name: digest(blob) for name, blob in captured.items()}, 'panel.csv': digest(panel_bytes)},
            'verifier_sha256': digest(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('reports/model-frontier-2026-10-03/denominator'))
    parser.add_argument('--panel', type=Path, default=Path('data/processed/panel.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.evidence, args.panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
