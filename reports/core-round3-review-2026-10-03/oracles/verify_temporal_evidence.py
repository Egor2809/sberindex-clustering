"""Independent linear-cost replay of new decomposition/context evidence; no fit."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sbercluster.io import CATEGORIES, TOTAL
OUT = Path(__file__).resolve().parent
RUN = ROOT / '.local/core-round3-20261003/temporal/run-v1'


def nearest(x, rule):
    # Independent complete squared-distance+bias rule, no predictor import.
    yy = x @ np.asarray(rule['transform'])
    sq = ((yy[:, None] - np.asarray(rule['centers'])[None]) ** 2).sum(axis=2)
    return (sq + np.asarray(rule['biases'])).argmin(axis=1)


def check_decomposition(a, b, row, rule):
    d = b - a
    mean = np.sum(d, axis=0) / len(d)
    remainder = d - mean
    total, common, individual = float((d * d).sum()), float(len(d) * (mean * mean).sum()), float((remainder * remainder).sum())
    np.testing.assert_allclose([total, common, individual], [row['total_squared_displacement'], row['common_squared_displacement'], row['individual_squared_displacement']], rtol=1e-13, atol=1e-10)
    np.testing.assert_allclose(mean, row['mean_common_displacement'], rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.median(d, axis=0), row['median_paired_displacement'], rtol=0, atol=1e-13)
    reference = nearest(a, rule)
    actual = nearest(b, rule) != reference
    common_change = nearest(a + mean, rule) != reference
    individual_change = nearest(a + remainder, rule) != reference
    wanted = {'actual_label_changes': actual.sum(), 'common_only_counterfactual_changes': common_change.sum(), 'individual_only_counterfactual_changes': individual_change.sum(),
              'actual_and_common_only': (actual & common_change).sum(), 'actual_and_individual_only': (actual & individual_change).sum()}
    for key, value in wanted.items():
        assert row[key] == int(value), key
    assert abs(total - common - individual) <= 1e-9
    assert row['scope'] == 'geometric_mean_displacement_decomposition;counterfactuals_not_causal_attribution'
    return {'total': total, 'common_fraction': common / total, 'changes': int(actual.sum()), 'identity_error': abs(total - common - individual)}


def main():
    panel = pd.read_csv(ROOT / 'data/processed/panel.csv', dtype={'entity_id': str, 'period': str})
    base = json.loads((ROOT / 'reports/model-frontier-2026-10-03/models/historical_kmeans4.json').read_text('utf-8'))
    scaler, rule = base['feature_scaler'], base['rule']
    tensor, ids = {}, None
    for year in [2023, 2024]:
        rows = []
        for month in range(1, 13):
            frame = panel[panel.period == f'{year}-{month:02d}-01'].sort_values('entity_id')
            if ids is None:
                ids = frame.entity_id.tolist()
            assert frame.entity_id.tolist() == ids
            logs = np.log(frame[CATEGORIES].to_numpy(float) / frame[TOTAL].to_numpy(float)[:, None])
            rows.append((logs - np.asarray(scaler['ratio_center'])) / np.asarray(scaler['ratio_iqr']) / np.sqrt(5))
        tensor[year] = np.asarray(rows)
    stored = json.loads((RUN / 'drift_decomposition.json').read_text('utf-8'))
    annual = check_decomposition(np.median(tensor[2023], axis=0), np.median(tensor[2024], axis=0), stored['annual'], rule)
    months = [check_decomposition(tensor[2023][i], tensor[2024][i], stored['same_calendar_month'][i], rule) for i in range(12)]
    shifts = np.median(tensor[2024], axis=1) - np.median(tensor[2023], axis=1)
    corrected = tensor[2024] - shifts[:, None]
    z0 = nearest(np.median(tensor[2023], axis=0), rule)
    relative = nearest(np.median(corrected, axis=0), rule)
    stored_relative = json.loads((RUN / 'relative_context.json').read_text('utf-8'))
    np.testing.assert_allclose(shifts, stored_relative['shared_drift_by_month'], rtol=0, atol=1e-13)
    with np.load(RUN / 'labels.npz', allow_pickle=False) as labels:
        np.testing.assert_array_equal(relative, labels['relative2024'])
        np.testing.assert_array_equal(z0, labels['2023__historical_kmeans4'])
    ari = float(adjusted_rand_score(z0, relative))
    assert abs(ari - stored_relative['relative_ARI']) < 1e-14
    assert int((relative != z0).sum()) == stored_relative['relative_changes']
    manifest = json.loads((RUN / 'manifest.json').read_text('utf-8'))['files_sha256']
    failures = [name for name, digest in manifest.items() if hashlib.sha256((RUN / name).read_bytes()).hexdigest() != digest]
    assert not failures, failures
    report = {'status': 'PASS', 'scope': 'Independent original-row geometric replay, no new fit/fullmetrics audit',
              'annual': annual, 'monthly_cases': len(months), 'largest_identity_error': max(r['identity_error'] for r in months),
              'relative_ARI': ari, 'relative_changes': int((relative != z0).sum()),
              'noncausal_nonadditive_counterfactual_counts': {'actual': stored['annual']['actual_label_changes'], 'common': stored['annual']['common_only_counterfactual_changes'], 'individual': stored['annual']['individual_only_counterfactual_changes']},
              'manifest_payloads_verified': len(manifest), 'input_sha256': hashlib.sha256((ROOT / 'data/processed/panel.csv').read_bytes()).hexdigest(),
              'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'temporal-evidence-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
