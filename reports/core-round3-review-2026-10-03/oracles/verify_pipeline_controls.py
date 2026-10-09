"""New full probe-fold outcome isolation and exact-consumed-byte CLI controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from oracle_controls import ROOT, fixture, assert_state_equal, expect_detection
from sbercluster.io import CATEGORIES
OUT = Path(__file__).resolve().parent


def fold_controls():
    from sbercluster.economic_generalization import fit_region_representation
    from scripts.validate_economic_generalization import evaluate_fold
    logs, region, ids, _ = fixture()
    # 96 expense entities let the actual pipeline's >=50 observed-train rule
    # operate while keeping this oracle far smaller than national computation.
    logs = np.concatenate([logs, logs + .008], axis=1)
    regions = np.r_[region, region]
    ids = np.array([f'tid_{i+1}' for i in range(logs.shape[1])])
    holdout = 10
    fitted = fit_region_representation(logs, regions, holdout, ids)
    numeric = np.column_stack([np.linspace(8., 11., len(ids)), np.linspace(9., 10., len(ids))])
    y = 9. + numeric[:, 0] * .03 + logs[0, :, 0] * .1
    y[17] = np.nan  # Expense fit retains this entity; wage fit must omit it.
    data = {'ids': ids, 'regions': regions, 'numeric': numeric,
            'types': np.where(np.arange(len(ids)) % 2, 'city', 'district'),
            'controls_ok': np.ones(len(ids), bool), 'y2023': y,
            'y2024': np.nan_to_num(y, nan=9.) + .04}
    frame, state = evaluate_fold(data, fitted, holdout, False)
    mutated = deepcopy(data)
    mutated['y2023'][regions == holdout] += 1000.
    mutated['y2024'][:] += 1000.
    second, second_state = evaluate_fold(mutated, fitted, holdout, False)
    assert_state_equal(state, second_state)
    pred_columns = [c for c in frame if c not in ['entity_id', 'held_out_region', 'observed2023', 'observed2024']]
    np.testing.assert_array_equal(frame[pred_columns], second[pred_columns])
    assert ids[17] in fitted['artifact']['training_entity_ids']
    assert ids[17] not in state['training_outcome_entities']
    assert not set(ids[regions == holdout]) & set(state['training_outcome_entities'])
    # No target-dependent renumbering: permuting the four dummy columns is
    # equivalent for OLS/quadratic and cannot invent a semantic cluster effect.
    renumbered = deepcopy(fitted)
    renumbered['labels']['kmeans4'] = np.array([2, 0, 3, 1])[fitted['labels']['kmeans4']]
    renamed, _ = evaluate_fold(data, renumbered, holdout, False)
    for c in ['linear_kmeans4', 'quadratic_kmeans4']:
        np.testing.assert_allclose(frame[c], renamed[c], rtol=0, atol=1e-10)
    return {'status': 'PASS', 'prediction_variants': len(pred_columns),
            'heldout2023_all2024_outcome_mutation': 'unchanged fitted probe states and predictions',
            'expense_fit_retains_outcome_missing_training_id': ids[17],
            'linear_quadratic_label_renaming': 'PASS'}


def cli_controls():
    from scripts.predict_temporal import predict_file
    from sbercluster.temporal_profiles import predict_temporal_panel
    _, _, _, panel = fixture()
    one = panel[panel.entity_id == 'tid_1'].copy()
    old = ROOT / '.local/core-round3-20261003/temporal/run-v1/models/calendar_only.json'
    model = json.loads(old.read_text('utf-8'))
    model['identity_mode'] = 'source_territory_id'
    expect_detection(lambda: predict_temporal_panel(one.assign(territory_id=999), model, 2024))
    expect_detection(lambda: predict_temporal_panel(one.drop(columns='territory_id'), model, 2024))
    work = OUT / 'cli-byte-control'
    work.mkdir(exist_ok=False)
    model_path, panel_path = work / 'model.json', work / 'panel.csv'
    model_path.write_text(json.dumps(model), encoding='utf-8')
    one.to_csv(panel_path, index=False)
    originals = {model_path: model_path.read_bytes(), panel_path: panel_path.read_bytes()}
    hashes = {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()}
    expected = predict_temporal_panel(one, model, 2024)
    changed = deepcopy(model)
    changed['centers'] = (np.asarray(changed['centers']) + 100.).tolist()
    replacements = {model_path: json.dumps(changed).encode(), panel_path: originals[panel_path] + b'\n'}
    read_count = {p: 0 for p in originals}
    original_read = Path.read_bytes
    def intercepted(path):
        returned = original_read(path)
        if path in originals:
            read_count[path] += 1
            path.write_bytes(replacements[path])
        return returned
    with patch.object(Path, 'read_bytes', intercepted):
        actual = predict_file(model_path, panel_path, work / 'prediction', 2024, hashes[model_path])
    np.testing.assert_array_equal(expected.cluster, actual.cluster)
    provenance = json.loads((work / 'prediction/provenance.json').read_text('utf-8'))
    assert provenance['model_sha256'] == hashes[model_path]
    assert provenance['input_sha256'] == hashes[panel_path]
    assert all(n == 1 for n in read_count.values())
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p] for p in originals)
    return {'status': 'PASS', 'exact_one_read_model_and_panel': True,
            'file_mutation_during_consumption': 'pinned bytes, predictions and provenance remain bound',
            'territory_identity_negativecases': 'PASS'}


def main():
    report = {'fold_controls': fold_controls(), 'cli_controls': cli_controls()}
    report['status'] = 'PASS'
    report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
        'sbercluster/economic_generalization.py', 'scripts/validate_economic_generalization.py',
        'sbercluster/temporal_profiles.py', 'scripts/predict_temporal.py']}
    report['oracle_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / 'pipeline-controls-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
