"""One historical no-fit paired-reference application and finite missing controls."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from sbercluster.dual_profiles import build_reference_lock, build_reference_packet, predict_dual_panel, gaussian_mmd2
from sbercluster.io import sha256, write_json
from sbercluster.input_contracts import validate_prepared_panel
from scripts.benchmark_temporal_profiles import metrics
from scripts.predict_dual import read_panel, write_immutable_json, prediction_records


def run(out):
    if out.exists(): raise FileExistsError('Use new dual output directory')
    out.mkdir(parents=True); started = time.perf_counter()
    protocol = {'scope': 'already_explored2024_historical_descriptive_not_holdout', 'no_fitting': True,
                'reference': 'all2016_fixed2023_ID_pairs;retrospective_complete_cohort',
                'estimand': 'median_of_paired_same_calendar_month_deltas;not_population_median_difference',
                'uncertainty': 'exact_missing_value_bounds_at_epsilon0;no_error_probability_or_support_policy',
                'controls_before_effects': ['January_reference_missing5/20/50/60percent;fixedseed1729_random_or_upper_delta_tail',
                    'January_leave_each_of_three_largest_reference_regions', 'all12immutablepackets_sealed_Aexact',
                    'publishedcontext_batch_invariance'],
                'stop': 'one_historical_application;no_fit;no_new_parameter_grid;no_ARI_promotion'}
    write_json(out / 'protocol.json', protocol)
    sources = ['sbercluster/dual_profiles.py', 'scripts/predict_dual.py', 'scripts/benchmark_dual_profiles.py', 'sbercluster/features.py']
    for source in sources:
        target = out / 'source' / source; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / source).read_bytes())
    source_hashes = {p: sha256(out / 'source' / p) for p in sources}
    panel, panel_hash = read_panel(ROOT / 'data/processed/panel.csv')
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text('utf-8'))
    if panel_hash != manifest['panel_sha256']: raise ValueError('Panel hash differs')
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8')); validate_prepared_panel(panel, manifest, cfg)
    model_path = ROOT / 'reports/model-frontier-2026-10-03/models/historical_kmeans4.json'
    artifact = json.loads(model_path.read_bytes())
    lock = build_reference_lock(panel, artifact, {'panel_sha256': panel_hash, 'model_file_sha256': sha256(model_path),
        'cohort_selection': 'retrospectively_complete2023_2024;not_prospective', 'data_as_of': 'unknown'})
    lock_hash = write_immutable_json(out / 'private-reference-lock.json', lock)
    packets, packet_manifest = {}, {}
    for m in range(1, 13):
        period = f'2024-{m:02d}-01'
        packets[period] = build_reference_packet(panel, lock, period, {'input_file_sha256': panel_hash, 'data_as_of': 'unknown'})
        location = f'packets/{period}.json'; digest = write_immutable_json(out / location, packets[period])
        packet_manifest[period] = {'path': location, 'sha256': digest}
    packet_manifest_hash = write_immutable_json(out / 'packet_manifest.json', {'packets': packet_manifest})
    baseline = predict_dual_panel(panel, lock, 2023)
    current = predict_dual_panel(panel, lock, 2024, packets)
    historical = json.loads((ROOT / 'reports/model-frontier-2026-10-03/annual-2024-application.json').read_text('utf-8'))
    expected = next(r['labels_2024'] for r in historical['models'] if r['id'] == 'historical_kmeans4')
    if not np.array_equal(current.absolute_label, expected): raise AssertionError('A changed')
    z0 = baseline.absolute_label.to_numpy(int); za = current.absolute_label.to_numpy(int); zr = current.relative_label.to_numpy(int)
    x0 = np.asarray(baseline.absolute_profile.tolist()); xa = np.asarray(current.absolute_profile.tolist()); xr = np.asarray(current.relative_profile.tolist())
    distance = cdist(x0, x0); bandwidth = float(np.median(distance[np.triu_indices(len(distance), 1)]))
    summary = {'scope': protocol['scope'], 'fit_executed': False, 'default_preserved': True,
        'reference_n': len(lock['entity_ids']), 'reference_regions': len(set(lock['regions'])),
        'absolute2024_labels_exact': True, 'absolute_ARI': float(adjusted_rand_score(z0, za)),
        'paired_relative_ARI': float(adjusted_rand_score(z0, zr)),
        'absolute_changes': int((za != z0).sum()), 'paired_relative_changes': int((zr != z0).sum()),
        'paired_common_absolute_geometry_metrics': metrics(xa, zr), 'paired_own_relative_geometry_metrics': metrics(xr, zr),
        'paired_vs_population_correction_max_coordinate_difference': float(max(np.max(np.abs(np.asarray(p['point_paired_median']) - p['population_median_difference'])) for p in packets.values())),
        'absolute_geometric_labels': int(current.absolute_geometric_label.notna().sum()),
        'paired_relative_geometric_labels': int(current.relative_geometric_label.notna().sum()),
        'geometry_certificate_scope': 'conditional_exact_box_only;no_support_or_measurement_probability',
        'confirmed_transition_count': 0, 'territory_transition_status': 'unresolved',
        'mmd_bandwidth_fit2023': bandwidth, 'absolute_mmd2': gaussian_mmd2(x0, xa, bandwidth),
        'relative_mmd2': gaussian_mmd2(x0, xr, bandwidth),
        'absolute_quantiles2023': np.quantile(x0, [.1, .25, .5, .75, .9], axis=0).tolist(),
        'absolute_quantiles2024': np.quantile(xa, [.1, .25, .5, .75, .9], axis=0).tolist(),
        'relative_quantiles2024': np.quantile(xr, [.1, .25, .5, .75, .9], axis=0).tolist()}
    jan = panel[panel.period == '2024-01-01'].sort_values('entity_id'); ids = lock['entity_ids']; n = len(ids)
    from sbercluster.features import transform_frozen
    delta = transform_frozen(jan, artifact['feature_scaler']) - np.asarray(lock['baseline_features'])[0]
    rng = np.random.default_rng(1729); random_order = rng.permutation(n); tail_order = np.argsort(delta[:, 0], kind='stable')[::-1]
    scenarios = []
    for kind, order in [('random', random_order), ('upper_delta_tail', tail_order)]:
        for fraction in (.05, .2, .5, .6):
            dropped = {ids[i] for i in order[:int(np.ceil(n * fraction))]}
            scenarios.append((f'{kind}_{fraction}', jan[~jan.entity_id.isin(dropped)]))
    ranked_regions = sorted(set(lock['regions']), key=lambda r: (-lock['regions'].count(r), r))[:3]
    for region in ranked_regions:
        dropped = {entity for entity, reg in zip(ids, lock['regions']) if reg == region}
        scenarios.append((f'leave_region_{region}', jan[~jan.entity_id.isin(dropped)]))
    controls = []
    truth = np.asarray(packets['2024-01-01']['point_paired_median'])
    for name, rows in scenarios:
        packet = build_reference_packet(rows, lock, '2024-01-01')
        lower = np.asarray([-np.inf if x is None else x for x in packet['correction_lower']])
        upper = np.asarray([np.inf if x is None else x for x in packet['correction_upper']])
        inclusion = bool((truth >= lower).all() and (truth <= upper).all())
        if not inclusion: raise AssertionError('Full reference median outside sharp bounds')
        controls.append({'scenario': name, 'reference_n': n, 'observed_n': packet['observed_valid_n'],
            'coverage': packet['coverage'], 'status': packet['status'], 'true_full_median_in_bounds': inclusion,
            'identified_point_null': packet['point_paired_median'] is None,
            'lower': packet['correction_lower'], 'upper': packet['correction_upper'],
            'regional_coverage': packet['regional_coverage']})
    current[['entity_id', 'year', 'absolute_label', 'relative_label', 'absolute_geometric_label', 'relative_geometric_label', 'transition_status']].to_csv(out / 'assignments.csv', index=False)
    write_json(out / 'summary.json', summary); write_json(out / 'missing_reference_controls.json', controls)
    write_json(out / 'reference_metadata.json', {k: v for k, v in lock.items() if k not in ('baseline_features', 'frozen_artifact')})
    write_json(out / 'provenance.json', {'panel_sha256': panel_hash, 'source_sha256': source_hashes,
        'reference_file_sha256': lock_hash, 'reference_content_sha256': lock['content_sha256'],
        'packet_manifest_sha256': packet_manifest_hash, 'model_file_sha256': sha256(model_path)})
    write_json(out / 'status.json', {'status': 'completed', 'elapsed_seconds': time.perf_counter() - started, 'pid': os.getpid()})
    write_json(out / 'manifest.json', {'files_sha256': {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({k: v for k, v in summary.items() if k not in ('absolute_quantiles2023','absolute_quantiles2024','relative_quantiles2024')}, ensure_ascii=False), flush=True)


def main():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True); kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000): raise OSError('Cannot set worker priority')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    with threadpool_limits(limits=1): run(args.output.resolve())


if __name__ == '__main__': main()
