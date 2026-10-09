"""Small, separately supervised exploratory checks after the frozen evaluation.

No parameter selection or confirmatory claims use these already inspected data.
Each stage is independently reproducible and never overwrites a previous run.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import subprocess
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from .features import make_slices
from .graph import knn_graph
from .io import CATEGORIES, TOTAL, sha256, write_json, code_revision
from .metrics import all_metrics, network_indices
from .models import require_execution, fit_temporal
from .research import fit_candidate
from .contest_graphs import read_road_graph, mix_layers
from .dynamics import transitions
from .joint import fit_graph_regularized_kmeans, joint_objective


def permute_within_groups(a, groups, seed):
    """Relabel a graph within region, retaining edges/weights and regional mixing.

    This destroys municipality-specific alignment while preserving regional
    topology. It does not preserve each municipality's degree or distance.
    """
    groups = np.asarray(groups)
    if a.shape != (len(groups), len(groups)) or pd.isna(groups).any():
        raise ValueError('One nonmissing stratum per vertex required')
    rng = np.random.default_rng(seed)
    order = np.arange(len(groups))
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        order[selected] = rng.permutation(selected)
    return a[order][:, order].tocsr(), order


def load_inputs(root, cfg):
    path = root / 'data/processed/panel.csv'
    manifest = json.loads((path.parent / 'manifest.json').read_text('utf-8'))
    if sha256(path) != manifest['panel_sha256']:
        raise ValueError('Prepared panel fingerprint differs')
    # Avoid carrying repeated long names/coordinates through all monthly copies.
    panel = pd.read_csv(path, usecols=['entity_id', 'period', 'region_code'] + CATEGORIES + [TOTAL],
                        dtype={'entity_id': str, 'period': str, 'region_code': str})
    slices, scaler = make_slices(panel, cfg)
    expected = pd.date_range('2023-01-01', '2024-12-01', freq='MS').strftime('%Y-%m-%d').tolist()
    ids = slices[0][1]
    if [s[0] for s in slices] != expected or any(s[1] != ids for s in slices):
        raise ValueError('Complete ordered monthly panel required')
    regions = panel.drop_duplicates('entity_id').set_index('entity_id').reindex(ids).region_code.to_numpy()
    if panel.groupby('entity_id').region_code.nunique().max() != 1:
        raise ValueError('Region changes require explicit crosswalk')
    del panel
    monthly = np.stack([s[2] for s in slices])
    annual = np.median(monthly[:12], axis=0)
    frozen_path = root / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json'
    frozen = json.loads(frozen_path.read_text('utf-8'))
    z = np.asarray(frozen['reference_labels'])
    if frozen['ids'] != ids or not np.array_equal(cdist(annual, frozen['centers']).argmin(axis=1), z):
        raise ValueError('Frozen model identity/reproduction failed')
    return slices, monthly, annual, ids, regions, z, scaler, manifest


def run(cfg, root):
    require_execution(cfg, True)
    root = Path(root)
    settings = cfg['followup']
    stage = settings['stage']
    if stage not in ('small_k', 'joint', 'temporal'):
        raise ValueError('Unknown followup stage')
    out = root / settings['output']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', {'status': 'running', 'stage': stage})
    print(json.dumps({'research_run': str(out), 'stage': stage}), flush=True)
    started = time.perf_counter()
    slices, monthly, x, ids, regions, reference, scaler, manifest = load_inputs(root, cfg)
    sources = list((root / 'sbercluster').glob('*.py')) + [root / 'scripts/run_bounded.py']
    hashes = {}
    for p in sources:
        relative = p.relative_to(root).as_posix()
        hashes[relative] = sha256(p)
        destination = out / 'implementation_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    import importlib.metadata
    import platform
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                  'scope': 'exploratory after 2024 inspection; frozen model is unchanged',
                  **code_revision(root),
                  'frozen_prototypes_sha256': sha256(root / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json'),
                  'panel_sha256': manifest['panel_sha256'], 'scaler': scaler,
                  'implementation': hashes, 'ids': ids, 'environment': {
                      'python': platform.python_version(), 'packages': {
                          n: importlib.metadata.version(n) for n in
                          ['numpy', 'pandas', 'scipy', 'scikit-learn', 'igraph', 'leidenalg']}}}
    write_json(out / 'provenance.json', provenance)
    attr, info = knn_graph(x, cfg['graph']['k'])
    graphs = {'attribute': attr}
    graph_info = {'attribute': info}
    rows = []
    labels = {}
    comparisons = []

    def record(name, z, spec, fit_info=None):
        row = {'id': name, **spec, **all_metrics(x, z, attr),
               'ARI_frozen_k4': float(adjusted_rand_score(reference, z)),
               'warnings': (fit_info or {}).get('warnings', []), 'fit_info': fit_info or {}}
        rows.append(row)
        labels[name] = list(map(int, z))
        comparisons.append({'id': name, **{key: network_indices(g, z) for key, g in graphs.items()}})
        # Partial completed results survive a later resource interruption.
        write_json(out / 'metrics.json', rows)
        write_json(out / 'labels.json', {'ids': ids, 'labels': labels})
        write_json(out / 'network_metrics.json', comparisons)
        print(json.dumps({'candidate': name, 'SW': row['SW'], 'k': row['k'],
                          'min_cluster_size': row['min_cluster_size']}), flush=True)

    def fit(name, spec, g=attr):
        z, fit_info = fit_candidate(x, g, spec, cfg['seed'], cfg)
        record(name, z, spec, fit_info)

    if stage == 'small_k':
        record('frozen_kmeans_k4', reference, {'method': 'frozen_kmeans', 'graph': 'attribute'})
        for method in ['kmeans', 'ward']:
            for k in [2, 3]:
                fit(f'{method}_k{k}', {'method': method, 'k': k, 'graph': 'attribute'})
    elif stage == 'joint':
        source = root / settings['source_dir'] / 'hackathonlicence/connection.parquet'
        road, _, road_info = read_road_graph(source, ids, cfg['graph']['k'])
        graphs['transport'] = road
        graph_info['transport'] = road_info
        write_json(out / 'graphs.json', graph_info)
        save_npz(out / 'transport_graph.npz', road)
        baseline_dir = root / settings['baseline_dir']
        baseline_status = json.loads((baseline_dir / 'status.json').read_text('utf-8'))
        baseline_provenance = json.loads((baseline_dir / 'provenance.json').read_text('utf-8'))
        baseline = json.loads((baseline_dir / 'labels.json').read_text('utf-8'))
        if (baseline_status['status'] != 'completed' or baseline['ids'] != ids
                or baseline_provenance['panel_sha256'] != manifest['panel_sha256']
                or baseline_provenance['config']['features'] != cfg['features']
                or baseline_provenance['config']['seed'] != cfg['seed']
                or baseline_status.get('stage') != 'small_k'):
            raise ValueError('Aligned completed small-K baseline with identical features required')
        provenance.update(road_sha256=sha256(source), road_date=road_info['as_of'],
                          baseline_labels_sha256=sha256(baseline_dir / 'labels.json'),
                          baseline_provenance_sha256=sha256(baseline_dir / 'provenance.json'),
                          graph_scope='2024 road accessibility; not 2023 prospective data or observed mobility',
                          objective='SSE/TSS + alpha * undirected_cut_weight/undirected_edge_mass')
        write_json(out / 'provenance.json', provenance)
        baselines = {k: np.asarray(baseline['labels'][f'kmeans_k{k}']) for k in [2, 3]}
        baselines[4] = reference
        new_partitions = 0
        for k, z0 in baselines.items():
            _, info0 = fit_graph_regularized_kmeans(x, road, z0, alpha=0)
            record(f'joint_alpha0_k{k}', z0, {'method': 'reused_kmeans', 'graph': 'none',
                   'alpha': 0, 'k': k}, info0)
            alpha = settings['alpha']
            z, details = fit_graph_regularized_kmeans(x, road, z0, alpha=alpha,
                         seed=cfg['seed'], max_sweeps=settings['max_sweeps'])
            record(f'joint_road_k{k}', z, {'method': 'graph_regularized_kmeans',
                   'graph': 'transport', 'alpha': alpha, 'k': k}, details)
            new_partitions += 1
        alpha = settings['sensitivity_alpha']
        z, details = fit_graph_regularized_kmeans(x, road, reference, alpha=alpha,
                     seed=cfg['seed'], max_sweeps=settings['max_sweeps'])
        record('joint_road_sensitivity_k4', z, {'method': 'graph_regularized_kmeans',
               'graph': 'transport', 'alpha': alpha, 'k': 4}, details)
        new_partitions += 1
        for seed in settings['permutation_seeds']:
            shuffled, order = permute_within_groups(road, regions, seed)
            z, details = fit_graph_regularized_kmeans(x, shuffled, reference,
                         alpha=settings['alpha'], seed=cfg['seed'], max_sweeps=settings['max_sweeps'])
            name = 'joint_region_shuffled_' + str(seed)
            details['objective_on_real_road'] = joint_objective(x, road, z, settings['alpha'])
            record(name, z, {'method': 'graph_regularized_kmeans', 'graph': 'region_shuffled_transport',
                   'alpha': settings['alpha'], 'permutation_seed': seed, 'k': 4}, details)
            write_json(out / (name + '_permutation.json'), order.tolist())
            new_partitions += 1
        fit('spectral_transport_k4', {'method': 'spectral_global', 'k': 4, 'graph': 'transport'}, road)
        new_partitions += 1
    else:
        # Independent phases can use a small pilot before all24 slices.
        indices = settings['month_indices']
        if len(indices) < 2 or indices != sorted(set(indices)) or any(not 0 <= i < 24 for i in indices):
            raise ValueError('Distinct sorted month indices required')
        selected = [slices[i] for i in indices]
        monthly_graphs = [knn_graph(s[2], cfg['graph']['k'])[0] for s in selected]
        if settings.get('graph_layer', 'attribute') == 'attribute_road':
            source = root / settings['source_dir'] / 'hackathonlicence/connection.parquet'
            road, _, road_info = read_road_graph(source, ids, cfg['graph']['k'])
            weight = settings['attribute_graph_weight']
            monthly_graphs = [mix_layers(g, road, weight) for g in monthly_graphs]
            graph_info['transport'] = road_info
            write_json(out / 'graphs.json', graph_info)
            provenance.update(road_sha256=sha256(source), road_date=road_info['as_of'],
                              graph_scope='Retrospective 2024 road layer used in every slice',
                              attribute_graph_weight=weight)
            write_json(out / 'provenance.json', provenance)
        elif settings.get('graph_layer', 'attribute') != 'attribute':
            raise ValueError('Unknown temporal graph layer')
        variants = []
        for omega in settings['omegas']:
            temporal_cfg = deepcopy(cfg)
            temporal_cfg['clustering']['temporal_coupling_relative'] = omega
            memberships, fit_info = fit_temporal(selected, monthly_graphs, temporal_cfg, execute=True)
            links = [transitions(ids, memberships[i-1], ids, memberships[i]) for i in range(1, len(selected))]
            per_month = [{'period': s[0], **all_metrics(s[2], z, g)}
                         for s, z, g in zip(selected, memberships, monthly_graphs)]
            sw = [m['SW'] for m in per_month if m['SW'] is not None]
            summary = {'omega_relative': omega, 'graph_layer': settings.get('graph_layer', 'attribute'), 'months': len(selected),
                       'mean_ARI_adjacent': float(np.mean([v['ARI'] for v in links])),
                       'mean_matched_churn': float(np.mean([v['matched_churn'] for v in links])),
                       'mean_SW': float(np.mean(sw)) if sw else None,
                       'groups_min': min(len(np.unique(z)) for z in memberships),
                       'groups_max': max(len(np.unique(z)) for z in memberships), **fit_info}
            variants.append(summary)
            write_json(out / f'temporal_omega{omega:g}.json', {'ids': ids,
                       'periods': [s[0] for s in selected], 'labels': [z.tolist() for z in memberships],
                       'transitions': links, 'monthly_metrics': per_month, 'summary': summary})
            write_json(out / 'temporal_summary.json', variants)
            print(json.dumps({'phase': 'temporal', **summary}), flush=True)
    write_json(out / 'graphs.json', graph_info)
    if rows:
        pd.DataFrame(rows).drop(columns=['warnings', 'S_Dbw_diagnostics', 'fit_info'], errors='ignore').to_csv(out / 'metrics.csv', index=False)
    summary = {'stage': stage, 'seconds': time.perf_counter()-started,
               'new_static_partitions': new_partitions if stage == 'joint' else max(0, len(rows)-1), 'n': len(ids),
               'temporal_variants': len(settings['omegas']) if stage == 'temporal' else 0}
    write_json(out / 'summary.json', summary)
    write_json(out / 'status.json', {'status': 'completed', **summary})
    print(json.dumps({'status': 'completed', **summary}), flush=True)
    return out
