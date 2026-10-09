"""Frozen-label administrative confounding audit; no clustering or prediction fit.

Partial R-squared is a descriptive least-squares decomposition. Both outcome
and cluster indicators are residualized on identical region/type controls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = 'reports/experiments/2026-09-23-v2/validation/reference_assignments.csv'
REGISTRY = 'artifacts/sources/acquisition/t_dict_municipal_districts.xlsx'
MARKET = 'data/processed/market_access_2024.csv'
PANEL = 'data/processed/panel.csv'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def join_registry(labels, registry, year=2023):
    """Exact stable ID + half-open registry validity; ambiguous versions fail."""
    valid = registry.loc[(registry.year_from <= year) & (year < registry.year_to)].copy()
    valid['entity_id'] = 'tid_' + valid.territory_id.astype(int).astype(str)
    if labels.entity_id.duplicated().any() or valid.entity_id.duplicated().any():
        raise ValueError('Duplicate entity/version key')
    merged = labels.merge(valid, on='entity_id', how='left', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('A frozen entity has no registry version')
    return merged.drop(columns='_merge')


def center_by(matrix, codes, n_groups):
    counts = np.bincount(codes, minlength=n_groups)
    sums = np.zeros((n_groups, matrix.shape[1]))
    np.add.at(sums, codes, matrix)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    return matrix - means[codes]


def residual_matrix(frame, outcome, controls):
    columns = [np.asarray(frame[outcome], float)]
    columns.extend((frame.cluster.to_numpy() == k).astype(float) for k in range(4))
    if controls == 'region_type_level':
        columns.append(np.asarray(frame.log_expense_2023, float))
    matrix = np.column_stack(columns)
    if controls in ('region', 'region_type', 'region_type_level'):
        stratum = frame.region_code.astype(str)
        if controls != 'region':
            stratum = stratum + '|' + frame.municipal_district_type
    elif controls == 'type':
        stratum = frame.municipal_district_type
    elif controls == 'none':
        stratum = pd.Series('all', index=frame.index)
    else:
        raise ValueError(controls)
    codes, levels = pd.factorize(stratum, sort=True)
    return center_by(matrix, codes, len(levels)), codes


def partial_r2(cross_product, adjust_level=False):
    """Partial R² from within-stratum cross-products; handles rank deficiency."""
    c = cross_product.copy()
    if adjust_level:
        level_variance = c[-1, -1]
        c = c[:-1, :-1] - (np.outer(c[:-1, -1], c[-1, :-1]) / level_variance if level_variance > 1e-12 else 0)
    total = c[0, 0]
    if total <= 1e-12:
        return None
    explained = float(c[0, 1:] @ np.linalg.pinv(c[1:, 1:], rcond=1e-12) @ c[1:, 0])
    return float(np.clip(explained / total, 0, 1))


def audit_effect(frame, outcome, controls, bootstrap=500, seed=1729):
    f = frame.loc[np.isfinite(frame[outcome])].copy()
    matrix, strata = residual_matrix(f, outcome, controls)
    region, regions = pd.factorize(f.region_code, sort=True)
    adjust = controls == 'region_type_level'
    observed = partial_r2(matrix.T @ matrix, adjust)
    # Region resampling changes stratum means for pooled controls; recenter them
    # with frequency weights. For region-based strata, means stay unchanged.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        counts = np.bincount(rng.integers(len(regions), size=len(regions)), minlength=len(regions))
        weights = counts[region].astype(float)
        m = matrix
        if controls in ('none', 'type'):
            ns = int(strata.max()) + 1
            sw = np.bincount(strata, weights=weights, minlength=ns)
            sums = np.zeros((ns, matrix.shape[1]))
            np.add.at(sums, strata, matrix * weights[:, None])
            means = np.divide(sums, sw[:, None], out=np.zeros_like(sums), where=sw[:, None] > 0)
            m = matrix - means[strata]
        value = partial_r2(m.T @ (m * weights[:, None]), adjust)
        if value is not None:
            draws.append(value)
    mixed = f.groupby(pd.Series(strata, index=f.index)).cluster.nunique()
    informative = np.isin(strata, mixed.index[mixed > 1])
    return {'outcome': outcome, 'controls': controls, 'n': len(f), 'regions': len(regions),
            'strata': int(strata.max()) + 1, 'mixed_cluster_strata': int((mixed > 1).sum()),
            'n_in_mixed_cluster_strata': int(informative.sum()), 'partial_r_squared': observed,
            'region_bootstrap_percentile_95': np.quantile(draws, [.025, .975]).tolist() if draws else None,
            'bootstrap_draws': bootstrap, 'valid_bootstrap_draws': len(draws), 'seed': seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/external-v4')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 20:
        raise ValueError('At least 20 bootstrap draws required')
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('External audit output must be empty or new')
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / LABELS)
    registry = pd.read_excel(ROOT / REGISTRY)
    bridge = join_registry(labels, registry)
    panel = pd.read_csv(ROOT / PANEL, usecols=['entity_id', 'period', 'Все категории'])
    panel = panel.loc[panel.period.str.startswith('2023')]
    counts = panel.groupby('entity_id').size()
    if not counts.eq(12).all():
        raise ValueError('Expected 12 monthly expense records per entity')
    expense = panel.groupby('entity_id')['Все категории'].median().rename('expense_2023')
    bridge = bridge.merge(expense, on='entity_id', validate='one_to_one')
    if len(bridge) != len(labels) or (bridge.expense_2023 <= 0).any():
        raise ValueError('Missing or invalid expense values')
    bridge['log_expense_2023'] = np.log(bridge.expense_2023)
    market = pd.read_csv(ROOT / MARKET)
    bridge = bridge.merge(market, on='territory_id', how='left', validate='one_to_one')
    bridge['intracity_moscow_petersburg'] = bridge.region_code.isin([77, 78]) & bridge.municipal_district_type.str.startswith('внутригородская')
    bridge['administrative_center'] = bridge.municipal_district_status.eq('административный_центр_субъекта')
    cols = ['entity_id', 'territory_id', 'cluster', 'region_code', 'region_name', 'oktmo', 'municipal_district_type', 'municipal_district_status', 'year_from', 'year_to', 'intracity_moscow_petersburg', 'administrative_center', 'expense_2023', 'log_expense_2023', 'market_access']
    bridge[cols].to_csv(out / 'municipality_bridge.csv', index=False)
    tables = {}
    for dimension in ['municipal_district_type', 'region_code', 'intracity_moscow_petersburg', 'administrative_center']:
        table = pd.crosstab(bridge[dimension], bridge.cluster).reindex(columns=range(4), fill_value=0)
        table.to_csv(out / ('composition_' + dimension + '.csv'))
        tables[dimension] = [{'value': str(i), 'cluster_counts': [int(v) for v in row]} for i, row in table.iterrows()]
    results = []
    coverage = []
    for subset, f in [('all', bridge), ('without_moscow_petersburg_intracity', bridge.loc[~bridge.intracity_moscow_petersburg])]:
        for cluster, rows in f.groupby('cluster'):
            coverage.append({'subset': subset, 'cluster': int(cluster), 'n': len(rows), 'regions': int(rows.region_code.nunique()), 'market_observed': int(rows.market_access.notna().sum()), 'market_missing': int(rows.market_access.isna().sum()), 'median_market_access': float(rows.market_access.median()), 'median_expense_2023': float(rows.expense_2023.median())})
        for outcome in ['market_access', 'log_expense_2023']:
            controls = ['none', 'region', 'type', 'region_type']
            if outcome == 'market_access':
                controls.append('region_type_level')
            for control in controls:
                result = audit_effect(f, outcome, control, args.bootstrap)
                result['subset'] = subset
                results.append(result)
    pd.DataFrame(coverage).to_csv(out / 'coverage_by_cluster.csv', index=False)
    sources = [{'path': p, 'sha256': digest(ROOT / p)} for p in [LABELS, REGISTRY, MARKET, PANEL]]
    result = {'scope': 'Exploratory fixed-label administrative confounding; no clustering fitted; no new independent wage, employment, sector or population data.',
              'registry_year': 2023, 'market_year': 2024, 'expense_year': 2023, 'cluster_display_offset': 1,
              'n': len(bridge), 'regions': int(bridge.region_code.nunique()), 'excluded_intracity_n': int(bridge.intracity_moscow_petersburg.sum()),
              'sources': sources, 'script_sha256': digest(__file__), 'composition': tables, 'coverage': coverage, 'effects': results,
              'limitations': ['Market access was previously examined; these are exploratory sensitivity checks.', 'No prospective holdout, causal interpretation or production-type validation.', 'Region-bootstrap intervals assume regions are resampling blocks and are conditional on fixed labels and observed sample; cross-region spatial dependence remains.', 'Partial R² is unadjusted for model degrees of freedom, nonnegative and upward biased under a null. Percentile intervals are not null-hypothesis tests.', 'Full region × type interaction may leave sparse or single-cluster strata. Their counts are reported.', 'Capital exclusion keeps the original labels and does not establish stability under retraining.', 'Status empty in source is interpreted as no designated administrative-center flag; no population-size control is available.']}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'n': result['n'], 'excluded_intracity_n': result['excluded_intracity_n'], 'effects': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
