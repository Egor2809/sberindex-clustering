"""Build the atlas research extension from published exploratory evidence."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)

def build(research_dir, output, external_results):
    research_dir, external_results = research_dir.resolve(), external_results.resolve()
    base = read(ROOT / 'reports/contest-v3/atlas_extension.json')
    small = research_dir / 'review-20260924-small-k'
    joint = research_dir / 'review-20260924-joint'
    temporal = research_dir / 'review-20260924-temporal'
    rows = read(joint / 'metrics.json')
    memberships = read(joint / 'labels.json.gz')
    ids, labels = memberships['ids'], memberships['labels']
    reference = read(ROOT / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json')
    if ids != reference['ids']:
        raise ValueError('Review and frozen reference identifiers differ')
    bridge_path = ROOT / 'reports/external-v4/municipality_bridge.csv'
    bridge = pd.read_csv(bridge_path,
        usecols=['entity_id', 'municipal_district_type', 'expense_2023'],
        dtype={'entity_id': str, 'municipal_district_type': str})
    if bridge.entity_id.isna().any() or bridge.entity_id.duplicated().any():
        raise ValueError('Comparability bridge requires unique nonmissing entity_id')
    bridge = bridge.set_index('entity_id').reindex(ids)
    expense = pd.to_numeric(bridge.expense_2023, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(expense).all() or (expense <= 0).any():
        raise ValueError('Missing or invalid 2023 expense for comparability')
    base['comparability'] = {
        entity_id: {
            'municipal_district_type': None if pd.isna(row.municipal_district_type)
                else str(row.municipal_district_type),
            'expense_2023': float(row.expense_2023),
        }
        for entity_id, row in bridge.iterrows()
    }
    base['comparability_definition'] = {
        'source': 'reports/external-v4/municipality_bridge.csv',
        'expense_2023': 'Median of the 12 monthly All categories values in 2023',
        'ratio_direction': 'neighbor expense_2023 / selected expense_2023',
        'used_to_filter_or_rank_neighbors': False,
    }
    z0 = np.asarray(reference['reference_labels'])
    panel = pd.read_csv(ROOT/'data/processed/panel.csv',usecols=['entity_id','period','Общественное питание','Все категории'])
    panel = panel[panel.period.str.startswith('2023')].copy()
    panel['service_ratio'] = panel['Общественное питание']/panel['Все категории']
    service_ratio = panel.groupby('entity_id').service_ratio.median().reindex(ids).to_numpy()
    if not np.isfinite(service_ratio).all():raise ValueError('Missing service ratios for model naming')
    profile_names = ['Сбалансированный: средние МО', 'Продуктовый: малые районы',
                     'Городской сервисный', 'Удалённый: мало маркетплейсов']
    base['profiles'][0]['name'] = profile_names[0]
    base['profiles'][0]['subtitle'] = 'Общепит и транспорт заметнее, чем в продуктовом типе; в составе много районов и городских округов.'
    map_models = {}
    for candidate in ['joint_road_k4', 'joint_road_k2']:
        z = np.asarray(labels[candidate], dtype=int)
        k = int(z.max()) + 1
        if k == 4:
            contingency = np.zeros((4, 4), dtype=int)
            np.add.at(contingency, (z0, z), 1)
            old, new = linear_sum_assignment(-contingency)
            mapping = dict(zip(new.tolist(), old.tolist()))
            display = np.array([mapping[int(v)] for v in z])
            names = ['Группа '+str(i+1)+' · '+profile_names[i].lower() for i in range(4)]
        else:
            # Label semantics follow the attribute prototypes; no production sector is inferred.
            services = [float(np.median(service_ratio[z == c])) for c in range(k)]
            order = np.argsort(services, kind='stable')
            mapping = {int(old):int(new) for new, old in enumerate(order)}
            display = np.array([mapping[int(v)] for v in z])
            names = ['Больше повседневных покупок', 'Больше услуг']
        map_models[candidate] = {
            'name': 'Расходы + дороги · '+str(k)+' группы',
            'labels': dict(zip(ids, display.tolist())),
            'display_label_mapping': mapping,
            'profiles': [{'id': c, 'name': names[c], 'n': int(np.sum(display == c)), 'median_foodservice_ratio': float(np.median(service_ratio[display == c]))} for c in range(k)],
            'note': 'Совместная модель: расходы 2023 года + дорожная доступность на 31.12.2024, α=0,25. Ретроспективное сравнение; центры опорной модели и её проверка 2024 года сохранены отдельно.'}
    base['map_models'] = map_models
    info = read(joint / 'graphs.json')
    base['network'] = {
        'candidates': [{**r, 'ARI_reference': r['ARI_frozen_k4']} for r in rows],
        'labels': {name: dict(zip(ids, values)) for name, values in labels.items()},
        'layers': [{'name': {'attribute': 'Расходы', 'transport': 'Дорожная доступность'}[name], **values}
                   for name, values in info.items()],
        'note': 'Один метод использует два источника: профиль расходов и дорожный граф. Показаны слабая и сильная регуляризация, исходный KMeans и три перестановки дорожных связей внутри регионов. Все индексы атрибутного сравнения вычислены на одном опорном графе.'}
    variants = read(temporal / 'temporal_summary.json')
    base['temporal'] = {'variants': [{**r, 'mean_churn': r['mean_matched_churn']} for r in variants],
        'note': '24 месяца, один и тот же смешанный граф расходов и дорог, связь соседних месяцев ω=0 или 0,1. Стабильность оптимизируется самим методом; это описание динамики, а не доказательство раннего предупреждения.'}
    baseline_rows = read(small / 'metrics.json')
    existing_ids = {r['candidate'] for r in base['metric_comparison']}
    base['metric_comparison'].extend({**r, 'candidate': r['id']} for r in baseline_rows
        if r['id'] != 'frozen_kmeans_k4' and r['id'] not in existing_ids)
    base['external_controls'] = read(external_results)
    stability_path = research_dir/'review-20260924-joint-stability/metrics.json'
    base['joint_stability'] = read(stability_path)
    base['review_scope'] = 'Exploratory extensions; the 2024 data were already inspected. Original frozen results remain unchanged.'
    base['source_hashes'] = {str(p.relative_to(ROOT)).replace(chr(92), '/'): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [small/'metrics.json', joint/'metrics.json', joint/'labels.json.gz', temporal/'temporal_summary.json', external_results, stability_path, bridge_path]}
    from extend_round2_atlas import extend
    extend(base, ROOT, ids, z0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(base, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'output': str(output), 'models': len(base['map_models']), 'comparisons': len(base['metric_comparison'])}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-dir', type=Path, default=ROOT/'reports/review-2026-09-24')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/review-2026-09-24/atlas_extension.json')
    parser.add_argument('--external-results', type=Path, default=ROOT/'reports/external-v4/results.json')
    args = parser.parse_args()
    build(args.research_dir, args.output, args.external_results)
