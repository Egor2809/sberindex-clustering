"""Synthetic transition-tracking test: temporal Leiden against monthly and frozen KMeans on planted groups and switches.

Group profiles are the mean shares of the seven stable temporal groups; the common monthly shift and the persistent and
monthly noise of the log shares are measured on the real panel around the monthly temporal Leiden groups. Temporal
Leiden runs with the configuration of the main model and is not tuned to the synthetic data.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from .contest_graphs import mix_layers
from .graph import knn_graph
from .io import sha256, write_json
from .models import fit_temporal
from .temporal_groups import CATEGORIES, TOTAL

RULE = ('Я сравниваю временной Leiden с каждой из двух базовых линий (KMeans, обучаемый заново каждый месяц, и замороженный '
        'KMeans) по трём метрикам: средний по месяцам ARI с истиной (больше лучше), доля настоящих переходов, найденных в '
        'пределах ±1 месяц (больше лучше), число ложных переключений (меньше лучше). Leiden лучше линии по метрике, если '
        'среднее по повторам генерации в его пользу и он строго лучше не меньше чем в 4 из 5 повторов. Leiden выигрывает у '
        'линии, если он лучше хотя бы по двум метрикам из трёх. Leiden выигрывает тест, если выигрывает у обеих линий; если '
        'только у одной, я пишу «частичный выигрыш»; если ни у одной, «не выигрывает». После прогона я не меняю ни это '
        'правило, ни параметры генерации, ни определения метрик.')
RULE_WRITTEN = '2026-10-09T13:18:18Z'
METHODS = ('temporal_leiden', 'kmeans_monthly', 'kmeans_frozen')
BASELINES = ('kmeans_monthly', 'kmeans_frozen')
METRICS = {'ari': True, 'recall': True, 'false_switches': False}
GENERATION = {'n': 2000, 'months': 24, 'transition_share': 0.1, 'first_transition': 3, 'last_transition': 21,
              'spatial_share': 0.6, 'centres_per_group': 3, 'centre_spread': 0.07, 'road_degree': 15,
              'calibration_months': 12, 'kmeans_n_init': 10, 'window': 1}


def periods(months):
    return pd.date_range('2023-01-01', periods=months, freq='MS').strftime('%Y-%m-%d').tolist()


def mad(values, axis):
    centre = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - centre), axis=axis)


def calibrate(panel_path, labels_path):
    """Common monthly shift of the median log share and robust persistent and monthly within-group spread."""
    with gzip.open(labels_path, 'rt', encoding='utf-8') as stream:
        stored = json.load(stream)
    ids, months = stored['ids'], stored['periods']
    panel = pd.read_csv(panel_path, usecols=['entity_id', 'period', *CATEGORIES, TOTAL], dtype={'entity_id': str, 'period': str})
    panel = panel.set_index(['period', 'entity_id'])
    logs = np.stack([np.log(panel.loc[p].reindex(ids)[list(CATEGORIES)].to_numpy(float)
                            / panel.loc[p].reindex(ids)[[TOTAL]].to_numpy(float)) for p in months])
    if not np.isfinite(logs).all():
        raise ValueError('Panel does not cover the temporal cohort')
    shift = np.median(logs, axis=1)
    shift -= shift.mean(axis=0)
    labels = np.asarray(stored['labels'])
    residual = np.empty_like(logs)
    for t in range(len(months)):
        for g in np.unique(labels[t]):
            mask = labels[t] == g
            residual[t, mask] = logs[t, mask] - np.median(logs[t, mask], axis=0)
    persistent = residual.mean(axis=0)
    return {'shift': np.round(shift, 6).tolist(), 'persistent_sd': np.round(mad(persistent, 0), 6).tolist(),
            'monthly_sd': np.round(mad((residual - persistent).reshape(-1, len(CATEGORIES)), 0), 6).tolist()}


def group_profiles(path):
    data = json.loads(Path(path).read_text('utf-8'))
    names = [g['name'] for g in data['groups']]
    shares = np.array([[g['mean_shares'][c] for c in CATEGORIES] for g in data['groups']])
    cells = np.array([g['cells'] for g in data['groups']], float)
    return names, shares, cells / cells.sum()


def road_graph(pos, degree):
    radius = math.sqrt(degree / (math.pi * len(pos)))
    pairs = cKDTree(pos).query_pairs(radius, output_type='ndarray')
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    w = np.exp(-np.square(d / radius))
    a = csr_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(len(pos),) * 2)
    return a


def standardise(logs, months):
    ref = logs[:months].reshape(-1, logs.shape[2])
    centre = np.median(ref, axis=0)
    iqr = np.quantile(ref, .75, axis=0) - np.quantile(ref, .25, axis=0)
    return (logs - centre) / iqr


def generate(seed, shares, weights, cal, p=GENERATION):
    """Planted groups, a share of nodes switching once to another group, a common shift and noise from the panel."""
    rng = np.random.default_rng(seed)
    n, months, k = p['n'], p['months'], len(shares)
    start = rng.choice(k, size=n, p=weights)
    movers = rng.choice(n, size=round(p['transition_share'] * n), replace=False)
    change = np.full(n, -1)
    change[movers] = rng.integers(p['first_transition'], p['last_transition'] + 1, size=len(movers))
    target = start.copy()
    target[movers] = (start[movers] + rng.integers(1, k, size=len(movers))) % k
    truth = np.where((change >= 0) & (np.arange(months)[:, None] >= change), target, start)
    shift = np.asarray(cal['shift'])[:months]
    unique = rng.normal(0, cal['persistent_sd'], size=(n, len(CATEGORIES)))
    noise = rng.normal(0, cal['monthly_sd'], size=(months, n, len(CATEGORIES)))
    logs = np.log(shares)[truth] + shift[:, None, :] + unique + noise
    centres = rng.uniform(0, 1, size=(k, p['centres_per_group'], 2))
    near = rng.uniform(size=n) < p['spatial_share']
    pos = rng.uniform(0, 1, size=(n, 2))
    pick = centres[start, rng.integers(0, p['centres_per_group'], size=n)] + rng.normal(0, p['centre_spread'], size=(n, 2))
    pos[near] = np.clip(pick[near], 0, 1)
    return {'x': standardise(logs, p['calibration_months']), 'truth': truth, 'change': change,
            'road': road_graph(pos, p['road_degree'])}


def align(previous, current):
    """Relabel current so that its groups carry the labels of the previous month they overlap most (Hungarian)."""
    a, b = np.unique(previous), np.unique(current)
    table = np.zeros((len(a), len(b)))
    np.add.at(table, (np.searchsorted(a, previous), np.searchsorted(b, current)), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {b[c]: a[r] for r, c in zip(rows, cols)}
    spare = max(a.max(), b.max()) + 1
    for v in b:
        if v not in mapping:
            mapping[v], spare = spare, spare + 1
    return np.array([mapping[v] for v in current])


def kmeans_monthly(x, k, seed, n_init):
    out = []
    for t in range(len(x)):
        z = KMeans(k, n_init=n_init, random_state=seed).fit_predict(x[t])
        out.append(z if not out else align(out[-1], z))
    return np.array(out)


def kmeans_frozen(x, k, seed, n_init, months):
    model = KMeans(k, n_init=n_init, random_state=seed).fit(x[:months].reshape(-1, x.shape[2]))
    return np.array([model.predict(x[t]) for t in range(len(x))])


def temporal_leiden(x, road, cfg):
    c = deepcopy(cfg)
    ids = [f'n{i:05d}' for i in range(x.shape[1])]
    slices = [(p, ids, x[t]) for t, p in enumerate(periods(len(x)))]
    graphs = [mix_layers(knn_graph(x[t], c['graph']['k'])[0], road, c['followup'].get('attribute_graph_weight', .5))
              for t in range(len(x))]
    memberships, info = fit_temporal(slices, graphs, c, execute=True)
    return np.array(memberships), info


def tracking(pred, change, window=1):
    """Recall of planted switches within +-window months and every other change of predicted group (false switches)."""
    switched = pred[1:] != pred[:-1]
    found = false = returns = 0
    for i in range(pred.shape[1]):
        times = np.flatnonzero(switched[:, i]) + 1
        hit = False
        if change[i] >= 0 and len(times):
            hit = bool(np.min(np.abs(times - change[i])) <= window)
        found += hit
        false += len(times) - hit
        labels = pred[:, i]
        returns += sum(labels[a - 1] == labels[b] for a, b in zip(times[:-1], times[1:]))
    movers = int((change >= 0).sum())
    stayers = change < 0
    return {'movers': movers, 'found': int(found), 'recall': round(found / movers, 6) if movers else None,
            'false_switches': int(false), 'false_per_1000_node_months': round(1000 * false / pred.size, 4),
            'stayers_with_switch': round(float(switched[:, stayers].any(axis=0).mean()), 6), 'returns': int(returns)}


def score(pred, truth, change, window=1):
    ari = [adjusted_rand_score(truth[t], pred[t]) for t in range(len(truth))]
    return {'ari': round(float(np.mean(ari)), 6), 'ari_min': round(float(np.min(ari)), 6),
            'groups_min': int(min(len(np.unique(z)) for z in pred)), 'groups_max': int(max(len(np.unique(z)) for z in pred)),
            **tracking(pred, change, window)}


def repeat(args):
    seed, shares, weights, cal, cfg, p = args
    data = generate(seed, shares, weights, cal, p)
    k, out, times = len(shares), {}, {}
    start = time.perf_counter()
    leiden, info = temporal_leiden(data['x'], data['road'], cfg)
    times['temporal_leiden'] = time.perf_counter() - start
    start = time.perf_counter()
    monthly = kmeans_monthly(data['x'], k, cfg['seed'], p['kmeans_n_init'])
    times['kmeans_monthly'] = time.perf_counter() - start
    start = time.perf_counter()
    frozen = kmeans_frozen(data['x'], k, cfg['seed'], p['kmeans_n_init'], p['calibration_months'])
    times['kmeans_frozen'] = time.perf_counter() - start
    for name, pred in (('temporal_leiden', leiden), ('kmeans_monthly', monthly), ('kmeans_frozen', frozen)):
        out[name] = {**score(pred, data['truth'], data['change'], p['window']), 'seconds': round(times[name], 2)}
    out['truth'] = {'transitions': int((data['change'] >= 0).sum()), 'road_edges': int(data['road'].nnz // 2),
                    'leiden_omega_absolute': info['omega']}
    return seed, out


def better(leiden, base, higher):
    """Leiden is better on a metric when the mean paired difference favours it and it is strictly better in >= 80 % of repeats."""
    d = (np.asarray(leiden, float) - np.asarray(base, float)) * (1 if higher else -1)
    need = math.ceil(0.8 * len(d))
    return {'mean_difference': round(float(d.mean()), 6), 'repeats_better': int((d > 0).sum()), 'needed': need,
            'better': bool(d.mean() > 0 and (d > 0).sum() >= need)}


def verdict(per_seed):
    seeds = list(per_seed)
    out = {}
    for base in BASELINES:
        rows = {m: better([per_seed[s]['temporal_leiden'][m] for s in seeds], [per_seed[s][base][m] for s in seeds], higher)
                for m, higher in METRICS.items()}
        out[base] = {'metrics': rows, 'wins': sum(r['better'] for r in rows.values()) >= 2}
    won = sum(out[b]['wins'] for b in BASELINES)
    out['result'] = 'выигрывает' if won == len(BASELINES) else 'частичный выигрыш' if won else 'не выигрывает'
    return out


def aggregate(per_seed):
    out = {}
    for name in METHODS:
        out[name] = {}
        for key in ('ari', 'ari_min', 'recall', 'false_switches', 'false_per_1000_node_months', 'stayers_with_switch', 'returns',
                    'groups_min', 'groups_max', 'seconds'):
            v = np.array([per_seed[s][name][key] for s in per_seed], float)
            out[name][key] = {'mean': round(float(v.mean()), 4), 'sd': round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                              'min': round(float(v.min()), 4), 'max': round(float(v.max()), 4)}
    return out


def run(cfg, root, output, seeds, profiles, labels, workers=1):
    root = Path(root)
    s = cfg['followup']
    names, shares, weights = group_profiles(root / profiles)
    cal = calibrate(root / 'data/processed/panel.csv', root / labels)
    p = dict(GENERATION)
    start = time.perf_counter()
    jobs = [(seed, shares, weights, cal, cfg, p) for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = dict(pool.map(repeat, jobs))
    else:
        results = dict(map(repeat, jobs))
    per_seed = {str(seed): results[seed] for seed in seeds}
    summary = {'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'rule': RULE, 'rule_written_utc': RULE_WRITTEN, 'rule_sha256': hashlib.sha256(RULE.encode('utf-8')).hexdigest(),
               'generation': p, 'generation_seeds': list(seeds), 'groups': names, 'group_weights': np.round(weights, 6).tolist(),
               'group_shares': np.round(shares, 6).tolist(), 'categories': list(CATEGORIES), 'calibration': cal,
               'leiden': {'resolution': cfg['clustering']['resolution'], 'omega_relative': cfg['clustering']['temporal_coupling_relative'], 'seed': cfg['seed'],
                          'iterations': cfg['clustering']['leiden_iterations'], 'knn_k': cfg['graph']['k'],
                          'attribute_graph_weight': s.get('attribute_graph_weight', .5)},
               'kmeans': {'k': len(shares), 'n_init': p['kmeans_n_init'], 'random_state': cfg['seed'],
                          'frozen_fit_months': p['calibration_months']},
               'per_seed': per_seed, 'aggregate': aggregate(per_seed), 'verdict': verdict(per_seed),
               'seconds': round(time.perf_counter() - start, 1),
               'inputs': {rel: sha256(root / rel) for rel in ('data/processed/panel.csv', profiles, labels)}}
    summary['chance'] = chance(summary)
    out = root / output
    write_json(out / 'summary.json', summary)
    (out / 'README.md').write_text(readme(summary), encoding='utf-8')
    return out


def _n(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')


def _cell(agg, key, digits=3, scale=1):
    return f'{_n(scale * agg[key]["mean"], digits)} ± {_n(scale * agg[key]["sd"], digits)}'


LABELS = {'temporal_leiden': 'Временной Leiden', 'kmeans_monthly': 'KMeans каждый месяц', 'kmeans_frozen': 'KMeans замороженный'}
METRIC_LABELS = {'ari': 'ARI', 'recall': 'найденные переходы', 'false_switches': 'ложные переключения'}


def chance(summary):
    """Monthly switch rate of each method and the share of planted switches a random switcher at that rate would hit."""
    p, a = summary['generation'], summary['aggregate']
    movers, pairs = round(p['transition_share'] * p['n']), p['n'] * (p['months'] - 1)
    out = {}
    for name in METHODS:
        rate = (a[name]['false_switches']['mean'] + a[name]['recall']['mean'] * movers) / pairs
        out[name] = {'switch_rate': round(rate, 4), 'chance_recall': round(1 - (1 - rate) ** (2 * p['window'] + 1), 4)}
    return out


def _inline(name):
    return 'временной Leiden' if name == 'temporal_leiden' else LABELS[name]


def _groups(m):
    lo, hi = int(m['groups_min']['min']), int(m['groups_max']['max'])
    return str(lo) if lo == hi else f'{lo}–{hi}'


def readme(summary):
    p, a, v = summary['generation'], summary['aggregate'], summary['verdict']
    c = chance(summary)
    reps = len(summary['generation_seeds'])
    out = ['# Синтетический тест «отслеживание переходов»', '',
           f'## Правило выигрыша (записано до прогона, {summary["rule_written_utc"]})', '', summary['rule'], '',
           f'SHA-256 текста правила: `{summary["rule_sha256"]}`.', '',
           '## Итог', '',
           f'По этому правилу временной Leiden **{v["result"]}**.', '',
           '| Метод | ARI с истиной | Найдено переходов (±1 мес.) | Ложные переключения | Неперешедших с переключением | Возвраты туда-обратно | Групп в месяц |',
           '|---|---|---|---|---|---|---|']
    for name in METHODS:
        m = a[name]
        out.append(f'| {LABELS[name]} | {_cell(m, "ari")} | {_cell(m, "recall", 1, 100)} % | {_cell(m, "false_switches", 0)} | '
                   f'{_cell(m, "stayers_with_switch", 1, 100)} % | {_cell(m, "returns", 0)} | '
                   f'{_groups(m)} |')
    out += ['', f'Среднее ± стандартное отклонение по {reps} повторам генерации (seed {", ".join(map(str, summary["generation_seeds"]))}).', '',
            '| Против линии | Метрика | Средняя разница в пользу Leiden | Повторов, где Leiden лучше | Лучше? |', '|---|---|---|---|---|']
    for base in BASELINES:
        for m, row in v[base]['metrics'].items():
            out.append(f'| {LABELS[base]} | {METRIC_LABELS[m]} | {_n(row["mean_difference"], 4)} | {row["repeats_better"]} из {reps} | '
                       f'{"да" if row["better"] else "нет"} |')
    out += ['', 'Выигрыш у линии: ' + '; '.join(f'{LABELS[b]}: {"да" if v[b]["wins"] else "нет"}' for b in BASELINES) + '.', '',
            '## Как читать', '',
            'Доля найденных переходов зависит от того, как часто метод вообще меняет группу узла. Если метод меняет группу в доле '
            f'r пар соседних месяцев, окно ±{p["window"]} месяц случайно ловит смену с вероятностью 1 − (1 − r)^{2 * p["window"] + 1}. '
            + 'По методам: ' + '; '.join(f'{_inline(n)} r = {_n(100 * c[n]["switch_rate"], 1)} %, случайная доля {_n(100 * c[n]["chance_recall"], 1)} %, '
                        f'найдено {_n(100 * a[n]["recall"]["mean"], 1)} %' for n in METHODS) + '. '
            'Превышение над случайной долей: ' + '; '.join(
                f'{_inline(n)} {_n(100 * (a[n]["recall"]["mean"] - c[n]["chance_recall"]), 1)} п. п.' for n in METHODS) + ' '
            'Значит, KMeans не только чаще переключает узлы, но и действительно реагирует на переход, а временной Leiden '
            'почти не отличает настоящий переход от фона. Вероятная причина в том, что связь между месяцами и дороги, которые '
            'тянут узел к прежней группе, удерживают перешедший узел на месте. Временной Leiden делает в '
            f'{_n(a["kmeans_monthly"]["false_switches"]["mean"] / a["temporal_leiden"]["false_switches"]["mean"], 1)} раза меньше '
            'ложных переключений, чем KMeans каждый месяц, при близком ARI, но находит меньше половины настоящих переходов и не '
            'выигрывает по заранее записанному правилу. При шуме, измеренном на реальной панели, ARI всех трёх методов с '
            'истиной низкий (около 0,3).', '',
            '## Как устроены данные', '',
            f'Я генерирую {p["n"]} узлов на {p["months"]} месяцев. Каждый узел получает одну из семи групп с вероятностью, '
            'пропорциональной числу пар «МО × месяц» группы в `reports/temporal-groups/profiles.json`; профиль группы равен средним '
            'долям пяти категорий этой группы. Логарифм доли категории узла в месяце равен логарифму доли его группы плюс общий '
            'сдвиг месяца, плюс постоянное отклонение узла, плюс месячный шум. Общий сдвиг, разброс постоянных отклонений и месячного '
            'шума я измерил на реальной панели: сдвиг как медиану логарифма доли по МО в каждом месяце, разбросы как робастное '
            'стандартное отклонение (1,4826 × MAD) остатков вокруг медиан месячных групп временного Leiden seed 1729. Сдвиг '
            f'маркетплейсов за два года составляет {_n(max(r[1] for r in summary["calibration"]["shift"]) - min(r[1] for r in summary["calibration"]["shift"]), 2)} '
            'по логарифму доли и одинаков для всех групп.', '',
            f'{round(100 * p["transition_share"])} % узлов в случайный месяц с {p["first_transition"] + 1}-го по {p["last_transition"] + 1}-й '
            'переходят в случайную другую группу и остаются в ней; постоянное отклонение узла сохраняется. Признаки стандартизованы '
            f'медианой и межквартильным размахом первых {p["calibration_months"]} месяцев, как в основной модели.', '',
            f'Слой «дорог» это случайный геометрический граф: {round(100 * p["spatial_share"])} % узлов лежат возле одного из '
            f'{p["centres_per_group"]} центров своей исходной группы, остальные равномерно в квадрате; радиус подобран под среднюю '
            f'степень около {p["road_degree"]}, вес ребра exp(−(d/r)²). После перехода узел остаётся на месте, поэтому дороги '
            'тянут его к прежней группе.', '',
            '## Методы и метрики', '',
            f'Временной Leiden вызывается тем же кодом (`fit_temporal`), что и в основной модели: kNN-граф профиля с k = '
            f'{summary["leiden"]["knn_k"]}, смесь с дорогами с весом {_n(summary["leiden"]["attribute_graph_weight"], 1)}, '
            f'γ = {_n(summary["leiden"]["resolution"], 1)}, ω = {_n(summary["leiden"]["omega_relative"], 2)} от медианной силы '
            f'вершины первых 12 месяцев, {summary["leiden"]["iterations"]} итерации, seed {summary["leiden"]["seed"]}. Число групп '
            f'он выбирает сам. KMeans получает истинное K = {summary["kmeans"]["k"]} (это фора базовым линиям), n_init = '
            f'{summary["kmeans"]["n_init"]}. KMeans каждый месяц обучается заново, метки соседних месяцев сопоставлены венгерским '
            f'алгоритмом по максимальному пересечению. Замороженный KMeans обучен на всех парах «узел × месяц» первых '
            f'{summary["kmeans"]["frozen_fit_months"]} месяцев и применяется к каждому месяцу.', '',
            'ARI считается с истинными группами в каждом месяце и усредняется. Переход найден, если метод меняет группу узла в '
            f'пределах ±{p["window"]} месяц от настоящего перехода. Ложное переключение: любая смена группы узла, кроме одной, '
            'засчитанной как найденный переход; сюда входят переключения неперешедших узлов и возвраты туда-обратно. Возвраты '
            '(узел уходит из группы и при следующей смене возвращается) показаны отдельно и уже входят в ложные переключения.', '',
            '## Повтор', '', '```sh', 'python scripts/synthetic_transitions.py', '```', '',
            f'Полный прогон занял {round(summary["seconds"])} с. Метрики по каждому повтору лежат в `summary.json` (`per_seed`).', '']
    return '\n'.join(out)
