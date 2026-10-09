"""Temporal Leiden of the main model (omega 0.01, gamma 0.7) refitted with 20 seeds: agreement between seeds, how many of the
seven stable groups of the main run each seed finds, and how many municipalities keep the same group in every seed."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib.metadata
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score
from .contest_graphs import mix_layers
from .followup import load_inputs
from .graph import knn_graph
from .io import sha256, write_json
from .method_comparison import temporal_tracks
from .models import fit_temporal
from .published_inputs import load_published

_STATE = {}


def build(cfg, root):
    slices, _, _, ids, _, _, _, manifest = load_inputs(root, cfg)
    road = load_published(root, cfg, ids, manifest)[0]
    graphs = [mix_layers(knn_graph(s[2], cfg['graph']['k'])[0], road, cfg['followup'].get('attribute_graph_weight', .5)) for s in slices]
    return slices, ids, [s[0] for s in slices], graphs


def _init(cfg, root):
    _STATE['cfg'] = cfg
    _STATE['slices'], _STATE['ids'], _STATE['periods'], _STATE['graphs'] = build(cfg, Path(root))


def _fit(seed):
    c = deepcopy(_STATE['cfg'])
    c['seed'] = seed
    start = time.perf_counter()
    memberships, info = fit_temporal(_STATE['slices'], _STATE['graphs'], c, execute=True)
    return seed, [z.tolist() for z in memberships], {**info, 'seconds': round(time.perf_counter() - start, 1)}


def save_labels(path, ids, periods, labels, fit):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({'ids': ids, 'periods': periods, 'labels': labels, 'fit': fit}, ensure_ascii=False, separators=(',', ':'))
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
        stream.write(data.encode('utf-8'))


def read_labels(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def fit_seeds(cfg, root, seeds, out, workers, ids, periods):
    pending = [s for s in seeds if not (out / f'seed{s}.json.gz').exists()]
    if pending:
        with ProcessPoolExecutor(min(workers, len(pending)), initializer=_init, initargs=(cfg, str(root))) as pool:
            for seed, labels, fit in pool.map(_fit, pending):
                save_labels(out / f'seed{seed}.json.gz', ids, periods, labels, fit)
                print(json.dumps({'seed': seed, 'seconds': fit['seconds']}), flush=True)
    return {s: out / f'seed{s}.json.gz' for s in seeds}


def pairwise_ari(series):
    """Monthly ARI over every pair of seeds and month, and the mean over months for every pair."""
    monthly, per_pair = [], []
    for a, b in combinations(series, 2):
        values = [adjusted_rand_score(x, y) for x, y in zip(series[a], series[b])]
        monthly += values
        per_pair.append(float(np.mean(values)))
    stats = lambda v: {'mean': round(float(np.mean(v)), 4), 'median': round(float(np.median(v)), 4), 'min': round(float(np.min(v)), 4),
                       'max': round(float(np.max(v)), 4)}
    return {'pairs': len(per_pair), 'monthly': stats(monthly), 'per_pair_mean_over_months': stats(per_pair)}


def cell_labels(cells, n, months, mapping):
    out = np.full((months, n), -1)
    for track, own in cells.items():
        for i, t in own:
            out[t, i] = mapping.get(track, -1)
    return out


def best_match(own, cells):
    scores = {k: len(own & other) / len(own | other) for k, other in cells.items()}
    return max(scores.items(), key=lambda kv: (kv[1], -kv[0])) if scores else (None, 0.0)


def modal(labels):
    out = []
    for column in labels.T:
        counts = Counter(int(v) for v in column if v >= 0)
        out.append(max(counts, key=lambda c: (counts[c], -c)) if counts else -1)
    return np.array(out)


def analyse(root, paths, main_path, t, names_by_cells):
    main = read_labels(main_path)
    ids, periods = main['ids'], main['periods']
    n, months = len(ids), len(periods)
    rel = lambda p: str(Path(p).relative_to(root))
    _, base = temporal_tracks(root, rel(main_path), ids, periods, t['jaccard_threshold'], t['size_change'], t['min_months'], t['rejoin_jaccard'])
    order = sorted(base, key=lambda k: -len(base[k]))
    base_index = {k: j for j, k in enumerate(order)}
    per_seed, mapped, found = {}, {}, {j: [] for j in range(len(order))}
    series = {}
    for seed, path in paths.items():
        stored = read_labels(path)
        if stored['ids'] != ids or stored['periods'] != periods:
            raise ValueError('Seed labels do not match the main cohort')
        series[seed] = [np.asarray(z) for z in stored['labels']]
        info, cells = temporal_tracks(root, rel(path), ids, periods, t['jaccard_threshold'], t['size_change'], t['min_months'], t['rejoin_jaccard'])
        best = {}
        for k in order:
            _, score = best_match(base[k], cells)
            best[base_index[k]] = round(score, 4)
            if score >= t['match_jaccard']:
                found[base_index[k]].append(seed)
        to_base = {}
        for track, own in cells.items():
            j, score = best_match(own, {base_index[k]: base[k] for k in order})
            if score >= t['match_jaccard']:
                to_base[track] = j
        mapped[seed] = cell_labels(cells, n, months, to_base)
        per_seed[str(seed)] = {**info, 'best_jaccard_to_main_groups': best, 'seconds': stored['fit'].get('seconds')}
    stack = np.stack(list(mapped.values()))
    same_cells = (stack >= 0).all(axis=0) & (stack == stack[0]).all(axis=0)
    modes = np.stack([modal(m) for m in mapped.values()])
    same_modal = (modes >= 0).all(axis=0) & (modes == modes[0]).all(axis=0)
    main_modal = modal(cell_labels(base, n, months, base_index))
    groups = []
    for j, k in enumerate(order):
        groups.append({'name': names_by_cells.get(len(base[k]), f'группа {j + 1}'), 'size_cells': len(base[k]),
                       'municipalities_modal': int((main_modal == j).sum()), 'seeds_found': len(found[j]),
                       'median_best_jaccard': round(float(np.median([per_seed[str(s)]['best_jaccard_to_main_groups'][j] for s in paths])), 4),
                       'min_best_jaccard': round(float(np.min([per_seed[str(s)]['best_jaccard_to_main_groups'][j] for s in paths])), 4),
                       'same_modal_share': round(float(same_modal[main_modal == j].mean()), 4) if (main_modal == j).any() else None})
    return {'ids_n': n, 'months': months, 'pairwise_ari': pairwise_ari(series), 'groups': groups,
            'groups_found_in_all_seeds': sum(g['seeds_found'] == len(paths) for g in groups),
            'groups_found_in_at_least_90pct': sum(g['seeds_found'] >= 0.9 * len(paths) for g in groups),
            'same_group_all_seeds': {'municipalities_modal': round(float(same_modal.mean()), 4),
                                     'municipalities_modal_count': int(same_modal.sum()),
                                     'cells': round(float(same_cells.mean()), 4)},
            'stable_groups_per_seed': dict(Counter(v['stable'] for v in per_seed.values())),
            'per_seed': per_seed}, series


def reproduction(series, stored, paths):
    out = {}
    for seed, path in stored.items():
        published = read_labels(path)
        old = [np.asarray(z) for z in published['labels']]
        values = [adjusted_rand_score(a, b) for a, b in zip(series[seed], old)]
        out[str(seed)] = {'identical': all(np.array_equal(a, b) for a, b in zip(series[seed], old)),
                          'omega_published': published['fit']['omega'], 'omega_refit': read_labels(paths[seed])['fit']['omega'],
                          'monthly_ari_mean': round(float(np.mean(values)), 4), 'monthly_ari_min': round(float(np.min(values)), 4)}
    return out


def run(cfg, root, mc, seeds, output, workers, names_by_cells):
    root = Path(root)
    out = root / output
    t = mc['temporal']
    main_path = root / t['labels'][t['base_seed']]
    start = time.perf_counter()
    main = read_labels(main_path)
    paths = fit_seeds(cfg, root, seeds, out / 'labels', workers, main['ids'], main['periods'])
    summary, series = analyse(root, paths, main_path, t, names_by_cells)
    stored = {int(s): root / p for s, p in t['labels'].items()}
    summary = {'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'seeds': list(seeds),
               'omega_relative': cfg['clustering']['temporal_coupling_relative'], 'resolution': cfg['clustering']['resolution'],
               'iterations': cfg['clustering']['leiden_iterations'], 'main_run': str(main_path.relative_to(root)),
               'tracking': {k: t[k] for k in ('jaccard_threshold', 'size_change', 'min_months', 'rejoin_jaccard', 'match_jaccard')},
               'reproduction_of_published_seeds': reproduction(series, stored, paths),
               'three_seeds': {'published': pairwise_ari({s: [np.asarray(z) for z in read_labels(p)['labels']] for s, p in stored.items()}),
                               'refit': pairwise_ari({s: series[s] for s in stored})}, **summary,
               'platform': {'python': platform.python_version(), 'machine': platform.machine(), 'system': platform.system(),
                            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'igraph', 'leidenalg')}},
               'seconds_analysis_and_new_fits': round(time.perf_counter() - start, 1),
               'labels_sha256': {str(s): sha256(p) for s, p in paths.items()}}
    write_json(out / 'summary.json', summary)
    (out / 'README.md').write_text(readme(summary), encoding='utf-8')
    return out


def _n(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def readme(s):
    a, same, k = s['pairwise_ari'], s['same_group_all_seeds'], len(s['seeds'])
    rep = s['reproduction_of_published_seeds']
    out = ['# Временной Leiden на 20 seed', '',
           f'Я переобучил временной Leiden основной модели (ω = {_n(s["omega_relative"])}, γ = {_n(s["resolution"], 1)}, '
           f'{s["iterations"]} итерации, те же 24 помесячных графа: kNN профиля и дороги) с {k} seed: '
           f'{", ".join(map(str, s["seeds"]))}. Группы прослежены тем же кодом, что в `reports/method-comparison` '
           f'(`temporal_tracks`: Жаккар событий {_n(s["tracking"]["jaccard_threshold"], 1)}, устойчивая группа живёт не меньше '
           f'{s["tracking"]["min_months"]} месяцев), а группа seed совпадает с группой основного запуска, если Жаккар по парам '
           f'«МО × месяц» не ниже {_n(s["tracking"]["match_jaccard"], 1)}.', '',
           '## Согласие seed между собой', '',
           f'Попарный ARI по {a["pairs"]} парам seed и 24 месяцам: среднее {_n(a["monthly"]["mean"])}, медиана '
           f'{_n(a["monthly"]["median"])}, минимум {_n(a["monthly"]["min"])}. Если сначала усреднить ARI по месяцам внутри '
           f'пары: среднее {_n(a["per_pair_mean_over_months"]["mean"])}, медиана {_n(a["per_pair_mean_over_months"]["median"])}, '
           f'минимум {_n(a["per_pair_mean_over_months"]["min"])}. Число устойчивых групп по seed: '
           + ', '.join(f'{g} групп в {c} seed' for g, c in sorted(s['stable_groups_per_seed'].items(), key=lambda x: int(x[0]))) + '.', '',
           '## Семь групп основного запуска', '',
           f'Основной запуск это опубликованные метки seed 1729 (`{s["main_run"]}`).', '',
           '| Группа | Пар «МО × месяц» | МО (самая частая группа) | В скольких seed найдена | Медиана лучшего Жаккара | Минимум | МО с той же группой во всех seed |',
           '|---|---|---|---|---|---|---|']
    for g in s['groups']:
        share = 'нет МО' if g['same_modal_share'] is None else f'{_n(100 * g["same_modal_share"], 1)} %'
        out.append(f'| {g["name"]} | {g["size_cells"]} | {g["municipalities_modal"]} | {g["seeds_found"]} из {k} | '
                   f'{_n(g["median_best_jaccard"])} | {_n(g["min_best_jaccard"])} | {share} |')
    out += ['', f'Во всех {k} seed находятся {s["groups_found_in_all_seeds"]} из {len(s["groups"])} групп основного запуска; '
                f'не меньше чем в {math.ceil(0.9 * k)} seed из {k} тоже {s["groups_found_in_at_least_90pct"]}.', '',
            '## Одинаковая группа во всех seed', '',
            'Каждую устойчивую группу seed я отображаю на группу основного запуска с наибольшим Жаккаром (если он не ниже '
            f'{_n(s["tracking"]["match_jaccard"], 1)}), иначе считаю её несопоставленной. Самая частая за 24 месяца '
            f'сопоставленная группа МО одинакова во всех {k} seed у {same["municipalities_modal_count"]} МО из {s["ids_n"]} '
            f'({_n(100 * same["municipalities_modal"], 1)} %). На уровне пар «МО × месяц» одна и та же сопоставленная группа во '
            f'всех seed у {_n(100 * same["cells"], 1)} % пар.', '',
            '## Повтор опубликованных seed', '',
            'Опубликованные метки seed 1729, 2718 и 3141 посчитаны 24 сентября 2026 года; платформа в их provenance не записана. '
            f'Мой повтор на {s["platform"]["system"]} {s["platform"]["machine"]}, Python {s["platform"]["python"]} с теми же версиями '
            'пакетов (numpy, scipy, igraph, leidenalg) побайтово с ними не совпадает: '
            + '; '.join(f'seed {seed}: ARI с опубликованными метками по месяцам в среднем {_n(r["monthly_ari_mean"])}, минимум '
                        f'{_n(r["monthly_ari_min"])}' for seed, r in rep.items()) + '. '
            f'Уже абсолютный ω расходится в последнем знаке ({next(iter(rep.values()))["omega_published"]!r} в опубликованном '
            f'запуске и {next(iter(rep.values()))["omega_refit"]!r} у меня), то есть медиана силы вершин считается с другим '
            'округлением, а Leiden чувствителен к таким различиям. Поэтому для 20 seed я беру свои повторы всех seed, '
            'посчитанные в одном окружении, а группы основного запуска беру из опубликованных меток seed 1729. На тех же трёх '
            f'seed мой повтор даёт попарный ARI по месяцам в среднем {_n(s["three_seeds"]["refit"]["monthly"]["mean"])} с минимумом '
            f'{_n(s["three_seeds"]["refit"]["monthly"]["min"])}, опубликованные метки {_n(s["three_seeds"]["published"]["monthly"]["mean"])} '
            f'и {_n(s["three_seeds"]["published"]["monthly"]["min"])}. На 20 seed среднее ниже, а минимум заметно ниже, потому что '
            'пар больше и среди них есть менее согласные.', '',
            '## Повтор', '', '```sh', 'python scripts/temporal_seeds.py --workers 6', '```', '',
            'Метки каждого seed лежат в `labels/seed<N>.json.gz` (SHA-256 в `summary.json`); уже посчитанные seed скрипт не '
            'переобучает. Один seed считается около минуты на одном ядре.', '']
    return '\n'.join(out)
