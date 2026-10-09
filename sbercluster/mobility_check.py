"""SberIndex consumer mobility index as an external reference for the temporal groups and KMeans4.

The open dataset is read through the public SberIndex API, normalised to a sorted CSV whose SHA-256 is recorded and checked,
matched to the 2016 municipalities by name within the regions it covers, and compared across groups with eta squared and the
Kruskal-Wallis test.
"""
from __future__ import annotations
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
import uuid
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from .io import sha256, write_json
from .temporal_groups import event_owners, rejoin, track_key, track_labels

API = 'https://sberindex.ru/api/sowa'
DATASET = 'indeks-mobilnosti'
TYPES = re.compile(r'внутригородская территория города федерального значения|муниципальный район|муниципальный округ|'
                   r'городской округ|город|поселок')
MIN_GROUP = 5
CITY = 'Санкт-Петербург'


def decode(value, kind=None):
    """Values of the API envelope are tagged strings (__string__<base64>, __number__<text>, __null__) inside typed objects."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict) and value.get('type') == 'object':
        out = {}
        for item in value['value']:
            if item.get('type') == 'longstring':
                return ''.join(decode(item['value'], 'longstring'))
            out[item['key']] = decode(item['value'], item.get('type'))
        return out
    text = str(value)
    if not text.startswith('__'):
        return value
    tag, _, raw = text[2:].partition('__')
    tag = kind if kind not in (None, 'longstring') else tag
    if tag == 'null':
        return None
    if tag == 'number':
        return float(raw)
    if tag == 'boolean':
        return raw == 'true'
    if tag == 'string':
        return base64.b64decode(raw).decode('utf-8')
    return raw


def envelope(route):
    return json.dumps({'SOWA': {'method': 'GET', 'route': base64.b64encode(route.encode()).decode(),
                                'data': {'type': 'object', 'value': []}}}).encode()


def post(route, cafile=None):
    """POST through urllib; when Python cannot build the certificate chain (the server omits its intermediate), use curl,
    which relies on the system trust store. The content is checked by SHA-256 afterwards either way."""
    headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0', 'RqUID': uuid.uuid4().hex}
    context = ssl.create_default_context()
    if cafile:
        context.load_verify_locations(cafile=str(cafile))
    try:
        request = urllib.request.Request(API, data=envelope(route), headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            raw = response.read()
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLError) or shutil.which('curl') is None:
            raise
        command = ['curl', '-sS', '--fail', '-m', '120', '-X', 'POST', API, '--data-binary', '@-']
        for key, value in headers.items():
            command += ['-H', f'{key}: {value}']
        if cafile:
            command += ['--cacert', str(cafile)]
        raw = subprocess.run(command, input=envelope(route), capture_output=True, check=True).stdout
    return decode(json.loads(raw.decode('utf-8'))['SOWA']['data'])


def fetch(dataset=DATASET, cafile=None, limit=1000):
    rows, offset, fields = [], 0, None
    while True:
        page = post(f'/dataset/v1/{dataset}?limit={limit}&offset={offset}', cafile)
        fields = fields or page['fields']
        rows += page['data']
        offset += limit
        if offset >= int(page['pagination']['total_records']) or not page['data']:
            return fields, rows


def normalise(fields, rows):
    """One row per area and year (the API stamps the year end in UTC, so the local year is taken from period + 3 h)."""
    frame = pd.DataFrame(rows, columns=fields)
    year = (pd.to_datetime(frame['period'], utc=True) + pd.Timedelta(hours=3)).dt.year
    out = pd.DataFrame({'indicator_id': frame['indicator_id'], 'kpi_id': frame['kpi_id'], 'area': frame['ref_area'],
                        'period': frame['period'], 'year': year, 'value': frame['value'].astype(float),
                        'unit': frame['unit_measure'], 'freq': frame['freq'], 'obs_status': frame['obs_status']})
    if out.duplicated(['area', 'year']).any():
        raise ValueError('Mobility index has duplicate area-year rows')
    return out.sort_values(['area', 'year']).reset_index(drop=True)


def stem(name):
    text = TYPES.sub(' ', name.lower().replace('ё', 'е'))
    return ' '.join(text.split())


def match(areas, cohort, regions):
    """Match each area to the cohort by exact name, then by the name without municipal type words, within the covered
    regions. Only unique matches are kept."""
    pool = cohort[cohort['region_name'].isin(regions)]
    exact = pool.groupby('municipal_district_name')['entity_id'].apply(list).to_dict()
    stems = pool.assign(stem=pool['municipal_district_name'].map(stem)).groupby('stem')['entity_id'].apply(list).to_dict()
    out = {}
    for area in areas:
        found = exact.get(area) or stems.get(stem(area), [])
        out[area] = found[0] if len(found) == 1 else None
    used = Counter(v for v in out.values() if v)
    if any(n > 1 for n in used.values()):
        raise ValueError('Two mobility areas matched one municipality')
    return out


def modal_groups(cfg, root):
    """Most frequent stable temporal group of each municipality over 24 months (as in scripts/group_profiles.py)."""
    with gzip.open(root / cfg['temporal_labels'], 'rt', encoding='utf-8') as stream:
        temporal = json.load(stream)
    events = json.loads((root / cfg['events']).read_text('utf-8'))['temporal_leiden']
    ids, periods, labels = temporal['ids'], temporal['periods'], temporal['labels']
    owners, _ = event_owners(events['sizes'], events['steps'])
    tracks = rejoin(track_labels(labels, owners), cfg['rejoin_jaccard'])
    alive = Counter(track for row in tracks for track in set(row))
    names = {k: cfg['names'][track_key(k, owners, periods)] for k, n in alive.items() if n >= cfg['min_months']}
    modal = {}
    for i, key in enumerate(ids):
        counts = Counter(tracks[t][i] for t in range(len(periods)) if tracks[t][i] in names)
        if counts:
            modal[key] = names[max(counts, key=lambda c: (counts[c], -c))]
    return modal


def eta_squared(values, labels):
    values, labels = np.asarray(values, float), np.asarray(labels)
    total = ((values - values.mean()) ** 2).sum()
    between = sum((labels == g).sum() * (values[labels == g].mean() - values.mean()) ** 2 for g in np.unique(labels))
    return float(between / total)


def compare(values, labels, min_group=MIN_GROUP):
    """eta squared (raw and log) and Kruskal-Wallis with epsilon squared over groups with at least min_group members."""
    values, labels = np.asarray(values, float), np.asarray(labels).astype(str)
    sizes = Counter(labels)
    keep = np.array([sizes[g] >= min_group for g in labels])
    v, z = values[keep], labels[keep]
    groups = sorted(set(z))
    out = {'n': int(keep.sum()), 'groups': len(groups), 'dropped_small_groups': {g: n for g, n in sorted(sizes.items()) if n < min_group}}
    if len(groups) < 2:
        return {**out, 'eta2': None, 'eta2_log': None, 'kruskal_H': None, 'kruskal_p': None, 'epsilon2': None}
    h, p = kruskal(*[v[z == g] for g in groups])
    return {**out, 'eta2': round(eta_squared(v, z), 4), 'eta2_log': round(eta_squared(np.log(v), z), 4),
            'kruskal_H': round(float(h), 3), 'kruskal_p': float(f'{p:.3g}'),
            'epsilon2': round(float((h - len(groups) + 1) / (len(v) - len(groups))), 4),
            'medians': {g: round(float(np.median(v[z == g])), 3) for g in groups},
            'sizes': {g: int((z == g).sum()) for g in groups}}


def download(root, path, sources_path, cafile=None):
    fields, rows = fetch(cafile=cafile)
    frame = normalise(fields, rows)
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.6g', lineterminator='\n')
    digest = sha256(path)
    sources = root / sources_path
    if sources.exists():
        expected = json.loads(sources.read_text('utf-8'))['sha256']
        if digest != expected:
            raise ValueError(f'Mobility index SHA-256 {digest} differs from the recorded {expected}; the dataset may have been revised')
    else:
        write_json(sources, {'dataset': DATASET, 'api': API, 'route': f'/dataset/v1/{DATASET}', 'file': str(Path(path).relative_to(root)),
                             'rows': len(frame), 'sha256': digest,
                             'downloaded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    return path, digest


def run(cfg, root, source, sources_path, output, year, regions):
    root = Path(root)
    path = root / source
    expected = json.loads((root / sources_path).read_text('utf-8'))
    if sha256(path) != expected['sha256']:
        raise ValueError('Mobility index file differs from the recorded SHA-256')
    frame = pd.read_csv(path, dtype={'indicator_id': str, 'kpi_id': str})
    indicators = pd.read_csv(root / cfg['panel'], usecols=['entity_id', 'region_name', 'municipal_district_name'],
                             dtype=str).drop_duplicates('entity_id')
    modal = modal_groups(cfg, root)
    kmeans = pd.read_csv(root / cfg['reference'], dtype={'entity_id': str}).set_index('entity_id')['cluster'].astype(str)
    cohort = indicators[indicators['entity_id'].isin(modal)]
    if len(cohort) != len(modal):
        raise ValueError('Panel does not cover the cohort')
    areas = sorted(frame['area'].unique())
    matched = match(areas, cohort, regions)
    excluded = set(pd.read_csv(root / 'reports/excluded_names.csv')['mo'])
    unmatched = [a for a in areas if matched[a] is None]
    names = cohort.set_index('entity_id')
    table = frame.assign(entity_id=frame['area'].map(matched)).dropna(subset=['entity_id'])
    table = table.assign(region_name=table['entity_id'].map(names['region_name']), group=table['entity_id'].map(modal),
                         kmeans4=table['entity_id'].map(kmeans))
    results = {}
    for y in sorted(table['year'].unique()):
        part = table[table['year'] == y]
        rest = part[part['region_name'] != CITY]
        results[str(int(y))] = {**{by: compare(part['value'], part[by]) for by in ('group', 'kmeans4', 'region_name')},
                                'without_city': {by: compare(rest['value'], rest[by]) for by in ('group', 'kmeans4', 'region_name')},
                                'city_municipalities': int((part['region_name'] == CITY).sum())}
    covered = cohort[cohort['region_name'].isin(regions)]
    summary = {'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dataset': DATASET, 'api': API,
               'source': {'file': source, 'sha256': expected['sha256'], 'rows': len(frame), 'downloaded_utc': expected['downloaded_utc']},
               'unit': sorted(frame['unit'].unique()), 'years': sorted(int(y) for y in frame['year'].unique()),
               'areas': len(areas), 'regions': list(regions), 'cohort': len(modal),
               'cohort_in_regions': len(covered), 'matched': int(sum(v is not None for v in matched.values())),
               'unmatched': {'excluded_incomplete_panel': sorted(a for a in unmatched if a in excluded),
                             'not_in_contest_data': sorted(a for a in unmatched if a not in excluded)},
               'cohort_in_regions_without_index': sorted(covered.loc[~covered['entity_id'].isin(table['entity_id']), 'municipal_district_name']),
               'main_year': year, 'min_group': MIN_GROUP, 'results': results,
               'group_counts_in_regions': dict(Counter(modal[e] for e in covered['entity_id']).most_common()),
               'inputs': {rel: sha256(root / rel) for rel in (cfg['temporal_labels'], cfg['events'], cfg['reference'], cfg['panel'])}}
    out = root / output
    write_json(out / 'summary.json', summary)
    table[['entity_id', 'area', 'region_name', 'year', 'value', 'group', 'kmeans4']].sort_values(['entity_id', 'year']).to_csv(
        out / 'matched.csv', index=False, float_format='%.6g', lineterminator='\n')
    (out / 'README.md').write_text(readme(summary), encoding='utf-8')
    return out


def _n(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')


def _p(value):
    if value < 1e-4:
        mantissa, exponent = f'{value:.1e}'.split('e')
        return f'{mantissa.replace(".", ",")}·10^{int(exponent)}'
    return _n(value, 4)


LABELS = {'group': 'Семь групп временного Leiden', 'kmeans4': 'KMeans4 (фиксированные центры)', 'region_name': 'Регион'}


def readme(summary):
    main = summary['results'][str(summary['main_year'])]
    other = [y for y in summary['results'] if y != str(summary['main_year'])]
    g, k = main['group'], main['kmeans4']
    out = ['# Индекс покупательской мобильности СберИндекса как внешний эталон', '',
           '## Что я проверил', '',
           f'Открытый датасет `{summary["dataset"]}` СберИндекса доступен через публичный API сайта ({summary["api"]}, '
           f'маршрут `/dataset/v1/{summary["dataset"]}`). Строк в нём {summary["source"]["rows"]}: {summary["areas"]} '
           f'муниципальных образований, по одному значению за каждый из годов {", ".join(map(str, summary["years"]))} '
           f'(значение последнего года датировано 31 октября), единица измерения «{", ".join(summary["unit"])}». Все МО датасета '
           'относятся к 11 регионам Северо-Западного федерального округа, других регионов в нём нет. Поэтому проверка возможна '
           'только на этой части страны.', '',
           f'Я сохранил нормализованную таблицу (одна строка на МО и год, сортировка по названию) и записал её SHA-256 в '
           f'`sources.json`: `{summary["source"]["sha256"]}`. Повторная загрузка сверяет хеш и останавливается при расхождении.', '',
           '## Сопоставление с 2 016 МО', '',
           f'В датасете нет ОКТМО, только название МО. Я сопоставляю его с названием в данных конкурса внутри '
           f'{len(summary["regions"])} регионов СЗФО: сначала точное совпадение, затем совпадение без слов о типе МО '
           '(«муниципальный район», «муниципальный округ», «городской округ», «город», «поселок», «внутригородская территория '
           'города федерального значения»), потому что часть районов в 2024–2025 годах стала округами. Беру только однозначные '
           f'совпадения. Сопоставлено {summary["matched"]} из {summary["areas"]} МО. Из несопоставленных '
           f'{len(summary["unmatched"]["excluded_incomplete_panel"])} исключены из когорты за неполную 24-месячную панель '
           f'(`reports/excluded_names.csv`), {len(summary["unmatched"]["not_in_contest_data"])} нет в данных конкурса. '
           f'Из {summary["cohort_in_regions"]} МО когорты в этих регионах без значения индекса остались '
           f'{len(summary["cohort_in_regions_without_index"])}: {", ".join(summary["cohort_in_regions_without_index"])}.', '',
           f'Группа МО равна самой частой устойчивой группе временного Leiden за 24 месяца, как в `scripts/group_profiles.py`. '
           f'Группы, в которые попало меньше {summary["min_group"]} сопоставленных МО, я не включаю в η² и тест Краскела–Уоллиса '
           '(список в `summary.json`).', '',
           f'## Результат за {summary["main_year"]} год', '',
           '| Разбиение | МО | Групп | η² | η² (log) | H Краскела–Уоллиса | p | ε² |', '|---|---|---|---|---|---|---|---|']
    for by in ('group', 'kmeans4', 'region_name'):
        r = main[by]
        out.append(f'| {LABELS[by]} | {r["n"]} | {r["groups"]} | {_n(r["eta2"])} | {_n(r["eta2_log"])} | {_n(r["kruskal_H"], 1)} | '
                   f'{_p(r["kruskal_p"])} | {_n(r["epsilon2"])} |')
    out += ['', 'Медиана индекса по группам временного Leiden: ' + '; '.join(
        f'{name} {_n(v, 2)} ({g["sizes"][name]} МО)' for name, v in sorted(g['medians'].items(), key=lambda x: -x[1])) + '.', '']
    lead = 'почти столько же, сколько KMeans4' if abs(g['eta2'] - k['eta2']) < 0.02 else 'больше' if g['eta2'] > k['eta2'] else 'меньше'
    w = main['without_city']
    out += [f'Семь групп объясняют {_n(100 * g["eta2"], 1)} % разброса индекса мобильности, KMeans4 {_n(100 * k["eta2"], 1)} %, '
            f'то есть группы объясняют {lead}. Регион для сравнения объясняет {_n(100 * main["region_name"]["eta2"], 1)} %. '
            f'Заметную часть эффекта даёт Санкт-Петербург: {main["city_municipalities"]} его внутригородских МО попадают в «Города и '
            f'Москва» и имеют самые высокие значения индекса. Без Санкт-Петербурга η² групп падает до {_n(w["group"]["eta2"])}, '
            f'KMeans4 до {_n(w["kmeans4"]["eta2"])}:', '',
            '| Разбиение | МО | Групп | η² | η² (log) | p Краскела–Уоллиса | ε² |', '|---|---|---|---|---|---|---|']
    for by in ('group', 'kmeans4', 'region_name'):
        r = w[by]
        out.append(f'| {LABELS[by]} | {r["n"]} | {r["groups"]} | {_n(r["eta2"])} | {_n(r["eta2_log"])} | {_p(r["kruskal_p"])} | '
                   f'{_n(r["epsilon2"])} |')
    out += ['', 'Медиана индекса без Санкт-Петербурга: ' + '; '.join(
        f'{name} {_n(v, 2)} ({w["group"]["sizes"][name]} МО)' for name, v in sorted(w['group']['medians'].items(), key=lambda x: -x[1])) + '.', '']
    for y in other:
        r = summary['results'][y]
        out.append(f'За {y} год: η² групп {_n(r["group"]["eta2"])}, KMeans4 {_n(r["kmeans4"]["eta2"])}, региона '
                   f'{_n(r["region_name"]["eta2"])}; без Санкт-Петербурга {_n(r["without_city"]["group"]["eta2"])}, '
                   f'{_n(r["without_city"]["kmeans4"]["eta2"])} и {_n(r["without_city"]["region_name"]["eta2"])}.')
    out += ['', '## Ограничения', '',
            f'Проверка покрывает {summary["matched"]} МО одного федерального округа из {summary["cohort"]:,} МО когорты'.replace(',', ' ') + ', и состав групп '
            'в СЗФО отличается от страны в целом (`group_counts_in_regions`): из семи групп в анализ входят только четыре. Индекс '
            'измерен в километрах, то есть описывает расстояния, а не структуру трат, по которой строились группы, поэтому '
            'проверка не круговая. Источник у него тот же (данные СберИндекса), так что независимость неполная. Методика '
            'индекса на уровне МО в датасете не описана.', '',
            '## Повтор', '', '```sh', 'python scripts/mobility_check.py --download', 'python scripts/mobility_check.py', '```', '',
            'Первая команда скачивает датасет в `artifacts/sources/mobility/` и сверяет SHA-256 с `sources.json`. Если Python не '
            'доверяет сертификату сайта (сервер не отдаёт промежуточный сертификат), загрузчик повторяет запрос через `curl` с '
            'системным хранилищем сертификатов; можно также передать свой PEM через `--cafile`. Вторая команда считает таблицы. '
            'Сопоставленные значения лежат в `matched.csv`.', '']
    return '\n'.join(out)
