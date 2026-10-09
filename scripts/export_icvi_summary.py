"""Expose the chosen MQ convention without altering historical metric records."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'NewmanQ')


def validate_records(records: list[dict]) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty list of metric records.')
    names = set()
    bounds = {'SW': (-1, 1), 'CH': (0, math.inf), 'S_Dbw': (0, math.inf),
              'AVI': (0, 1), 'AVU': (0, 1), 'NewmanQ': (-1, 1)}
    for record in records:
        name = record['candidate']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Candidate names must be nonempty and unique.')
        names.add(name)
        n, k, minimum = (record[key] for key in ('n', 'k', 'min_cluster_size'))
        if any(type(value) is not int or value < 1 for value in (n, k, minimum)):
            raise ValueError(f'{name}: sizes must be positive integers.')
        if k > n or k * minimum > n:
            raise ValueError(f'{name}: cluster sizes are inconsistent with N and K.')
        for key, (lower, upper) in bounds.items():
            value = record[key]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name}: {key} must be finite or null.')
            if not lower - 1e-12 <= value <= upper + 1e-12:
                raise ValueError(f'{name}: {key} is outside its mathematical range.')


def export_summary(source: Path, output: Path, fragment: Path) -> None:
    source_bytes = source.read_bytes()
    records = json.loads(source_bytes)
    validate_records(records)
    rows = []
    for record in records:
        rows.append({
            'candidate': record['candidate'], 'n': record['n'], 'k': record['k'],
            'min_cluster_size': record['min_cluster_size'],
            **{key: record[key] for key in FIELDS},
            'MQ_selected': record['NewmanQ'],
            'MQ_convention': 'Newman Q; gamma=1; binary undirected graph; no loops',
            'official_formula_confirmed': False,
            'legacy_MQ': record['MQ'], 'legacy_MQ_status': record['MQ_status'],
            'S_Dbw_status': record['S_Dbw_status'],
        })
    document = {
        'source': source.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'method': 'Projection of saved metrics; no fitting, metric recomputation or replacement.',
        'contest_definition': 'Modularity Quality, regulation section 6.1; exact formula unspecified.',
        'rows': rows,
    }
    selected = [r for r in rows if r['k'] == 4]
    if not selected or len({r['n'] for r in selected}) != 1:
        raise ValueError('Expected a nonempty comparison on one population at K=4.')
    if any(r[key] is None for r in selected for key in ('SW', 'NewmanQ')):
        raise ValueError('The K=4 comparison requires defined SW and NewmanQ.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    labels = {'kmeans_k4': 'KMeans', 'gmm_full_k4': 'Gaussian mixture',
              'spectral_global_k4': 'Spectral · global', 'spectral_local_k4': 'Spectral · local',
              'ward_k4': 'Ward'}
    def fmt(value):
        return 'не определено' if value is None else f'{value:.4f}'.replace('.', ',')
    header = '| Метод | Минимальная группа, МО | SW ↑ | CH ↑ | S_Dbw ↓ | AVI ↑ | AVU ↓ | MQ · Newman Q ↑ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    markdown_rows = ''.join('| ' + ' | '.join([labels.get(r['candidate'], r['candidate']), str(r['min_cluster_size'])] + [fmt(r[k]) for k in FIELDS]) + ' |\n' for r in selected)
    note = ('В §6.1 Положения MQ назван Modularity Quality, но формула не задана. '
            'Здесь выбрана модульность Newman при gamma=1 на бинарном неориентированном графе без петель. '
            'Соответствие точной формуле организатора не подтверждено. Числа взяты из сохранённого поля NewmanQ; '
            'историческое MQ=null не заменено.')
    reading = ('SW и CH описывают компактность расходных профилей, S_Dbw добавляет разброс и плотность между группами. '
               'AVI, AVU и MQ описывают связи на общем опорном графе. Большая модульность сама по себе не доказывает '
               'экономический смысл групп. Методы сравниваются при одинаковых N и K; единого произвольного среднего из индексов нет.')
    best_sw = max(selected, key=lambda r: r['SW'])
    best_q = max(selected, key=lambda r: r['NewmanQ'])
    finding = (f"При K=4 лучший SW в этой таблице у {labels.get(best_sw['candidate'], best_sw['candidate'])} "
               f"({fmt(best_sw['SW'])}, минимальная группа {best_sw['min_cluster_size']} МО), лучший MQ у {labels.get(best_q['candidate'], best_q['candidate'])} "
               f"({fmt(best_q['NewmanQ'])}). Эти показатели проверяют разные свойства разбиения; "
               'окончательный выбор также учитывает размеры групп, временную устойчивость и внешние экономические проверки.')
    readme = ('# Шесть показателей качества\n\n'
              'Сводка опубликованных разбиений на общей выборке и опорном графе. Числа не пересчитывались.\n\n'
              + note + '\n\n' + header + markdown_rows + '\n' + finding + '\n\n' + reading + '\n\n'
              'Для других K и статусов неопределённости см. [summary.json](summary.json). '
              'Прочерк в S_Dbw означает нулевой знаменатель плотности, а не нулевую ошибку.\n\n'
              'Воспроизведение из корня репозитория:\n\n```sh\npython -m scripts.export_icvi_summary\n```\n\n'
              'Формулы и источники: [METRICS.md](../../docs/METRICS.md). '
              f"Вход: `{document['source']}`; SHA-256 `{document['source_sha256']}`.\n")
    (output / 'README.md').write_text(readme, encoding='utf-8', newline='\n')
    table_rows = ''.join('<tr><th scope="row">' + html.escape(labels.get(r['candidate'], r['candidate'])) + '</th>'
                         + '<td class="num">' + str(r['min_cluster_size']) + '</td>'
                         + ''.join('<td class="num">' + fmt(r[k]) + '</td>' for k in FIELDS) + '</tr>' for r in selected)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    n_display = f'{selected[0]["n"]:,}'.replace(',', ' ')
    fragment.write_text(
        '<section class="panel result-block" aria-labelledby="icvi-title">\n'
        '<div class="section-head"><div><p class="eyebrow">Проверка качества</p>'
        '<h2 id="icvi-title">Что говорят шесть метрик</h2></div></div>\n'
        f'<p>{html.escape(finding)}</p>\n'
        '<div class="table-scroll" tabindex="0" role="region" aria-label="Сравнение шести метрик">'
        f'<table><caption>Опубликованные разбиения: {n_display} МО, четыре группы, общий опорный граф расходов</caption>'
        '<thead><tr><th scope="col">Метод</th><th scope="col">Малая группа, МО</th><th scope="col">SW ↑</th><th scope="col">CH ↑</th>'
        '<th scope="col">S_Dbw ↓</th><th scope="col">AVI ↑</th><th scope="col">AVU ↓</th>'
        '<th scope="col">MQ · Newman Q ↑</th></tr></thead><tbody>' + table_rows + '</tbody></table></div>\n'
        f'<p class="control-help">{html.escape(reading)}</p>\n'
        '<details><summary>Определение MQ и происхождение чисел</summary>'
        f'<p>{html.escape(note)}</p><p>Источник: reports/contest-v3/published_partition_metrics.json. '
        'Формулы и проверяемые примеры приведены в docs/METRICS.md.</p></details>\n</section>\n', encoding='utf-8', newline='\n')
    assert source.read_bytes() == source_bytes
    print(json.dumps({'rows': len(rows), 'k4_rows': len(selected), 'source_sha256': document['source_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/contest-v3/published_partition_metrics.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/icvi-2026-10-07')
    parser.add_argument('--fragment', type=Path, default=ROOT / 'web/icvi_results.html')
    args = parser.parse_args()
    export_summary(args.source.resolve(), args.output.resolve(), args.fragment.resolve())
