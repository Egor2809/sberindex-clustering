"""Describe temporal Leiden groups through event-based identity, without refitting."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import median

from sklearn.metrics import adjusted_rand_score

LIFECYCLE = {"continue", "grow", "shrink", "split", "merge", "birth", "death"}
CATEGORIES = ("Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт")
TOTAL = "Все категории"


def event_owners(sizes: list[dict], steps: list[dict]) -> tuple[list[dict[int, str]], list[dict]]:
    if len(steps) != len(sizes) - 1:
        raise ValueError("Lifecycle steps do not join consecutive months")
    owners: list[dict[int, str]] = []
    marks = []

    def start(label: str, t: int) -> int:
        owners.append({t: label})
        return len(owners) - 1

    current = {label: start(label, 0) for label in sorted(sizes[0], key=int)}
    for t, step in enumerate(steps, 1):
        following: dict[str, int] = {}

        def assign(label: str, track: int) -> None:
            if label not in sizes[t]:
                raise ValueError(f"Unknown lifecycle successor at step {t}: {label}")
            if label in following:
                return
            following[label] = track
            owners[track][t] = label

        for event in step["events"]:
            kind, before, after = event["event"], [str(x) for x in event["from"]], [str(x) for x in event["to"]]
            if kind not in LIFECYCLE or any(label not in current for label in before):
                raise ValueError(f"Unknown lifecycle event at step {t}: {kind}")
            if kind in ("continue", "grow", "shrink"):
                assign(after[0], current[before[0]])
            elif kind == "merge":
                keep = max(before, key=lambda label: (sizes[t - 1][label], -int(label)))
                assign(after[0], current[keep])
            elif kind == "split":
                keep = max(after, key=lambda label: (sizes[t][label], -int(label)))
                assign(keep, current[before[0]])
                for label in after:
                    if label != keep:
                        assign(label, start(label, t))
            elif kind == "birth":
                assign(after[0], start(after[0], t))
            if kind in ("split", "merge", "birth", "death"):
                marks.append({"t": t, "event": kind, "size_before": event["size_before"], "size_after": event["size_after"]})
        if set(following) != set(sizes[t]):
            raise ValueError(f"Lifecycle events do not cover every group at step {t}")
        current = following
    return owners, marks


def track_labels(labels: list[list[int]], owners: list[dict[int, str]]) -> list[list[int]]:
    lookup = [{} for _ in labels]
    for track, owned in enumerate(owners):
        for t, label in owned.items():
            lookup[t][int(label)] = track
    try:
        return [[lookup[t][label] for label in row] for t, row in enumerate(labels)]
    except KeyError as error:
        raise ValueError(f"Leiden label {error} has no lifecycle track") from None


def rejoin(tracks: list[list[int]], threshold: float) -> list[list[int]]:
    members: dict[int, dict[int, set[int]]] = {}
    for t, row in enumerate(tracks):
        for i, track in enumerate(row):
            members.setdefault(track, {}).setdefault(t, set()).add(i)
    parent, tail = {}, {}
    for track in sorted(members, key=lambda k: (min(members[k]), k)):
        first, cells = min(members[track]), members[track][min(members[track])]
        scores = [(len(tail[root][1] & cells) / len(tail[root][1] | cells), -root) for root in tail if tail[root][0] < first]
        score, root = max(scores, default=(0.0, 0))
        root = -root if score >= threshold else track
        parent[track] = root
        last = max(members[track])
        tail[root] = (last, members[track][last])
    return [[parent[track] for track in row] for row in tracks]


def flows(tracks: list[list[int]]) -> list[dict[tuple[int, int], int]]:
    return [dict(sorted(Counter(zip(a, b)).items())) for a, b in zip(tracks, tracks[1:])]


def track_key(track: int, owners: list[dict[int, str]], periods: list[str]) -> str:
    start = min(owners[track])
    return f"{periods[start][:7]}:{owners[track][start]}"


def _share(counter: Counter, total: int) -> dict:
    return {str(key): round(value / total, 4) for key, value in sorted(counter.items())}


def _read_panel(path: Path, ids: set[str], periods: list[str]) -> tuple[dict, dict]:
    ratios, regions = {}, {}
    wanted = set(periods)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["entity_id"]
            if key not in ids or row["period"] not in wanted:
                continue
            total = float(row[TOTAL])
            ratios[key, row["period"]] = [float(row[name]) / total for name in CATEGORIES]
            regions[key] = row["region_name"]
    if len(ratios) != len(ids) * len(periods):
        raise ValueError("Panel does not cover every territory and month of the temporal labels")
    return ratios, regions


def _read_indicators(path: Path) -> dict:
    values = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            values[row["entity_id"]] = {key: float(row[key]) if row[key] else None for key in ("population", "wage_2023")}
    return values


def _median(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return round(median(values), 4) if values else None


def describe(config: dict, root: Path) -> dict:
    with gzip.open(root / config["temporal_labels"], "rt", encoding="utf-8") as stream:
        temporal = json.load(stream)
    events = json.loads((root / config["events"]).read_text("utf-8"))["temporal_leiden"]
    ids, periods, labels = temporal["ids"], temporal["periods"], temporal["labels"]
    if events["sizes"] != [{str(k): v for k, v in sorted(Counter(row).items())} for row in labels]:
        raise ValueError("Lifecycle sizes differ from the stored Leiden labels")
    owners, _ = event_owners(events["sizes"], events["steps"])
    tracks = rejoin(track_labels(labels, owners), config["rejoin_jaccard"])
    with (root / config["reference"]).open(encoding="utf-8", newline="") as stream:
        reference = {row["entity_id"]: int(row["cluster"]) for row in csv.DictReader(stream)}
    if set(reference) != set(ids):
        raise ValueError("Reference types do not match the temporal cohort")
    ratios, regions = _read_panel(root / config["panel"], set(ids), periods)
    indicators = _read_indicators(root / config["indicators"])
    months = len(periods)
    paths = [[tracks[t][i] for t in range(months)] for i in range(len(ids))]
    alive = Counter(track for row in tracks for track in set(row))
    stable = sorted(k for k, n in alive.items() if n >= config["min_months"])
    names = config["names"]
    groups = []
    for k in stable:
        key = track_key(k, owners, periods)
        cells = [(i, t) for i, path in enumerate(paths) for t, track in enumerate(path) if track == k]
        members = {i for i, _ in cells}
        months_in = Counter(i for i, _ in cells)
        types = Counter(reference[ids[i]] for i, _ in cells)
        region, region_n = Counter(regions[ids[i]] for i, _ in cells).most_common(1)[0]
        groups.append({
            "key": key, "name": names.get(key), "track": k, "start": min(owners[k]),
            "end": max(t for t, row in enumerate(tracks) if k in row), "months": alive[k],
            "sizes": [sum(track == k for track in tracks[t]) for t in range(months)],
            "territories": len(members), "core": sum(n == alive[k] for n in months_in.values()),
            "types": _share(types, len(cells)), "top_type": max(types, key=lambda c: (types[c], -c)),
            "shares": {name: _median([ratios[ids[i], periods[t]][j] for i, t in cells]) for j, name in enumerate(CATEGORIES)},
            "population": _median([indicators.get(ids[i], {}).get("population") for i in members]),
            "wage_2023": _median([indicators.get(ids[i], {}).get("wage_2023") for i in members]),
            "top_region": region, "top_region_share": round(region_n / len(cells), 4)})
    missing = [group["key"] for group in groups if not group["name"]]
    if missing or set(names) != {group["key"] for group in groups}:
        raise ValueError(f"Configured names must cover exactly the stable groups: {missing or sorted(names)}")
    district = {group["track"] for group in groups if group["top_type"] in (0, 1)}
    moves = [(a, b) for path in paths for a, b in zip(path, path[1:]) if a != b]
    ari = [adjusted_rand_score([reference[key] for key in ids], row) for row in labels]
    switches = [sum(a != b for a, b in zip(path, path[1:])) for path in paths]
    in_stable = [sum(track in set(stable) for track in path) for path in paths]
    return {
        "source": config["temporal_labels"], "events": config["events"], "omega": config["omega"], "seed": config["seed"], "n": len(ids), "periods": periods,
        "min_months": config["min_months"], "rejoin_jaccard": config["rejoin_jaccard"], "tracks": len(alive), "stable_groups": len(stable),
        "groups_per_month": [len(set(row)) for row in labels],
        "stable_cover": round(sum(in_stable) / (len(ids) * months), 4),
        "ari_reference": {"monthly": [round(value, 4) for value in ari], "median": round(median(ari), 4)},
        "multi_group_share": round(sum(len(set(path)) >= 2 for path in paths) / len(ids), 4),
        "multi_group": sum(len(set(path)) >= 2 for path in paths),
        "switches": {"median": median(switches), "zero": sum(s == 0 for s in switches), "total": sum(switches),
                     "between_district_groups": round(sum(a in district and b in district for a, b in moves) / len(moves), 4)},
        "groups": groups}


def _number(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def _percent(value: float) -> str:
    return f"{_number(100 * value, 1)} %"


def _month(period: str) -> str:
    months = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
    return f"{months[int(period[5:7]) - 1]} {period[:4]}"


def readme(summary: dict, type_names: dict[str, str]) -> str:
    periods, groups = summary["periods"], sorted(summary["groups"], key=lambda g: -sum(g["sizes"]))
    rows = []
    for g in groups:
        alive = [size for size in g["sizes"] if size]
        span = f"с {_month(periods[g['start']])} по {_month(periods[g['end']])}" + ("" if g["months"] == g["end"] - g["start"] + 1 else f", {g['months']} мес.")
        top = str(g["top_type"])
        rows.append(f"| {g['name']} | {span} | {_number(median(alive))} | {type_names[top]} ({_percent(g['types'][top])}) | "
                    f"{_percent(g['shares']['Продовольствие'])} | {_percent(g['shares']['Маркетплейсы'])} | "
                    f"{_number(g['population'])} | {_number(g['wage_2023'])} | {g['top_region']} ({_percent(g['top_region_share'])}) |")
    full = [g for g in groups if g["months"] == len(periods)]
    full_share = sum(g["sizes"][t] for g in full for t in range(len(periods))) / (summary["n"] * len(periods))
    lines = [
        "# Группы временного Leiden",
        "",
        f"Я разобрал, из каких территорий состоят группы временного Leiden (ω = {_number(summary['omega'], 2)}, seed {summary['seed']}) за 24 месяца. "
        "Модель не переобучалась: скрипт читает сохранённые метки и события жизненного цикла.",
        "",
        f"Номера групп Leiden в разные месяцы переиспользуются, поэтому группу я веду по событиям жизненного цикла (Жаккар ≥ 0,3), "
        f"а группу, которая пропала и вернулась почти тем же составом (Жаккар ≥ {_number(summary['rejoin_jaccard'], 1)}), считаю той же. "
        f"Всего получилось {summary['tracks']} групп, из них {summary['stable_groups']} устойчивых (живут не меньше {summary['min_months']} месяцев). "
        f"Они покрывают {_percent(summary['stable_cover'])} всех пар «территория × месяц».",
        "",
        "В таблице размер группы равен медиане числа МО в месяцы, когда она существует; доли расходов, население и зарплата даны медианами по участникам группы.",
        "",
        "| Группа | Когда есть | МО в месяц | Преобладающий тип 2023 | Продукты | Маркетплейсы | Население | Зарплата 2023, ₽ | Крупнейший регион |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Вывод",
        "",
        f"Согласие групп с четырьмя типами низкое: медианный ARI по месяцам {_number(summary['ari_reference']['median'], 2)}. "
        f"Динамическая сеть делит территории по-другому: {len(full)} группы живут все 24 месяца и держат {_percent(full_share)} наблюдений, "
        "а остальные устойчивые группы сезонные или региональные, которых в годовых типах нет. "
        f"Траектория {_number(summary['multi_group'])} из {_number(summary['n'])} МО ({_percent(summary['multi_group_share'])}) проходит через две и больше групп. "
        f"{_percent(summary['switches']['between_district_groups']).capitalize()} всех переходов приходится на обмен между группами, где преобладают сбалансированный и продуктовый типы: "
        "эти похожие районы сеть от месяца к месяцу делит немного по-разному. "
        "Поэтому типы я использую как устойчивую годовую опору, а временной Leiden как способ увидеть сезонные и региональные сдвиги, которые годовой профиль сглаживает.",
        "",
        "Пересчёт: `python scripts/describe_temporal_groups.py` (несколько секунд, без обучения моделей). Числа лежат в `summary.json`.",
        "",
    ]
    return "\n".join(lines)
