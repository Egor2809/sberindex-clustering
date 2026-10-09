"""Build descriptive municipal cases from saved atlas data without fitting models."""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/practical-cases-2026-10-07"


def read_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const packed\s*=\s*(\{.*?\});", text, re.S)
    if match is None:
        raise ValueError("Atlas data package not found")
    packed = json.loads(match.group(1))
    if packed.get("encoding") == "gzip-base64":
        return json.loads(gzip.decompress(base64.b64decode(packed["data"], validate=True)))
    if "entities" not in packed:
        raise ValueError("Unknown atlas data encoding")
    return packed


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def unique_by_id(rows: list[dict], key: str = "entity_id") -> dict:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate entity ID")
    return result


def distance(a: dict, b: dict) -> float:
    return sum((x - y) ** 2 for x, y in zip(a["features"], b["features"], strict=True)) ** 0.5


def eligible(a: dict, b: dict, factor: float = 1.25) -> bool:
    if factor <= 1 or a["expense_2023"] <= 0 or b["expense_2023"] <= 0:
        raise ValueError("Invalid expense comparison")
    ratio = b["expense_2023"] / a["expense_2023"]
    return a["municipal_district_type"] == b["municipal_district_type"] and 1 / factor <= ratio <= factor


def compile_cases(data: dict, bridge: list[dict], cohort: list[dict], paired: list[dict], abazinsky: dict) -> dict:
    entities = unique_by_id(data["entities"], "id")
    context = unique_by_id(bridge)
    outside = unique_by_id(cohort)
    paired_map = unique_by_id(paired)
    ids = set(entities)
    if set(context) != ids or set(paired_map) != ids:
        raise ValueError("Municipal context or paired labels do not join exactly")
    for row in context.values():
        row["expense_2023"] = float(row["expense_2023"])
    periods = data["periods"]
    expected_periods = [f"{year}-{month:02d}-01" for year in (2023, 2024) for month in range(1, 13)]
    if periods != expected_periods:
        raise ValueError("Expected unique ordered monthly calendar for 2023 and 2024")
    months_2023 = [i for i, value in enumerate(periods) if value.startswith("2023-")]
    months_2024 = [i for i, value in enumerate(periods) if value.startswith("2024-")]
    if len(months_2023) != 12 or len(months_2024) != 12:
        raise ValueError("Expected complete 2023 and 2024 monthly panel")
    for entity in entities.values():
        if len(entity["features"]) != 5 or len(entity["totals"]) != 24 or len(entity["ratios"]) != 24 or len(entity["annual_ratios"]) != 5:
            raise ValueError("Invalid profile shape")
        ratio_values = entity["annual_ratios"] + [value for row in entity["ratios"] for value in row]
        if any(len(row) != 5 for row in entity["ratios"]) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in ratio_values
        ):
            raise ValueError("Expected finite five-dimensional category ratios")
        # Both monthly percentages and annual medians are independently rounded to six decimals.
        for category, annual in enumerate(entity["annual_ratios"]):
            monthly_median = median(entity["ratios"][i][category] for i in months_2023)
            if not math.isclose(annual, monthly_median, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("Annual ratios differ from the 2023 monthly median")
        expected = median(entity["totals"][i] for i in months_2023)
        if abs(expected - context[entity["id"]]["expense_2023"]) > 1e-6:
            raise ValueError("Expense context differs from atlas panel")
    profiles = []
    for group in data["contest"]["profiles"]:
        members = [e for e in entities.values() if e["reference_cluster"] == group["id"]]
        wage_values = [float(outside[e["id"]]["wage_total"]) for e in members if e["id"] in outside and outside[e["id"]].get("wage_total")]
        market_values = [float(context[e["id"]]["market_access"]) for e in members if context[e["id"]].get("market_access")]
        if len(members) != group["n"]:
            raise ValueError("Group membership differs from published size")
        profiles.append({
            "group_zero_based": group["id"], "group_number": group["id"] + 1,
            "name": group["name"], "n": len(members),
            "median_ratios_pct": [median(e["annual_ratios"][j] for e in members) for j in range(5)],
            "median_monthly_expense_2023_rubles": median(context[e["id"]]["expense_2023"] for e in members),
            "market_access_2024_median": median(market_values), "market_access_n": len(market_values),
            "wage_2023_median_rubles": median(wage_values) if wage_values else None, "wage_n": len(wage_values),
            "municipal_types": group["municipal_types"], "top_regions": group["top_regions"], "examples": group["examples"],
        })
    target = entities["tid_196"]
    ranked = sorted((e for e in entities.values() if e["id"] != target["id"]), key=lambda e: (distance(target, e), e["id"]))
    rows = []
    for rank, e in enumerate(ranked[:15], 1):
        source, neighbor = context[target["id"]], context[e["id"]]
        rows.append({"rank": rank, "entity_id": e["id"], "name": e["name"], "region": e["region"],
                     "municipality_type": neighbor["municipal_district_type"], "distance": distance(target, e),
                     "expense_2023_rubles": neighbor["expense_2023"], "expense_ratio_to_selected": neighbor["expense_2023"] / source["expense_2023"],
                     "same_type": source["municipal_district_type"] == neighbor["municipal_district_type"],
                     "same_region": target["region"] == e["region"], "eligible_factor_1_25": eligible(source, neighbor),
                     "annual_ratios_pct": e["annual_ratios"],
                     "category_delta_pp": [b - a for a, b in zip(target["annual_ratios"], e["annual_ratios"], strict=True)]})
    if [row["entity_id"] for row in rows] != [row["entity_id"] for row in abazinsky["neighbors"]]:
        raise ValueError("Neighbors differ from independently saved Abazinsky case")
    for new, saved in zip(rows, abazinsky["neighbors"], strict=True):
        if abs(new["distance"] - saved["distance"]) > 1e-10:
            raise ValueError("Neighbor distance differs from saved case")
    sensitivity = []
    for factor in (1.1, 1.25, 1.5, 2.0):
        all_filtered = [e for e in ranked if eligible(context[target["id"]], context[e["id"]], factor)]
        sensitivity.append({"factor": factor, "eligible_in_first_15": sum(eligible(context[target["id"]], context[e["id"]], factor) for e in ranked[:15]),
                            "eligible_in_all_2015": len(all_filtered), "same_region_eligible": sum(e["region"] == target["region"] for e in all_filtered),
                            "first_3_ids": [e["id"] for e in all_filtered[:3]]})
    dynamic = []
    counts = {"raw_changed": 0, "population_relative_changed": 0, "paired_relative_changed": 0}
    for e in entities.values():
        old = e["reference_cluster"]
        raw = data["contest"]["assignments"]["raw"][e["id"]]
        relative = data["contest"]["assignments"]["relative"][e["id"]]
        pair = int(paired_map[e["id"]]["relative_label"])
        if int(paired_map[e["id"]]["absolute_label"]) != raw:
            raise ValueError("Raw labels differ across temporal sources")
        counts["raw_changed"] += old != raw
        counts["population_relative_changed"] += old != relative
        counts["paired_relative_changed"] += old != pair
        if old != raw and old == relative and old == pair:
            dynamic.append({"entity_id": e["id"], "name": e["name"], "region": e["region"],
                            "group_2023": old + 1, "raw_group_2024": raw + 1, "population_relative_group_2024": relative + 1, "paired_relative_group_2024": pair + 1,
                            "marketplaces_2023_pct": median(e["ratios"][i][1] for i in months_2023),
                            "marketplaces_2024_pct": median(e["ratios"][i][1] for i in months_2024)})
    if counts != {"raw_changed": 380, "population_relative_changed": 207, "paired_relative_changed": 218}:
        raise ValueError("Temporal counts differ from saved results")
    dynamic.sort(key=lambda row: row["entity_id"])
    low = entities["tid_616"]
    low_external = outside.get(low["id"], {})
    return {"categories": data["categories"], "n": len(entities), "profiles": profiles,
            "analogue_case": {"entity_id": target["id"], "name": target["name"], "region": target["region"],
                             "expense_2023_rubles": context[target["id"]]["expense_2023"], "annual_ratios_pct": target["annual_ratios"],
                             "neighbors": rows, "sensitivity": sensitivity},
            "low_marketplace_case": {"entity_id": low["id"], "name": low["name"], "region": low["region"],
                                     "group_number": low["reference_cluster"] + 1, "annual_ratios_pct": low["annual_ratios"],
                                     "expense_2023_rubles": context[low["id"]]["expense_2023"],
                                     "market_access_2024": float(context[low["id"]]["market_access"]) if context[low["id"]].get("market_access") else None,
                                     "wage_2023_rubles": float(low_external["wage_total"]) if low_external.get("wage_total") else None,
                                     "population_2023": float(low_external["population_total"]) if low_external.get("population_total") else None,
                                     "source_oktmo8": low_external.get("source_oktmo8"),
                                     "boundary_history_flag": low_external.get("source_boundary_history_flag")},
            "dynamics": {"counts": counts, "raw_change_removed_in_both_adjustments_n": len(dynamic),
                         "example_selection": "First entity_id in the raw-change-only subset for both adjustments; illustrative retrospective choice", "examples": dynamic[:2]},
            "scope": "Descriptive historical cases. No new fitting, no causal effect, no industry labels, no 2025 outcomes."}


def number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "нет данных"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def markdown_report(result: dict) -> str:
    case = result["analogue_case"]
    nogai = next(row for row in case["neighbors"] if row["entity_id"] == "tid_201")
    dynamic = result["dynamics"]["examples"][0]
    low = result["low_marketplace_case"]
    low_wage = (
        f"Зарплата организаций без малого бизнеса самого района в сохранённом источнике 2023 года {number(low['wage_2023_rubles'])} руб.; низкая интенсивность маркетплейсов не означает низких зарплат."
        if low["wage_2023_rubles"] is not None
        else "В сохранённом источнике нет данных о зарплате организаций без малого бизнеса этого района за 2023 год. Низкая интенсивность маркетплейсов не определяет уровень зарплат."
    )
    lines = ["# Как использовать потребительские профили местной экономики", "",
             "Четыре группы описывают наблюдаемый состав безналичных потребительских расходов муниципалитетов. По ним можно выбрать сравнение и уточнить вопрос для дальнейшей проверки. Отраслевая специализация, причины изменения и эффект муниципальных мер из этих меток не установлены.", "",
             "## Что означает каждая группа", "",
             "Профиль 2023 года: медиана 12 месячных отношений каждой категории к общему показателю «Все категории», затем медиана по МО группы. Знаменатель каждого отношения: показатель «Все категории» того же месяца. Категории могут пересекаться, а их сумма не обязана совпадать с итогом, поэтому пять отношений я не складываю в 100%. «Все категории»: модельная оценка средних месячных безналичных потребительских расходов жителей МО в рублях; на население я её дополнительно не делю. Уровень в таблице: медиана 12 месячных значений 2023 года, затем медиана по группе.", "",
             "| Группа | МО | Продовольствие, % | Общепит, % | Маркетплейсы, % | Расходы, руб. | Доступность рынков 2024; покрытие | Зарплата 2023, руб.; покрытие |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for p in result["profiles"]:
        r = p["median_ratios_pct"]
        lines.append(f"| {p['group_number']}. {p['name']} | {p['n']} | {number(r[3])} | {number(r[2])} | {number(r[1])} | {number(p['median_monthly_expense_2023_rubles'])} | {number(p['market_access_2024_median'])}; {p['market_access_n']}/{p['n']} | {number(p['wage_2023_median_rubles'])}; {p['wage_n']}/{p['n']} |")
    lines.extend(["", "Доступность рынков взята из сохранённого внешнего показателя, зарплата из муниципальных таблиц Росстата и «Если быть точным». Зарплата относится к организациям без малого бизнеса и не измеряет доход всех жителей. Это сырые описательные медианы доступной части каждой группы. Различия региона, типа МО, пропуски и пространственная зависимость не устранены; таблица не доказывает независимую связь или причины. Год 2024 у доступности рынков следует за периодом группировки.", ""])
    for p in result["profiles"]:
        examples = "; ".join(f"{e['name']} ({e['id']}, {e['region']})" for e in p["examples"][:2])
        types = "; ".join(f"{name}: {n}/{p['n']}" for name, n in p["municipal_types"].items())
        lines.extend([f"Группа {p['group_number']}: {examples}. Состав по типам: {types}.", ""])
    lines.extend(["Название «Городской сервисный» относится к потреблению. В этом типе есть 36 муниципальных районов из 386; слово «городской» не заменяет муниципальный тип. В типе «Удалённый: мало маркетплейсов» 26 МО из Якутии, но низкое отношение расходов на маркетплейсах к общему показателю «Все категории» не доказывает транспортный барьер или промышленную специализацию.", "",
                  "## Подобрать территорию для сравнения", "",
                  f"Абазинский муниципальный район ({case['entity_id']}, {case['region']}): медианный месячный уровень расходов 2023 года {number(case['expense_2023_rubles'])} руб. Из 15 ближайших по пяти координатам соседей только один имеет тот же муниципальный тип. Это Ногайский муниципальный район ({nogai['entity_id']}) из того же региона: {number(nogai['expense_2023_rubles'])} руб., отношение уровней {number(nogai['expense_ratio_to_selected'], 4)}, расстояние профилей {number(nogai['distance'], 6)}.", "",
                  "Для предварительного сравнения я беру тот же тип МО и симметричное отношение уровней расходов от 1/1,25 до 1,25, то есть от 0,8 до 1,25. Это рабочее правило, его пригодность для решений не проверена. Регион я показываю отдельно: одинаковый регион облегчает сравнение административных условий, другой регион помогает искать отличающиеся условия. Порядок по-прежнему задаёт близость пяти координат. Новое правило действует только в этой таблице; интерактивный список атласа по-прежнему показывает 15 соседей по составу.", "",
                  "| Предел отношения | Подходят из первых 15 | Подходят из остальных 2015 МО | Из них того же региона | Первые три ID после ограничения |", "|---|---:|---:|---:|---|"])
    for row in case["sensitivity"]:
        lines.append(f"| {number(row['factor'])} | {row['eligible_in_first_15']} | {row['eligible_in_all_2015']} | {row['same_region_eligible']} | {', '.join(row['first_3_ids'])} |")
    lines.extend(["", f"Отношения категорий к общему показателю «Все категории» Абазинского и Ногайского районов: продовольствие {number(case['annual_ratios_pct'][3])}% и {number(nogai['annual_ratios_pct'][3])}%; маркетплейсы {number(case['annual_ratios_pct'][1])}% и {number(nogai['annual_ratios_pct'][1])}%. Разница продовольствия {number(nogai['category_delta_pp'][3])} п.п. Эту пару можно взять для дальнейшего сравнения структуры торговли и доступности услуг. Перед предложением меры нужно сверить население, охват платежей, границы и размещение объектов. Само различие отношений не означает дефицита или эффекта меры.", "",
                  "## Уточнить вопрос по группе", "",
                  f"Усть-Камчатский муниципальный район (tid_616, Камчатский край) входит в группу {low['group_number']}. Его медианное отношение маркетплейсов к общему показателю «Все категории» {number(low['annual_ratios_pct'][1])}%, медианный месячный уровень расходов {number(low['expense_2023_rubles'])} руб. Для группы отношение маркетплейсов 4,60%, а доступность рынков 139,30 при покрытии 117 из 127 МО. {low_wage} Это повод сопоставить территорию с МО той же группы и отдельно проверить наблюдаемые условия торговли. Нужны данные об охвате, ассортименте, доставке и ценах, чтобы отличить спрос от доступности. Сценарий помогает сформулировать вопрос и выбрать показатели для проверки. Экономический эффект по этим данным не оценён.", "",
                  "## Разобрать смену профиля", "",
                  f"{dynamic['name']} ({dynamic['entity_id']}, {dynamic['region']}): группа {dynamic['group_2023']} в 2023 году, сырая группа {dynamic['raw_group_2024']} в 2024 году, группа {dynamic['population_relative_group_2024']} после прежней поправки и {dynamic['paired_relative_group_2024']} после парной поправки. Медианное отношение маркетплейсов к общему показателю «Все категории» изменилось с {number(dynamic['marketplaces_2023_pct'])}% до {number(dynamic['marketplaces_2024_pct'])}%. Это иллюстративный ретроспективный пример: первый ID среди смен, исчезающих при обеих поправках.", "",
                  "Во всём составе 2016 МО сырые метки меняются у 380; прежняя разность популяционных медиан даёт 207, парная медиана изменений фиксированного референса даёт 218. Это разные способы поправки. Общая поправка показывает изменение положения профиля относительно типичного сдвига. Рост отношения расходов категории к итогу остаётся отдельным наблюдением. Исчезновение смены после поправки не доказывает её причину. Аналитик может сначала изучить месяцы и категории, затем проверить локальные события и охват данных; прогноза и вывода о перестройке экономики здесь нет.", "",
                  "## Воспроизведение и границы", "", "```text", "python -X utf8 scripts/build_practical_cases.py", "python -m unittest tests.test_practical_cases -v", "```", "",
                  "Скрипт декодирует gzip-base64 из docs/index.html, не обучает модели и не читает исходы 2025. cases.json хранит полные числа и ID, group_profiles.csv содержит профили и покрытия, analogue_screen.csv содержит всех 15 соседей, provenance.json сохраняет пути и SHA256 входов, правила и команду запуска. Сценарии выбраны для объяснения уже известных данных; они не служат заранее зарегистрированной проверкой эффективности.", "",
                  "Сохранённая отраслевая проверка содержит 1797 допустимых МО. Наблюдаемые доли доступны для меньших выборок; все 36 направленных границ сравнения на полном составе включают ноль. Поэтому группы остаются типами наблюдаемого потребительского профиля местной экономики. Производственная специализация не установлена. [Подробности внешней проверки](ECONOMIC_GENERALIZATION.md).", ""])
    return "\n".join(lines)


def html_block(result: dict) -> str:
    case = result["analogue_case"]
    low = result["low_marketplace_case"]
    nogai = next(row for row in case["neighbors"] if row["entity_id"] == "tid_201")
    dynamic = result["dynamics"]["examples"][0]
    p = next(row for row in result["profiles"] if row["group_number"] == low["group_number"])
    dynamic_name = escape(dynamic["name"])
    group = {row["group_number"]: escape(row["name"]) for row in result["profiles"]}
    return f'''<section class="card practical-cases" aria-labelledby="practical-title">
<div class="section-head"><h2 id="practical-title">Три способа использовать результат</h2><p>Типы KMeans4 описывают наблюдаемый потребительский профиль местной экономики. Они помогают выбрать сравнение и вопрос для проверки. Отраслевая специализация и эффект мер не установлены.</p></div>
<div class="practical-grid">
<article class="practical-card" aria-labelledby="practical-analogue-title">
<h3 id="practical-analogue-title">Выбрать сравнение для района</h3>
<p>Абазинский район, Карачаево-Черкесия: {number(case['expense_2023_rubles'])} руб. Из 15 ближайших соседей один имеет тот же муниципальный тип: Ногайский район ({nogai['entity_id']}), {number(nogai['expense_2023_rubles'])} руб., тот же регион.</p>
<div class="table-wrap"><table class="practical-table"><caption>Отношения категорий к общему показателю «Все категории», медианы месяцев 2023</caption><thead><tr><th scope="col">Категория</th><th scope="col">Абазинский</th><th scope="col">Ногайский</th></tr></thead><tbody><tr><th scope="row">Продовольствие</th><td>{number(case['annual_ratios_pct'][3])}%</td><td>{number(nogai['annual_ratios_pct'][3])}%</td></tr><tr><th scope="row">Маркетплейсы</th><td>{number(case['annual_ratios_pct'][1])}%</td><td>{number(nogai['annual_ratios_pct'][1])}%</td></tr></tbody></table></div>
<p class="practical-note">Правило для этой пары: тот же тип МО, отношение уровней от 0,8 до 1,25; проходит 1 из первых 15. Порог рабочий, пригодность для решений не проверена. Уровень: медиана 12 модельных оценок средних месячных безналичных расходов жителей МО, без деления на население.</p>
<p>Можно изучить причины разницы продовольствия {number(nogai['category_delta_pp'][3])} п.п. Перед предложением меры нужны население, охват платежей и размещение услуг.</p>
<a href="#neighbors-title">Проверить исходный список аналогов</a>
</article>
<article class="practical-card" aria-labelledby="practical-profile-title">
<h3 id="practical-profile-title">Поставить вопрос о торговле</h3>
<p>Усть-Камчатский район ({low['entity_id']}, Камчатский край), тип «{group[low['group_number']]}»: маркетплейсы {number(low['annual_ratios_pct'][1])}%, уровень расходов {number(low['expense_2023_rubles'])} руб.</p>
<p>В этом типе медиана маркетплейсов {number(p['median_ratios_pct'][1])}%; доступность рынков 2024 года {number(p['market_access_2024_median'])}, данные есть для {p['market_access_n']} из {p['n']} МО.</p>
<p>Можно подобрать территории с похожим профилем, затем сравнить ассортимент, доставку, цены и охват. Низкое отношение само по себе не объясняет условия покупок.</p>
<p class="practical-note">Внешний показатель описывает доступную часть группы. Региональные различия и пропуски сохраняются. Медиана типа не заменяет показатель района.</p>
<a href="#territory-title">Найти район и открыть его профиль</a>
</article>
<article class="practical-card" aria-labelledby="practical-dynamics-title">
<h3 id="practical-dynamics-title">Разобрать смену метки</h3>
<p>{dynamic_name} ({dynamic['entity_id']}, {escape(dynamic['region'])}): тип «{group[dynamic['group_2023']]}» в 2023 году, «{group[dynamic['raw_group_2024']]}» в 2024 году без поправки. После прежней и парной поправок тип остаётся прежним.</p>
<p>Маркетплейсы: {number(dynamic['marketplaces_2023_pct'])}% → {number(dynamic['marketplaces_2024_pct'])}%. Аналитик может проверить месяцы и категории, прежде чем связывать переход с локальным событием.</p>
<p class="practical-note">Во всём составе 2016 МО: 380 сырых смен, 207 после прежней поправки, 218 после парной. Это разные методы. Пример выбран ретроспективно; исчезновение смены не доказывает её причину.</p>
<a href="#flows-title">Сопоставить временные каналы</a>
</article>
</div>
<details class="practical-note"><summary>Данные, правила и границы применения</summary><p>Сценарии рассчитаны скриптом build_practical_cases.py из сохранённого снимка без обучения и исходов 2025. Полные ID, числа, покрытия и чувствительность порога находятся в reports/practical-cases-2026-10-07, происхождение и SHA256 в provenance.json, описание в docs/ECONOMIC_CASES.md. Каждая категория делится на общий показатель «Все категории» того же месяца; категории могут пересекаться, а их сумма не обязана совпадать с итогом. Отраслевые сравнения полной допустимой совокупности не подтверждают превосходство: все 36 границ включают ноль.</p></details>
</section>
'''


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(atlas: Path, output: Path, *, data_only: bool = False, check_data: bool = False) -> dict:
    input_paths = {
        "atlas": atlas,
        "bridge": ROOT / "reports/external-v4/municipality_bridge.csv",
        "cohort": ROOT / "reports/external-national-2026-10-03/cohort-2023.csv",
        "paired": ROOT / "reports/temporal-core-2026-10-03/paired-reference/assignments.csv",
        "abazinsky": ROOT / "reports/round2-cases/abazinsky.json",
        "sector_validation": ROOT / "reports/economic-generalization-2026-10-03/sectors/results.json",
        "external_manifest": ROOT / "reports/external-national-2026-10-03/manifest.json",
    }
    data = read_payload(atlas)
    result = compile_cases(data, read_csv(input_paths["bridge"]), read_csv(input_paths["cohort"]),
                           read_csv(input_paths["paired"]), json.loads(input_paths["abazinsky"].read_text("utf-8")))
    if check_data:
        saved = json.loads((output / "cases.json").read_text("utf-8"))
        if saved != result:
            raise ValueError("Saved descriptive cases do not reproduce from current scientific inputs")
        return result
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8", newline="\n")
    profiles = [{"group": p["group_number"], "name": p["name"], "n": p["n"],
                 **{f"category_{i}_pct": value for i, value in enumerate(p["median_ratios_pct"])},
                 "expense_2023_rubles": p["median_monthly_expense_2023_rubles"], "market_access_2024": p["market_access_2024_median"],
                 "market_access_n": p["market_access_n"], "wage_2023_rubles": p["wage_2023_median_rubles"], "wage_n": p["wage_n"]}
                for p in result["profiles"]]
    write_csv(output / "group_profiles.csv", profiles)
    write_csv(output / "analogue_screen.csv", [{k: v for k, v in row.items() if k not in ("annual_ratios_pct", "category_delta_pp")} for row in result["analogue_case"]["neighbors"]])
    payload_bytes = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    provenance = {"inputs": [{"role": role, "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for role, path in input_paths.items()],
                  "decoded_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                  "authoritative_atlas_input": "decoded_payload_sha256; this is independent of HTML presentation",
                  "atlas_html_hash_role": "Historical input file at calculation time, not a required hash of the final assembled HTML",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "command": "python -X utf8 scripts/build_practical_cases.py",
                  "profile_rule": "Median of monthly category/All categories ratios in 2023, then median across municipalities in a frozen group. Ratios need not be additive or disjoint.",
                  "expense_semantics": "Model estimate of average monthly cashless consumer expenses of municipal residents; no repeated division by population",
                  "wage_semantics": "Average monthly organization wage excluding small businesses, not all resident income",
                  "analogue_rule": "Euclidean distance over saved 5 standardized annual features, ID ascending tie break. Same municipal type, symmetric expense factor 1.25; region reported, not filtered.",
                  "scope": result["scope"], "selection": result["dynamics"]["example_selection"]}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")
    if not data_only:
        (ROOT / "docs/ECONOMIC_CASES.md").write_text(markdown_report(result), "utf-8", newline="\n")
        (ROOT / "web/practical_cases.html").write_text(html_block(result), "utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ROOT / "docs/index.html")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data-only", action="store_true", help="Refresh tables and provenance without overwriting edited presentation text")
    parser.add_argument("--check-data", action="store_true", help="Recompute and compare cases.json without writing files")
    args = parser.parse_args()
    built = build(args.atlas, args.output, data_only=args.data_only, check_data=args.check_data)
    print(f"Saved {built['n']} municipal profiles, 4 groups, 15 screened analogues; dynamics {built['dynamics']['counts']}")
