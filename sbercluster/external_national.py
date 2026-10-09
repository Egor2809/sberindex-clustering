"""Strict extraction of annual independent Rosstat PMO observations.

Source identity is the explicit OKTMO/year request and exact year-valid registry
join. Names provide a second check; they never assign a territory. Suppression,
missing values, and conflicts are retained rather than filled or averaged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


def clean(text: str) -> str:
    return " ".join(text.split())


def name_key(text: str) -> str:
    text = clean(text).casefold().replace("ё", "е")
    text = re.sub(r"\((?:до|с)\s+\d[^)]*\)", " ", text)
    text = text.replace("город-герой", "город")
    for phrase in ("закрытое административно-территориальное образование",
                   "муниципальное образование", "муниципальный район",
                   "муниципальный округ", "городской округ", "городское поселение",
                   "внутригородская территория", "внутригородское муниципальное образование"):
        text = text.replace(phrase, " ")
    text = re.sub(r"\b(?:город|район|г|зато)\b", " ", text)
    return re.sub(r"[^а-яa-z0-9]", "", text)


def number_status(text: str) -> tuple[float | None, str]:
    text = clean(text)
    if not text:
        return None, "missing"
    if text in {"...", "…", "х", "x", "X", "Х"}:
        return None, "suppressed"
    if text in {"SD", "UD"}:
        return None, "suppressed_or_unknown"
    if text in {"ND", "CD", "NR"}:
        return None, "not_reported"
    if text in {"-", "–", "—"}:
        return None, "not_applicable_or_absent"
    candidate = text.replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", candidate):
        return None, "unparsed"
    value = float(candidate)
    if not (-1e20 < value < 1e20):
        return None, "unparsed"
    return value, "observed"


def indicator_kind(text: str) -> str | None:
    text = clean(text).casefold()
    if text.startswith("среднемесячная заработная плата работников организаций") and "без субъектов малого" in text:
        return "wage"
    if text.startswith("среднесписочная численность работников организаций") and "без субъектов малого" in text:
        return "headcount"
    if text.startswith("среднемесячная номинальная начисленная заработная плата работников крупных"):
        return "efficiency_wage"
    if text == "оценка численности населения на 1 января текущего года":
        return "population"
    if text.startswith("численность всего населения по полу и возрасту на 1 января текущего года"):
        return "population_age_sex"
    return None


@dataclass(frozen=True)
class Passport:
    status: str
    source_name: str | None
    source_type: str | None
    source_region_caption: str | None
    observations: tuple[dict, ...]


class _PassportHTML(HTMLParser):
    """Read text cells without executing HTML or depending on browser parsers."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.divs: list[str] = []
        self.rows: list[list[tuple[dict, str]]] = []
        self.text_parts: list[str] = []
        self._div_stack: list[list[str]] = []
        self._row: list[tuple[dict, str]] | None = None
        self._cell: tuple[dict, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            self._div_stack.append([])
        elif tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = (dict(attrs), [])
        elif tag == "br":
            self.handle_data(" ")

    def handle_data(self, data):
        self.text_parts.append(data)
        for div in self._div_stack:
            div.append(data)
        if self._cell is not None:
            self._cell[1].append(data)

    def handle_endtag(self, tag):
        if tag == "div" and self._div_stack:
            self.divs.append(clean(" ".join(self._div_stack.pop())))
        elif tag == "td" and self._cell is not None and self._row is not None:
            self._row.append((self._cell[0], clean(" ".join(self._cell[1]))))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def parse_passport(raw: bytes, year: int) -> Passport:
    if not raw or len(raw) > 8 * 1024 * 1024:
        raise ValueError("Nonempty bounded passport required")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("windows-1251")
    tree = _PassportHTML()
    tree.feed(text)
    divs = tree.divs
    all_text = " ".join(tree.text_parts)
    if "Файл таблицы не найден" in all_text:
        return Passport("table_absent", None, None, None, ())
    periods = [i for i, t in enumerate(divs) if re.fullmatch(r"за \d{4} год", t)]
    if len(periods) != 1 or divs[periods[0]] != f"за {year} год":
        return Passport("wrong_or_missing_year", None, None, None, ())
    i = periods[0]
    if i < 3 or "Паспорт муниципального образования" not in all_text:
        return Passport("invalid_passport_header", None, None, None, ())
    source_name, source_type = divs[i - 1], divs[i - 2]
    region_caption = next((v for v in divs[:i] if v.startswith("БД ПМО")), None)
    observations = []
    current_indicator = ""
    kind = None
    hierarchy: list[tuple[float, str]] = []
    for row_number, cells in enumerate(tree.rows, 1):
        if len(cells) != 3:
            continue
        values = [c[1] for c in cells]
        if cells[0][0].get("class") == "pok":
            current_indicator = values[0]
            kind = indicator_kind(current_indicator)
            hierarchy = []
            continue
        if kind is None or cells[0][0].get("class") != "prizn":
            continue
        indent_match = re.search(r"padding-left:\s*([\d.]+)pt", cells[0][0].get("style", ""), re.I)
        indent = float(indent_match[1]) if indent_match else 0
        if not values[1] and not values[2]:
            hierarchy = [(level, label) for level, label in hierarchy if level < indent]
            hierarchy.append((indent, values[0]))
            continue
        period = values[0]
        annual = period.casefold() == "январь-декабрь"
        population_point = kind in {"population", "population_age_sex"} and period.casefold() == "на 1 января"
        if not annual and not population_point:
            continue
        expected_unit = "рубль" if kind in {"wage", "efficiency_wage"} else "человек"
        value, status = number_status(values[2])
        if values[1].casefold() != expected_unit:
            status = "unexpected_unit"
            value = None
        sector_path = [label for level, label in hierarchy if level < indent]
        observations.append({"kind": kind, "indicator": current_indicator,
                             "sector_path": sector_path,
                             "sector": sector_path[-1] if sector_path else "",
                             "period": period, "unit": values[1],
                             "value": value, "value_status": status,
                             "source_row": row_number})
    return Passport("parsed", source_name, source_type, region_caption, tuple(observations))


def select_unique(observations: tuple[dict, ...], kind: str, sector_test) -> tuple[float | None, str]:
    rows = [row for row in observations if row["kind"] == kind and sector_test(row)]
    if not rows:
        return None, "not_published"
    if len(rows) != 1:
        return None, "duplicate_or_conflicting_source_rows"
    return rows[0]["value"], rows[0]["value_status"]


def total_sector(row: dict) -> bool:
    sector = row["sector"].casefold()
    return len(row["sector_path"]) <= 1 and (sector.startswith("всего") or sector == "все население" or not sector)


SECTOR_PREFIXES = {
    "A": "сельское, лесное хозяйство", "B": "добыча полезных ископаемых",
    "C": "обрабатывающие производства", "D": "обеспечение электрической энергией",
    "E": "водоснабжение", "F": "строительство", "G": "торговля оптовая и розничная",
    "H": "транспортировка и хранение", "I": "деятельность гостиниц и предприятий общественного питания",
    "J": "деятельность в области информации и связи", "K": "деятельность финансовая и страховая",
    "L": "деятельность по операциям с недвижимым имуществом",
    "M": "деятельность профессиональная, научная и техническая",
    "N": "деятельность административная и сопутствующие дополнительные услуги",
    "O": "государственное управление и обеспечение военной безопасности",
    "P": "образование", "Q": "деятельность в области здравоохранения и социальных услуг",
    "R": "деятельность в области культуры, спорта", "S": "предоставление прочих видов услуг",
}


def broad_sector(row: dict) -> str | None:
    if len(row["sector_path"]) != 1:
        return None
    sector = row["sector"].casefold()
    sector = re.sub(r"^раздел\s+[a-zа-я]\.?\s*", "", sector)
    for code, prefix in SECTOR_PREFIXES.items():
        if sector.startswith(prefix):
            return code
    return None
