# Сравнение годовых моделей и границы переноса

Два конечных варианта повысили силуэт годовых потребительских профилей в тех же
координатах. Huber75 + ограниченный поиск SW даёт **SW=0,26032357** против
**0,24023677** у исторического KMeans4. CH снижается с 779,82704 до 740,90907;
AVU растёт с 0,52963533 до 0,54330275. Эти потери входят в заранее заданные
практические допуски сравнения. [Полное объяснение](../../docs/MODEL_FRONTIER.md).

Исторический KMeans остаётся моделью по умолчанию. Huber75 сохранён как годовая
исследовательская альтернатива, обычный вариант — как сравнение, а контрастные
K2/K4 — как проверка иной семантики признаков. Национальная зарплата и структура
занятости проверяются в [отдельном независимом слое](../external-national-2026-10-03/README.md);
выбор новых кандидатов состоялся до просмотра этих исходов.

Проверка 2023 использует уже просмотренные данные и ретроспективно полную
панель 2 016 МО × 24 месяца. Четыре оставленных квартала, 30 парных замен
трёхмесячных блоков и два последовательных окна дают свидетельства
чувствительности и переноса. Их нельзя читать как нетронутую подтверждающую
выборку. Годовой прототип также не гарантирует достаточное наполнение кластеров
в отдельном будущем месяце: минимальная группа новых вариантов в ноябре
содержит 20/19 МО, а S_Dbw HV01 там не определён. Все эти значения сохранены.

## Состав пакета

| Путь | Содержание |
|---|---|
| [protocol.json](protocol.json), [конфигурационная запись](../../configs/model_frontier_20261003.json) | Зафиксированные конечные варианты и критерии; исходная конфигурация подготовки остаётся research_validation.json |
| [screen/screen.json](screen/screen.json) | Все десять первоначальных вариантов, включая отрицательные результаты |
| [constrained/screen.json](constrained/screen.json) | Три варианта с ограничением CH ≥ 95% опорного значения; исходные метки и полные ICVI |
| [validation/validation.json](validation/validation.json) | Четыре парных квартальных переноса, 500 региональных повторений, 30 парных месячных bootstrap-опытов |
| [denominator/comparisons.json](denominator/comparisons.json) | Общая исходная и собственная контрастная геометрия, одномерные отрицательные контроли, η² и ARI |
| [denominator/stress.json](denominator/stress.json) | Изменение общего знаменателя при фиксированном scaler; точная инвариантность четырёх контрастов |
| [forward/summary.json](forward/summary.json), [forward/results.json](forward/results.json) | Последовательные окна с обучающим IQR, фактические месячные SW/CH/S_Dbw и размеры групп |
| [annual-2024-application.json](annual-2024-application.json) | Фактическое годовое применение сохранённых правил 2023 к 12 месяцам 2024 без обучения; общие ICVI, размеры, метки, ARI и переходы |
| [models/verification.json](models/verification.json) | Три полных экспортированных правила, SHA256 и точное повторение 2 016 меток из raw-панели |
| [multiscale_cross_tabs.json](multiscale_cross_tabs.json) | Сопоставление сохранённых K2 и K4 без предположения строгой вложенности |
| [geometry-axis-audit.json](geometry-axis-audit.json) | Независимое согласование CH, T/B, осевых долей, η² и двух порядков агрегирования |
| [current_source_sha256.json](current_source_sha256.json), source/ | Точный текущий код новых модулей и тестов |
| [lineage.json](lineage.json) | Связь научных копий с исходными манифестами расчётов |

Каждый расчёт содержит свой исходный код, признаки, scaler, метки и манифест.
Из научных копий убраны частные пути исполняемого Python и идентификаторы
процессов. Новые манифесты проверяют содержимое копий, а lineage.json сохраняет
хеши исходных манифестов. Исторические архивы результатов остаются неизменными.

Исходный panel SHA256:
`a175b7646a528e900d61dd4ade8d84fb4549ffcbb977d845340107b4f3b3c0fa`.
Проверенная среда: Python 3.10.1, NumPy 2.2.6, pandas 2.2.3, SciPy 1.14.1,
scikit-learn 1.5.2, threadpoolctl 3.7.0. Подготовка raw→panel описана в
[REPRODUCE.md](../../docs/REPRODUCE.md).

## Готовое применение

Артефакт имеет тип `prototype_frontier`, версию формата 1 и полный scaler,
преобразование перед медианой, матрицу расстояния, центры и добавочные смещения.
Для каждого ID в выбранном календарном году требуются все 12 месяцев.
Команда проверяет переданный хеш и записывает хеши реально прочитанных байтов.

```powershell
python -m scripts.predict_frontier `
  --model reports/model-frontier-2026-10-03/models/sw_constrained_huber75.json `
  --expected-sha256 41a1071c7d35019f93218d09b6b5abc293bec3b3b4fafdc740cf2c7ac572003a `
  --panel data/processed/panel.csv --year 2023 --output runs/frontier-prediction
```

Индексы кластеров стабильно соответствуют сохранённому правилу. Применение к
другому году описывает потребительский профиль в прежнем пространстве; наличие
ID не подтверждает неизменность границ территории. Обычный frozen API
проверяет тип модели и отвергает этот отдельный формат.

## Фактическое годовое применение в 2024

Сохранённые три правила применены к медиане всех 12 наблюдаемых месяцев 2024.
Scaler и параметры остаются рассчитанными по 2023; для сравнения используется
один общий опорный граф 2024 в этих же пяти координатах. Обучение и выбор
параметров отсутствуют. Значения 2024 уже просмотрены, поэтому этот расчёт
уточняет границу годового применения и сохраняет прежнюю модель по умолчанию.

| Модель | SW ↑ | CH ↑ | S_Dbw HV01 ↓ | AVI ↑ | AVU ↓ | Размеры 2024 | ARI 2023→2024 |
|---|---:|---:|---:|---:|---:|---|---:|
| KMeans4 | 0,26968 | 814,27 | 0,92485 | 0,82019 | 0,51859 | 413 / 1 162 / 414 / 27 | 0,55970 |
| Huber75 + поиск SW | 0,28316 | 739,70 | 0,85282 | 0,82351 | 0,53548 | 289 / 1 363 / 336 / 28 | 0,65344 |
| Обычные прототипы + поиск SW | 0,27961 | 719,31 | 0,87009 | 0,81841 | 0,53742 | 248 / 1 392 / 348 / 28 | 0,64684 |

Все четыре кластера остаются заняты, но минимальная годовая группа всех
вариантов существенно меньше обучающего порога 3%. Huber75 сохраняет больший
силуэт и ARI при ухудшении CH и AVU; CH составляет около 90,84% опорного
значения, поэтому обучающий допуск 95% не переносится автоматически.
С сохранёнными индексами меняются 380 / 295 / 298 меток соответственно.
Опорный результат точно совпадает с прежним расчётом 2024. Полные таблицы
переходов и метки включены в JSON; точный скрипт расчёта сохранён в
[source/scripts/verify_frontier_annual_application.py](source/scripts/verify_frontier_annual_application.py).

## Повторение конечного эксперимента

Команды выполняются из корня новой рабочей копии с подготовленной панелью.
Выходные каталоги должны быть новыми. Числовая конфигурационная запись — это
протокол фактических вариантов; скрипты загружают preprocessing из
`configs/research_validation.json`. Пути ниже сохраняют последовательность
входов первоначального расчёта.

```powershell
python -m venv .venv-frontier
.venv-frontier/Scripts/python.exe -m pip install -r requirements-models.txt -r requirements-visuals.txt
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:POLARS_MAX_THREADS='1'
.venv-frontier/Scripts/python.exe -m scripts.benchmark_frontier --stage screen --output .local/frontier-20261003/benchmark-screen
.venv-frontier/Scripts/python.exe -m scripts.benchmark_frontier --stage screen --family constrained --output .local/frontier-20261003/benchmark-constrained
.venv-frontier/Scripts/python.exe -m scripts.benchmark_frontier --stage validate --screen .local/frontier-20261003/benchmark-constrained --bootstrap 30 --output .local/frontier-20261003/benchmark-validation
.venv-frontier/Scripts/python.exe -m scripts.export_frontier --screen .local/frontier-20261003/benchmark-constrained --validation .local/frontier-20261003/benchmark-validation --panel data/processed/panel.csv --output runs/frontier-models
.venv-frontier/Scripts/python.exe -m scripts.benchmark_contrasts --bootstrap 30 --output runs/denominator-ablation
.venv-frontier/Scripts/python.exe -m scripts.benchmark_forward --output runs/forward-windows
.venv-frontier/Scripts/python.exe -m scripts.verify_frontier_geometry --output runs/geometry-axis-audit.json
.venv-frontier/Scripts/python.exe -m unittest discover -s tests -p 'test_selection_*.py' -v
```

Критерии продолжения: SW выше хотя бы на 0,005; CH не ниже 95%; AVI не ниже
опорного на 0,03; AVU не выше опорного на 0,03; S_Dbw не выше 110%; размер
каждой обучающей группы ≥ 3%; средний квартальный выигрыш SW ≥ 0,002 с нижней
границей регионального интервала выше нуля; медиана ARI месячного bootstrap
не ниже опорной на 0,02. Это согласованные практические допуски исследования,
а не конкурсная формула или формальная гарантия после выбора модели.

Для контрастов обе геометрии показываются отдельно. Каждый месяц сначала
преобразуется в сохранённый ортонормальный базис, затем берётся медиана.
Преобразование годовой медианы даёт другой результат. K2 остаётся крупным
масштабом потребительских профилей; K4 уточняет его без требования вложенности.
