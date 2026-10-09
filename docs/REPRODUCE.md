# Воспроизведение

Все команды выполняются из корня репозитория. Поддерживается Python 3.10–3.12; версии библиотек закреплены в `requirements*.txt`. Статусы фактических проверок выпуска записаны в [протоколе](../reports/dmon-plateau-2026-09-25/verification.json).

## Одна команда

GitHub Actions (`.github/workflows/tests.yml`) на чистом Linux повторяет шаги 1–7, 11 и 12: загрузку с проверкой SHA-256, подготовку, все тесты, точное переобучение KMeans4, правила рёбер, группы временного Leiden (с проверкой, что файлы не изменились), побайтовую сборку атласа и проверку сохранённых моделей. Шаги 8–10 там не идут: Leiden годового каркаса на Linux даёт немного другое разбиение (в `reports/network-core/profiles.json` меняется число МО по регионам), поэтому их побайтовая сверка проходит на macOS, где получены опубликованные файлы.

```sh
bash scripts/reproduce.sh
```

Скрипт `scripts/reproduce.sh` печатает заголовок каждого шага и выполняет по порядку:

1. ставит зависимости из `requirements-models.txt` и `requirements-visuals.txt`;
2. загружает три исходных архива СберИндекса и проверяет их SHA-256;
3. собирает панель 2 016 МО × 24 месяца и графы; контрольный SHA-256 панели: `a175b7646a528e900d61dd4ade8d84fb4549ffcbb977d845340107b4f3b3c0fa`. Отчёты подготовки (`data_audit.json`, `source_alignment.json`, `graph_preparation.json`) пишутся в `artifacts/preparation/`, опубликованные версии в `reports/` не перезаписываются. `scripts/check_preparation.py` сверяет их с опубликованными без учёта времени расчёта, хешей кода и полей, которые есть только при приватной CSV-выгрузке;
4. прогоняет основной набор тестов `python -m unittest discover -s tests`;
5. заново обучает опорный KMeans4 и сверяет все 2 016 меток с опубликованными;
6. повторяет сравнение правил рёбер, сводную таблицу индексов и события групп в `artifacts/edge-rules-repeated` и сверяет результат с `reports/edge-rules` через `scripts/compare_edge_results.py`: числа с допуском 1e-4, метки по ARI не ниже 0,99;
7. описывает группы временного Leiden по сохранённым меткам;
8. заново строит сетевую типологию и побайтово сверяет её с `reports/network-core`;
9. считает индексы по месяцам и выбор ω и побайтово сверяет их с `reports/temporal-quality`;
10. оценивает все 56 годовых разбиений, включая KEFRiN, на общих осях и сравнивает трекинг временного Leiden на трёх seed (`scripts/method_comparison.py`), побайтово сверяет с `reports/method-comparison`;
11. пересобирает атлас и побайтово сравнивает его с `docs/index.html`; при совпадении печатает `Atlas rebuilt byte-for-byte`;
12. автономно проверяет техническое ядро (`scripts/verify_technical_core.py`, раздел 21): сохранённые модели, применение без обучения и отказы команд на неверном вводе.

`reproduce.sh` проверяет все основные результаты из списка выше; шаг 7 пересчитывает описание групп временного Leiden без сверки. DMoN и временной Leiden он не переобучает: это отдельные команды из разделов 9–10. Проверки стыка 2023/2024 и экономической валидации типов описаны в разделе 12, устойчивость к базе и выбор K в разделе 13. Конфиги и скрипты прошлых этапов, которые для этой проверки не нужны, лежат в `archive/` ([описание](../archive/README.md)).

Скрипт останавливается на первой ошибке. Дольше всего считается сравнение правил рёбер: около 10 минут (точное время записано в `reports/edge-rules/provenance.json`). Время загрузки зависит от сети. Другой интерпретатор задаётся переменной `PYTHON=python3.12 bash scripts/reproduce.sh`. Для распаковки RAR нужен `bsdtar`, `unrar`, `7z` или Windows `tar` (в Debian/Ubuntu это пакет `libarchive-tools`).

## Сравнение правил рёбер

```sh
python scripts/compare_edge_rules.py --output artifacts/edge-rules-repeated
```

Конфигурация лежит в `configs/edge_rules.json`: k = 15, ряды за 24 месяца, лаги до 3 месяцев, полоса DTW 2 месяца, K = 4 для спектральной кластеризации, сетка k = 5, 10, 15, 30, порог Жаккара 0,3 и порог изменения размера 10% для событий. Нужны подготовленная панель и `artifacts/sources/acquisition` (дорожные расстояния). Без `--output` результат пишется в `reports/edge-rules`.

Выход:

- `edge_rules.json`: структура шести графов, пересечения рёбер, спектральные и Leiden-разбиения с индексами качества;
- `k_sensitivity.json`: чувствительность к k для евклидова и пирсоновского правил;
- `icvi_table.csv` и `icvi_table.json`: SW, CH, обе конвенции S_Dbw, AVI, AVU, MQ для 43 опубликованных разбиений на одном опорном графе;
- `cluster_events.json`: события групп по месяцам для фиксированных типов (с сезонностью и без неё) и временного Leiden;
- `edge_rule_partitions.csv.gz`: метки всех МО на каждом графе;
- `provenance.json`: конфигурация, SHA-256 панели и модулей, время расчёта.

Повтор сравнивается с опубликованным каталогом скриптом `scripts/compare_edge_results.py`.

## 1. Подготовка данных

```sh
python -m pip install -r requirements-models.txt -r requirements-visuals.txt
python scripts/download_sources.py
python scripts/prepare.py
python scripts/prepare_graphs.py
python scripts/check_preparation.py
```

Загрузчик проверяет SHA-256 исходных архивов. При несовпадении хеша панели новая выгрузка не считается тем же набором данных.

## 2. Точная сборка опубликованного атласа

```sh
python scripts/describe_temporal_groups.py
python scripts/build_research_atlas.py --reference-run reports/experiments/2026-09-23-v2 --validation-run reports/experiments/2026-09-23-v2/validation --contest-run reports/review-2026-09-24/atlas_extension.json --map-data reports/contest-v3/municipal_map.json --edge-rules reports/edge-rules --temporal-groups reports/temporal-groups --external-municipal reports/external-municipal --output artifacts/atlas-preview/index.html
cmp artifacts/atlas-preview/index.html docs/index.html
```

Сборка использует опубликованные метки и результаты и ничего не переобучает. `describe_temporal_groups.py` за несколько секунд описывает группы временного Leiden по сохранённым меткам и пишет `reports/temporal-groups/summary.json`. Атлас открывается в современном браузере без сети; встроенный gzip-пакет распаковывается через `DecompressionStream`, флаг `--uncompressed` собирает несжатую версию.

Независимое повторение производных таблиц и карты:

```sh
python scripts/evaluate_published.py
python scripts/describe_frozen.py
python scripts/build_map.py --gpkg artifacts/sources/acquisition/t_dict_municipal_districts_poly.gpkg --output artifacts/rebuilt-map.json
```

## 3. Точное переобучение опорного KMeans4

```sh
python scripts/run_research.py --config configs/reproduce_kmeans4.json --execute-clustering
python scripts/run_research.py --compare-run runs/research-<напечатанный-идентификатор>
```

Первая команда обучает годовой KMeans4 с исходными параметрами (seed 1729, n_init 20). Вторая сверяет все 2 016 идентификаторов и меток с опубликованными, сообщает буквальное совпадение и совпадение с точностью до перестановки номеров групп. Ненулевой код возврата означает расхождение.

## 4. Полная исходная серия

```sh
python scripts/run_research.py --config configs/research_annual.json --execute-clustering
python scripts/run_research.py --config archive/configs/research_months.json --execute-clustering
python scripts/run_research.py --config archive/configs/research_stability.json --execute-clustering
python scripts/run_research.py --config configs/research_validation.json --execute-clustering
```

Годовая и месячная стадии дают 26 и 78 разбиений, стадия устойчивости 56 возмущений, валидация повторяет протокол 2024 года. Каждый запуск печатает новый каталог `runs/research-<timestamp>`; эти пути передаются в `archive/scripts/export_research_results.py` через `--annual`, `--months`, `--stability`, `--validation` и новый `--output`. Архивные результаты не перезаписываются.

## 5. Малые K, совместная и временная модели

```sh
python scripts/run_contest.py --config configs/followup_small_k.json --execute-clustering
python scripts/run_contest.py --config configs/followup_joint.json --execute-clustering
python scripts/run_contest.py --config configs/followup_stability.json --execute-clustering
python scripts/run_contest.py --config configs/followup_temporal_pilot.json --execute-clustering
python scripts/run_contest.py --config configs/followup_temporal.json --execute-clustering
python scripts/export_followup.py --run runs/review-20260924-small-k --run runs/review-20260924-joint --run runs/review-20260924-joint-stability --run runs/review-20260924-temporal-pilot --run runs/review-20260924-temporal --output reports/review-repeated
```

Порядок первых двух обязателен: совместная модель переиспользует KMeans2/3. Стадия устойчивости берёт кэш дорожного графа из совместной модели, сверяет хеши и делает восемь запусков с другими стартами и порядком обхода. Существующие каталоги не перезаписываются: для повтора сохраните копии конфигов с новыми `followup.output`, `baseline_dir` и `joint_dir`.

## 6. Внешняя проверка и иллюстрации

```sh
python scripts/external_validation_v4.py --output artifacts/external-repeated --bootstrap 500
python archive/scripts/build_review_figures.py --external-results artifacts/external-repeated/results.json --output-dir artifacts/figures-repeated
```

Внешний контроль использует фиксированные метки, регион, тип МО, уровень расходов и индекс доступности рынков; новой кластеризации он не обучает. Чтобы перенести пересчёт в атлас, передайте `--external-results artifacts/external-repeated/results.json` в `build_review_extension.py`; без флага используется опубликованный `reports/external-v4/results.json`.

## 7. Тесты и браузер

```sh
python -m unittest discover -s tests -v
python -m compileall -q sbercluster scripts
npm install --no-save --package-lock=false playwright@1.62.1
npx playwright install chromium
node tests/browser/map_regressions.cjs docs/index.html chromium
```

Основной тестовый прогон запускает `python -m unittest discover -s tests`. В чистом клоне проверки, которым нужна локальная панель, пропускаются до её подготовки. Браузерный тест использует настоящие щелчки, перетаскивания, касания и клавиатуру; вместо `chromium` можно указать `chrome` или `msedge`.

## 8. Docker и CI

```sh
docker build -t economic-neighbors .
docker run --rm economic-neighbors python -m unittest discover -s tests -v
```

Для обучения смонтируйте `data/processed` в `/app/data/processed` и выходной `runs` в `/app/runs`; для дорожных экспериментов нужен также `artifacts/sources/acquisition`. Аргумент сборки `SOURCE_REVISION` записывает идентификатор коммита в происхождение.

GitHub Actions собирает контейнер, обучает KMeans4 на Linux, сверяет метки и выполняет малый пилот. Число вычислительных потоков ограничено; необязательный [Windows supervisor](RESOURCE_LIMITS.md) включается флагом `--supervisor PROFILE`.

## 9. Региональный контроль, DMoN и временная сетка

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-dmon.txt
python -m unittest discover -s tests -p test_dmon.py -v
python -m unittest discover -s tests -p test_round2.py -v
python scripts/run_contest.py --config configs/round2_region_control.json --execute-clustering
python scripts/run_contest.py --config configs/round2_dmon_pilot.json --execute-clustering
python scripts/run_contest.py --config configs/round2_temporal_pilot.json --execute-clustering
python scripts/run_contest.py --config configs/round2_dmon.json --execute-clustering
python scripts/run_contest.py --config configs/round2_temporal_grid.json --execute-clustering
python scripts/export_round2.py --run runs/round2-20260924-region_control
python scripts/export_round2.py --run runs/round2-20260924-dmon
python scripts/export_round2.py --run runs/round2-20260924-temporal_grid
python archive/scripts/build_round2_figures.py
```

Серия использует опубликованные исходные разбиения и дорожную матрицу из `reports/graphs`; загрузчик проверяет порядок ID, признаки, параметры и SHA-256. Пилоты: DMoN K = 2, seed 1729, 200 эпох; временная модель на трёх месяцах, ω = 0,03, три seed. Временная сетка: ω = 0; 0,01; 0,03; 0,06; 0,1, три seed, 24 месяца, четыре итерации Leiden; два прежних результата переиспользуются с проверкой хешей. Workflow `Round two experiments` запускается вручную и пропускает завершённые этапы только после сверки хешей кода, конфигурации и файлов.

Региональный слой Росстата готовится отдельно, без обучения:

```sh
python archive/scripts/download_external_round2.py
python archive/scripts/prepare_external_round2.py
```

Правила сопоставления и охват описаны в `reports/external-round2/README.md`.

## 10. DMoN с остановкой по плато

```sh
python scripts/run_contest.py --config configs/dmon_plateau.json --execute-clustering
python scripts/export_round2.py --run runs/dmon-plateau-20260925 --output reports/dmon-plateau-2026-09-25 --compact-traces
python archive/scripts/build_dmon_plateau.py
python scripts/build_review_extension.py
```

Зависимости и тесты DMoN те же, что в разделе 9. Для повтора используйте новый выходной каталог: экспорт не перезаписывает опубликованные результаты. Правило остановки задано в конфиге до обучения: после 3 000 эпох каждые 500 эпох, три подряд окна с улучшением лучшего loss не более 0,0001 и сменой не более 0,1% меток лучшего checkpoint; предохранитель срабатывает на 50 000 эпох. Первые 1 000 эпох повторяются с тем же seed, чтобы восстановить состояние Adam; совпадение с прежней серией записано в `prefix_comparison.json`. Сборка HTML описана в разделе 2.

## 11. Население, зарплата и занятость по МО

```sh
python scripts/download_external_municipal.py --cafile russian_trusted_ca.pem
python scripts/prepare_external_municipal.py
python scripts/external_municipal_validation.py --config configs/external_municipal.json
```

Первая команда скачивает бюллетень Росстата и нужные CSV БД ПМО из архивов «Если быть точным» и сверяет их SHA-256 с `reports/external-municipal/sources.json`. Сайт rosstat.gov.ru использует сертификат Russian Trusted CA. Если его нет в системном хранилище, передайте PEM-бандл с корневым и промежуточным сертификатами через `--cafile` (https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt и `russian_trusted_sub_ca_2024_pem.crt`); без проблем с сертификатом флаг не нужен. Вторая команда сопоставляет МО по ОКТМО и пишет `reports/external-municipal/municipal_indicators.csv`, третья считает медианы по типам и η² (`summary.csv`, `results.json`). Атлас берёт эти таблицы через `--external-municipal reports/external-municipal` (раздел 2). Метки типов не пересчитываются. Тесты: `python -m unittest tests.test_external_municipal`.

## 12. Стык 2023/2024 и экономическая проверка типов

```sh
python scripts/january_check.py --config configs/january_check.json
python scripts/type_validation.py --config configs/type_validation.json
python -m unittest tests.test_type_validation
```

Первая команда сравнивает помесячные медианные доли категорий и размеры типов на стыке лет с изменениями внутри каждого года и пишет `reports/january-check/results.json`. Вторая заново строит три набора аналогов на 1 774 МО, сверяет их средние ошибки с опубликованными (2,679 / 2,721 / 2,812 п.п.) и считает bootstrap-интервалы по регионам. Затем она оценивает прирост R² от типа сверх региона и логарифма населения. Результат пишется в `reports/type-validation/results.json`, seed 1729 и число повторов заданы в конфиге. Расчёт занимает около полутора минут. Нужны подготовленная панель и `reports/external-municipal/municipal_indicators.csv` (раздел 11).

## 13. Устойчивость к базовому периоду и выбор K

```sh
python scripts/robustness_checks.py
python -m unittest tests.test_robustness
```

Параметры заданы в `configs/robustness.json`: три базы 2023 года (год, I и II полугодие), сетка K = 2…8, число бутстреп-повторов, seed, правило выбора K. Скрипт пишет `reports/robustness/results.json` и два рисунка, расчёт занимает около 20 секунд. Флаг `--no-figures` пропускает рисунки. В `reproduce.sh` этой команды нет: в `results.json` записывается время расчёта (`runtime_seconds`), поэтому файл при каждом запуске немного меняется. Остальные числа при том же seed повторяются. Нужна подготовленная панель.


## 14. Проверка исправленного технического ядра, 3 октября

После подготовки той же панели:

```sh
python -m unittest discover -s tests -v
python -m compileall -q sbercluster scripts
python scripts/verify_joint_refinement.py --output artifacts/joint-refinement-repeated --bootstrap 500
```

Последняя команда выполняет восемь ограниченных подгонок совместной модели: дорожные K2/K3/K4 и региональный K4, каждый из первоначального старта и сохранённого конечного разбиения. Параметры `α=0,25`, seed=1729 и максимум 50 проходов закреплены; перенос учитывает точное изменение обоих центров. Проверяются все допустимые одиночные переносы после остановки. Выходной каталог должен быть новым. DMoN и временная сетка этой командой не переобучаются.

Выход содержит метки, признаки, значения цели, внешнее сравнение с 500 парными региональными bootstrap-выборками, версии и копии исходных модулей, входные хеши и manifest. Проверенный снимок: [joint-refinement](../reports/technical-core-2026-10-03/joint-refinement/manifest.json); [изменения и границы проверки](../reports/technical-core-2026-10-03/README.md).

Текущие загрузчики сверяют идентичность панели и её manifest до вычислений. Сравнение KMeans требует завершённого запуска и уникальных ID. Ошибки и прерывания новых исследовательских запусков записываются в `status.json`; повторное использование требует согласованных настроек, источников и полного набора свидетельств. Экспорт сначала собирается во временном каталоге и появляется под окончательным именем только после успешной проверки. Исторические источники сохранены побайтно, поэтому новое ядро и старые архивы различаются по хешам; это не основание перезаписывать прежние результаты.

## 15. Применение замороженной модели без обучения

`FrozenProfileModel` применяет сохранённые центры и полную настройку признаков. Параметры не подгоняются на новых наблюдениях. Вход: CSV с `entity_id`, `territory_id`, `period` и шестью категориями исходной панели. Идентификатор должен иметь вид `tid_<territory_id>`, период задаётся первым числом месяца в формате `YYYY-MM-01`; повторные пары территория–месяц отклоняются.

```sh
python scripts/predict_frozen.py --model reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json --input data/processed/panel.csv --output artifacts/frozen-monthly
python scripts/predict_frozen.py --model reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json --input data/processed/panel.csv --aggregation annual --output artifacts/frozen-annual
python scripts/verify_frozen_inference.py --output artifacts/frozen-verification
```

Каждый выходной каталог должен быть новым. Годовое применение требует все 12 месяцев каждого года для каждой территории и использует медиану преобразованных месячных признаков. Проверочная команда сравнивает все месячные и оба годовых разбиения с архивом; обучение не запускается.

Для встроенного архивного пути модель автоматически проверяется по `SHA256.json`. Для собственной модели можно передать `--expected-model-sha256 <hash>`: ожидаемый хеш должен быть взят из доверенного источника. Без него произвольный модельный файл проходит структурную проверку, а его фактический хеш записывается в результат. Хеш-манифест связывает версии файлов, но не служит криптографической подписью автора.

Результат содержит `assignments.csv`, точную копию модели, исходники использованных модулей, версии зависимостей и манифест хешей. `relative_distance_margin` означает относительный разрыв расстояний до двух ближайших центров, а не вероятность. Новые территории разрешены и отмечаются `in_reference_cohort=false`; сопоставимость их границ и исходной статистики проверяется отдельно.

## 16. Сбой подготовки и восстановление

`python scripts/prepare.py` сначала проверяет источники и строит результаты во временном каталоге. Исходные файлы проверяются по хешам до чтения и перед публикацией. Одновременно разрешена одна подготовка на рабочую копию: файл `.prepare.lock` предотвращает второго писателя. При обычном сбое прежние результаты восстанавливаются, манифест заменяется последним. Отсутствующие координаты в официальном справочнике допустимы: они не входят в признаки модели.

При ошибке самого восстановления сохраняются каталог `.prepare-*` с резервными файлами и блокировка; исключение сообщает путь. Перед ручным восстановлением нужно убедиться, что процесс подготовки завершён, сверить резервные файлы и восстановить согласованный комплект данных с манифестом. Блокировку снимают после восстановления. Автоматически удалять её по возрасту нельзя. Замена набора файлов не атомарна при отключении питания; потребители должны запускаться после успешного завершения подготовки.

## 17. Национальная внешняя проверка

После подготовки исходной панели в среде `requirements.txt`:

```sh
python -m scripts.download_external_national_archive
python -m scripts.prepare_external_national
python -m scripts.validate_external_national --permutations 999 --bootstrap 1999
python -m scripts.validate_external_national_robustness
python -m scripts.summarize_external_national
```

Загрузчик проверяет закреплённые SHA-256 архивов Росстата и «Если быть точным». Ключи связываются по наблюдаемому ОКТМО и году, без подстановки современного кода вместо исторического. Экономические исходы не участвуют в обучении кластеров. В обоих годах исходов используются фиксированные расходные профили 2023 года; экономические регрессии обучаются отдельно. Перестановочные p-values описывают только исследовательский случайный контроль; основные результаты дают ошибки при исключении регионов и парные интервалы. [Полные определения, покрытие, первоисточник и ограничения](../reports/external-national-2026-10-03/README.md).

## 18. Применение годовой альтернативы Huber75

Основная модель остаётся прежней. Для отдельной годовой альтернативы с улучшенным силуэтом используется собственный формат и полное правило назначения, включающее смещения границ:

```sh
python -m scripts.predict_frontier --model reports/model-frontier-2026-10-03/models/sw_constrained_huber75.json --expected-sha256 41a1071c7d35019f93218d09b6b5abc293bec3b3b4fafdc740cf2c7ac572003a --panel data/processed/panel.csv --year 2023 --output runs/frontier-prediction
python -m scripts.verify_frontier_geometry --output runs/frontier-geometry-audit.json
```

Каталог предсказаний и файл аудита должны быть новыми. В выбранном году каждой территории нужны все 12 месяцев. Хешируется содержимое, которое действительно прочитано для применения; неверный формат или хеш отклоняется. Старый `predict_frozen.py` не принимает формат `prototype_frontier`.

Применение к новым годовым данным не гарантирует прежние размеры групп и индексы качества. Проверка отдельных месяцев обнаружила сезонные малые группы, поэтому альтернативе не приписывается гарантия месячной устойчивости. Выход описывает расходы, а не производственный тип или прогноз зарплаты. [Полный конечный эксперимент, конфигурации, контрастная альтернатива и команды обучения](../reports/model-frontier-2026-10-03/README.md).


## 19. Два временных канала и конечная проверка

Новый парный канал вычитает медиану изменений фиксированного референса; исходный канал сохраняется. Получение текущей поправки использует текущие записи референса, а применение зафиксированного пакета детерминировано. Оба режима описательные: 2024 просмотрен, неизменность границ не доказана. [Создание референса, пакетов и применение с проверкой хешей](TEMPORAL_CORE.md).

```powershell
python -m scripts.benchmark_dual_profiles --output runs/dual-reproduced
python -m scripts.benchmark_temporal_profiles --output runs/temporal-reproduced
python -m unittest discover -s tests -p test_dual_profiles.py -v
python -m unittest discover -s tests -p test_temporal_profiles.py -v
```

Каталоги должны быть новыми. Второй benchmark повторяет ровно конечный набор трёх исследовательских вариантов; все три не прошли совместный отбор, новых рекомендуемых моделей он не создаёт. [Исходные свидетельства и обе геометрии](../reports/temporal-core-2026-10-03/README.md).

## 20. Полная региональная экономическая проверка и границы пропусков

Нужны исходная панель и национальные `cohort-2023.csv`, `cohort-2024.csv`, `frozen-labels.json` с хешами из архивного протокола. Зарплатный сценарий исключает регион из всех новых обучающих стадий. Отраслевой сценарий повторно использует расходные разбиения и предсказывает также неизвестные исходы полного состава с доступными контролями.

```powershell
python -m scripts.validate_economic_generalization --output runs/economic-generalization-reproduced --include-h75
python -m scripts.validate_economic_sector_generalization --expense-folds runs/economic-generalization-reproduced --output runs/sector-generalization-reproduced
python -m unittest discover -s tests -p "test_economic*.py" -v
```

Для проверки стоимости первого зарплатного разбиения добавьте `--max-new-folds 1`; дальнейшая команда без ограничения переиспользует только кэш совпадающего протокола/входов/исходников. Изменение снимка требует нового каталога. Все прогнозы 2024 в этом сценарии используют только 2023 признаки и 2023 обучающие зарплаты; прежняя национальная проверка заново обучала экономическую модель на исходах каждого года и сохранена как другой эксперимент.

Один показатель расходов в контролях: log медианы месячного TOTAL 2023, не годовая сумма. Условные зарплатные bootstrap-интервалы и точные отраслевые границы неизвестных долей имеют разные смыслы. [Метод и ограничения](ECONOMIC_GENERALIZATION.md) · [полный пакет и manifest](../reports/economic-generalization-2026-10-03/README.md).

## 21. Автономная проверка поставляемого ядра

```sh
python -m scripts.verify_technical_core --output artifacts/technical-acceptance.json
```

Нужен новый выходной файл. Команда проверяет четыре пакета свидетельств, сохранённые правила и реальные CLI на небольшом синтетическом вводе, без скачивания данных и обучения. Для одной целостности файлов доступен `--integrity-only`; предсказания в этом режиме имеют статус `NOT_RUN`. Подробный контракт и коды завершения описаны в [TECHNICAL_ACCEPTANCE.md](TECHNICAL_ACCEPTANCE.md).
