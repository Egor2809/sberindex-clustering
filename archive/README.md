# Архив

Здесь лежат конфиги и скрипты прошлых этапов работы. Я убрал их из `configs/` и `scripts/`, чтобы там остался только действующий конвейер: то, что запускает `scripts/reproduce.sh`, что нужно тестам, коду `sbercluster`, CI и документам.

Что в архиве:

- `configs/research_months.json`, `configs/research_stability.json`: месячная стадия и стадия устойчивости исходной серии экспериментов 23 сентября;
- `configs/research_screen.json`, `configs/research_smoke.json`, `configs/contest_full.json`, `configs/pilot_bounded.json`, `configs/development_check_bounded.json`: ранние пилоты и проверочные запуски;
- `scripts/export_research_results.py`, `scripts/build_research_figures.py`: экспорт и рисунки исходной серии;
- `scripts/build_review_figures.py`, `scripts/build_round2_figures.py`, `scripts/build_dmon_plateau.py`: рисунки и сводки второго раунда и DMoN;
- `scripts/download_external_round2.py`, `scripts/prepare_external_round2.py`: загрузка и подготовка региональных данных Росстата;
- `scripts/build_dashboard.py`, `scripts/import_csv.py`: первая версия дашборда и импорт приватной CSV-выгрузки.

Результаты этих запусков лежат в `reports/` как исторические и не перезаписываются. Для проверки итога архив не нужен: `bash scripts/reproduce.sh` его не использует.

Скрипты запускаются из корня репозитория, например `python archive/scripts/build_round2_figures.py`, конфиги передаются как `--config archive/configs/research_months.json`. Пути внутри опубликованных `provenance.json` остались прежними (`configs/...`, `scripts/...`), потому что записаны в момент расчёта.
