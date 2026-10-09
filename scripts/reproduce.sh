#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "== 1. Зависимости"
"$PYTHON" -m pip install -r requirements-models.txt -r requirements-visuals.txt

echo "== 2. Исходные архивы и проверка SHA-256"
"$PYTHON" scripts/download_sources.py

echo "== 3. Панель и помесячные графы"
"$PYTHON" scripts/prepare.py
"$PYTHON" scripts/prepare_graphs.py
"$PYTHON" scripts/check_preparation.py

echo "== 4. Тесты"
"$PYTHON" -m unittest discover -s tests

echo "== 5. Переобучение опорного KMeans4 и сверка меток"
"$PYTHON" scripts/run_research.py --config configs/reproduce_kmeans4.json --execute-clustering
"$PYTHON" scripts/run_research.py --compare-run "$(ls -td runs/research-* | head -1)"

echo "== 6. Сравнение правил рёбер и сверка с reports/edge-rules"
"$PYTHON" scripts/compare_edge_rules.py --output artifacts/edge-rules-repeated
"$PYTHON" scripts/compare_edge_results.py reports/edge-rules artifacts/edge-rules-repeated

echo "== 7. Группы временного Leiden"
"$PYTHON" scripts/describe_temporal_groups.py
"$PYTHON" scripts/group_profiles.py > /dev/null

echo "== 8. Сетевая типология и сверка с reports/network-core"
"$PYTHON" scripts/network_core.py --config configs/network_core.json --output artifacts/network-core-repeated
for f in assignments.csv candidates.csv summary.json profiles.json; do cmp "reports/network-core/$f" "artifacts/network-core-repeated/$f"; done
echo "Network typology matches the published files"

echo "== 9. Качество помесячных разбиений и сверка с reports/temporal-quality"
"$PYTHON" scripts/temporal_quality.py --config configs/temporal_quality.json --output artifacts/temporal-quality-repeated
for f in summary.json monthly_icvi.csv omega_tradeoff.csv; do cmp "reports/temporal-quality/$f" "artifacts/temporal-quality-repeated/$f"; done
echo "Monthly quality matches the published files"

echo "== 10. Сравнение всех методов на общих осях и сверка с reports/method-comparison"
"$PYTHON" scripts/method_comparison.py --output artifacts/method-comparison-repeated
for f in table.csv summary.json README.md; do cmp "reports/method-comparison/$f" "artifacts/method-comparison-repeated/$f"; done
echo "Method comparison matches the published files"

echo "== 11. Сборка атласа и побайтовое сравнение с docs/index.html"
"$PYTHON" scripts/build_research_atlas.py --reference-run reports/experiments/2026-09-23-v2 --validation-run reports/experiments/2026-09-23-v2/validation --contest-run reports/review-2026-09-24/atlas_extension.json --map-data reports/contest-v3/municipal_map.json --edge-rules reports/edge-rules --temporal-groups reports/temporal-groups --external-municipal reports/external-municipal --network-core reports/network-core --temporal-quality reports/temporal-quality --output artifacts/atlas-preview/index.html
cmp artifacts/atlas-preview/index.html docs/index.html && echo "Atlas rebuilt byte-for-byte"

echo "== 12. Автономная проверка ядра: сохранённые модели и применение без обучения"
"$PYTHON" -m scripts.verify_technical_core --output "artifacts/technical-acceptance-$(date +%Y%m%dT%H%M%S).json"
