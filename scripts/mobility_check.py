"""SberIndex consumer mobility index (open data, municipal level) as an external reference for the temporal groups and KMeans4."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NORTHWEST = ('Архангельская область', 'Вологодская область', 'Калининградская область', 'Ленинградская область',
             'Мурманская область', 'Ненецкий автономный округ', 'Новгородская область', 'Псковская область',
             'Республика Карелия', 'Республика Коми', 'Санкт-Петербург')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/temporal_groups.json')
    p.add_argument('--download', action='store_true', help='Fetch the dataset through the SberIndex API and check its SHA-256')
    p.add_argument('--cafile', help='PEM bundle for the site certificate chain')
    p.add_argument('--source', default='artifacts/sources/mobility/indeks-mobilnosti.csv')
    p.add_argument('--sources', default='reports/mobility-check/sources.json')
    p.add_argument('--year', type=int, default=2024)
    p.add_argument('--output', default='reports/mobility-check')
    args = p.parse_args()
    from sbercluster.mobility_check import download, run
    if args.download:
        print(download(ROOT, args.source, args.sources, args.cafile))
    else:
        cfg = json.loads((ROOT / args.config).read_text('utf-8'))
        print(run(cfg, ROOT, args.source, args.sources, args.output, args.year, NORTHWEST))
