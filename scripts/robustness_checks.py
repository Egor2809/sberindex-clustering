import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbercluster.features import make_slices
from sbercluster.io import CATEGORIES, sha256, write_json
from sbercluster.robustness import base_robustness, k_selection, market_trajectory, month_indices, ratio_tensor


def figures(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fafaf6', 'axes.facecolor': '#fafaf6'})
    market = result['market_trajectory']
    months = np.arange(24)
    labels = [p[2:7] for p in market['periods']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(months, market['median_share_pct'], color='#1f3a5f', marker='o', ms=3, label='Факт, медиана по МО')
    ext = market['extrapolation']
    ax.plot(np.arange(12, 24), ext['predicted_2024_pct'], color='#b5523b', ls='--', label='Тренд II полугодия 2023, продолжение')
    ax.axvspan(6, 11, color='#d9d4c7', alpha=.5, lw=0)
    ax.set(title='Маркетплейсы / все категории, %', xticks=months[::3], xticklabels=labels[::3])
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    colors = {'annual_2023': '#1f3a5f', 'h1_2023': '#6b8f71', 'h2_2023': '#b5523b'}
    names = {'annual_2023': 'База: год 2023', 'h1_2023': 'База: I полугодие 2023', 'h2_2023': 'База: II полугодие 2023'}
    for name, base in result['bases'].items():
        ax.plot(months, base['monthly_remote_counts'], color=colors.get(name, '#444'), marker='o', ms=3, label=names.get(name, name))
    ax.set(title='Число МО удалённого типа по месяцам', xticks=months[::3], xticklabels=labels[::3])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'base-sensitivity.png', dpi=150)
    plt.close(fig)
    rows = result['k_selection']['rows']
    gap = result['k_selection']['gap']
    ks = [r['K'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(ks, [r['SW'] for r in rows], marker='o', color='#1f3a5f')
    axes[0].set(title='Силуэт (SW)', xlabel='K')
    axes[1].errorbar(gap['K'], gap['gap'], yerr=gap['s'], marker='o', color='#1f3a5f', capsize=3)
    axes[1].set(title='Gap statistic ± s', xlabel='K')
    axes[2].plot(ks, [r['bootstrap_ari_median'] for r in rows], marker='o', color='#1f3a5f', label='Бутстреп МО')
    axes[2].plot(ks, [r['seed_ari_median'] for r in rows], marker='s', color='#b5523b', label='Смена seed')
    axes[2].axhline(.8, color='#888', lw=.8, ls=':')
    axes[2].set(title='Устойчивость, медианный ARI', xlabel='K', ylim=(0, 1.02))
    axes[2].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'k-selection.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/robustness.json')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    started = perf_counter()
    cfg = json.loads((ROOT / args.config).read_text('utf-8'))
    feature_cfg = json.loads((ROOT / cfg['feature_config']).read_text('utf-8'))
    path = ROOT / cfg['panel']
    manifest = json.loads((path.parent / 'manifest.json').read_text('utf-8'))
    if sha256(path) != manifest['panel_sha256']:
        raise ValueError('Panel snapshot differs')
    panel = pd.read_csv(path, dtype={'entity_id': str, 'period': str, 'oktmo': str})
    slices, scaler = make_slices(panel, feature_cfg)
    ids = slices[0][1]
    if any(s[1] != ids for s in slices) or len(slices) != 24:
        raise ValueError('Expected 24 aligned months')
    periods = [s[0] for s in slices]
    monthly = np.stack([s[2] for s in slices])
    frozen = json.loads((ROOT / cfg['reference']).read_text('utf-8'))
    if frozen['ids'] != ids:
        raise ValueError('Frozen IDs differ')
    reference = np.array(frozen['reference_labels'])
    centers = np.array(frozen['centers'])
    seed, k, n_init = cfg['seed'], cfg['kmeans']['k'], cfg['kmeans']['n_init']
    trend_idx = month_indices(periods, cfg['trend_window'])
    if month_indices(periods, cfg['comparison']) != list(range(12, 24)):
        raise ValueError('Comparison window must be calendar 2024')
    market = market_trajectory(panel, periods, cfg['market_category'], cfg['trend_window'])
    bases, shifts = base_robustness(monthly, ratio_tensor(panel, periods, ids), reference, centers, cfg['bases'],
                                    periods, trend_idx, cfg['remote_type'], seed, n_init, k)
    annual = np.median(monthly[month_indices(periods, cfg['bases']['annual_2023'])], axis=0)
    selection, _ = k_selection(annual, reference, cfg['k_selection'], seed, n_init)
    result = {'config': args.config, 'panel_sha256': manifest['panel_sha256'], 'reference_sha256': frozen['freeze_sha256'],
              'n': len(ids), 'features': CATEGORIES, 'feature_scaler': scaler,
              'reference_sizes': np.bincount(reference, minlength=k).tolist(),
              'market_trajectory': market, 'national_shifts_standardized': shifts, 'bases': bases,
              'k_selection': selection, 'runtime_seconds': None}
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures(result, out)
    result['runtime_seconds'] = round(perf_counter() - started, 1)
    write_json(out / 'results.json', result)
    print(json.dumps({'runtime_seconds': result['runtime_seconds'], 'chosen_K': selection['selection']['chosen_K'],
                      'gap_one_se_K': selection['gap']['one_se_K']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
