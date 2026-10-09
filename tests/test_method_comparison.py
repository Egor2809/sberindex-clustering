import json
import unittest
from pathlib import Path

from sbercluster.method_comparison import AXES, FIRST_SELECTION, axis_role, family, neutral_rank, winners

ROOT = Path(__file__).resolve().parents[1]


def row(name, k, min_size, **values):
    _, fitted = family(name)
    out = {'partition': name, 'family': family(name)[0], 'k': k, 'min_size': min_size}
    for axis in AXES:
        out[f'{axis}_role'] = axis_role(axis, fitted, FIRST_SELECTION.get(name, set()))
    out.update(values)
    return out


class MethodComparisonTests(unittest.TestCase):
    def test_fitted_axes_are_marked_circular(self):
        self.assertEqual(family('kmeans_k4')[1], {'X23', 'P23'})
        self.assertIn('R', family('joint_road_k4')[1])
        self.assertIn('R', family('dmon_k4_seed1729')[1])
        self.assertIn('C23', family('network_typology')[1])
        self.assertTrue({'R', 'X24', 'P24'} <= family('temporal_leiden_seed1729')[1])
        self.assertEqual(axis_role('C24', family('network_typology')[1]), 'year_2024')
        self.assertEqual(axis_role('R', family('kmeans_k4')[1]), 'external')
        with self.assertRaises(ValueError):
            family('unknown_method')

    def test_winner_ignores_fitted_axis_and_small_groups(self):
        rows = [row('joint_road_k4', 4, 99, R_MQ=0.9), row('kmeans_k4', 4, 127, R_MQ=0.3),
                row('ward_k4', 4, 138, R_MQ=0.2), row('spectral_global_k4', 4, 8, R_MQ=0.8)]
        best = winners(rows, 4, {'R': ('MQ',)}, 20)['R_MQ']
        self.assertEqual(best['partition'], 'kmeans_k4')
        self.assertEqual(best['candidates'], 2)

    def test_spectral3_and_registry_families(self):
        self.assertEqual(family('spectral3_profile_comovement')[1], {'X23', 'P23', 'C23'})
        self.assertEqual(family('registry_type3')[1], set())

    def test_neutral_rank_skips_fitted_and_first_selection_axes(self):
        keys = ['R_MQ', 'X24_SW']
        rows = [row('kmeans_k3', 3, 496, R_MQ=0.30, X24_SW=0.24), row('network_typology', 3, 249, R_MQ=0.21, X24_SW=0.26),
                row('joint_road_k3', 3, 391, R_MQ=0.36, X24_SW=0.25)]
        self.assertEqual(rows[1]['X24_role'], 'selection')
        ranked = {r['partition']: r for r in neutral_rank(rows, 3, keys, 20)}
        self.assertEqual(ranked['network_typology']['places'], {'R_MQ': [2, 2]})
        self.assertEqual(ranked['joint_road_k3']['places'], {'X24_SW': [1, 2]})
        self.assertEqual(ranked['kmeans_k3']['mean_rank'], 1.5)
        self.assertEqual(winners(rows, 3, {'X': ('SW',)}, 20)['X24_SW']['partition'], 'joint_road_k3')

    def test_degenerate_avu_is_not_ranked(self):
        rows = [row('kmeans_k3', 3, 496, R_AVU=2 / 3, R_MQ=0.3), row('ward_k3', 3, 289, R_AVU=2 / 3 + 1e-16, R_MQ=0.2)]
        best = winners(rows, 3, {'R': ('AVU', 'MQ')}, 20)
        self.assertNotIn('R_AVU', best)
        self.assertIn('R_MQ', best)

    def test_published_summary_is_consistent(self):
        path = ROOT / 'reports/method-comparison/summary.json'
        if not path.exists():
            self.skipTest('method comparison not published')
        summary = json.loads(path.read_text('utf-8'))
        for k, best in summary['winners'].items():
            for key, item in best.items():
                self.assertGreaterEqual(item['candidates'], 2, (k, key))
        seeds = summary['temporal_seeds']
        self.assertEqual(set(seeds['per_seed']), {'1729', '2718', '3141'})
        self.assertEqual(seeds['per_seed']['1729']['stable'], 7)
        self.assertLessEqual(seeds['base_groups_found_in_all_seeds'], seeds['per_seed']['1729']['stable'])
        for k, ranked in summary['neutral_rank'].items():
            for r in ranked:
                self.assertFalse(set(FIRST_SELECTION.get(r['partition'], ())) & {key.split('_')[0] for key in r['places']}, (k, r['partition']))
        self.assertIn('P23_AVU', summary['degenerate']['3'])


if __name__ == '__main__':
    unittest.main()
