"""Small inference-only regressions; no real outcome data or fitting."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sbercluster.continuous_atlas import (
    boundary_radius, convex_encode, encode_panel, load_model, read_panel_csv,
)
from sbercluster.io import CATEGORIES, TOTAL


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = '6000163eb591530a6cb8135c6e3c3c0bd5e8f1ee17f7b2eaa92b38304b5bae38'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContinuousGeometryTests(unittest.TestCase):
    def test_projection_matches_independent_constrained_optimizer(self):
        rng = np.random.default_rng(382)
        centers = rng.normal(size=(4, 5))
        points = np.vstack([centers, centers.mean(axis=0), rng.normal(size=(7, 5))])
        result = convex_encode(points, centers)
        for point, actual in zip(points, result['reconstructed']):
            solution = minimize(lambda w: np.sum((w @ centers - point) ** 2),
                                np.full(4, .25), method='SLSQP',
                                bounds=[(0., 1.)] * 4,
                                constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.},
                                options={'ftol': 1e-12, 'maxiter': 300})
            self.assertTrue(solution.success, solution.message)
            np.testing.assert_allclose(actual, solution.x @ centers, atol=2e-6)
        np.testing.assert_allclose(result['weights'].sum(axis=1), 1., atol=1e-12)
        self.assertTrue((result['weights'] >= 0.).all())
        np.testing.assert_allclose(result['reconstructed'] + result['residual'], points)

    def test_analytic_projection_duplicate_vertices_and_boundary(self):
        centers = np.array([[0., 0.], [2., 0.], [0., 2.], [0., 0.]])
        points = np.array([[.5, .5], [2., 2.], [-1., -1.]])
        result = convex_encode(points, centers)
        np.testing.assert_allclose(result['reconstructed'], [[.5, .5], [1., 1.], [0., 0.]], atol=1e-12)
        np.testing.assert_allclose(boundary_radius([[0.], [1.], [2.]], [[0.], [2.]]), [1., 0., 1.])
        np.testing.assert_allclose(boundary_radius([[0.]], [[0.], [0.]]), [0.])


class ContinuousAtlasTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = load_model(ROOT / 'models/continuous_atlas.json', MODEL_SHA)
        self.model_path = self.root / 'model.json'
        self.model_path.write_text(json.dumps(self.model), encoding='utf-8')
        scaler = self.model['feature_scaler']
        self.target = np.array([.21, -.17, .32, .04, -.08])
        values = 100. * np.exp(np.array(scaler['ratio_center']) +
                               np.array(scaler['ratio_iqr']) * np.sqrt(5) * self.target)
        self.panel = pd.DataFrame([
            {'entity_id': 'tid_999999', 'territory_id': '999999', 'period': f'2023-{m:02d}-01',
             TOTAL: 100., **dict(zip(CATEGORIES, values))} for m in range(1, 13)])
        self.panel_path = self.root / 'panel.csv'
        self.panel.to_csv(self.panel_path, index=False)

    def test_all_codecs_preserve_coordinates_and_original_label_rules(self):
        original = copy.deepcopy(self.model)
        for codec in ('legacy_centers', 'observed_full', 'observed_core95'):
            with self.subTest(codec=codec):
                result = encode_panel(self.panel, self.model, 2023, codec)
                point = result[[f'x{j}' for j in range(5)]].to_numpy()[0]
                np.testing.assert_allclose(point, self.target, atol=1e-14)
                weights = result[[f'weight{j}' for j in range(4)]].to_numpy()
                reconstruction = result[[f'reconstructed{j}' for j in range(5)]].to_numpy()
                residual = result[[f'residual{j}' for j in range(5)]].to_numpy()
                np.testing.assert_allclose(weights @ self.model['codecs'][codec]['centers'], reconstruction)
                np.testing.assert_allclose(reconstruction + residual, [self.target], atol=1e-14)
                self.assertAlmostEqual(float(weights.sum()), 1.)
                self.assertTrue((weights >= 0.).all())
                self.assertAlmostEqual(result.residual_norm.iloc[0], np.linalg.norm(residual))
                self.assertTrue(result.new_entity.iloc[0])
                self.assertEqual(result.month_coverage.iloc[0], 12)
                self.assertEqual(result.weight_semantics.iloc[0], 'convex_coordinates_not_probabilities')
                self.assertEqual(result.territory_transition_status.iloc[0], 'unresolved')
                for name, rule in self.model['legacy_rules'].items():
                    scores = np.square(np.array(rule['centers']) - point).sum(axis=1) + rule['biases']
                    self.assertEqual(result[name].iloc[0], int(np.argmin(scores)))
        self.assertEqual(self.model, original)

    def test_calendar_ids_expenses_and_explicit_selection_rejections(self):
        bad_panels = [self.panel.iloc[:-1], pd.concat([self.panel, self.panel.iloc[[0]]])]
        for column, value in [('entity_id', 'tid_01'), ('territory_id', '01'),
                              ('period', '2023-01-02'), ('period', '2025-01-01'),
                              (TOTAL, 0.), (TOTAL, float('inf')), (CATEGORIES[0], -1.),
                              (CATEGORIES[0], float('nan')), (CATEGORIES[0], 'bad')]:
            bad = self.panel.copy()
            if isinstance(value, str) and column in CATEGORIES + [TOTAL]:
                bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            bad_panels.append(bad)
        for bad in bad_panels:
            with self.subTest(first_row=bad.iloc[0].to_dict()), self.assertRaises(ValueError):
                encode_panel(bad, self.model, 2023, 'observed_full')
        for year, codec in [(2025, 'observed_full'), (2023, 'implicit')]:
            with self.assertRaises(ValueError):
                encode_panel(self.panel, self.model, year, codec)

    def test_year_selection_monthly_median_and_labels_are_not_weight_argmax(self):
        model = copy.deepcopy(self.model)
        # Controlled five-dimensional segment: interior point has largest weight
        # at vertex zero, while a saved bias makes legacy label one win.
        centers = [[0.] * 5, [1.] * 5, [2.] * 5, [3.] * 5]
        model['codecs']['legacy_centers']['centers'] = centers
        for rule in model['legacy_rules'].values():
            rule['centers'] = centers
            rule['biases'] = [0., -20., 0., 0.]
        scaler = model['feature_scaler']
        panel = self.panel.copy()
        for month in range(12):
            target = .1 if month < 7 else .9
            values = 100. * np.exp(np.array(scaler['ratio_center']) +
                                  np.array(scaler['ratio_iqr']) * np.sqrt(5) * target)
            panel.loc[month, CATEGORIES] = values
        other = panel.copy(); other['period'] = other.period.str.replace('2023', '2024')
        result = encode_panel(pd.concat([other, panel.iloc[::-1]]), model, 2023, 'legacy_centers')
        np.testing.assert_allclose(result[[f'x{j}' for j in range(5)]], [[.1] * 5], atol=1e-14)
        self.assertEqual(np.argmax(result[[f'weight{j}' for j in range(4)]].to_numpy()[0]), 0)
        self.assertEqual(result.historical_kmeans4.iloc[0], 1)
        selected = encode_panel(other, model, 2024, 'legacy_centers')
        self.assertEqual(selected.year.iloc[0], 2024)

    def test_model_validation_and_hash_pins(self):
        for field, value in [('revision', 'unknown'), ('training_year', 2024),
                             ('training_entity_ids', ['tid_1', 'tid_1']),
                             ('training_entity_ids', ['not_a_source_id'])]:
            model = copy.deepcopy(self.model); model[field] = value
            self.check_invalid_model(model)
        for field, value in [('ratio_iqr', [0.] * 5), ('ratio_center', [0.] * 4),
                             ('level_weight', 1.), ('calibration_end', '2024-12-01')]:
            model = copy.deepcopy(self.model); model['feature_scaler'][field] = value
            self.check_invalid_model(model)
        for group, name, field, value in [
                ('codecs', 'observed_full', 'centers', [[0.] * 4] * 4),
                ('legacy_rules', 'historical_kmeans4', 'biases', [0.] * 3),
                ('legacy_rules', 'historical_kmeans4', 'transform', [[0.] * 5] * 5)]:
            model = copy.deepcopy(self.model); model[group][name][field] = value
            self.check_invalid_model(model)
        for content in ['{"revision":"a","revision":"b"}', '{"value":NaN}', '{"value":Infinity}']:
            self.model_path.write_text(content, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_model(self.model_path, digest(self.model_path))
        for pin in ['0' * 64, 'bad', 'A' * 64]:
            with self.assertRaises(ValueError):
                load_model(self.model_path, pin)
            with self.assertRaises(ValueError):
                read_panel_csv(self.panel_path, pin)

    def check_invalid_model(self, model):
        self.model_path.write_text(json.dumps(model), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_model(self.model_path, digest(self.model_path))

    def test_csv_structure_rejections(self):
        source = self.panel_path.read_text(encoding='utf-8').splitlines()
        cases = ['\n'.join([source[0] + ',entity_id'] + source[1:]),
                 '\n'.join([source[0].replace('entity_id', 'unknown', 1)] + source[1:]),
                 '\n'.join([source[0], source[1] + ',extra'] + source[2:]),
                 '\n'.join([source[0], source[1].rsplit(',', 1)[0]] + source[2:]),
                 '\n'.join([source[0], '"unterminated'] + source[2:])]
        for content in cases:
            with self.subTest(content=content[:100]):
                self.panel_path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_panel_csv(self.panel_path, digest(self.panel_path))

    def cli(self, output, *extra):
        return subprocess.run([sys.executable, '-m', 'scripts.predict_continuous',
            '--model', str(self.model_path), '--expected-model-sha256', digest(self.model_path),
            '--panel', str(self.panel_path), '--expected-panel-sha256', digest(self.panel_path),
            '--year', '2023', '--codec', 'observed_full', '--output', str(output), *extra],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')

    def test_cli_success_no_clobber_and_refusal_leaves_no_output(self):
        output = self.root / 'result'
        run = self.cli(output)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        metadata = json.loads(before['metadata.json'])
        self.assertEqual(metadata['operation'], 'inference_only_no_fit')
        self.assertEqual(metadata['csv_sha256'], hashlib.sha256(before['atlas.csv']).hexdigest())
        self.assertNotEqual(self.cli(output).returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        for extra in [('--year', '2025'), ('--expected-panel-sha256', '0' * 64)]:
            target = self.root / ('refused' + str(len(extra[1])))
            run = self.cli(target, *extra)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
