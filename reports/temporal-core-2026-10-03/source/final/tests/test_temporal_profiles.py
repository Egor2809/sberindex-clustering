"""Independent invariants of temporal calibration and annual aggregation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sbercluster.io import CATEGORIES, TOTAL
from sbercluster.temporal_profiles import (fit_temporal_calibration, transform_temporal_rows,
    fit_temporal_model, predict_temporal_panel, validate_temporal_model, batch_relative_context)
from scripts.predict_temporal import predict_file


class TemporalProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/research_validation.json').read_text('utf-8'))
        rng = np.random.default_rng(1987)
        cls.periods = [f'2023-{m:02d}-01' for m in range(1, 13)]
        rows = []
        fixed = rng.normal(size=(20, 5)) * .3
        for i in range(20):
            for m, period in enumerate(cls.periods):
                ratio = np.exp(-3 + fixed[i] + .12 * np.sin(m + np.arange(5)) + rng.normal(size=5) * .01)
                rows.append({'entity_id': f'tid_{i}', 'territory_id': str(i), 'period': period,
                             TOTAL: 1000., **dict(zip(CATEGORIES, 1000. * ratio))})
        cls.panel = pd.DataFrame(rows)
        cls.artifact = fit_temporal_model(cls.panel, cls.cfg, .5)

    def test_metric_precision_amplifies_less_temporally_noisy_axis(self):
        rng = np.random.default_rng(131)
        x = rng.normal(size=(12, 80, 2)) * np.array([.05, 2.])
        calibration = fit_temporal_calibration(x, self.periods, 1.)
        metric = np.asarray(calibration['metric'])
        self.assertGreater(metric[0, 0], metric[1, 1])
        self.assertGreater(np.linalg.eigvalsh(metric).min(), 0.)
        self.assertAlmostEqual(float(np.sum(metric ** 2)), 2., places=12)

    def test_calendar_control_removes_only_saved_seasonality(self):
        rng = np.random.default_rng(98)
        entities = rng.normal(size=(40, 3))
        calendar = rng.normal(size=(12, 3))
        x = entities[None, :, :] + calendar[:, None, :]
        calibration = fit_temporal_calibration(x, self.periods, 0.)
        transformed = np.stack([transform_temporal_rows(rows, [period] * len(rows), calibration)
                                for period, rows in zip(self.periods, x)])
        np.testing.assert_allclose(transformed, np.broadcast_to(transformed[0], transformed.shape), atol=1e-14)
        shifted = transform_temporal_rows(x[0] + 2., ['2024-01-01'] * len(entities), calibration)
        np.testing.assert_allclose(shifted - transformed[0], 2., atol=1e-14)

    def test_preaggregation_calendar_transform_does_not_commute_with_median(self):
        rng = np.random.default_rng(1)
        calibration = fit_temporal_calibration(rng.normal(size=(12, 8, 2)), self.periods, 0.)
        offsets = np.zeros((12, 2)); offsets[:, 0] = [-6.] * 4 + [3.] * 8
        calibration['calendar_offsets'] = offsets.tolist()
        rows = np.zeros((12, 2)); rows[:, 0] = [0.] * 8 + [9.] * 4
        correct = np.median(transform_temporal_rows(rows, self.periods, calibration), axis=0)
        wrong = np.median(rows, axis=0) - np.median(offsets, axis=0)
        self.assertAlmostEqual(correct[0], 6.)
        self.assertAlmostEqual(wrong[0], -3.)

    def test_future_values_cannot_change_fitted_artifact(self):
        future = self.panel.copy(); future['period'] = future.period.str.replace('2023', '2024')
        future[CATEGORIES + [TOTAL]] *= 1.e7
        combined = pd.concat([self.panel, future], ignore_index=True)
        fitted = fit_temporal_model(combined, self.cfg, .5)
        self.assertEqual(json.dumps(fitted, sort_keys=True), json.dumps(self.artifact, sort_keys=True))

    def test_complete_year_batch_and_order_invariance(self):
        full = predict_temporal_panel(self.panel, self.artifact, 2023)
        shuffled = predict_temporal_panel(self.panel.sample(frac=1., random_state=17), self.artifact, 2023)
        pd.testing.assert_frame_equal(full, shuffled)
        target = self.panel[self.panel.entity_id == 'tid_0']
        pd.testing.assert_frame_equal(full[full.entity_id == 'tid_0'].reset_index(drop=True), predict_temporal_panel(target, self.artifact, 2023))

    def test_new_entity_is_flagged_and_source_key_mismatch_rejected(self):
        changed = self.panel[self.panel.entity_id == 'tid_0'].copy()
        changed['entity_id'], changed['territory_id'] = 'tid_999', '999'
        prediction = predict_temporal_panel(changed, self.artifact, 2023)
        self.assertTrue(prediction.new_entity.all())
        changed['territory_id'] = '998'
        with self.assertRaisesRegex(ValueError, 'source territory'):
            predict_temporal_panel(changed, self.artifact, 2023)

    def test_invalid_training_source_identity_rejected_before_fit(self):
        training = self.panel.copy(); training.loc[0, 'territory_id'] = '999'
        with self.assertRaisesRegex(ValueError, 'source territory'):
            fit_temporal_model(training, self.cfg, .5)

    def test_missing_month_duplicate_and_partial_calibration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'all12'):
            predict_temporal_panel(self.panel.iloc[:-1], self.artifact, 2023)
        with self.assertRaisesRegex(ValueError, 'unique'):
            predict_temporal_panel(pd.concat([self.panel, self.panel.iloc[:1]]), self.artifact, 2023)
        cfg = copy.deepcopy(self.cfg); cfg['features']['calibration_end'] = '2023-06-01'
        with self.assertRaisesRegex(ValueError, 'all12'):
            fit_temporal_model(self.panel, cfg, .5)

    def test_model_metadata_and_hash_guards(self):
        bad = copy.deepcopy(self.artifact); bad['calibration']['current_period_recalibration'] = True
        with self.assertRaises(ValueError): validate_temporal_model(bad)
        bad = copy.deepcopy(self.artifact); bad['training_entity_ids'][0] = 'tid_invalid'
        with self.assertRaises(ValueError): validate_temporal_model(bad)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'; model.write_text(json.dumps(self.artifact), encoding='utf-8')
            panel = Path(directory) / 'panel.csv'; self.panel.to_csv(panel, index=False)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                predict_file(model, panel, Path(directory) / 'output', 2023, '0' * 64)

    def test_cli_uses_one_exact_read_for_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'; raw = json.dumps(self.artifact).encode(); model.write_bytes(raw)
            panel = Path(directory) / 'panel.csv'; self.panel.to_csv(panel, index=False); panel_raw = panel.read_bytes()
            real_read = Path.read_bytes
            counts = {model: 0, panel: 0}
            def read_once(path):
                if path in counts:
                    counts[path] += 1
                    if counts[path] > 1: raise AssertionError('Read input twice')
                return real_read(path)
            with patch.object(Path, 'read_bytes', read_once):
                predict_file(model, panel, Path(directory) / 'output', 2023, hashlib.sha256(raw).hexdigest())
            provenance = json.loads((Path(directory) / 'output/provenance.json').read_text())
            self.assertEqual(provenance['input_sha256'], hashlib.sha256(panel_raw).hexdigest())
            self.assertEqual(counts, {model: 1, panel: 1})

    def test_relative_context_reports_a_shift_it_removes(self):
        rng = np.random.default_rng(19); original = rng.normal(size=(51, 3))
        shared = np.array([.7, -.3, .2])
        context = batch_relative_context(original + shared, original + shared, np.median(original, axis=0))
        np.testing.assert_allclose(context['corrected_features'], original, atol=1e-14)
        np.testing.assert_allclose(context['common_location_drift'], shared, atol=1e-14)
        self.assertIn('transductive', context['scope'])
        subset = batch_relative_context(original + shared, (original + shared)[:10], np.median(original, axis=0))
        self.assertGreater(float(np.linalg.norm(subset['common_location_drift'] - shared)), .1)


if __name__ == '__main__': unittest.main()
