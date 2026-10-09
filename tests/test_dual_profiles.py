"""Paired estimand, immutable reference context and conditional uncertainty."""
import copy
import itertools
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sbercluster.io import CATEGORIES, TOTAL
from sbercluster.selection_artifacts import make_artifact
from sbercluster.dual_profiles import (build_reference_lock, build_reference_packet, predict_dual_panel,
    median_missing_bounds, classify_profile_box, validate_reference_lock, validate_reference_packet,
    gaussian_mmd2, _seal)
from scripts.predict_dual import write_immutable_json, read_pinned_json, prediction_records
import hashlib
import tempfile


class DualProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scaler = {'mode': 'log_ratios_to_total', 'level_weight': 0., 'calibration_end': '2023-12-01',
                      'ratio_center': [0.] * 5, 'ratio_iqr': [1.] * 5, 'level_center': 0., 'level_iqr': 1.}
        cls.rule = {'revision': 'prototype_frontier_v1', 'centers': [[p, 0., 0., 0., 0.] for p in (-3., -1., 1., 3.)],
                    'transform': np.eye(5).tolist(), 'biases': [0.] * 4}
        cls.frozen = make_artifact(cls.rule, cls.scaler, provenance={'scope': 'synthetic_not_real_panel'})
        profiles = [-3.4, -2.6, -1.4, -.6, .6, 1.4, 2.6, 3.4]
        cls.base = pd.DataFrame([{'entity_id': f'tid_{i}', 'territory_id': str(i), 'region_code': str(i % 2),
                                  'period': f'2023-{m:02d}-01', TOTAL: 1.,
                                  **{name: float(np.exp(np.sqrt(5) * (p if j == 0 else 0.))) for j, name in enumerate(CATEGORIES)}}
                                 for i, p in enumerate(profiles) for m in range(1, 13)])
        cls.lock = build_reference_lock(cls.base, cls.frozen)
        cls.current = cls.base.copy(); cls.current.period = cls.current.period.str.replace('2023', '2024')
        cls.current[CATEGORIES[0]] *= np.exp(np.sqrt(5))
        cls.packets = {f'2024-{m:02d}-01': build_reference_packet(cls.current, cls.lock, f'2024-{m:02d}-01') for m in range(1, 13)}

    def test_real_shared_shift_absolute_transitions_and_relative_normalization_both_retained(self):
        base = predict_dual_panel(self.base, self.lock, 2023)
        current = predict_dual_panel(self.current, self.lock, 2024, self.packets)
        self.assertEqual(int((base.absolute_label != current.absolute_label).sum()), 3)
        np.testing.assert_array_equal(current.relative_label, base.absolute_label)
        for removed in current.effective_removed: np.testing.assert_allclose(removed, [1., 0., 0., 0., 0.], atol=1e-14)
        self.assertTrue(current.transition_status.eq('unresolved').all())

    def test_paired_median_not_difference_of_population_medians(self):
        base = self.base[self.base.entity_id.isin(['tid_0', 'tid_1', 'tid_2'])].copy()
        for entity, p in zip(['tid_0', 'tid_1', 'tid_2'], [0., 10., 11.]):
            base.loc[base.entity_id == entity, CATEGORIES[0]] = np.exp(np.sqrt(5) * p)
        lock = build_reference_lock(base, self.frozen)
        current = base.copy(); current.period = current.period.str.replace('2023', '2024')
        for entity, p in zip(['tid_0', 'tid_1', 'tid_2'], [9., 10., 20.]):
            current.loc[current.entity_id == entity, CATEGORIES[0]] = np.exp(np.sqrt(5) * p)
        packet = build_reference_packet(current, lock, '2024-01-01')
        self.assertAlmostEqual(packet['point_paired_median'][0], 9.)
        self.assertAlmostEqual(packet['population_median_difference'][0], 0.)

    def test_missing_bounds_cover_all_enumerated_completions(self):
        for n in range(1, 7):
            for missing in range(n + 1):
                observed = np.arange(n - missing, dtype=float)[:, None]
                lower, upper = median_missing_bounds(observed, n)
                for completion in itertools.product([-9., 0., 9.], repeat=missing):
                    truth = np.median(np.concatenate([observed[:, 0], completion]))
                    self.assertLessEqual(lower[0], truth); self.assertGreaterEqual(upper[0], truth)

    def test_partial_reference_denominator_and_region_coverage_are_fixed(self):
        partial = self.current[self.current.entity_id != 'tid_0']
        packet = build_reference_packet(partial, self.lock, '2024-01-01')
        self.assertEqual(packet['reference_n'], 8); self.assertEqual(packet['observed_valid_n'], 7)
        self.assertIsNone(packet['point_paired_median']); self.assertIsNotNone(packet['representative_correction'])
        self.assertEqual(packet['regional_coverage']['0']['coverage'], .75)
        incomplete = build_reference_packet(self.current[self.current.entity_id.isin(['tid_0', 'tid_1'])], self.lock, '2024-01-01')
        self.assertEqual(incomplete['status'], 'reference_unbounded')
        self.assertIsNone(incomplete['representative_correction'])
        json.dumps(incomplete, allow_nan=False)

    def test_invalid_reference_counts_missing_and_conflicting_duplicate_refuses(self):
        bad = self.current.copy(); bad.loc[(bad.entity_id == 'tid_0') & (bad.period == '2024-01-01'), CATEGORIES[0]] = 0.
        packet = build_reference_packet(bad, self.lock, '2024-01-01')
        self.assertEqual(packet['invalid_ids'], ['tid_0']); self.assertEqual(packet['observed_valid_n'], 7)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_reference_packet(pd.concat([self.current, self.current.iloc[:1]]), self.lock, '2024-01-01')

    def test_packet_batch_and_order_invariance_and_no_reference_admission(self):
        packet = build_reference_packet(self.current, self.lock, '2024-01-01')
        shuffled = build_reference_packet(self.current.sample(frac=1., random_state=31), self.lock, '2024-01-01')
        self.assertEqual(packet, shuffled)
        new = self.current[self.current.entity_id == 'tid_0'].copy(); new.entity_id = 'tid_999'; new.territory_id = '999'
        new[CATEGORIES] *= 1.e9
        augmented = build_reference_packet(pd.concat([self.current, new]), self.lock, '2024-01-01')
        self.assertEqual(packet, augmented)
        full = predict_dual_panel(self.current, self.lock, 2024, self.packets)
        target = self.current[self.current.entity_id == 'tid_0']
        pd.testing.assert_frame_equal(full[full.entity_id == 'tid_0'].reset_index(drop=True), predict_dual_panel(target, self.lock, 2024, self.packets))

    def test_wrong_packet_absence_and_missing_entity_month_keep_explicit_A(self):
        absent = predict_dual_panel(self.current, self.lock, 2024)
        self.assertTrue(absent.absolute_label.notna().all()); self.assertTrue(absent.relative_label.isna().all())
        wrong = dict(self.packets); wrong['2024-01-01'] = self.packets['2024-02-01']
        refused = predict_dual_panel(self.current, self.lock, 2024, wrong)
        self.assertTrue(refused.relative_label.isna().all())
        missing = self.current.drop(self.current.index[0]); result = predict_dual_panel(missing, self.lock, 2024, self.packets)
        row = result[result.entity_id == 'tid_0'].iloc[0]
        self.assertEqual(row.status, 'incomplete_entity_year'); self.assertTrue(pd.isna(row.absolute_label))

    def test_sealed_snapshot_revisions_do_not_mutate_prior_packets(self):
        before = copy.deepcopy(self.packets['2024-01-01'])
        revised = self.current.copy(); revised.loc[revised.period == '2024-01-01', CATEGORIES[0]] *= 2.
        after = build_reference_packet(revised, self.lock, '2024-01-01')
        self.assertNotEqual(before['content_sha256'], after['content_sha256'])
        self.assertEqual(before, self.packets['2024-01-01'])
        bad = copy.deepcopy(before); bad['coverage'] = 1.e-3
        with self.assertRaisesRegex(ValueError, 'SHA256'): validate_reference_packet(bad, self.lock)
        badlock = copy.deepcopy(self.lock); badlock['baseline_features'][0][0][0] += 1.
        with self.assertRaisesRegex(ValueError, 'SHA256'): validate_reference_lock(badlock)

    def test_box_certificate_matches_vertices_and_preserves_nonmidpoint_label(self):
        rng = np.random.default_rng(911)
        for _ in range(30):
            centers = rng.normal(size=(4, 5)); lower = rng.normal(size=5); upper = lower + rng.uniform(size=5)
            point = lower + .2 * (upper - lower)
            certificate = classify_profile_box(point, lower, upper, centers)
            vertices = np.asarray(list(itertools.product(*zip(lower, upper))))
            distances = np.sum((vertices[:, None] - centers[None, :]) ** 2, axis=2)
            chosen = certificate['nearest_label']
            true_margin = min(np.min(distances[:, other] - distances[:, chosen]) for other in range(4) if other != chosen)
            self.assertAlmostEqual(certificate['min_squared_margin'], true_margin, places=12)
            if certificate['geometric_label'] is not None: self.assertTrue(np.all(distances.argmin(axis=1) == chosen))
        result = classify_profile_box([.1], [0.], [3.], [[0.], [2.]])
        self.assertEqual(result['nearest_label'], 0); self.assertIsNone(result['geometric_label'])

    def test_future_data_cannot_change_locked_baseline(self):
        future = self.current.copy(); future.entity_id = 'invalid_future_id'; future.territory_id = 'bad'
        future[CATEGORIES] *= 1.e6
        augmented = build_reference_lock(pd.concat([self.base, future]), self.frozen)
        self.assertEqual(self.lock, augmented)

    def test_sealed_malformed_packet_refuses_relative_and_preserves_absolute(self):
        malformed = copy.deepcopy(self.packets['2024-01-01']); malformed.pop('content_sha256')
        malformed['representative_correction'] = [0.]
        packets = dict(self.packets); packets['2024-01-01'] = _seal(malformed)
        result = predict_dual_panel(self.current, self.lock, 2024, packets)
        self.assertTrue(result.absolute_label.notna().all()); self.assertTrue(result.relative_label.isna().all())
        json.dumps(prediction_records(result), allow_nan=False)

    def test_marginal_distribution_and_paired_motion_are_distinct(self):
        first = np.array([[-1., -1.], [1., 1.]])
        permuted = first[::-1]
        self.assertAlmostEqual(gaussian_mmd2(first, permuted, 1.), 0., places=14)
        self.assertGreater(np.linalg.norm(permuted - first), 0.)
        dependence_changed = np.array([[-1., 1.], [1., -1.]])
        np.testing.assert_array_equal(np.median(dependence_changed - first, axis=0), [0., 0.])
        self.assertGreater(gaussian_mmd2(first, dependence_changed, 1.), .1)

    def test_source_log_audit_retains_scale_invisible_to_ratio_profiles(self):
        scaled = self.current.copy(); scaled[CATEGORIES + [TOTAL]] *= 10.
        original = predict_dual_panel(self.current, self.lock, 2024, self.packets)
        changed = predict_dual_panel(scaled, self.lock, 2024, self.packets)
        np.testing.assert_allclose(np.asarray(original.absolute_profile.tolist()), np.asarray(changed.absolute_profile.tolist()), atol=1e-14)
        for old, new in zip(original.log_input_annual_medians, changed.log_input_annual_medians):
            self.assertAlmostEqual(new[TOTAL] - old[TOTAL], np.log(10.))

    def test_file_revision_is_exclusive_and_one_byte_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'packet.json'; digest = write_immutable_json(path, self.packets['2024-01-01'])
            self.assertEqual(read_pinned_json(path, digest), self.packets['2024-01-01'])
            with self.assertRaises(FileExistsError): write_immutable_json(path, self.packets['2024-01-01'])
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'SHA256'): read_pinned_json(path, digest)


if __name__ == '__main__': unittest.main()
