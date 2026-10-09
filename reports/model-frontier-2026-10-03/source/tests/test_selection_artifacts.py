import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sbercluster.features import transform_frozen
from sbercluster.io import CATEGORIES, TOTAL, sha256
from sbercluster.selection_frontier import REVISION, predict_frontier
from sbercluster.selection_artifacts import load_artifact, make_artifact, predict_annual_panel
from sbercluster.selection_contrasts import contrast_projection


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.scaler = {'mode': 'log_ratios_to_total', 'level_weight': 0., 'ratio_center': [0.] * 5,
                       'ratio_iqr': [1.] * 5, 'calibration_end': '2023-12-01'}
        self.rule = {'revision': REVISION, 'transform': np.eye(5).tolist(),
                     'centers': [[-1.] * 5, [1.] * 5], 'biases': [.1, -.1]}
        self.artifact = make_artifact(self.rule, self.scaler, provenance={'test': True})
        rows = []
        for entity, value in [('a', 20.), ('b', 400.)]:
            for month in range(1, 13):
                rows.append({'entity_id': entity, 'period': f'2023-{month:02d}-01', TOTAL: 100.,
                             **{name: value for name in CATEGORIES}})
        self.panel = pd.DataFrame(rows)

    def test_hash_checked_roundtrip_and_complete_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            path.write_text(json.dumps(self.artifact), encoding='utf-8')
            restored = load_artifact(path, sha256(path))
            got = predict_annual_panel(self.panel, restored, 2023)
            self.assertEqual(got.entity_id.tolist(), ['a', 'b'])
            self.assertEqual(got.cluster.tolist(), [0, 1])
            fingerprint = sha256(path)
            path.write_text(path.read_text('utf-8') + ' ', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_artifact(path, fingerprint)

    def test_full_year_is_required_without_dropping_entities(self):
        with self.assertRaisesRegex(ValueError, 'twelve'):
            predict_annual_panel(self.panel.iloc[1:], self.artifact, 2023)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            predict_annual_panel(pd.concat([self.panel, self.panel.iloc[:1]]), self.artifact, 2023)
        with self.assertRaisesRegex(ValueError, 'absent'):
            predict_annual_panel(self.panel, self.artifact, 2024)

    def test_optional_transform_precedes_month_aggregation(self):
        rng = np.random.default_rng(14)
        panel = self.panel.copy()
        panel[CATEGORIES] *= np.exp(rng.normal(size=(24, 5)))
        projection = contrast_projection(self.scaler)
        artifact = make_artifact(self.rule, self.scaler, provenance={}, pre_aggregation_transform=projection)
        annual = np.array([np.median(transform_frozen(rows, self.scaler) @ projection, axis=0)
                           for _, rows in panel.groupby('entity_id')])
        np.testing.assert_array_equal(predict_annual_panel(panel, artifact, 2023).cluster,
                                      predict_frontier(annual, self.rule))
        shifted = panel.copy()
        shifted[TOTAL] *= np.exp(rng.normal(size=24))
        np.testing.assert_array_equal(predict_annual_panel(shifted, artifact, 2023).cluster,
                                      predict_annual_panel(panel, artifact, 2023).cluster)

    def test_wrong_kind_and_mapping_fail(self):
        with self.assertRaisesRegex(ValueError, 'prototype_frontier'):
            predict_annual_panel(self.panel, {**self.artifact, 'model_kind': 'frozen_profile'}, 2023)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            predict_annual_panel(self.panel, {**self.artifact, 'cluster_mapping': [1, 0]}, 2023)
        for version in (True, 1.0):
            with self.assertRaisesRegex(ValueError, 'prototype_frontier'):
                predict_annual_panel(self.panel, {**self.artifact, 'format_version': version}, 2023)

    def test_parse_consumes_the_same_verified_model_bytes(self):
        raw = json.dumps(self.artifact).encode()
        import hashlib
        with patch.object(Path, 'read_bytes', return_value=raw) as read:
            with patch.object(Path, 'read_text', side_effect=AssertionError('second read')):
                loaded = load_artifact('unused.json', hashlib.sha256(raw).hexdigest())
        self.assertEqual(loaded['model_kind'], 'prototype_frontier')
        self.assertEqual(read.call_count, 1)


if __name__ == '__main__':
    unittest.main()
