import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sbercluster.inference import FrozenProfileModel
from sbercluster.io import CATEGORIES, TOTAL, sha256
from scripts.predict_frozen import predict_file


class FrozenInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'model.json'
        self.model_data = {'centers': [[0.] * 5, [2.] * 5],
            'ids': ['tid_1', 'tid_2', 'tid_3'], 'reference_labels': [0, 0, 1],
            'scaler': {'mode': 'log_ratios_to_total', 'level_weight': 0.,
                       'ratio_center': [-2.] * 5, 'ratio_iqr': [1.] * 5,
                       'level_center': 0., 'level_iqr': 1., 'calibration_end': '2023-12-01'}}
        self.write_model()
        self.frame = pd.DataFrame([{'entity_id': 'tid_1', 'territory_id': '1', 'period': f'2024-{m:02d}-01',
                                  TOTAL: 100., **{c: 10. for c in CATEGORIES}} for m in range(1, 13)])

    def write_model(self):
        self.path.write_text(json.dumps(self.model_data), encoding='utf-8')

    def test_frozen_prediction_never_fits_and_preserves_annual_aggregation(self):
        model = FrozenProfileModel.load(self.path, sha256(self.path))
        before = copy.deepcopy(model.scaler)
        with patch('sklearn.cluster.KMeans.fit_predict', side_effect=AssertionError('training forbidden')):
            monthly = model.predict(self.frame)
            annual = model.predict(self.frame, 'annual')
        self.assertEqual(len(monthly), 12)
        self.assertEqual(len(annual), 1)
        self.assertEqual(annual.observations.iloc[0], 12)
        self.assertEqual(annual.cluster.iloc[0], monthly.cluster.iloc[0])
        self.assertEqual(model.scaler, before)
        mutated = model.scaler; mutated['ratio_center'][0] = 999
        self.assertEqual(model.scaler, before)

    def test_annual_profiles_require_each_complete_year(self):
        model = FrozenProfileModel.load(self.path)
        with self.assertRaisesRegex(ValueError, 'Complete calendar year'):
            model.predict(self.frame.iloc[:-1], 'annual')
        with self.assertRaisesRegex(ValueError, 'duplicate entity-month'):
            model.predict(pd.concat([self.frame, self.frame.iloc[[0]]]), 'annual')

    def test_new_entity_flag_and_identity_consistency(self):
        model = FrozenProfileModel.load(self.path)
        frame = self.frame.copy(); frame['entity_id'] = 'tid_99'; frame['territory_id'] = '99'
        self.assertFalse(model.predict(frame).in_reference_cohort.any())
        frame['territory_id'] = '1'
        with self.assertRaisesRegex(ValueError, 'differ from source'):
            model.predict(frame)

    def test_centroid_tie_has_zero_margin_and_stable_label(self):
        self.model_data['centers'] = [[0.] * 5, [0.] * 5]; self.write_model()
        result = FrozenProfileModel.load(self.path).predict(self.frame)
        self.assertTrue(result.cluster.eq(0).all())
        self.assertTrue(result.relative_distance_margin.eq(0).all())

    def test_model_fingerprint_dimensions_and_reference_labels(self):
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            FrozenProfileModel.load(self.path, '0' * 64)
        for field, value in [('centers', [[0.] * 4, [1.] * 4]),
                             ('reference_labels', [0, 1]), ('ids', ['tid_1', 'tid_1', 'tid_3'])]:
            original = copy.deepcopy(self.model_data)
            self.model_data[field] = value; self.write_model()
            with self.assertRaises(ValueError):
                FrozenProfileModel.load(self.path)
            self.model_data = original

    def test_nonfinite_or_duplicate_model_fields_rejected(self):
        self.path.write_text('{"centers": NaN}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            FrozenProfileModel.load(self.path)
        self.path.write_text('{"centers": [], "centers": []}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            FrozenProfileModel.load(self.path)

    def test_cli_output_contains_verified_model_sources_and_no_overwrite(self):
        input_path = self.root / 'input.csv'; output = self.root / 'result'
        self.frame.to_csv(input_path, index=False)
        result = predict_file(self.path, input_path, output, 'annual')
        self.assertEqual(result['rows'], 1)
        self.assertFalse(result['fitting_executed'])
        manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
        for name, value in manifest['files_sha256'].items():
            self.assertEqual(sha256(output / name), value)
        self.assertEqual(sha256(output / 'model.json'), sha256(self.path))
        with self.assertRaises(FileExistsError):
            predict_file(self.path, input_path, output)

    def test_published_path_is_manifest_bound_without_an_explicit_pin(self):
        archive = self.root / 'reports/experiments/2026-09-23-v2'
        published = archive / 'validation/frozen_prototypes.json'
        published.parent.mkdir(parents=True)
        published.write_bytes(self.path.read_bytes())
        (archive / 'SHA256.json').write_text(json.dumps({
            'validation/frozen_prototypes.json': sha256(published)}), encoding='utf-8')
        self.model_data['centers'][0][0] += .001
        published.write_text(json.dumps(self.model_data), encoding='utf-8')
        input_path = self.root / 'input.csv'; self.frame.to_csv(input_path, index=False)
        with patch('scripts.predict_frozen.ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                predict_file(published, input_path, self.root / 'result')
        self.assertFalse((self.root / 'result').exists())

    def test_input_duplicate_header_is_rejected_before_output(self):
        input_path = self.root / 'input.csv'; output = self.root / 'result'
        self.frame.to_csv(input_path, index=False)
        text = input_path.read_text('utf-8').splitlines()
        text[0] += ',period'
        input_path.write_text('\n'.join(text), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate input column'):
            predict_file(self.path, input_path, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
