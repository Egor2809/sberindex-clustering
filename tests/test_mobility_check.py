import base64
import json
import unittest
from pathlib import Path

import pandas as pd

from sbercluster import mobility_check as mc

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'mobility-check'


def tag(text):
    return '__string__' + base64.b64encode(text.encode('utf-8')).decode()


class MobilityCheckTests(unittest.TestCase):
    def test_decode_envelope(self):
        raw = {'type': 'object', 'value': [{'key': 'fields', 'type': 'object', 'value': [tag('ref_area'), tag('value')]},
                                           {'key': 'data', 'type': 'object', 'value': [[tag('Кемский муниципальный район'), '__number__1.5']]},
                                           {'key': 'next', 'value': '__null__'}]}
        out = mc.decode(raw)
        self.assertEqual(out['fields'], ['ref_area', 'value'])
        self.assertEqual(out['data'], [['Кемский муниципальный район', 1.5]])
        self.assertIsNone(out['next'])

    def test_normalise_takes_local_year(self):
        fields = ['indicator_id', 'kpi_id', 'period', 'value', 'obs_status', 'source', 'ref_area', 'freq', 'decimals', 'unit_measure', 'unit_mult']
        rows = [['a', '1', '2024-12-30T21:00:00.000Z', 2.0, 'A', 's', 'Б', 'Год', '3', 'км', '0'],
                ['a', '1', '2025-10-31T21:00:00.000Z', 3.0, 'A', 's', 'Б', 'Год', '3', 'км', '0']]
        self.assertEqual(mc.normalise(fields, rows)['year'].tolist(), [2024, 2025])

    def test_match_ignores_municipal_type_and_other_regions(self):
        cohort = pd.DataFrame({'entity_id': ['a', 'b', 'c'], 'region_name': ['Псковская область', 'Новгородская область', 'Тверская область'],
                               'municipal_district_name': ['Невельский муниципальный район', 'Пестовский муниципальный район',
                                                           'Кировский муниципальный район']})
        out = mc.match(['муниципальный округ Невельский', 'Пестовский муниципальный район', 'Кировский муниципальный район'], cohort,
                       ('Псковская область', 'Новгородская область'))
        self.assertEqual(out, {'муниципальный округ Невельский': 'a', 'Пестовский муниципальный район': 'b',
                               'Кировский муниципальный район': None})

    def test_compare_drops_small_groups(self):
        out = mc.compare([1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 9], ['x'] * 5 + ['y'] * 5 + ['z'])
        self.assertEqual(out['groups'], 2)
        self.assertEqual(out['dropped_small_groups'], {'z': 1})
        self.assertAlmostEqual(out['eta2'], 1.0)

    def test_published_summary(self):
        summary = json.loads((REPORT / 'summary.json').read_text('utf-8'))
        sources = json.loads((REPORT / 'sources.json').read_text('utf-8'))
        self.assertEqual(summary['source']['sha256'], sources['sha256'])
        matched = pd.read_csv(REPORT / 'matched.csv', dtype={'entity_id': str})
        self.assertEqual(matched['entity_id'].nunique(), summary['matched'])
        self.assertFalse(matched.duplicated(['entity_id', 'year']).any())
        main = summary['results'][str(summary['main_year'])]
        for by in ('group', 'kmeans4', 'region_name'):
            self.assertTrue(0 <= main[by]['eta2'] <= 1)
        self.assertLess(main['without_city']['group']['n'], main['group']['n'])


if __name__ == '__main__':
    unittest.main()
