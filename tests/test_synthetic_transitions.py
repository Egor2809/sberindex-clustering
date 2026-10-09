import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from sbercluster import synthetic_transitions as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'synthetic-transitions'


class SyntheticTransitionsTests(unittest.TestCase):
    def test_align_restores_permuted_labels(self):
        previous = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(st.align(previous, np.array([2, 2, 0, 0, 1, 1])).tolist(), previous.tolist())

    def test_tracking_counts_hits_false_switches_and_returns(self):
        pred = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
        out = st.tracking(pred, np.array([2, -1, -1]))
        self.assertEqual((out['found'], out['false_switches'], out['returns']), (1, 2, 1))
        self.assertEqual(out['stayers_with_switch'], 0.5)
        late = st.tracking(np.array([[0], [0], [0], [1]]), np.array([1]))
        self.assertEqual((late['found'], late['false_switches']), (0, 1))

    def test_generation_plants_one_lasting_switch(self):
        cal = {'shift': np.zeros((24, 5)).tolist(), 'persistent_sd': [0.1] * 5, 'monthly_sd': [0.05] * 5}
        shares = np.array([[0.05, 0.2, 0.02, 0.5, 0.05], [0.06, 0.1, 0.08, 0.3, 0.08], [0.04, 0.15, 0.03, 0.45, 0.04]])
        p = {**st.GENERATION, 'n': 200}
        data = st.generate(3, shares, np.array([0.5, 0.3, 0.2]), cal, p)
        self.assertEqual(data['x'].shape, (24, 200, 5))
        movers = np.flatnonzero(data['change'] >= 0)
        self.assertEqual(len(movers), 20)
        changes = (data['truth'][1:] != data['truth'][:-1]).sum(axis=0)
        self.assertTrue((changes[movers] == 1).all() and (np.delete(changes, movers) == 0).all())
        self.assertTrue((data['change'][movers] >= 3).all() and (data['change'][movers] <= 21).all())
        self.assertTrue((data['road'] != data['road'].T).nnz == 0)

    def test_rule_needs_four_of_five_repeats(self):
        self.assertTrue(st.better([2, 2, 2, 2, 0], [1, 1, 1, 1, 1], True)['better'])
        self.assertFalse(st.better([2, 2, 2, 0, 0], [1, 1, 1, 1, 1], True)['better'])
        self.assertTrue(st.better([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], False)['better'])

    def test_published_summary(self):
        summary = json.loads((REPORT / 'summary.json').read_text('utf-8'))
        self.assertEqual(summary['rule'], st.RULE)
        self.assertEqual(summary['rule_sha256'], hashlib.sha256(st.RULE.encode('utf-8')).hexdigest())
        self.assertIn(st.RULE, (REPORT / 'README.md').read_text('utf-8'))
        self.assertEqual(summary['verdict'], st.verdict(summary['per_seed']))
        self.assertEqual(len(summary['per_seed']), 5)
        self.assertEqual(summary['leiden']['omega_relative'], 0.01)
        self.assertEqual(summary['leiden']['resolution'], 0.7)


if __name__ == '__main__':
    unittest.main()
