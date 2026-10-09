"""Check public plateau results without refitting the models."""
import gzip,hashlib,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/dmon-plateau-2026-09-25'
RUN=REPORT/'dmon-plateau-20260925'

@unittest.skipUnless(RUN.exists(),'plateau evidence not available')
class PlateauEvidence(unittest.TestCase):
    def test_manifest_and_all_scientific_sources(self):
        m=json.loads((RUN/'manifest.json').read_text())
        for name,h in m['files_sha256'].items():self.assertEqual(hashlib.sha256((RUN/name).read_bytes()).hexdigest(),h)
        for h in m['source_objects'].values():self.assertEqual(hashlib.sha256((REPORT/'source_objects'/f'{h}.py').read_bytes()).hexdigest(),h)

    def test_plateau_objective_checks_reproduce_the_full_trace(self):
        rows=json.loads((REPORT/'comparison.json').read_text())
        self.assertEqual({(r['requested_k'],r['seed']) for r in rows},{(k,s) for k in [2,4] for s in [1729,2718,3141]})
        for row in rows:
            path=RUN/row['id'];info=json.loads((path/'info.json').read_text());trace=json.loads(gzip.decompress((path/'trace.json.gz').read_bytes()))
            self.assertEqual(len(trace),info['epochs']+1)
            objectives=np.array([r['objective'] for r in trace]);incumbent=np.minimum.accumulate(objectives)
            self.assertEqual(info['best_objective'],min(objectives))
            for check in info['plateau_checks']:
                epoch=check['epoch'];self.assertEqual(check['best_objective'],incumbent[epoch])
                if epoch:self.assertAlmostEqual(check['best_loss_gain'],incumbent[epoch-500]-incumbent[epoch],places=12)
            self.assertTrue(info['plateau_reached']);self.assertEqual(info['stop_reason'],'training_plateau')
            for check in info['plateau_checks'][-3:]:
                self.assertGreaterEqual(check['epoch'],3000);self.assertLessEqual(check['best_loss_gain'],1e-4);self.assertLessEqual(check['hard_label_change_fraction'],.001)
            soft=np.load(path/'assignments.npy',allow_pickle=False);labels=np.load(path/'labels.npy',allow_pickle=False)
            np.testing.assert_array_equal(soft.argmax(axis=1),labels)
            self.assertEqual(len(np.unique(labels)),row['requested_k'])
