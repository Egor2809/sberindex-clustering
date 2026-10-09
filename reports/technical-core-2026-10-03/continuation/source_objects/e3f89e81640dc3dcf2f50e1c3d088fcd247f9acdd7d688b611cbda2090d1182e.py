import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sbercluster.io import read_verified_artifact


class ArtifactIntegrityTests(unittest.TestCase):
    def test_changed_centers_rejected_even_when_reference_labels_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'validation').mkdir()
            path = root / 'validation/model.json'
            old = np.array([[0., 0.], [2., 2.]])
            probes = np.array([[.1, .1], [1.9, 1.9], [.2, .2]])
            path.write_text(json.dumps({'centers': old.tolist()}), encoding='utf-8')
            manifest = {'validation/model.json': hashlib.sha256(path.read_bytes()).hexdigest()}
            (root / 'SHA256.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(json.loads(read_verified_artifact(root, 'validation/model.json'))['centers'], old.tolist())
            altered = old + .001
            np.testing.assert_array_equal(cdist(probes, old).argmin(1), cdist(probes, altered).argmin(1))
            path.write_text(json.dumps({'centers': altered.tolist()}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                read_verified_artifact(root, 'validation/model.json')

    def test_manifest_requires_exact_entry_and_rejects_duplicate_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data.json').write_text('{}', encoding='utf-8')
            (root / 'SHA256.json').write_text('{}', encoding='utf-8')
            for relative in ['data.json', '../data.json']:
                with self.assertRaises(ValueError):
                    read_verified_artifact(root, relative)
            digest = hashlib.sha256(b'{}').hexdigest()
            (root / 'SHA256.json').write_text('{"data.json":"' + digest + '","data.json":"' + digest + '"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate manifest'):
                read_verified_artifact(root, 'data.json')


if __name__ == '__main__':
    unittest.main()
