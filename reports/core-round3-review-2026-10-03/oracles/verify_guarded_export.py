"""Guard-only derivative audit, deliberately without fitting or metric replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
old = ROOT / '.local/core-round3-20261003/temporal/run-v1'
new = ROOT / '.local/core-round3-20261003/temporal/guarded-export-v1'
count = 0
with np.load(old / 'labels.npz', allow_pickle=False) as labels:
    for path in sorted((new / 'models').glob('*.json')):
        arm = path.stem
        former = json.loads((old / 'models' / path.name).read_text('utf-8'))
        current = json.loads(path.read_text('utf-8'))
        assert current.pop('identity_mode') == 'source_territory_id'
        assert current == former, f'{arm}: numerical model changed during guard-only export'
        for year in [2023, 2024]:
            frame = pd.read_csv(new / f'{year}_{arm}' / 'assignments.csv')
            np.testing.assert_array_equal(frame.cluster, labels[f'{year}__{arm}'])
            assert frame.entity_id.tolist() == former['training_entity_ids']
            assert not frame.new_entity.any()
            provenance = json.loads((new / f'{year}_{arm}' / 'provenance.json').read_text('utf-8'))
            assert provenance['model_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert provenance['input_sha256'] == hashlib.sha256((ROOT / 'data/processed/panel.csv').read_bytes()).hexdigest()
            assert provenance['fitting_executed'] is False and provenance['recalibration_executed'] is False
            count += len(frame)
manifest = json.loads((new / 'manifest.json').read_text('utf-8'))['files_sha256']
assert all(hashlib.sha256((new / name).read_bytes()).hexdigest() == digest for name, digest in manifest.items())
verification = json.loads((new / 'verification.json').read_text('utf-8'))
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in verification['source_hashes'].items())
report = {'status': 'PASS', 'no_fit': True, 'exact_labels': count, 'models': 3,
          'only_model_difference': 'identity_mode=source_territory_id', 'guarded_payload_hashes': len(manifest),
          'current_guard_sources_match_export_snapshot': True,
          'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'guarded-export-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
