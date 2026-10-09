"""Export checked complete annual frontier artifacts from a completed benchmark."""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster.io import sha256, write_json
from sbercluster.selection_artifacts import make_artifact, load_artifact, predict_annual_panel


def checked_json(directory, filename, manifest):
    raw = (directory / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files_sha256'][filename]:
        raise ValueError('Benchmark artifact fingerprint differs: ' + filename)
    return json.loads(raw)


def export(screen, validation, panel_path, output):
    screen, validation, output = Path(screen), Path(validation), Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    screen_raw = (screen / 'manifest.json').read_bytes()
    validation_raw = (validation / 'manifest.json').read_bytes()
    screen_manifest, validation_manifest = json.loads(screen_raw), json.loads(validation_raw)
    if checked_json(screen, 'status.json', screen_manifest)['status'] != 'completed':
        raise ValueError('Completed screen required')
    if checked_json(validation, 'status.json', validation_manifest)['status'] != 'completed':
        raise ValueError('Completed validation required')
    prior = checked_json(screen, 'provenance.json', screen_manifest)
    validation_prior = checked_json(validation, 'provenance.json', validation_manifest)
    screen_digest = hashlib.sha256(screen_raw).hexdigest()
    if validation_prior['screen_manifest_sha256'] != screen_digest:
        raise ValueError('Validation and screen linkage differs')
    models = checked_json(screen, 'models.json', screen_manifest)
    data = checked_json(screen, 'labels.json', screen_manifest)
    names = checked_json(screen, 'shortlist.json', screen_manifest)['ids']
    outcomes = checked_json(validation, 'validation.json', validation_manifest)
    accepted = {row['id']: row['accepted_continuation'] for row in outcomes}
    panel_bytes = Path(panel_path).read_bytes()
    panel_digest = hashlib.sha256(panel_bytes).hexdigest()
    if panel_digest != prior['panel_sha256']:
        raise ValueError('Panel fingerprint differs')
    panel = pd.read_csv(BytesIO(panel_bytes), dtype={'entity_id': str, 'period': str})
    output.mkdir(parents=True)
    write_json(output / 'status.json', {'status': 'running'})
    verification = []
    try:
        for name in names:
            artifact = make_artifact(models[name], prior['scaler'], provenance={
                'candidate_id': name, 'training_year': 2023,
                'training_scope': 'already_inspected_retrospective_2023_complete_panel',
                'screen_manifest_sha256': screen_digest,
                'validation_manifest_sha256': hashlib.sha256(validation_raw).hexdigest(),
                'training_features_sha256': screen_manifest['files_sha256']['features.npy'],
                'policy': prior['policy'], 'limitations': prior['limitations'],
                'continuation_guards_passed': accepted[name]})
            model_path = output / (name + '.json')
            write_json(model_path, artifact)
            model_digest = sha256(model_path)
            restored = load_artifact(model_path, model_digest)
            got = predict_annual_panel(panel, restored, 2023)
            if got.entity_id.tolist() != data['ids'] or got.cluster.tolist() != data['labels'][name]:
                raise ValueError('Raw-panel replay differs: ' + name)
            verification.append({'id': name, 'artifact': model_path.name, 'sha256': model_digest,
                                 'n': len(got), 'raw_panel_replay_exact': True})
        write_json(output / 'verification.json', {'models': verification, 'panel_sha256': panel_digest})
        write_json(output / 'status.json', {'status': 'completed'})
        write_json(output / 'manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): sha256(p)
                                                              for p in sorted(output.rglob('*')) if p.is_file()}})
        print(json.dumps({'status': 'completed', 'models': verification}), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                           else 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.screen, args.validation, args.panel, args.output)


if __name__ == '__main__':
    main()
