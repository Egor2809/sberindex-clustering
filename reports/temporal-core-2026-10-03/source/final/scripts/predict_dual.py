"""Content-pinned dual-profile reference construction, packets and inference."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sbercluster.dual_profiles import build_reference_lock, build_reference_packet, predict_dual_panel


def read_pinned_json(path, expected_sha256):
    raw = Path(path).read_bytes(); actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Pinned file SHA256 differs: ' + str(path))
    return json.loads(raw)


def read_panel(path):
    raw = Path(path).read_bytes()
    return pd.read_csv(BytesIO(raw), dtype={'entity_id': str, 'period': str, 'territory_id': str, 'region_code': str}), hashlib.sha256(raw).hexdigest()


def write_immutable_json(path, contents):
    """Exclusive file creation: publishing a revision requires a new filename."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(contents, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream: stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prediction_records(frame):
    # JSON has null, no pandas NaN; preserve original Python float precision.
    def clean(value):
        if isinstance(value, dict): return {key: clean(val) for key, val in value.items()}
        if isinstance(value, list): return [clean(val) for val in value]
        if isinstance(value, float) and pd.isna(value): return None
        return value
    return clean(frame.to_dict(orient='records'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock')
    lock.add_argument('--model', required=True); lock.add_argument('--expected-model-sha256', required=True)
    for current in (lock,):
        current.add_argument('--panel', required=True); current.add_argument('--output', required=True)
    packet = sub.add_parser('packet')
    packet.add_argument('--reference', required=True); packet.add_argument('--expected-reference-sha256', required=True)
    packet.add_argument('--panel', required=True); packet.add_argument('--period', required=True); packet.add_argument('--output', required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--reference', required=True); predict.add_argument('--expected-reference-sha256', required=True)
    predict.add_argument('--panel', required=True); predict.add_argument('--year', type=int, required=True); predict.add_argument('--output', required=True)
    predict.add_argument('--packet-manifest'); predict.add_argument('--expected-packet-manifest-sha256')
    args = parser.parse_args()
    panel, input_hash = read_panel(args.panel)
    if args.command == 'lock':
        artifact = read_pinned_json(args.model, args.expected_model_sha256)
        result = build_reference_lock(panel, artifact, {'panel_sha256': input_hash,
            'model_file_sha256': args.expected_model_sha256,
            'cohort_selection': 'retrospectively_complete2023_2024;not_prospective'})
    else:
        reference = read_pinned_json(args.reference, args.expected_reference_sha256)
        if args.command == 'packet':
            result = build_reference_packet(panel, reference, args.period, {'input_file_sha256': input_hash})
        else:
            packets = {}
            if args.packet_manifest:
                if not args.expected_packet_manifest_sha256:
                    parser.error('--expected-packet-manifest-sha256 is required with --packet-manifest')
                manifest_path = Path(args.packet_manifest).resolve()
                manifest = read_pinned_json(manifest_path, args.expected_packet_manifest_sha256)
                for period, item in manifest['packets'].items():
                    location = (manifest_path.parent / item['path']).resolve()
                    if manifest_path.parent not in location.parents:
                        raise ValueError('Packet path must stay inside manifest directory')
                    packets[period] = read_pinned_json(location, item['sha256'])
            frame = predict_dual_panel(panel, reference, args.year, packets)
            result = {'scope': 'descriptive_absolute_and_paired_reference;no_confirmed_territorial_transition',
                      'reference_file_sha256': args.expected_reference_sha256, 'input_file_sha256': input_hash,
                      'fitting_executed': False, 'records': prediction_records(frame)}
    output_hash = write_immutable_json(args.output, result)
    print(json.dumps({'output': args.output, 'sha256': output_hash, 'fitting_executed': False}))


if __name__ == '__main__': main()
