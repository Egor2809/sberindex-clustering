"""Inspect and summarize saved monthly cluster perturbations without fitting."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbercluster.selection_frontier import predict_frontier
from sbercluster.published_inputs import read_archive

FRONTIER = ROOT / 'reports/model-frontier-2026-10-03'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_labels(values, size=None):
    z = np.asarray(values)
    if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer) or z.size == 0:
        raise ValueError('Labels must be a nonempty one-dimensional integer array')
    if size is not None and z.size != size:
        raise ValueError('Label length differs from the common ID universe')
    if np.any(z < 0):
        raise ValueError('Noise or missing labels are not supported')
    return z


def contingency(reference, perturbed):
    reference = validate_labels(reference)
    perturbed = validate_labels(perturbed, reference.size)
    ref_labels, ri = np.unique(reference, return_inverse=True)
    new_labels, ni = np.unique(perturbed, return_inverse=True)
    counts = np.zeros((ref_labels.size, new_labels.size), dtype=np.int64)
    np.add.at(counts, (ri, ni), 1)
    return ref_labels, new_labels, counts, ri, ni


def cluster_matches(reference, perturbed):
    """Match each reference group independently, as in Hennig's clusterwise score."""
    ref_labels, new_labels, counts, _, _ = contingency(reference, perturbed)
    old_sizes, new_sizes = counts.sum(axis=1), counts.sum(axis=0)
    scores = counts / (old_sizes[:, None] + new_sizes[None, :] - counts)
    matches = []
    for i, label in enumerate(ref_labels):
        # Preserve all maximizers: a numerical label must not resolve scientific ties.
        winners = np.flatnonzero(scores[i] == scores[i].max())
        matches.append({'reference_cluster': int(label), 'reference_size': int(old_sizes[i]),
                        'best_jaccard': float(scores[i].max()),
                        'matched_labels': new_labels[winners].tolist(),
                        'matched_sizes': new_sizes[winners].tolist(),
                        'intersections': counts[i, winners].tolist()})
    return matches


def entity_neighborhoods(reference, perturbed):
    """Compare same-group peers of each entity, excluding the entity itself."""
    _, _, counts, ri, ni = contingency(reference, perturbed)
    old_peers = counts.sum(axis=1)[ri] - 1
    new_peers = counts.sum(axis=0)[ni] - 1
    intersection = counts[ri, ni] - 1
    union = old_peers + new_peers - intersection
    jaccard = np.divide(intersection, union, out=np.ones(union.size), where=union > 0)
    retention = np.divide(intersection, old_peers, out=np.full(union.size, np.nan), where=old_peers > 0)
    return jaccard, retention


def pair_frequency_histograms(reference, repeats):
    """Return exact frequency histograms of distinct, unordered entity pairs."""
    reference = validate_labels(reference)
    repeats = [validate_labels(z, reference.size) for z in repeats]
    if not repeats or len(repeats) > np.iinfo(np.uint16).max:
        raise ValueError('Unsupported number of repeats')
    joint = np.zeros((reference.size, reference.size), dtype=np.uint16)
    for z in repeats:
        joint += z[:, None] == z[None, :]
    labels = np.unique(reference)
    rows = []
    for i, left in enumerate(labels):
        li = np.flatnonzero(reference == left)
        for right in labels[i:]:
            rj = np.flatnonzero(reference == right)
            values = joint[np.ix_(li, rj)]
            values = values[np.triu_indices(li.size, 1)] if left == right else values.ravel()
            histogram = np.bincount(values, minlength=len(repeats) + 1)
            for frequency, number in enumerate(histogram):
                rows.append({'reference_left': int(left), 'reference_right': int(right),
                             'joint_repeats': frequency, 'repeats': len(repeats),
                             'pairs': int(number)})
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_saved(frontier):
    manifest_path = frontier / 'manifest.json'
    pins = json.loads(manifest_path.read_text('utf-8'))['files_sha256']
    consumed = {'manifest.json': sha256(manifest_path)}

    def checked(relative):
        path = frontier / relative
        digest = sha256(path)
        if pins.get(relative) != digest:
            raise ValueError('Saved input fingerprint differs: ' + relative)
        consumed[relative] = digest
        return path

    def read(relative):
        return json.loads(checked(relative).read_text('utf-8'))

    labels = read('constrained/labels.json')
    provenance = read('constrained/provenance.json')
    shortlist = read('constrained/shortlist.json')['ids']
    metadata = read('validation/stability.json')
    checked('constrained/source/scripts/benchmark_frontier.py')
    ids = labels['ids']
    if len(set(ids)) != len(ids) or provenance['ids'] != ids:
        raise ValueError('Nonunique or differently ordered saved IDs')
    x = np.load(checked('constrained/features.npy'), allow_pickle=False)
    if x.shape != (len(ids), 5) or not np.isfinite(x).all():
        raise ValueError('Saved reference feature format differs')
    models, repeats = {}, {}
    common_draws = None
    for name in shortlist:
        model = read('models/' + name + '.json')
        z = validate_labels(labels['labels'][name], len(ids))
        if np.unique(z).tolist() != [0, 1, 2, 3]:
            raise ValueError('Reference must contain four groups numbered 0 through 3')
        if model['feature_scaler'] != provenance['scaler']:
            raise ValueError('Exported reference scaler differs')
        predicted = predict_frontier(x, model['rule'])
        mapped = np.asarray(model['cluster_mapping'])[predicted]
        if not np.array_equal(mapped, z):
            raise ValueError('Exported reference model does not reproduce labels')
        records = metadata[name]
        signatures = [(r['draw'], r['seed'], r['month_indices']) for r in records]
        if [r['draw'] for r in records] != list(range(30)):
            raise ValueError('Expected all 30 archived draws in order')
        if common_draws is not None and signatures != common_draws:
            raise ValueError('Model perturbations are not paired')
        common_draws = signatures
        repeats[name] = []
        for record in records:
            draw = record['draw']
            months = record['month_indices']
            if record['seed'] != 2200 + draw or len(months) != 12 or any(m not in range(12) for m in months):
                raise ValueError('Month perturbation metadata differs')
            if any(months[i + j] != (months[i] + j) % 12 for i in range(0, 12, 3) for j in range(3)):
                raise ValueError('Draw does not contain circular three-month blocks')
            arr = validate_labels(np.load(checked(f'validation/{name}__month_draw{draw}.npy'),
                                          allow_pickle=False), len(ids))
            if np.unique(arr).tolist() != [0, 1, 2, 3]:
                raise ValueError('Saved draw does not contain the expected four groups')
            if np.bincount(arr, minlength=4).tolist() != record['cluster_sizes']:
                raise ValueError('Saved draw sizes differ from metadata')
            saved_ari = record['ARI']
            if not np.isfinite(saved_ari):
                raise ValueError('Saved draw ARI must be finite')
            if abs(adjusted_rand_score(z, arr) - saved_ari) > 1e-12:
                raise ValueError('Saved draw ARI differs from metadata')
            repeats[name].append(arr)
        models[name] = model
    return ids, labels['labels'], repeats, consumed, provenance, models


def published_alignment(ids, reference):
    published = read_archive(ROOT, 'review-20260924-joint', 'labels.json.gz')
    if published['ids'] != ids or published['labels']['joint_alpha0_k4'] != list(reference):
        raise ValueError('Historical partition differs from the published KMeans4 by ID')
    frozen_path = ROOT / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json'
    atlas_path = ROOT / 'reports/review-2026-09-24/atlas_extension.json'
    frozen = json.loads(frozen_path.read_text('utf-8'))
    atlas = json.loads(atlas_path.read_text('utf-8'))
    if frozen['ids'] != ids:
        raise ValueError('Frozen atlas reference IDs differ')
    matches = cluster_matches(reference, frozen['reference_labels'])
    profiles = {p['id']: p['name'] for p in atlas['profiles']}
    mapping = {}
    for match in matches:
        if match['best_jaccard'] != 1. or len(match['matched_labels']) != 1:
            raise ValueError('Atlas group membership is not an exact permutation')
        public_label = match['matched_labels'][0]
        mapping[match['reference_cluster']] = {'atlas_cluster': public_label,
                                              'atlas_name': profiles[public_label]}
    paths = [ROOT / 'reports/review-2026-09-24/review-20260924-joint/manifest.json',
             ROOT / 'reports/review-2026-09-24/review-20260924-joint/labels.json.gz',
             frozen_path, atlas_path]
    return mapping, {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def report(frontier, output):
    ids, references, repeats, consumed, provenance, models = load_saved(frontier)
    alignment, published_hashes = published_alignment(ids, references['historical_kmeans4'])
    summary, detail, entities, histograms = [], [], [], []
    for name, draws in repeats.items():
        reference = np.asarray(references[name])
        matched = [cluster_matches(reference, z) for z in draws]
        peer_jaccard, peer_retention = zip(*(entity_neighborhoods(reference, z) for z in draws))
        mean_peer_jaccard = np.mean(peer_jaccard, axis=0)
        mean_peer_retention = np.mean(peer_retention, axis=0)
        histogram = pair_frequency_histograms(reference, draws)
        histograms.extend({'model': name, **r} for r in histogram)
        for cluster in range(4):
            scores = np.asarray([m[cluster]['best_jaccard'] for m in matched])
            in_group = reference == cluster
            within = [r for r in histogram if r['reference_left'] == cluster and r['reference_right'] == cluster]
            pair_count = sum(r['pairs'] for r in within)
            joint = sum(r['pairs'] * r['joint_repeats'] for r in within)
            summary.append({'model': name, 'reference_cluster': cluster,
                            'reference_size': int(in_group.sum()), 'repeats': len(draws),
                            'jaccard_mean': float(scores.mean()), 'jaccard_median': float(np.median(scores)),
                            'jaccard_min': float(scores.min()), 'jaccard_max': float(scores.max()),
                            'draws_jaccard_le_0_5': int((scores <= .5).sum()),
                            'within_pair_coassignment_mean': joint / (pair_count * len(draws)),
                            'peer_jaccard_mean': float(mean_peer_jaccard[in_group].mean()),
                            'entity_peer_retention_min': float(mean_peer_retention[in_group].min())})
        for draw, matches in enumerate(matched):
            for match in matches:
                detail.append({'model': name, 'draw': draw, **match})
        entities.extend({'model': name, 'entity_id': entity, 'reference_cluster': int(reference[i]),
                         'peer_jaccard_mean': float(mean_peer_jaccard[i]),
                         'original_peer_retention_mean': float(mean_peer_retention[i])}
                        for i, entity in enumerate(ids))
        print(json.dumps({'model': name, 'groups': summary[-4:]}, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'clusters.csv', summary)
    write_csv(output / 'entities.csv', entities)
    write_csv(output / 'pair-frequency-histograms.csv', histograms)
    (output / 'draws.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2) + '\n', 'utf-8', newline='\n')
    result = {'status': 'descriptive_saved_month_perturbations', 'year': 2023,
              'entities': len(ids), 'repeats': 30, 'circular_block_months': 3,
              'reference_model': 'historical_kmeans4', 'cluster_mapping': models['historical_kmeans4']['cluster_mapping'],
              'matching': 'independent maximum Jaccard per reference group; all ties preserved',
              'id_alignment': 'common ordered ids from saved labels and provenance; verified archived model predictions',
              'no_refit': True, 'future_outcome_2025_accessed': False,
              'neighbor_scope': 'same-group peers excluding self; no nearest-15 list stability estimate',
              'limitations': ['30 conditional month perturbations, not independent entity sampling',
                              '2023 only; no confidence interval or evidence of real economic types',
                              'draw label files have no embedded IDs; alignment follows archived source and common provenance',
                              'archived root manifest verifies bytes, not independent truth of historical ordering'],
              'calibration_end': provenance['scaler']['calibration_end'], 'groups': summary,
              'atlas_alignment_by_exact_id_membership': alignment,
              'published_input_sha256': published_hashes,
              'input_sha256': consumed, 'script_sha256': sha256(Path(__file__)),
              'environment': {'python': sys.version.split()[0],
                              **{name: importlib.metadata.version(name)
                                 for name in ('numpy', 'scipy', 'scikit-learn')}}}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8', newline='\n')
    print(json.dumps({'status': 'completed', 'entities': len(ids), 'models': len(repeats),
                      'verified_frontier_inputs': len(consumed) - 1, 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', type=Path, default=FRONTIER)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/cluster-stability-2026-10-07')
    args = parser.parse_args()
    report(args.frontier, args.output)
