"""Read fingerprinted published baselines without refitting or private runtime files."""
from pathlib import Path
import gzip, json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from .io import sha256

ARCHIVE = 'reports/review-2026-09-24'

def read_archive(root, name, filename):
    directory = Path(root) / ARCHIVE / name
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    path = directory / filename
    if sha256(path) != manifest['files_sha256'][filename]:
        raise ValueError('Published artifact fingerprint differs: ' + filename)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw)

def load_published(root, cfg, ids, manifest):
    s = cfg['followup']; root = Path(root)
    joint = 'review-20260924-joint'; base = 'review-20260924-small-k'
    read = lambda name, file: read_archive(root, name, file)
    jp,bp = [read(name,'provenance.json') for name in [joint,base]]
    jz,bz = [read(name,'labels.json.gz') for name in [joint,base]]
    for name,p,z in [(joint,jp,jz),(base,bp,bz)]:
        if read(name,'status.json')['status'] != 'completed' or p['ids'] != ids or z['ids'] != ids:
            raise ValueError('Completed aligned published baseline required')
        if p['panel_sha256'] != manifest['panel_sha256']:
            raise ValueError('Published panel fingerprint differs')
        for key in ['features','graph','seed']:
            if p['config'][key] != cfg[key]: raise ValueError('Published preprocessing differs: '+key)
    bm = json.loads((root/ARCHIVE/base/'manifest.json').read_text('utf-8'))
    if (jp['baseline_labels_sha256'] != bm['source_files_sha256']['labels.json'] or
            jp['baseline_provenance_sha256'] != bp['source_provenance_sha256'] or
            jp['road_sha256'] != s['expected_road_sha256'] or jp['road_date'] != '2024-12-31' or
            jp['config']['followup']['alpha'] != s['alpha'] or
            jp['config']['followup']['max_sweeps'] != s['max_sweeps']):
        raise ValueError('Published baseline linkage or road settings differ')
    roadpath=root/'reports/graphs/road_2024_2016.npz'
    if sha256(roadpath) != s['expected_transport_graph_sha256']: raise ValueError('Road cache fingerprint differs')
    a=load_npz(roadpath).tocsr()
    if a.shape != (len(ids),len(ids)): raise ValueError('Road cache shape differs')
    archive=root/s['ward_archive']
    if sha256(archive) != s['expected_ward_archive_sha256']: raise ValueError('Ward archive fingerprint differs')
    frame=pd.read_csv(archive);ward=frame[(frame.representation=='annual_2023') & (frame.candidate=='ward_k4')]
    if ward.entity_id.tolist()!=ids: raise ValueError('Ward IDs differ')
    initial={2:{'kmeans':np.asarray(bz['labels']['kmeans_k2']),'ward':np.asarray(bz['labels']['ward_k2'])},
             4:{'kmeans':np.asarray(jz['labels']['joint_alpha0_k4']),'ward':ward.cluster.to_numpy()}}
    original={k:np.asarray(jz['labels'][f'joint_road_k{k}']) for k in [2,4]}
    provenance={'input_mode':'verified published baseline artifacts; no refitting',
        'road_sha256':jp['road_sha256'],'road_date':jp['road_date'],
        'transport_graph_sha256':sha256(roadpath),
        'published_joint_manifest_sha256':sha256(root/ARCHIVE/joint/'manifest.json'),
        'published_baseline_manifest_sha256':sha256(root/ARCHIVE/base/'manifest.json'),
        'ward_archive_sha256':sha256(archive)}
    return a,read(joint,'graphs.json')['transport'],initial,original,provenance
