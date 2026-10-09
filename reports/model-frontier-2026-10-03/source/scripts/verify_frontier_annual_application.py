"""Retrospective 2024 annual application of frozen 2023 rules; no fitting."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from sbercluster.features import transform_frozen, scalers_equal
from sbercluster.graph import knn_graph
from sbercluster.metrics import all_metrics
from sbercluster.selection_artifacts import load_artifact, predict_annual_panel
from sbercluster.io import write_json

root = Path.cwd()
package = root / 'reports/model-frontier-2026-10-03'
output = package / 'annual-2024-application.json'
if output.exists():
    raise FileExistsError('The retrospective application report already exists')
panel_bytes = (root / 'data/processed/panel.csv').read_bytes()
panel_digest = hashlib.sha256(panel_bytes).hexdigest()
prior = json.loads((package / 'constrained/provenance.json').read_bytes())
if panel_digest != prior['panel_sha256']:
    raise ValueError('Prepared panel differs')
panel = pd.read_csv(BytesIO(panel_bytes), dtype={'entity_id': str, 'period': str})
reference = json.loads((package / 'constrained/labels.json').read_bytes())
ids = reference['ids']
models_manifest = json.loads((package / 'models/manifest.json').read_bytes())
names = ['historical_kmeans4', 'sw_constrained_huber75', 'sw_constrained_ordinary']
models = {name: load_artifact(package / 'models' / (name + '.json'),
                             models_manifest['files_sha256'][name + '.json']) for name in names}
scaler = models[names[0]]['feature_scaler']
if any(not scalers_equal(model['feature_scaler'], scaler) for model in models.values()):
    raise ValueError('Frozen scalers differ across compared models')
frames = []
for month in range(1, 13):
    frame = panel[panel.period == f'2024-{month:02d}-01'].sort_values('entity_id')
    if frame.entity_id.tolist() != ids:
        raise ValueError('2024 monthly IDs differ from the saved reference')
    frames.append(transform_frozen(frame, scaler))
x2024 = np.median(np.stack(frames), axis=0)
with threadpool_limits(limits=1):
    graph, graph_info = knn_graph(x2024, 15, 'union')
    rows = []
    for name, model in models.items():
        annual = predict_annual_panel(panel, model, 2024)
        if annual.entity_id.tolist() != ids:
            raise ValueError('2024 frozen annual prediction IDs differ')
        old = np.asarray(reference['labels'][name], dtype=int)
        new = annual.cluster.to_numpy(dtype=int)
        cross = np.zeros((4, 4), dtype=int)
        np.add.at(cross, (old, new), 1)
        left, right = linear_sum_assignment(-cross)
        scores = all_metrics(x2024, new, graph)
        rows.append({'id': name, 'model_sha256': models_manifest['files_sha256'][name + '.json'],
                     'requested_k': 4, 'occupied_k': int(len(np.unique(new))),
                     'cluster_sizes_2024': np.bincount(new, minlength=4).tolist(),
                     'common_2024_metrics': scores,
                     'ARI_2023_to_2024': float(adjusted_rand_score(old, new)),
                     'changed_with_saved_label_indices': int(np.count_nonzero(old != new)),
                     'changed_after_optimal_label_matching': int(len(ids) - cross[left, right].sum()),
                     'cross_tab_2023_rows_2024_columns': cross.tolist(), 'labels_2024': new.tolist()})
    result = {'status': 'completed', 'scope': 'already_inspected_2024_retrospective_annual_frozen_application',
              'no_fitting_or_model_selection': True, 'baseline_default_preserved': True,
              'ids': ids, 'panel_sha256_consumed': panel_digest, 'frozen_scaler': scaler,
              'features': 'coordinatewise median of 12 observed 2024 months transformed with the frozen 2023 scaler',
              'reference_graph': graph_info,
              'metric_geometry': 'same five frozen 2023 coordinates and same 2024 reference kNN graph for all models',
              'models': rows,
              'limits': ['2024 values were inspected before this application; this is not untouched confirmation.',
                         'No promotion or hyperparameter selection uses these scores.',
                         'Persistent territory IDs do not establish stable administrative boundaries.',
                         'Consumer-profile geometry does not prove economic production types.']}
    write_json(output, result)
    print(json.dumps({'status': 'completed', 'models': [{key: row[key] for key in
                           ('id', 'cluster_sizes_2024', 'ARI_2023_to_2024', 'changed_with_saved_label_indices')}
                          | {'SW': row['common_2024_metrics']['SW'], 'CH': row['common_2024_metrics']['CH'],
                             'S_Dbw': row['common_2024_metrics']['S_Dbw'], 'AVI': row['common_2024_metrics']['AVI'],
                             'AVU': row['common_2024_metrics']['AVU']} for row in rows]}), flush=True)
