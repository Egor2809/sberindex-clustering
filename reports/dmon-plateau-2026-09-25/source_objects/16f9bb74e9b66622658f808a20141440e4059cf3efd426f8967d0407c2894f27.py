# Copyright 2026 The Google Research Authors.
# Copyright 2026 SberAI contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Full license: docs/DMON_REFERENCE.md.
"""Sparse CPU DMoN: paper Eq. (5), with the official GCN architecture.

PyTorch adaptation, not bitwise reproduction of the TensorFlow code. The
normalization and self-loop differences are documented in DMON_REFERENCE.md.
No outcome labels, feature reconstruction, entropy or orthogonality loss.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn

SOURCE_REVISION = "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
METHOD = "dmon_paper_eq5_sparse_gcn"


def _validated_inputs(x, adjacency):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite node-by-feature matrix")
    if not sparse.issparse(adjacency):
        raise ValueError("adjacency must be a scipy sparse matrix")
    a = sparse.csr_matrix(adjacency, dtype=np.float32, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if a.shape != (len(x), len(x)):
        raise ValueError("adjacency shape must match the feature rows")
    if not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError("adjacency must have finite nonnegative weights")
    if np.any(a.diagonal() != 0):
        raise ValueError("input adjacency must not contain self loops")
    delta = a - a.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-6:
        raise ValueError("adjacency must be symmetric")
    # Remove only accepted floating-point asymmetry, without changing inputs.
    a = ((a + a.T) * 0.5).tocsr()
    volume = float(a.sum(dtype=np.float64))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("modularity requires positive finite graph volume")
    return x, a


def _torch_sparse(a, dtype=torch.float32):
    coo = a.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    return torch.sparse_coo_tensor(
        indices, torch.as_tensor(coo.data, dtype=dtype), a.shape,
        dtype=dtype, device="cpu").coalesce()


def normalized_adjacency(adjacency):
    """D^-1/2 A D^-1/2, without added loops; zero rows for isolated nodes."""
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degrees, dtype=float)
    np.divide(1.0, np.sqrt(degrees), out=inverse, where=degrees > 0)
    return sparse.diags(inverse) @ adjacency @ sparse.diags(inverse)


def dmon_loss(assignments, adjacency, degrees=None):
    """Return (total, negative soft modularity, collapse), JMLR Eq. (5).

    adjacency is a symmetric sparse torch tensor, with no self loops and
    positive volume. assignments is a row-stochastic n-by-k tensor. These
    preconditions are checked by fit_dmon; this differentiable kernel assumes
    them. The dense n-by-n modularity matrix is never materialized.
    """
    if degrees is None:
        degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
    volume = degrees.sum()  # 2m, NOT m; also total strength for weighted A.
    observed = (assignments * torch.sparse.mm(adjacency, assignments)).sum()
    cluster_volume = degrees @ assignments
    expected = cluster_volume.square().sum() / volume
    spectral = -(observed - expected) / volume
    sizes = assignments.sum(dim=0)
    collapse = sizes.norm() * math.sqrt(assignments.shape[1]) / len(assignments) - 1
    return spectral + collapse, spectral, collapse


class SparseDMoN(nn.Module):
    """One SELU GCN with a trainable per-channel skip, then softmax."""

    def __init__(self, n_features, hidden_dim, n_clusters):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.skip_weight = nn.Parameter(torch.empty(hidden_dim))
        self.assignment = nn.Linear(hidden_dim, n_clusters)
        nn.init.xavier_uniform_(self.kernel)
        nn.init.uniform_(self.skip_weight, -math.sqrt(3 / hidden_dim),
                         math.sqrt(3 / hidden_dim))
        nn.init.orthogonal_(self.assignment.weight)
        nn.init.zeros_(self.assignment.bias)

    def forward(self, x, norm_adjacency):
        projected = x @ self.kernel
        hidden = torch.nn.functional.selu(
            torch.sparse.mm(norm_adjacency, projected)
            + projected * self.skip_weight + self.bias)
        return torch.softmax(self.assignment(hidden), dim=1)



class PlateauMonitor:
    """Stop only after consecutive small loss gains AND stable hard labels."""
    def __init__(self, settings):
        self.settings = dict(settings)
        for name in ('check_every', 'min_epochs', 'patience'):
            value = self.settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('min_delta', 'max_label_change_fraction'):
            value = self.settings[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.settings['max_label_change_fraction'] > 1:
            raise ValueError('label-change threshold must not exceed one')
        self.previous = None
        self.streak = 0
        self.history = []

    def observe(self, epoch, best_objective, labels):
        if epoch % self.settings['check_every']:
            return False
        labels = np.asarray(labels).copy()
        gain = drift = None
        if self.previous is not None:
            old_loss, old_labels = self.previous
            gain = old_loss - best_objective
            drift = float(np.mean(labels != old_labels))
        eligible = epoch >= self.settings['min_epochs'] and gain is not None
        quiet = eligible and 0 <= gain <= self.settings['min_delta'] and drift <= self.settings['max_label_change_fraction']
        self.streak = self.streak + 1 if quiet else 0
        self.history.append({'epoch': epoch, 'best_objective': best_objective,
                             'best_loss_gain': gain, 'hard_label_change_fraction': drift,
                             'consecutive_plateau_windows': self.streak})
        self.previous = best_objective, labels
        return self.streak >= self.settings['patience']


def fit_dmon(x, sparse_adjacency, n_clusters, seed, epochs,
             hidden_dim=64, learning_rate=0.001, *,
             return_assignments=False, output_dir=None, plateau=None):
    """Fit to a fixed budget or an explicit loss/label plateau; select minimum loss.

    Returns (labels, info), or (labels, info, soft_assignments) if requested.
    Input attributes are used unchanged: preprocessing belongs to the caller.
    Epoch 0 is the initialization; epochs 1..epochs follow Adam updates.
    Ties retain the earliest checkpoint. Empty hard clusters are reported,
    never repaired. No convergence or global-optimality assertion is made.

    With output_dir, save trace.json, info.json, best_model.npz,
    assignments.npy and labels.npy. Checkpoints contain plain parameter arrays
    (no pickle) and are intended for inference, not exact optimizer resumption.
    With plateau settings, stopping uses only training loss and label stability.
    It is an empirical stopping rule, not a proof of a stationary/global optimum.
    CPU thread count and process resource policy are controlled by the caller.
    """
    started = time.perf_counter()
    monitor = PlateauMonitor(plateau) if plateau is not None else None
    x, a = _validated_inputs(x, sparse_adjacency)
    for value, name, lower, upper in (
        (n_clusters, "n_clusters", 2, len(x)),
        (hidden_dim, "hidden_dim", 1, None),
        (epochs, "epochs", 1, None),
        (seed, "seed", 0, 2**63 - 1),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if value < lower or (upper is not None and value > upper):
            raise ValueError(f"{name} is outside its valid range")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if output_dir is not None:
        destination = Path(output_dir)
        filenames = ("best_model.npz", "trace.json", "info.json", "labels.npy", "assignments.npy")
        if any((destination / name).exists() for name in filenames):
            raise FileExistsError("DMoN output files already exist; use a new output_dir")
    features = torch.from_numpy(x.copy())
    graph = _torch_sparse(a)
    normalized = _torch_sparse(normalized_adjacency(a))
    degrees = torch.from_numpy(np.asarray(a.sum(axis=1)).ravel())
    trace = []
    best_objective = float("inf")
    best_state = None
    best_epoch = None
    best_labels = None
    stop_reason = "fixed_epoch_budget"
    # Isolate CPU RNG state and avoid initializing CUDA on a CPU-only fit.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = SparseDMoN(x.shape[1], int(hidden_dim), int(n_clusters))
        # Keras Adam's default epsilon is 1e-7 (PyTorch's default is 1e-8).
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
        for epoch in range(int(epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            assignments = model(features, normalized)
            loss, spectral, collapse = dmon_loss(assignments, graph, degrees)
            values = [float(v.detach()) for v in (loss, spectral, collapse)]
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError(f"nonfinite DMoN objective at epoch {epoch}")
            trace.append({"epoch": epoch, "objective": values[0],
                          "negative_soft_modularity": values[1],
                          "collapse_loss": values[2]})
            if values[0] < best_objective:
                best_objective = values[0]
                best_epoch = epoch
                best_state = {name: value.detach().clone()
                              for name, value in model.state_dict().items()}
            if best_epoch == epoch:
                best_labels = assignments.detach().argmax(dim=1).numpy().copy()
            should_stop = monitor.observe(epoch, best_objective, best_labels) if monitor else False
            if monitor and epoch % monitor.settings['check_every'] == 0:
                progress = {'epoch': epoch, 'best_epoch': best_epoch,
                            'best_objective': best_objective,
                            'elapsed_seconds': time.perf_counter() - started,
                            'plateau_windows': monitor.streak}
                if output_dir is not None:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination/'progress.json').write_text(json.dumps(progress)+'\n', encoding='utf-8')
                print(json.dumps({'dmon_progress': progress, 'k': int(n_clusters), 'seed': int(seed)}), flush=True)
            if monitor and epoch == 1000 and output_dir is not None:
                np.save(destination/'epoch1000_labels.npy', best_labels, allow_pickle=False)
            if should_stop:
                stop_reason = 'training_plateau'
                break
            if epoch < epochs:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError(f"nonfinite DMoN gradient at epoch {epoch}")
                optimizer.step()
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            soft = model(features, normalized).numpy().copy()
    labels = soft.argmax(axis=1).astype(np.int64)
    sizes = np.bincount(labels, minlength=n_clusters)
    info = {
        "method": METHOD, "reference_revision": SOURCE_REVISION,
        "seed": int(seed), "epochs": int(epoch), "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate), "collapse_weight": 1.0,
        "dropout_rate": 0.0, "added_self_loops": False,
        "selection": "minimum_training_objective_including_initialization",
        "stop_reason": stop_reason, "best_epoch": best_epoch,
        "initial_objective": trace[0]["objective"],
        "final_objective": trace[-1]["objective"],
        "best_objective": best_objective,
        "best_negative_soft_modularity": trace[best_epoch]["negative_soft_modularity"],
        "best_collapse_loss": trace[best_epoch]["collapse_loss"],
        "n_nodes": len(x), "n_features": x.shape[1], "n_clusters": int(n_clusters),
        "n_edges": a.nnz // 2, "graph_volume": float(degrees.sum()),
        "isolated_nodes": int(np.count_nonzero(np.asarray(a.sum(axis=1)).ravel() == 0)),
        "hard_cluster_sizes": sizes.tolist(), "occupied_clusters": int((sizes > 0).sum()),
        "soft_cluster_sizes": soft.sum(axis=0, dtype=np.float64).tolist(),
        "torch_version": torch.__version__, "dtype": "float32", "device": "cpu",
        "trace": trace,
    }
    if monitor:
        info.update({'max_epochs': int(epochs), 'plateau_rule': monitor.settings,
                     'plateau_checks': monitor.history,
                     'plateau_reached': stop_reason == 'training_plateau',
                     'training_seconds': time.perf_counter() - started,
                     'initialization': 'replayed_from_seed_to_preserve_Adam_trajectory'})
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths = [destination / name for name in
                 ("best_model.npz", "trace.json", "info.json", "labels.npy", "assignments.npy")]
        if any(path.exists() for path in paths):
            raise FileExistsError("DMoN output files already exist; use a new output_dir")
        np.savez(paths[0], **{name: value.numpy() for name, value in best_state.items()})
        paths[1].write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths[2].write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.save(paths[3], labels, allow_pickle=False)
        np.save(paths[4], soft, allow_pickle=False)
    return (labels, info, soft) if return_assignments else (labels, info)
