"""Graph learners p_omega(.) (Sec 4.1 of the SUBLIME paper).

A graph learner maps the node features X (and, for the GNN learner, the
original adjacency matrix A) to a *sketched* adjacency matrix S~:

    FGP learner (Eq. 1):  S~ = sigma(Omega)
    Metric learners (Eq. 2): S~ = phi(h_omega(X, A)) = phi(E)

phi is the (non-parametric) cosine similarity between node embeddings E.
The post-processor (post_processor.py) turns S~ into the learned structure
S via sparsification, activation, symmetrization and normalization
(Eq. 6-8).

Two exceptions to that clean split, both following the paper:
  - The FGP learner applies its own activation (ELU + 1, Sec 4.2) and is
    never sparsified, so its `forward` already returns the post-activation
    S~ that the post-processor should only symmetrize/normalize.
  - When `sparse=True`, the metric learners perform a locality-sensitive
    kNN sparsification themselves (Sec 4.5) and return S~ as a sparse COO
    tensor containing only the top-k entries per row, since the full n x n
    similarity matrix would not fit in memory for large graphs.
"""
from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _apply_activation(x, name):
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"unsupported activation: {name!r}")


def _cosine_similarity(embeddings):
    """Dense phi(E): pairwise cosine similarity, Eq. 2."""
    embeddings = F.normalize(embeddings, dim=1, p=2)
    return embeddings @ embeddings.t()


def _sparse_topk_similarity(embeddings, k, batch_size=1000):
    """Locality-sensitive approximation of phi(E) + kNN sparsification (Sec 4.5).

    Computes cosine similarities batch-by-batch so the full n x n matrix is
    never materialized, keeping only the top-k neighbors per node (in both
    directions, to keep S~ symmetric).
    """
    embeddings = F.normalize(embeddings, dim=1, p=2)
    n = embeddings.shape[0]

    rows, cols, values = [], [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = embeddings[start:end] @ embeddings.t()
        vals, inds = sims.topk(k=k, dim=-1)
        rows.append(torch.arange(start, end, device=embeddings.device).repeat_interleave(k))
        cols.append(inds.reshape(-1))
        values.append(vals.reshape(-1))

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    values = torch.cat(values)

    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    values_sym = torch.cat([values, values])
    indices = torch.stack([rows_sym, cols_sym])
    return torch.sparse_coo_tensor(indices, values_sym, (n, n)).coalesce()


def _knn_init_adj(features, k, metric):
    """Binary, symmetric kNN graph over the raw input features.

    Used to initialize the FGP learner's parameters: edges of the kNN graph
    start at 1, all other entries start at 0 (Sec 4.3.1).
    """
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    adj = kneighbors_graph(features, k, metric=metric, include_self=False)
    adj = adj.toarray().astype(np.float32)
    return np.maximum(adj, adj.T)


def _normalize_adj(adj, sparse):
    """Symmetric normalization with self-loops: D~^-1/2 (A + I) D~^-1/2 (Eq. 5)."""
    n = adj.shape[0]
    if sparse:
        adj = adj.coalesce()
        eye_idx = torch.arange(n, device=adj.device)
        eye = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx]), torch.ones(n, device=adj.device), (n, n)
        )
        adj = (adj + eye).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        row, col = adj.indices()
        values = adj.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(adj.indices(), values, (n, n)).coalesce()

    adj = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = adj.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


class _Attentive(nn.Module):
    """One layer of the GAT-like embedding network (Eq. 3): rescales each
    feature dimension by a learned weight (Hadamard product)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _GCNConv(nn.Module):
    """One GCN layer (Eq. 5): h' = norm_adj @ (h @ W + b)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, norm_adj):
        h = self.linear(x)
        if norm_adj.is_sparse:
            return torch.sparse.mm(norm_adj, h)
        return norm_adj @ h


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------

class GraphLearner(nn.Module, abc.ABC):
    """Base class for the graph structure learners p_omega(.) of Sec 4.1."""

    def __init__(self, sparse=False):
        super().__init__()
        self.sparse = sparse

    @abc.abstractmethod
    def forward(self, features, adj=None):
        """Compute the sketched adjacency matrix S~.

        Args:
            features: node feature matrix X, shape (n, d).
            adj: original adjacency matrix A. Required by the GNN learner
                (structure refinement only); ignored otherwise.

        Returns:
            S~: a dense (n, n) tensor, or - for the metric learners with
            `sparse=True` - a sparse COO (n, n) tensor with k entries per row.
        """


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------

class FGPLearner(GraphLearner):
    """Full graph parameterization learner (Eq. 1).

    Each entry of the adjacency matrix is an independent learnable
    parameter Omega_ij, initialized from a kNN graph over the input features
    (Sec 4.3.1). sigma is ELU + 1 (Sec 4.2): this is applied here directly
    because FGP skips the post-processor's sparsification/activation steps,
    to keep the gradient flowing to every entry of S~.
    """

    def __init__(self, features, k, knn_metric="cosine"):
        super().__init__(sparse=False)
        init_adj = _knn_init_adj(features, k, knn_metric)
        self.omega = nn.Parameter(torch.from_numpy(init_adj).float())

    def forward(self, features=None, adj=None):
        return F.elu(self.omega) + 1


class AttentiveLearner(GraphLearner):
    """GAT-like attentive metric learner (Eq. 2 & 3).

    Each layer rescales every feature dimension by a learned weight vector,
    assuming features contribute independently to the existence of an edge
    with no correlation between them. Weights are initialized to 1, so
    E = X at the start of training (Sec 4.3.1).
    """

    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        super().__init__(sparse=sparse)
        self.layers = nn.ModuleList(_Attentive(in_dim) for _ in range(n_layers))
        self.k = k
        self.activation = activation

    def _embed(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = _apply_activation(h, self.activation)
        return h

    def forward(self, features, adj=None):
        embeddings = self._embed(features)
        if self.sparse:
            return _sparse_topk_similarity(embeddings, self.k)
        return _cosine_similarity(embeddings)


class MLPLearner(GraphLearner):
    """MLP-based metric learner (Eq. 2 & 4).

    Each layer is a square linear map, additionally modeling correlations
    and combinations between feature dimensions compared to the attentive
    learner. Weights are initialized to the identity (and biases to zero),
    so E = X at the start of training (Sec 4.3.1).
    """

    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        super().__init__(sparse=sparse)
        self.layers = nn.ModuleList(nn.Linear(in_dim, in_dim) for _ in range(n_layers))
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(in_dim))
            nn.init.zeros_(layer.bias)
        self.k = k
        self.activation = activation

    def _embed(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = _apply_activation(h, self.activation)
        return h

    def forward(self, features, adj=None):
        embeddings = self._embed(features)
        if self.sparse:
            return _sparse_topk_similarity(embeddings, self.k)
        return _cosine_similarity(embeddings)


class GNNLearner(GraphLearner):
    """GCN-based metric learner (Eq. 2 & 5), structure refinement only.

    Embeds features together with the (normalized) original adjacency
    matrix via GCN layers, so the learned similarities also reflect the
    original topology. Weights are initialized to the identity (and biases
    to zero), as for the MLP learner.
    """

    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        super().__init__(sparse=sparse)
        self.layers = nn.ModuleList(_GCNConv(in_dim, in_dim) for _ in range(n_layers))
        for layer in self.layers:
            layer.linear.weight = nn.Parameter(torch.eye(in_dim))
            nn.init.zeros_(layer.linear.bias)
        self.k = k
        self.activation = activation

    def _embed(self, features, norm_adj):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h, norm_adj)
            if i != len(self.layers) - 1:
                h = _apply_activation(h, self.activation)
        return h

    def forward(self, features, adj):
        if adj is None:
            raise ValueError("GNNLearner requires the original adjacency matrix `adj`")
        norm_adj = _normalize_adj(adj, self.sparse)
        embeddings = self._embed(features, norm_adj)
        if self.sparse:
            return _sparse_topk_similarity(embeddings, self.k)
        return _cosine_similarity(embeddings)
