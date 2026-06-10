"""Graph structure learners p_omega(.).

A learner maps node features X (and, for the GNN learner, the original
adjacency A) to a sketched adjacency matrix S~:

    FGP:     S~ = sigma(Omega)
    Metric:  S~ = phi(h_omega(X, A))    with phi = cosine similarity

The post-processor (post_processor.py) then turns S~ into the final structure
S via sparsify -> activate -> symmetrize -> normalize.

Two exceptions to that clean split:
  - FGP applies its own activation (ELU + 1) and is never sparsified, so its
    forward already returns the activated S~.
  - For sparse=True, the metric learners do top-k + degree-norm themselves
    because the full n x n similarity matrix won't fit in memory.
"""
from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph

# FGP init shift. The paper doesn't name this constant; main.py hardcodes 6.
# After kNN+I is scaled by (x * I - I), edges land at 0 and non-edges at -I,
# so elu+1 gives ~1 for edges and ~0 for non-edges at init.
_FGP_INIT_SHIFT = 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# between-layer activation for the metric learners
def _apply_activation(x, name):
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"unsupported activation: {name!r}")


# dense phi(E): pairwise cosine similarity
def _cosine_similarity(embeddings):
    embeddings = F.normalize(embeddings, dim=1, p=2)
    return embeddings @ embeddings.t()


# batched cosine + symmetric top-k for the large-graph sparse path.
# k+1 because every row's top entry is its self-similarity (=1), so we keep
# k actual neighbours per node, matching the dense path's top_k(sim, k+1).
# After the top-k, values are degree-normalized: v *= D_row^-0.5 * D_col^-0.5,
# where D = row_sum + col_sum of the raw top-k similarities (matching knn_fast).
#
# IMPORTANT: values are detached from the computation graph. This matches the
# original knn_fast() which writes into a pre-allocated torch.zeros() tensor,
# effectively breaking gradient flow to the learner parameters through the edge
# values. The learner's sparse path does NOT receive gradients from the encoder
# in the original code (DGL message passing on detached edata['w']).
def _sparse_topk_similarity(embeddings, k, batch_size=1000):
    embeddings = F.normalize(embeddings, dim=1, p=2)
    n = embeddings.shape[0]
    keep = k + 1

    # pre-allocate output arrays (matching knn_fast's pattern)
    total = n * keep
    values = torch.zeros(total, device=embeddings.device)
    rows = torch.zeros(total, dtype=torch.long, device=embeddings.device)
    cols = torch.zeros(total, dtype=torch.long, device=embeddings.device)
    norm_row = torch.zeros(n, device=embeddings.device)
    norm_col = torch.zeros(n, device=embeddings.device)

    # walk the rows in chunks so the full n x n sim matrix is never materialized
    index = 0
    while index < n:
        end = min(index + batch_size, n)
        sims = embeddings[index:end] @ embeddings.t()
        vals, inds = sims.topk(k=keep, dim=-1)
        chunk = (end - index) * keep
        values[index * keep:end * keep] = vals.reshape(-1)
        cols[index * keep:end * keep] = inds.reshape(-1)
        rows[index * keep:end * keep] = torch.arange(index, end, device=embeddings.device).view(-1, 1).repeat(1, keep).reshape(-1)
        # accumulate degree
        norm_row[index:end] = vals.sum(dim=1)
        norm_col.index_add_(-1, inds.reshape(-1), vals.reshape(-1))
        index += batch_size

    # symmetric degree normalization on the values (same as knn_fast)
    norm = norm_row + norm_col
    values = values * (norm[rows].pow(-0.5) * norm[cols].pow(-0.5))

    # relu: zero any negative values (the original applies relu via
    # apply_non_linearity with non_linearity='relu')
    values = F.relu(values)

    # also keep the transposed entries so S~ stays symmetric
    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    values_sym = torch.cat([values, values])
    indices = torch.stack([rows_sym, cols_sym])
    return torch.sparse_coo_tensor(indices, values_sym.detach(), (n, n)).coalesce()



# kNN graph over raw features + self-loops, shifted so that after FGP's elu+1
# activation, edges start at ~1 and non-edges at ~0
def _fgp_init(features, k, metric, shift):
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    adj = kneighbors_graph(features, k, metric=metric, include_self=False)
    adj = adj.toarray().astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)   # add self-loops
    return adj * shift - shift                            # edges -> 0, non-edges -> -shift


# one GAT-like layer: rescale every feature dim by a learned scalar
class _Attentive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


# one GCN layer: h' = norm_adj @ (h W + b)
class _GCNConv(nn.Module):
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

# every learner returns S~: dense (n, n), or sparse COO (n, n) with k entries per row
class GraphLearner(nn.Module, abc.ABC):
    def __init__(self, sparse=False):
        super().__init__()
        self.sparse = sparse

    # adj is only used by GNNLearner (structure refinement), ignored elsewhere
    @abc.abstractmethod
    def forward(self, features, adj=None):
        ...


# ---------------------------------------------------------------------------
# FGP: every entry of A is a free parameter, init from kNN over X
# ---------------------------------------------------------------------------

# sigma = ELU + 1 is applied here directly; FGP skips the post-processor's
# sparsify/activate steps to keep gradients flowing to every entry of S~
class FGPLearner(GraphLearner):
    def __init__(self, features, k, knn_metric="cosine", init_shift=_FGP_INIT_SHIFT):
        super().__init__(sparse=False)
        init_adj = _fgp_init(features, k, knn_metric, init_shift)
        self.omega = nn.Parameter(torch.from_numpy(init_adj).float())

    def forward(self, features=None, adj=None):
        return F.elu(self.omega) + 1


# ---------------------------------------------------------------------------
# metric learners: S~ = cosine(h_omega(X)), three flavours of h_omega
# ---------------------------------------------------------------------------

# Base for ATT / MLP / GNN: build layers, embed, then cosine sim.
# Subclasses just define how to build and apply a layer.
class _MetricLearner(GraphLearner):
    # GNN overrides to True so adj is normalized and passed into every layer
    needs_adj = False

    def __init__(self, n_layers, k, activation, sparse):
        super().__init__(sparse=sparse)
        self.k = k
        self.activation = activation
        self.layers = nn.ModuleList(self._make_layer() for _ in range(n_layers))

    # subclass: how to construct one layer
    @abc.abstractmethod
    def _make_layer(self):
        ...

    # subclass: how to apply one layer (ATT/MLP ignore norm_adj, GNN uses it)
    def _apply_layer(self, layer, h, norm_adj):
        return layer(h)

    def _embed(self, features, adj):
        # adj is used as-is by the GNN learner. The original passes
        # pre-normalized D^-1/2 A D^-1/2 adj to GNN_learner and runs layers
        # directly on it; we follow the same contract -- caller normalizes.
        h = features
        for i, layer in enumerate(self.layers):
            h = self._apply_layer(layer, h, adj)
            if i != len(self.layers) - 1:
                h = _apply_activation(h, self.activation)
        return h

    def forward(self, features, adj=None):
        if self.needs_adj and adj is None:
            raise ValueError(f"{type(self).__name__} requires the original adjacency `adj`")
        embeddings = self._embed(features, adj)
        if self.sparse:
            return _sparse_topk_similarity(embeddings, self.k)
        return _cosine_similarity(embeddings)


# Attentive: per-dim scalars, treats feature dims as independent.
# Weights init to 1 so E = X at the start of training.
class AttentiveLearner(_MetricLearner):
    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        self.in_dim = in_dim
        super().__init__(n_layers, k, activation, sparse)

    def _make_layer(self):
        return _Attentive(self.in_dim)


# Linear with weight = identity and bias = 0, so layer(X) = X at step 0.
# The paper specifies identity weights AND zero biases (so E = X at the first
# iteration); the original code only sets the weights and leaves PyTorch's
# random bias init. This file follows the paper.
def _identity_linear(in_dim):
    layer = nn.Linear(in_dim, in_dim)
    layer.weight = nn.Parameter(torch.eye(in_dim))
    nn.init.zeros_(layer.bias)
    return layer


# MLP: square linear maps, also model correlations between feature dims.
class MLPLearner(_MetricLearner):
    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        self.in_dim = in_dim
        super().__init__(n_layers, k, activation, sparse)

    def _make_layer(self):
        return _identity_linear(self.in_dim)


# GNN: GCN over the original adjacency, so S~ also reflects topology.
# SR only -- needs adj.
#
# Same init situation as the MLP: the paper wants E = A_hat X at step 0.
# The original code tries `layer.weight = eye`, but `layer` is the outer GCN
# module (no `weight` attr -- the real Linear lives at `layer.linear`), so
# identity init silently fails and the GCN starts random. We do what the paper
# says: set the inner Linear to identity.
class GNNLearner(_MetricLearner):
    needs_adj = True

    def __init__(self, in_dim, n_layers=2, k=30, knn_metric="cosine",
                 activation="relu", sparse=False):
        self.in_dim = in_dim
        super().__init__(n_layers, k, activation, sparse)

    def _make_layer(self):
        conv = _GCNConv(self.in_dim, self.in_dim)
        conv.linear = _identity_linear(self.in_dim)
        return conv

    def _apply_layer(self, layer, h, norm_adj):
        return layer(h, norm_adj)
