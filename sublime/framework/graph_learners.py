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



# Laplacian Positional Encoding: k_lap smallest non-trivial eigenvectors of
# L = I - norm_adj (norm_adj is already D^{-1/2} A D^{-1/2}).
# Handles both dense and sparse norm_adj to avoid materialising n×n matrices
# for large graphs (e.g. Pubmed 19k nodes).  Uses scipy ARPACK for n > 3000.
def _compute_lap_pe(norm_adj, k_lap):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = norm_adj.shape[0]
    k_req = min(k_lap, n - 2)  # need ≥2 eigenvalues to skip the trivial one
    if k_req <= 0:
        return torch.zeros(n, k_lap)

    if norm_adj.is_sparse:
        coo = norm_adj.coalesce().cpu()
        rows = coo.indices()[0].numpy()
        cols = coo.indices()[1].numpy()
        vals = coo.values().float().numpy()
        A_sp = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
        L_sp = sp.eye(n, dtype=np.float32, format="csr") - A_sp
        eigvals, eigvecs = spla.eigsh(L_sp, k=k_req + 1, which="SM", tol=1e-3)
        order = np.argsort(eigvals)
        pe = eigvecs[:, order[1:k_req + 1]].astype(np.float32)
    else:
        a = norm_adj.cpu().float().numpy()
        L = np.eye(n, dtype=np.float32) - a
        if n <= 3000:
            eigvals, eigvecs = np.linalg.eigh(L.astype(np.float64))
            pe = eigvecs[:, 1:k_req + 1].astype(np.float32)
        else:
            L_sp = sp.csr_matrix(L)
            eigvals, eigvecs = spla.eigsh(L_sp, k=k_req + 1, which="SM", tol=1e-3)
            order = np.argsort(eigvals)
            pe = eigvecs[:, order[1:k_req + 1]].astype(np.float32)

    # Sign normalisation: make the element with the largest absolute value
    # in each column positive (deterministic convention).
    signs = np.sign(pe[np.abs(pe).argmax(axis=0), np.arange(pe.shape[1])])
    signs[signs == 0.0] = 1.0
    pe *= signs[None, :]

    # Pad to exactly k_lap columns if the graph was too small.
    if pe.shape[1] < k_lap:
        pe = np.concatenate(
            [pe, np.zeros((n, k_lap - pe.shape[1]), dtype=np.float32)], axis=1
        )

    return torch.from_numpy(pe)


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

class _TransformerEncoderLayer(nn.Module):
    """Wraps nn.TransformerEncoderLayer to expose the (n, d) -> (n, d) interface
    expected by _MetricLearner._apply_layer.

    PyTorch <1.9 uses (seq, batch, d) convention; we treat the n nodes as the
    sequence and use a singleton batch dimension."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x):
        # x: (n, d) -> (n, 1, d) [seq, batch, d] -> (n, d)
        return self.layer(x.unsqueeze(1)).squeeze(1)


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

# Transformer encoder block: global self-attention over all n nodes as a sequence.
# nhead=1 is the safe default.
#
# k_lap controls Laplacian Positional Encoding (LapPE):
#   k_lap=0  — no LapPE (default; required for structure inference where no
#               initial graph is available).
#   k_lap>0  — add LapPE before the first encoder layer.  Call set_lap_pe()
#               once before training to pre-compute and cache the eigenvectors
#               from the normalized anchor adjacency (structure refinement only).
class TransformerLearner(_MetricLearner):
    def __init__(self, in_dim, n_layers=2, k=30, nhead=1,
                 dim_feedforward=None, dropout=0.1,
                 activation="relu", sparse=False, k_lap=0):
        if in_dim % nhead != 0:
            raise ValueError(
                f"in_dim ({in_dim}) must be divisible by nhead ({nhead})"
            )
        self.in_dim = in_dim
        self.nhead = nhead
        self.dim_feedforward = 4 * in_dim if dim_feedforward is None else dim_feedforward
        self.dropout = dropout
        self.k_lap = k_lap
        super().__init__(n_layers, k, activation, sparse)   # builds self.layers
        # LapPE projection and buffer — registered after nn.Module is initialised
        self.register_buffer("_lap_pe", None)
        if k_lap > 0:
            self.lap_proj = nn.Linear(k_lap, in_dim, bias=False)

    def set_lap_pe(self, norm_adj):
        """Pre-compute LapPE from the (already symmetrically-normalised) anchor adjacency.

        Call once after construction and before moving the model to device.
        Only meaningful for structure refinement; in structure inference the
        anchor is the identity, for which LapPE is degenerate.
        """
        if self.k_lap == 0:
            return
        pe = _compute_lap_pe(norm_adj, self.k_lap)   # (n, k_lap) float32 CPU
        # register_buffer replaces the placeholder None with the real tensor
        self.register_buffer("_lap_pe", pe)

    def _make_layer(self):
        return _TransformerEncoderLayer(
            self.in_dim, self.nhead, self.dim_feedforward, self.dropout
        )

    def _embed(self, features, adj):
        h = features
        # Add LapPE if it was pre-computed (structure refinement only)
        if self.k_lap > 0 and self._lap_pe is not None:
            h = h + self.lap_proj(self._lap_pe.to(h.device))
        # Each TransformerEncoderLayer already contains its own FFN activation,
        # LayerNorm, and residual connections — no inter-layer activation needed.
        for layer in self.layers:
            h = self._apply_layer(layer, h, adj)
        return h


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
