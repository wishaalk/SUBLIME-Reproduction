"""Post-processor: turns S~ from a graph learner into the final structure S.

Dense path (small graphs: cora, citeseer, wine, 20news, etc.):
    S~ -> top_k -> relu -> symmetrize -> normalize
    In the original code, top_k and relu live inside the learner's forward;
    we split them out here for clarity.

Sparse path (pubmed, ogbn-arxiv):
    The learner already does top-k + degree-norm in _sparse_topk_similarity.
    The original code's main.py never calls symmetrize/normalize on the sparse
    graph either -- it goes straight into the GCN encoder. So this is a no-op.

FGP path:
    Already activated (elu+1), never sparsified.
    Only symmetrize + normalize.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EOS = 1e-10  # avoid division by zero (same constant as in the original utils.py)


# ---------------------------------------------------------------------------
# individual ops
# ---------------------------------------------------------------------------

# keep only top-k entries per row, zero the rest
def _top_k(adj, k):
    n = adj.shape[0]
    values, indices = adj.topk(k=int(k), dim=-1)
    mask = torch.zeros_like(adj)
    mask.scatter_(1, indices, 1.0)
    return adj * mask


# (A + A^T) / 2
def _symmetrize(adj):
    return (adj + adj.T) / 2


# D^-1/2 A D^-1/2 (symmetric normalization, no self-loop addition here)
def _normalize_sym(adj):
    deg = adj.sum(dim=1) + EOS
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


# ---------------------------------------------------------------------------
# full post-processing pipelines
# ---------------------------------------------------------------------------

def post_process_dense(adj, k, is_fgp=False, *,
                       do_topk=True, do_relu=True,
                       do_sym=True, do_norm=True):
    """Dense pipeline: top_k -> relu -> symmetrize -> normalize.

    FGP skips top_k and relu unconditionally (already activated via elu+1).

    The four `do_*` toggles let an ablation study disable individual steps.
    Defaults preserve the original pipeline.
    """
    if not is_fgp:
        if do_topk:
            adj = _top_k(adj, k + 1)    # +1 for the self-entry
        if do_relu:
            adj = F.relu(adj)           # zero negative cosines
    if do_sym:
        adj = _symmetrize(adj)
    if do_norm:
        adj = _normalize_sym(adj)
    return adj


def post_process_sparse(adj):
    """Sparse pipeline: no-op. Learner already handled top-k + degree-norm."""
    return adj
