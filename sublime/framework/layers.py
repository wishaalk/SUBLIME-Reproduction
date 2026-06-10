"""Shared GCN building blocks. Same as the original layers.py, but without the DGL path."""
from __future__ import annotations

import torch
import torch.nn as nn


# One GCN conv: h' = adj @ (h W + b).
# The original splits this in two: GCNConv_dense for torch tensor adj (with a
# `sparse` flag selecting matmul vs torch.sparse.mm) and GCNConv_dgl for DGL
# graph adj. We don't carry DGL through our learners, so a single conv with
# is_sparse dispatch covers everything we need.
#
# IMPORTANT: In the sparse path, adj.values() must NOT require grad. The original
# avoids this problem by using DGL message passing; our version achieves the same
# effect by detaching values in _sparse_topk_similarity (see graph_learners.py).
# If values ever required grad here, torch.sparse.mm backward would try to
# materialize a dense N x N gradient and OOM on large graphs.
class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = self.linear(x)
        if adj.is_sparse:
            return torch.sparse.mm(adj, h)
        return adj @ h


# Edge dropout on a sparse COO adjacency. nn.Dropout would densify it, so we
# do it by masking the value array directly and rescaling by 1/keep_prob.
#
# Matches the original: no self.training gate, edges drop in eval too. Two
# changes are unavoidable: torch.sparse.FloatTensor is deprecated, and the
# original's torch.rand on CPU crashes when x lives on GPU.
class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super().__init__()
        self.kprob = 1 - dprob

    def forward(self, x):
        values = x._values()
        mask = ((torch.rand(values.size(), device=values.device) + self.kprob).floor()).bool()
        rc = x._indices()[:, mask]
        val = values[mask] * (1.0 / self.kprob)
        return torch.sparse_coo_tensor(rc, val, x.shape)
