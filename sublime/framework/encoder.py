"""Graph encoder f_theta and the contrastive loss used to train it.

GraphEncoder is a stack of GCN layers followed by a 2-layer MLP projection
head; forward returns (z, embedding) where embedding is the raw GCN output
and z is the projected vector that feeds the loss.

GCL is a thin wrapper that holds the encoder and exposes calc_loss
(NT-Xent, temperature set as 0.2 as in the paper).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GCNConv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Edge dropout for a sparse COO adjacency that respects self.training.
# This is what the original encoder's forward actually does: it calls
# F.dropout on the DGL edata['w'] with training=self.training. (The original
# also allocates a SparseDropout module in __init__, but never calls it.)
def _sparse_edge_dropout(adj, p, training):
    if p == 0.0 or not training:
        return adj
    adj = adj.coalesce()
    new_values = F.dropout(adj.values(), p=p, training=training)
    return torch.sparse_coo_tensor(adj.indices(), new_values, adj.shape).coalesce()


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------

# n_layers GCN -> emb_dim, then a 2-layer MLP -> proj_dim.
# dropout is applied to features between GCN layers; dropout_adj is applied
# to the adjacency before each forward pass.
class GraphEncoder(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, emb_dim, proj_dim,
                 dropout, dropout_adj, sparse):
        super().__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        # in -> hidden -> ... -> hidden -> emb. Matches the original's layer
        # construction: first and last are always appended, middle layers are
        # range(n_layers - 2). With n_layers <= 2 you get exactly 2 layers.
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, emb_dim))

        # used only in the dense path; sparse goes through _sparse_edge_dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)

        # projection head g_psi: Linear -> ReLU -> Linear
        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x, adj):
        if self.sparse:
            adj = _sparse_edge_dropout(adj, self.dropout_adj_p, self.training)
        else:
            adj = self.dropout_adj(adj)

        # ReLU + feature dropout after every GCN layer except the last
        for conv in self.layers[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = self.layers[-1](x, adj)

        z = self.proj_head(embedding)
        return z, embedding


# ---------------------------------------------------------------------------
# contrastive loss
# ---------------------------------------------------------------------------

# Wraps GraphEncoder and exposes the contrastive loss as a static method
# (called from the trainer with the two view embeddings).
class GCL(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, emb_dim, proj_dim,
                 dropout, dropout_adj, sparse):
        super().__init__()
        self.encoder = GraphEncoder(n_layers, in_dim, hidden_dim, emb_dim,
                                    proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, adj):
        return self.encoder(x, adj)

    # NT-Xent on projected embeddings. sym=True averages the loss over both
    # directions, which is what main.py does. Temperature default 0.2 also
    # matches main.py.
    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_0 = -torch.log(loss_0).mean()
            loss_1 = -torch.log(loss_1).mean()
            return (loss_0 + loss_1) / 2.0

        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        return -torch.log(loss_1).mean()
