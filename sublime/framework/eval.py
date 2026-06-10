"""Downstream evaluation: train a GCN classifier on the learned adjacency
(classification), or run KMeans on the learned embeddings (clustering).

Two entry points:
    evaluate_classification(adj, features, labels, masks, cfg) -> (val_acc, test_acc)
    evaluate_clustering(embedding, labels, n_classes, n_trials) -> ClusterScores

The GCN trainer mirrors main.py::evaluate_adj_by_cls: Adam + nll_loss,
early stopping on val accuracy every 10 epochs. The clustering metrics
match utils.py::clustering_metrics: KMeans then Hungarian alignment for ACC,
plus NMI, macro-F1, ARI from sklearn.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans

from .layers import GCNConv, SparseDropout


# ---------------------------------------------------------------------------
# downstream GCN classifier
# ---------------------------------------------------------------------------

# Same architecture as the encoder's GCN stack but with a held adjacency and
# a built-in edge-dropout step before each forward.
class _DownstreamGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers,
                 dropout, dropout_adj, adj, sparse):
        super().__init__()
        # in -> hidden -> ... -> hidden -> n_classes. Same as the encoder:
        # first and last always appended, middle is range(n_layers - 2).
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, n_classes))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse
        # adj is frozen for downstream eval -- no grads flow back to the learner here
        self.adj = adj
        self.adj.requires_grad = False
        # the original uses SparseDropout for the sparse path but calls F.dropout
        # on the edges directly, same as in the encoder; keep the module for parity
        self.dropout_adj = SparseDropout(dprob=dropout_adj) if sparse else nn.Dropout(p=dropout_adj)

    def forward(self, x):
        if self.sparse:
            adj = self.adj.coalesce()
            new_values = F.dropout(adj.values(), p=self.dropout_adj_p, training=self.training)
            adj = torch.sparse_coo_tensor(adj.indices(), new_values, adj.shape).coalesce()
        else:
            adj = self.dropout_adj(self.adj)

        for conv in self.layers[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, adj)


def _accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean()


def _cls_loss_acc(model, features, labels, mask):
    logits = model(features)
    logp = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(logp[mask], labels[mask], reduction="mean")
    acc = _accuracy(logp[mask], labels[mask])
    return loss, acc


@dataclass
class ClsConfig:
    hidden_dim: int
    n_layers: int
    dropout: float
    dropout_adj: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int


# Train a fresh GCN on (adj, features, labels) and return (best_val, test_acc).
# Early stopping checks val accuracy every 10 epochs and keeps the best model.
# All tensors are moved to CUDA inside if available, matching main.py.
def evaluate_classification(adj, features, labels, n_classes,
                            train_mask, val_mask, test_mask,
                            sparse, cfg: ClsConfig):
    model = _DownstreamGCN(
        in_dim=features.shape[1], hidden_dim=cfg.hidden_dim, n_classes=n_classes,
        n_layers=cfg.n_layers, dropout=cfg.dropout, dropout_adj=cfg.dropout_adj,
        adj=adj, sparse=sparse,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bad_counter = 0
    best_val = 0
    best_model = None

    if torch.cuda.is_available():
        model = model.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        features = features.cuda()
        labels = labels.cuda()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss, _ = _cls_loss_acc(model, features, labels, train_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            _, val_acc = _cls_loss_acc(model, features, labels, val_mask)
            if val_acc > best_val:
                bad_counter = 0
                best_val = val_acc
                best_model = copy.deepcopy(model)
            else:
                bad_counter += 1

            if bad_counter >= cfg.patience:
                break

    best_model.eval()
    _, test_acc = _cls_loss_acc(best_model, features, labels, test_mask)
    return best_val, test_acc


# ---------------------------------------------------------------------------
# clustering metrics
# ---------------------------------------------------------------------------

@dataclass
class ClusterScores:
    acc: float
    nmi: float
    f1: float
    ari: float


# Hungarian alignment of predicted clusters to true labels, then accuracy.
# Cost matrix counts overlaps and we negate it because Munkres minimizes.
# If KMeans collapsed and produced fewer unique clusters than true classes,
# we bail out with zeros, matching the original clustering_metrics behaviour.
def _hungarian_accuracy(true_label, pred_label):
    classes_true = list(set(true_label))
    classes_pred = list(set(pred_label))
    if len(classes_true) != len(classes_pred):
        return 0.0, None

    cost = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for i, c1 in enumerate(classes_true):
        rows = [r for r, e in enumerate(true_label) if e == c1]
        for j, c2 in enumerate(classes_pred):
            cost[i, j] = sum(1 for r in rows if pred_label[r] == c2)

    indices = Munkres().compute((-cost).tolist())
    relabeled = np.zeros(len(pred_label))
    for i, c1 in enumerate(classes_true):
        c2 = classes_pred[indices[i][1]]
        for r, e in enumerate(pred_label):
            if e == c2:
                relabeled[r] = c1
    return metrics.accuracy_score(true_label, relabeled), relabeled


# Run KMeans n_trials times with different seeds and average ACC/NMI/F1/ARI.
# Matches main.py: n_trials is configurable (default 5 in the original).
def evaluate_clustering(embedding, labels, n_classes, n_trials=5):
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    accs, nmis, f1s, aris = [], [], [], []
    for seed in range(n_trials):
        # n_init kept default; original doesn't override it
        kmeans = KMeans(n_clusters=n_classes, random_state=seed).fit(embedding)
        pred = kmeans.predict(embedding)
        acc, relabeled = _hungarian_accuracy(labels, pred)
        accs.append(acc)
        nmis.append(metrics.normalized_mutual_info_score(labels, pred))
        # f1 is also zero when alignment failed -- the original returns 0
        # for every relabeled-based metric in that case
        if relabeled is None:
            f1s.append(0.0)
        else:
            f1s.append(metrics.f1_score(labels, relabeled, average="macro"))
        aris.append(metrics.adjusted_rand_score(labels, pred))

    return ClusterScores(
        acc=float(np.mean(accs)),
        nmi=float(np.mean(nmis)),
        f1=float(np.mean(f1s)),
        ari=float(np.mean(aris)),
    )
