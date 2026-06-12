"""Training loop for SUBLIME.

One trial:
  1. Build anchor adjacency (identity for structure_inference, original A for
     structure_refinement) and normalize symmetrically.
  2. Build the structure learner (FGP / ATT / MLP / GNN) and the GCL encoder.
  3. For each epoch: augment features into two views (anchor view + learner
     view), run the encoder on both, NT-Xent loss, step both optimizers.
  4. Every c epochs, bootstrap the anchor: anchor = tau*anchor + (1-tau)*learned.
  5. Every eval_freq epochs, evaluate downstream (classification or clustering)
     on the learned adj and track the best.

Multiple trials run the same setup with different seeds; results are averaged.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset
from .framework.augment import mask_features
from .framework.encoder import GCL
from .framework.eval import (
    ClsConfig,
    ClusterScores,
    evaluate_classification,
    evaluate_clustering,
)
from .framework.graph_learners import (
    AttentiveLearner,
    FGPLearner,
    GNNLearner,
    GraphLearner,
    MLPLearner,
)
from .framework.post_processor import post_process_dense, post_process_sparse


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # mode
    sparse: bool
    gsl_mode: str               # 'structure_inference' or 'structure_refinement'
    task: str                   # 'classification' or 'clustering'

    # encoder (GCL)
    n_layers: int = 2
    hidden_dim: int = 512
    emb_dim: int = 64
    proj_dim: int = 64
    dropout: float = 0.5
    dropout_adj: float = 0.5

    # learner
    learner_type: str = "fgp"        # 'fgp' | 'att' | 'mlp' | 'gnn'
    learner_k: int = 30
    learner_activation: str = "relu" # between-layer activation for ATT/MLP/GNN
    learner_metric: str = "cosine"   # kNN metric for FGP init
    learner_n_layers: int = 2        # only used by ATT/MLP/GNN

    # augmentation
    maskfeat_rate_anchor: float = 0.2
    maskfeat_rate_learner: float = 0.2

    # optimization
    epochs: int = 1000
    lr: float = 0.01
    w_decay: float = 0.0
    contrast_batch_size: int = 0     # 0 = no batching; loss over all nodes at once

    # bootstrap
    tau: float = 1.0
    c: int = 0                       # bootstrap interval (0 means every epoch)

    # eval cadence
    eval_freq: int = 5
    n_clu_trials: int = 5            # KMeans trials for clustering task

    # post-processor toggles (dense path only; ablation use)
    pp_topk: bool = True
    pp_relu: bool = True
    pp_sym: bool = True
    pp_norm: bool = True

    # training-objective ablation:
    #   'contrastive' - SUBLIME's default NT-Xent on two augmented encoder views
    #   'feature_sim' - MSE between the learned adjacency and cos(X, X^T);
    #                   ignores the encoder, no bootstrap, no augmentation
    #   'none'        - no training; evaluate the learner at its random init
    loss_type: str = "contrastive"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# anchor adjacency for the contrastive setup:
#   SI mode -> I_n (no structure information, even if the dataset has an adj)
#   SR mode -> the dataset's adjacency (data.py already supplies I_n for the
#             UCI/20news datasets that lack an original graph)
# Both are then symmetrically normalized (D^-1/2 A D^-1/2).
def _build_anchor_adj(dataset: Dataset, gsl_mode: str, sparse: bool):
    n = dataset.n_nodes
    if gsl_mode == "structure_inference":
        adj = _identity(n, sparse)
    elif gsl_mode == "structure_refinement":
        adj = dataset.anchor_adj
        if sparse and not adj.is_sparse:
            adj = adj.to_sparse().coalesce()
        elif not sparse and adj.is_sparse:
            adj = adj.to_dense()
    else:
        raise ValueError(f"unknown gsl_mode: {gsl_mode!r}")

    return _normalize_sym(adj, sparse)


# n x n identity, sparse or dense
def _identity(n, sparse):
    if sparse:
        idx = torch.arange(n)
        indices = torch.stack([idx, idx])
        values = torch.ones(n)
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return torch.eye(n)


# D^-1/2 A D^-1/2 (no self-loop addition; SI's I already has self-loops, and
# SR's adj comes with whatever self-loops the dataset has). Mirrors
# utils.normalize(_, 'sym', sparse) in the original.
def _normalize_sym(adj, sparse):
    eps = 1e-10
    if sparse:
        adj = adj.coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = 1.0 / (deg.sqrt() + eps)
        row, col = adj.indices()
        values = adj.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(adj.indices(), values, adj.shape).coalesce()
    deg = adj.sum(dim=1)
    d_inv_sqrt = 1.0 / (deg.sqrt() + eps)
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


# Factory mapping the config string to the learner class.
# GNN learner additionally receives the (normalized) anchor adjacency.
def _build_learner(cfg: TrainConfig, features: torch.Tensor, gnn_adj: torch.Tensor) -> GraphLearner:
    in_dim = features.shape[1]
    common = dict(n_layers=cfg.learner_n_layers, k=cfg.learner_k,
                  knn_metric=cfg.learner_metric, activation=cfg.learner_activation,
                  sparse=cfg.sparse)
    if cfg.learner_type == "fgp":
        return FGPLearner(features.cpu(), k=cfg.learner_k, knn_metric=cfg.learner_metric)
    if cfg.learner_type == "att":
        return AttentiveLearner(in_dim=in_dim, **common)
    if cfg.learner_type == "mlp":
        return MLPLearner(in_dim=in_dim, **common)
    if cfg.learner_type == "gnn":
        # gnn_adj is stored implicitly: pipeline passes it on every forward
        return GNNLearner(in_dim=in_dim, **common)
    raise ValueError(f"unknown learner_type: {cfg.learner_type!r}")


# Two augmented views: features with a random subset of feature columns zeroed.
# If mask_rate == 0 the original returns the features unchanged (deepcopy).
def _augment(features, rate):
    if rate == 0:
        return features
    return mask_features(features, rate)


# Run the learner, then the post-processor. Returns S, the structure used by
# the encoder's "learner view". The post-processor is a no-op in the sparse
# path because the learner already does top-k + degree-norm itself.
def _learn_structure(learner, features, gnn_adj, cfg: TrainConfig):
    # GNN learner takes the (normalized) anchor adjacency on every forward;
    # others ignore the adj argument
    if isinstance(learner, GNNLearner):
        s_tilde = learner(features, gnn_adj)
    else:
        s_tilde = learner(features)

    if cfg.sparse:
        return post_process_sparse(s_tilde)
    return post_process_dense(
        s_tilde, cfg.learner_k,
        is_fgp=isinstance(learner, FGPLearner),
        do_topk=cfg.pp_topk, do_relu=cfg.pp_relu,
        do_sym=cfg.pp_sym, do_norm=cfg.pp_norm,
    )


# anchor = tau * anchor + (1 - tau) * learned, with learned detached.
# Runs every epoch when c == 0, otherwise every c epochs. tau == 1 disables
# bootstrapping entirely (the (1-tau) factor is zero, so anchor never moves).
def _bootstrap(anchor, learned, tau, sparse):
    learned = learned.detach()
    updated = anchor * tau + learned * (1.0 - tau)
    if sparse:
        return updated.coalesce()
    return updated


# Contrastive loss for one batch. If contrast_batch_size > 0 the loss is
# computed in chunks and weighted by chunk size, matching main.py.
def _contrastive_loss(model, z_anchor, z_learner, contrast_batch_size):
    if contrast_batch_size == 0:
        return GCL.calc_loss(z_anchor, z_learner)

    n = z_anchor.shape[0]
    idx = list(range(n))
    # original doesn't shuffle (the shuffle call is commented out in main.py)
    loss = 0.0
    for start in range(0, n, contrast_batch_size):
        batch = idx[start:start + contrast_batch_size]
        weight = len(batch) / n
        loss = loss + GCL.calc_loss(z_anchor[batch], z_learner[batch]) * weight
    return loss


# ---------------------------------------------------------------------------
# per-trial training
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    # for classification: best val/test acc seen at any eval checkpoint
    best_val_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    best_epoch: Optional[int] = None
    # for clustering: scores at the final eval checkpoint
    cluster_scores: Optional[ClusterScores] = None


def train_one_trial(dataset: Dataset, cfg: TrainConfig, cls_cfg: ClsConfig,
                    seed: int) -> TrialResult:
    _set_seed(seed)

    # data on device
    use_cuda = torch.cuda.is_available()
    features = dataset.features.cuda() if use_cuda else dataset.features
    labels = dataset.labels.cuda() if use_cuda else dataset.labels
    train_mask = dataset.train_mask.cuda() if use_cuda else dataset.train_mask
    val_mask = dataset.val_mask.cuda() if use_cuda else dataset.val_mask
    test_mask = dataset.test_mask.cuda() if use_cuda else dataset.test_mask

    # anchor adjacency: normalized once, possibly updated by bootstrap each epoch
    anchor_adj = _build_anchor_adj(dataset, cfg.gsl_mode, cfg.sparse)
    if use_cuda:
        anchor_adj = anchor_adj.cuda()

    # GNN learner uses the ORIGINAL normalized anchor at every forward and
    # never sees the bootstrapped version (the original stores it at init and
    # the bootstrap reassigns anchor_adj to a fresh tensor, so the learner's
    # reference is untouched). Snapshot it here.
    gnn_adj = anchor_adj.detach().clone() if isinstance(anchor_adj, torch.Tensor) else anchor_adj
    if cfg.sparse:
        gnn_adj = gnn_adj.coalesce()

    # learner + encoder
    learner = _build_learner(cfg, features, gnn_adj)
    model = GCL(n_layers=cfg.n_layers, in_dim=dataset.n_features, hidden_dim=cfg.hidden_dim,
                emb_dim=cfg.emb_dim, proj_dim=cfg.proj_dim,
                dropout=cfg.dropout, dropout_adj=cfg.dropout_adj, sparse=cfg.sparse)
    if use_cuda:
        learner = learner.cuda()
        model = model.cuda()

    opt_model = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
    opt_learner = torch.optim.Adam(learner.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)

    result = TrialResult()
    best_val = 0.0

    # ---- loss-ablation: 'none' = no training, just eval the initial learner ----
    if cfg.loss_type == "none":
        model.eval()
        learner.eval()
        with torch.no_grad():
            learned_adj = _learn_structure(learner, features, gnn_adj, cfg)
            if cfg.sparse:
                eval_adj = torch.sparse_coo_tensor(
                    learned_adj.indices(), learned_adj.values().detach(), learned_adj.shape
                ).coalesce()
            else:
                eval_adj = learned_adj.detach()
        # evaluate_classification trains a downstream GCN which needs autograd,
        # so it MUST run outside the no_grad block above.
        if cfg.task == "classification":
            val_acc, test_acc = evaluate_classification(
                eval_adj, features, labels, dataset.n_classes,
                train_mask, val_mask, test_mask,
                sparse=cfg.sparse, cfg=cls_cfg,
            )
            result.best_val_acc = float(val_acc) if torch.is_tensor(val_acc) else val_acc
            result.best_test_acc = float(test_acc) if torch.is_tensor(test_acc) else test_acc
            result.best_epoch = 0
        else:
            with torch.no_grad():
                _, embedding = model(features, learned_adj)
            result.cluster_scores = evaluate_clustering(
                embedding, labels, dataset.n_classes, n_trials=cfg.n_clu_trials,
            )
        return result

    # ---- loss-ablation: precompute cosine target for 'feature_sim' ----
    # MSE is computed on the RAW learner output (before post-processing), so
    # gradients flow through every entry of S~ instead of being masked out by
    # the non-differentiable top-k. Post-processing is still applied at eval
    # time via `_learn_structure` so the downstream classifier sees the same
    # structure as in the contrastive setting.
    #
    # Scale note: for ATT/MLP/GNN the learner returns cos(h_omega(X)), so both
    # pred and target live in [-1, 1]. For FGP the learner returns elu(omega)+1
    # (non-negative, no upper bound) which doesn't match the cosine range, but
    # MSE still pulls omega toward producing cos(X, X^T) at every entry --
    # FGP's init from kNN(X) already lives close to that target so this is a
    # near-no-op for FGP and a real training signal for ATT/MLP.
    target_sim = None
    if cfg.loss_type == "feature_sim":
        if cfg.sparse:
            raise ValueError("loss_type='feature_sim' requires -sparse 0 (dense path)")
        with torch.no_grad():
            feats_norm = F.normalize(features, p=2, dim=1)
            target_sim = feats_norm @ feats_norm.t()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        learner.train()

        if cfg.loss_type == "contrastive":
            # post-processed structure is needed for the encoder's learner view
            learned_adj = _learn_structure(learner, features, gnn_adj, cfg)

            # --- contrastive step ---
            feats_anchor = _augment(features, cfg.maskfeat_rate_anchor)
            feats_learner = _augment(features, cfg.maskfeat_rate_learner)

            z_anchor, _ = model(feats_anchor, anchor_adj)
            z_learner, _ = model(feats_learner, learned_adj)
            loss = _contrastive_loss(model, z_anchor, z_learner, cfg.contrast_batch_size)

            opt_model.zero_grad()
            opt_learner.zero_grad()
            loss.backward()
            opt_model.step()
            opt_learner.step()

            # --- structure bootstrap (contrastive-specific) ---
            if (1.0 - cfg.tau) > 0 and (cfg.c == 0 or epoch % cfg.c == 0):
                anchor_adj = _bootstrap(anchor_adj, learned_adj, cfg.tau, cfg.sparse)

        elif cfg.loss_type == "feature_sim":
            # Supervise the RAW learner output (pre-post-processing) so the
            # gradient reaches every entry of S~. For ATT/MLP/GNN this is
            # cos(h_omega(X)); for FGP it's elu(omega)+1.
            if isinstance(learner, GNNLearner):
                s_tilde_raw = learner(features, gnn_adj)
            else:
                s_tilde_raw = learner(features)
            loss = F.mse_loss(s_tilde_raw, target_sim)
            opt_learner.zero_grad()
            loss.backward()
            opt_learner.step()

            # Post-processed adjacency only needed at eval epochs; skip the
            # wasted compute (and the dangling graph) on every other epoch.
            if epoch % cfg.eval_freq == 0:
                with torch.no_grad():
                    learned_adj = _learn_structure(learner, features, gnn_adj, cfg)

        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type!r}")

        # --- periodic downstream eval ---
        if epoch % cfg.eval_freq == 0:
            model.eval()
            learner.eval()
            # The original evaluates on the SAME learned_adj from the training
            # step (not a fresh learner forward). It just detaches values.
            if cfg.sparse:
                eval_adj = learned_adj.coalesce()
                eval_adj = torch.sparse_coo_tensor(
                    eval_adj.indices(), eval_adj.values().detach(), eval_adj.shape
                ).coalesce()
            else:
                eval_adj = learned_adj.detach()

            if cfg.task == "classification":
                val_acc, test_acc = evaluate_classification(
                    eval_adj, features, labels, dataset.n_classes,
                    train_mask, val_mask, test_mask,
                    sparse=cfg.sparse, cfg=cls_cfg,
                )
                val_acc = float(val_acc) if torch.is_tensor(val_acc) else val_acc
                test_acc = float(test_acc) if torch.is_tensor(test_acc) else test_acc
                if val_acc > best_val:
                    best_val = val_acc
                    result.best_val_acc = val_acc
                    result.best_test_acc = test_acc
                    result.best_epoch = epoch
            else:  # clustering
                with torch.no_grad():
                    _, embedding = model(features, learned_adj)
                result.cluster_scores = evaluate_clustering(
                    embedding, labels, dataset.n_classes, n_trials=cfg.n_clu_trials,
                )

    return result


# ---------------------------------------------------------------------------
# multi-trial driver
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    trials: list = field(default_factory=list)

    # convenience accessors for classification trials
    def val_accs(self):
        return [t.best_val_acc for t in self.trials if t.best_val_acc is not None]

    def test_accs(self):
        return [t.best_test_acc for t in self.trials if t.best_test_acc is not None]

    def cluster_scores(self):
        return [t.cluster_scores for t in self.trials if t.cluster_scores is not None]


# Run ntrials independent trials with seeds 0, 1, ..., ntrials-1 (matches the
# original's setup_seed(trial)). For clustering the original uses ntrials=1
# and runs the KMeans inner loop n_clu_trials times instead -- the caller
# should set ntrials accordingly.
def train(dataset: Dataset, cfg: TrainConfig, cls_cfg: ClsConfig, n_trials: int = 1):
    result = TrainResult()
    for trial in range(n_trials):
        result.trials.append(train_one_trial(dataset, cfg, cls_cfg, seed=trial))
    return result
