"""Data loaders for all datasets used in the SUBLIME paper.
Datasets: cora, citeseer, pubmed, wine, cancer, digits, 20news, and ogbn-arxiv
"""
from __future__ import annotations

import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import sklearn.datasets as skd
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Dataset:
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    # anchor_adj: the input graph used in structure refinement (SR).
    # For datasets without an original graph (UCI, 20news) this is I_n;
    # the trainer ignores it under structure inference (SI).
    anchor_adj: torch.Tensor
    n_features: int
    n_classes: int
    n_nodes: int
    name: str


# ---------------------------------------------------------------------------
# helpers for adjacency, masks, splits
# ---------------------------------------------------------------------------

# scipy sparse matrix -> torch sparse tensor
def _scipy_to_torch_sparse(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape)).coalesce()


# n x n identity, can be sparse or dense
def _identity_adj(n, sparse):
    if sparse:
        return _scipy_to_torch_sparse(sp.eye(n, dtype=np.float32, format="csr"))
    return torch.eye(n, dtype=torch.float32)


# turn a list of indices into a boolean mask of length n
def _bool_mask(idx, n):
    m = np.zeros(n, dtype=bool)
    m[np.asarray(idx)] = True
    return torch.from_numpy(m)


# build train/val/test masks for datasets without a fixed split
def _stratified_split(labels, label_rate, val_size, n_classes, seed):
    # Appendix F.3 / Table 7 in the paper give the label rates per dataset.
    # The splitting technique is not specified, so we use LDS / SLAPS:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    n_train = max(n_classes, int(round(label_rate * n)))
    per_class = max(1, n_train // n_classes)

    train_idx = []
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        train_idx.extend(idx_c[:per_class].tolist())
    train_idx = np.array(sorted(train_idx[:n_train]))

    rest = np.setdiff1d(np.arange(n), train_idx)
    rng.shuffle(rest)
    val_idx = np.sort(rest[:val_size])
    test_idx = np.sort(rest[val_size:])
    return _bool_mask(train_idx, n), _bool_mask(val_idx, n), _bool_mask(test_idx, n)


# transform the feature or label arrays into a Dataset object with an identity anchor adj
def _wrap_dense(name, X, y, train_m, val_m, test_m, sparse):
    n = X.shape[0]
    return Dataset(
        features=torch.from_numpy(X.astype(np.float32)),
        labels=torch.from_numpy(y.astype(np.int64)),
        train_mask=train_m,
        val_mask=val_m,
        test_mask=test_m,
        anchor_adj=_identity_adj(n, sparse),
        n_features=X.shape[1],
        n_classes=int(y.max()) + 1,
        n_nodes=n,
        name=name,
    )


# ---------------------------------------------------------------------------
# citation networks (cora / citeseer / pubmed) - Planetoid split
#
# Same loading functionality as the original code (data_loader.py). That loader is
# itself the standard Planetoid reader used by most GNN papers: https://github.com/kimiyoung/planetoid
# Kept this part close to the original for fair reproductions, it's only refactored into smaller helpers
# ---------------------------------------------------------------------------

def _read_planetoid_file(path):
    with open(path, "rb") as f:
        if sys.version_info > (3, 0):
            return pkl.load(f, encoding="latin1")
        return pkl.load(f)


def _parse_index_file(path):
    return [int(line.strip()) for line in open(path)]


def _fix_citeseer_isolated_nodes(tx, ty, test_idx_reorder):
    # citeseer has about 15 isolated test nodes, we must insert zero rows so the shapes line up
    full_range = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_ext = sp.lil_matrix((len(full_range), tx.shape[1]))
    tx_ext[test_idx_reorder - min(test_idx_reorder), :] = tx
    ty_ext = np.zeros((len(full_range), ty.shape[1]))
    ty_ext[test_idx_reorder - min(test_idx_reorder), :] = ty
    return tx_ext, ty_ext


def _load_citation(name, sparse):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    x, y, tx, ty, allx, ally, graph = (
        _read_planetoid_file(_DATA_DIR / f"ind.{name}.{n}") for n in names
    )
    test_idx_reorder = _parse_index_file(_DATA_DIR / f"ind.{name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        tx, ty = _fix_citeseer_isolated_nodes(tx, ty, np.array(test_idx_reorder))

    # stack train+test feature blocks, then reorder test rows back into
    # their original node-id positions
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj_sp = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if sparse:
        anchor_adj = _scipy_to_torch_sparse(adj_sp)
    else:
        anchor_adj = torch.from_numpy(np.array(adj_sp.todense(), dtype=np.float32))

    labels_onehot = np.vstack((ally, ty))
    labels_onehot[test_idx_reorder, :] = labels_onehot[test_idx_range, :]
    # pin any all-zero rows (isolated citeseer nodes) to class 0
    row_sums = labels_onehot.sum(axis=1)
    labels_onehot[row_sums != 1] = 0
    labels_onehot[row_sums != 1, 0] = 1
    labels = torch.from_numpy(np.argmax(labels_onehot, axis=1).astype(np.int64))

    n = labels.shape[0]
    train_mask = _bool_mask(np.arange(len(y)), n)
    val_mask = _bool_mask(np.arange(len(y), len(y) + 500), n)
    test_mask = _bool_mask(test_idx_range, n)

    features_t = torch.from_numpy(np.asarray(features.todense(), dtype=np.float32))
    return Dataset(
        features=features_t,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        anchor_adj=anchor_adj,
        n_features=features_t.shape[1],
        n_classes=int(labels.max().item()) + 1,
        n_nodes=n,
        name=name,
    )


# ---------------------------------------------------------------------------
# UCI datasets (Wine, Cancer, Digits) — label rates from table 7 of the paper
# ---------------------------------------------------------------------------

# (sklearn loader, label_rate from table 7 of the paper, val_size we chose to use)
_UCI_CONFIG = {
    "wine":   ("load_wine",          0.056, 30),
    "cancer": ("load_breast_cancer", 0.018, 100),
    "digits": ("load_digits",        0.028, 300),
}


def _load_uci(name, sparse, seed):
    loader_name, label_rate, val_size = _UCI_CONFIG[name]
    raw = getattr(skd, loader_name)()
    X = scale(raw.data.astype(np.float32))
    y = raw.target.astype(np.int64)
    n_classes = int(y.max()) + 1

    train_m, val_m, test_m = _stratified_split(y, label_rate, val_size, n_classes, seed)
    return _wrap_dense(name, X, y, train_m, val_m, test_m, sparse)


# ---------------------------------------------------------------------------
# 20News dataset (10 categories, TF-IDF top-236 features)
# ---------------------------------------------------------------------------

# The paper does not specify which categories were used, so did some digging:
# they report 9607 samples / 236 features in table 7.
# The 10 categories below are the ones used by LDS: https://github.com/lucfra/LDS-GNN
# We can reproduce the same shape after applying TF-IDF top-236.
# LDS introduced this benchmark for GSL, so many methods including (speculatively) SUBLIME inherit
# its preprocessing to keep results comparable. 'The de-facto standard'
_NEWS20_CATEGORIES = [
    "alt.atheism", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
    "misc.forsale", "rec.autos", "rec.motorcycles",
    "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics",
]


def _load_20news(sparse, seed):
    train = fetch_20newsgroups(subset="train", categories=_NEWS20_CATEGORIES,
                               remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", categories=_NEWS20_CATEGORIES,
                              remove=("headers", "footers", "quotes"))
    texts = list(train.data) + list(test.data)
    y = np.concatenate([train.target, test.target]).astype(np.int64)

    vec = TfidfVectorizer(max_features=236, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    n_classes = int(y.max()) + 1

    train_m, val_m, test_m = _stratified_split(y, 0.010, 500, n_classes, seed)
    return _wrap_dense("20news", X, y, train_m, val_m, test_m, sparse)


# ---------------------------------------------------------------------------
# ogbn-arxiv (OGB node-classification benchmark, official train/val/test split)
# ---------------------------------------------------------------------------
# Using the canonical split that OGB ships with the dataset (dataset.get_idx_split()).

def _load_ogbn_arxiv(sparse, seed):
    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=str(_DATA_DIR / "ogb"))
    g, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    feats = g.ndata["feat"].numpy().astype(np.float32)
    y = labels.squeeze().long()
    n = feats.shape[0]

    # ogbn-arxiv ships as a directed graph; the standard preprocessing for
    # node classification is to symmetrise it and add self-loops.
    src, dst = (e.numpy() for e in g.edges())
    rows = np.concatenate([src, dst])         # (src,dst) + (dst,src) -> symmetric
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A + sp.eye(n, dtype=np.float32, format="csr")   # add self-loops
    A.data[:] = 1.0                                     # collapse duplicates to 1

    anchor_adj = _scipy_to_torch_sparse(A) if sparse else torch.from_numpy(A.toarray())

    train_m = _bool_mask(split_idx["train"].numpy(), n)
    val_m = _bool_mask(split_idx["valid"].numpy(), n)
    test_m = _bool_mask(split_idx["test"].numpy(), n)

    return Dataset(
        features=torch.from_numpy(feats),
        labels=y,
        train_mask=train_m,
        val_mask=val_m,
        test_mask=test_m,
        anchor_adj=anchor_adj,
        n_features=feats.shape[1],
        n_classes=int(y.max().item()) + 1,
        n_nodes=n,
        name="ogbn-arxiv",
    )


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

_CITATION = {"cora", "citeseer", "pubmed"}
_UCI = set(_UCI_CONFIG)


def load_dataset(name, sparse=False, seed=0):
    if name in _CITATION:
        return _load_citation(name, sparse=sparse)
    if name in _UCI:
        return _load_uci(name, sparse=sparse, seed=seed)
    if name == "20news":
        return _load_20news(sparse=sparse, seed=seed)
    if name == "ogbn-arxiv":
        return _load_ogbn_arxiv(sparse=sparse, seed=seed)
    raise KeyError(
        f"unknown dataset {name!r}; choose from "
        f"{sorted(_CITATION | _UCI | {'20news', 'ogbn-arxiv'})}"
    )
