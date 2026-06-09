"""Configurations for the reproduction experiments. This covers all the main tables and figures from the paper."""
from __future__ import annotations
from typing import Any

# I have taken the parameters from the original code and extended it
# to cover all the experiments + datasets in the paper

# ---------------------------------------------------------------------------
# Shared defaults - we can override these per experiment
# ---------------------------------------------------------------------------
_CLS_DEFAULTS = dict(
    epochs_cls=200,
    lr_cls=0.001,
    w_decay_cls=0.0005,
    hidden_dim_cls=32,
    dropout_cls=0.5,
    dropedge_cls=0.25,
    nlayers_cls=2,
    patience_cls=10,
)

_GCL_DEFAULTS = dict(
    nlayers=2,
    hidden_dim=512,
    rep_dim=256,
    proj_dim=256,
    dropout=0.5,
    sim_function="cosine",
    contrast_batch_size=0,
    c=0,
    downstream_task="classification",
    gpu=0,
)


def _cfg(**kw) -> dict[str, Any]:
    """Merge shared defaults with experiment-specific overrides."""
    cfg = {**_CLS_DEFAULTS, **_GCL_DEFAULTS}
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# Table 1 — Structure Inference
# ---------------------------------------------------------------------------
CONFIGS: dict[str, Any] = {}

CONFIGS["cora_si"] = _cfg(
    dataset="cora",
    gsl_mode="structure_inference",
    ntrials=5,
    sparse=0,
    # encoder
    epochs=4000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.5,
    # graph learner
    type_learner="fgp",
    k=30,
    activation_learner="relu",
    # augmentation
    maskfeat_rate_learner=0.5,
    maskfeat_rate_anchor=0.7,
    # bootstrapping
    tau=1,
    # eval
    eval_freq=20,
)

CONFIGS["citeseer_si"] = _cfg(
    dataset="citeseer",
    gsl_mode="structure_inference",
    ntrials=5,
    sparse=0,
    epochs=1000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.25,
    type_learner="att",
    k=20,
    activation_learner="tanh",
    maskfeat_rate_learner=0.8,
    maskfeat_rate_anchor=0.7,
    tau=0.9999,
    eval_freq=50,
    # classifier tweaks
    w_decay_cls=0.05,
    dropedge_cls=0.5,
)

CONFIGS["pubmed_si"] = _cfg(
    dataset="pubmed",
    gsl_mode="structure_inference",
    ntrials=5,
    sparse=1,
    # smaller model for pubmed
    hidden_dim=128,
    rep_dim=64,
    proj_dim=64,
    epochs=2000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.25,
    type_learner="att",
    k=15,
    activation_learner="tanh",
    maskfeat_rate_learner=0.8,
    maskfeat_rate_anchor=0.3,
    tau=1,
    contrast_batch_size=2000,
    eval_freq=20,
    lr_cls=0.01,
)

# ogbn-arxiv — the paper hasn't published hyperparameters for this dataset.
# GNN performed better than MLP and ATT on SR and SI, so using that as default here.
# Only nlayers_cls=3, hidden_dim_cls=256 are taken from the paper (appendix F.2)
CONFIGS["ogbn_arxiv_si"] = _cfg(
    dataset="ogbn-arxiv",
    gsl_mode="structure_inference",
    ntrials=1,
    sparse=1,
    epochs=1000,
    lr=0.0001,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="gnn",
    k=15,
    activation_learner="relu",
    maskfeat_rate_learner=0.3,
    maskfeat_rate_anchor=0.3,
    tau=0.99999,
    c=100,
    contrast_batch_size=2000,
    eval_freq=50,
    # ogbn-arxiv needs a deeper classifier (paper F.2)
    nlayers_cls=3,
    hidden_dim_cls=256,
    epochs_cls=500,
    lr_cls=0.01,
    w_decay_cls=0,
    patience_cls=50,
)

# UCI datasets (Wine, Cancer, Digits, 20news) — approximated configs.
# The paper does not publish per-dataset hyperparameters for any of these.
# only the search space is given in appendix F.3.
CONFIGS["wine_si"] = _cfg(
    dataset="wine",
    gsl_mode="structure_inference",
    ntrials=5,
    sparse=0,
    epochs=1000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="fgp",
    k=20,
    activation_learner="relu",
    maskfeat_rate_learner=0.5,
    maskfeat_rate_anchor=0.5,
    tau=0.9999,
    eval_freq=50,
)

CONFIGS["cancer_si"] = _cfg(**{**CONFIGS["wine_si"], "dataset": "cancer"})
CONFIGS["digits_si"] = _cfg(**{**CONFIGS["wine_si"], "dataset": "digits"})


# 20news uses sparse TF-IDF features (236-dim) so it's different from the Wine/Cancer/Digits configs
# MLP learner instead of FGP with some reasonable defaults for the other hyperparameters.
CONFIGS["news20_si"] = _cfg(
    dataset="20news",
    gsl_mode="structure_inference",
    ntrials=5,
    sparse=0,
    epochs=1000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="mlp",
    k=30,
    activation_learner="relu",
    maskfeat_rate_learner=0.5,
    maskfeat_rate_anchor=0.5,
    tau=0.9999,
    eval_freq=50,
)

# ---------------------------------------------------------------------------
# Table 2 — Structure Refinement
# ---------------------------------------------------------------------------

CONFIGS["cora_sr"] = _cfg(
    dataset="cora",
    gsl_mode="structure_refinement",
    ntrials=5,
    sparse=0,
    epochs=4000,
    lr=0.01,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="fgp",
    k=30,
    activation_learner="relu",
    maskfeat_rate_learner=0.7,
    maskfeat_rate_anchor=0.6,
    tau=0.9999,
    eval_freq=50,
    dropedge_cls=0.75,
)

CONFIGS["citeseer_sr"] = _cfg(
    dataset="citeseer",
    gsl_mode="structure_refinement",
    ntrials=5,
    sparse=0,
    epochs=1000,
    lr=0.001,
    w_decay=0.0,
    dropedge_rate=0.25,
    type_learner="att",
    k=20,
    activation_learner="tanh",
    maskfeat_rate_learner=0.6,
    maskfeat_rate_anchor=0.8,
    tau=0.9999,
    eval_freq=20,
    w_decay_cls=0.05,
    dropedge_cls=0.5,
)

CONFIGS["pubmed_sr"] = _cfg(
    dataset="pubmed",
    gsl_mode="structure_refinement",
    ntrials=5,
    sparse=1,
    hidden_dim=128,
    rep_dim=64,
    proj_dim=64,
    epochs=1500,
    lr=0.001,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="mlp",
    k=10,
    activation_learner="relu",
    maskfeat_rate_learner=0.4,
    maskfeat_rate_anchor=0.4,
    tau=0.999,
    c=50,
    contrast_batch_size=2000,
    eval_freq=20,
    lr_cls=0.01,
)

# ogbn-arxiv, same as its SI counterpart (see comment there), the paper does not specify the hyperparameters.
# These parameters are estimations of reasonable default.
CONFIGS["ogbn_arxiv_sr"] = _cfg(
    dataset="ogbn-arxiv",
    gsl_mode="structure_refinement",
    ntrials=1,
    sparse=1,
    epochs=1000,
    lr=0.001,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="gnn",
    k=15,
    activation_learner="relu",
    maskfeat_rate_learner=0.3,
    maskfeat_rate_anchor=0.3,
    tau=1,
    c=0,
    contrast_batch_size=2000,
    eval_freq=50,
    nlayers_cls=3,
    hidden_dim_cls=256,
    epochs_cls=500,
    lr_cls=0.01,
    w_decay_cls=0,
    patience_cls=50,
)

# ---------------------------------------------------------------------------
# Table 3 — Clustering (structure refinement)
# ---------------------------------------------------------------------------

CONFIGS["cora_clu"] = _cfg(
    dataset="cora",
    gsl_mode="structure_refinement",
    downstream_task="clustering",
    ntrials=10,
    sparse=0,
    epochs=2500,
    lr=0.001,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="fgp",
    k=20,
    activation_learner="relu",
    maskfeat_rate_learner=0.1,
    maskfeat_rate_anchor=0.8,
    tau=0.9999,
    eval_freq=100,
)

CONFIGS["citeseer_clu"] = _cfg(
    dataset="citeseer",
    gsl_mode="structure_refinement",
    downstream_task="clustering",
    ntrials=10,
    sparse=0,
    epochs=1000,
    lr=0.001,
    w_decay=0.0,
    dropedge_rate=0.5,
    type_learner="att",
    k=20,
    activation_learner="tanh",
    maskfeat_rate_learner=0.4,
    maskfeat_rate_anchor=0.9,
    tau=0.999,
    eval_freq=100,
)

# ---------------------------------------------------------------------------
# Table 4 / Fig 3 — Bootstrapping decay rate (tau) sweep, we only use the following tau's in accordance with TA: [1, 0.9999, 0.99]
# Run on Cora / Citeseer / Pubmed
# ---------------------------------------------------------------------------

_TAUS = [1, 0.9999, 0.99]

CONFIGS["fig3_cora_tau_sweep"] = [
    _cfg(**{**CONFIGS["cora_sr"], "tau": tau, "ntrials": 5})
    for tau in _TAUS
]

CONFIGS["fig3_citeseer_tau_sweep"] = [
    _cfg(**{**CONFIGS["citeseer_sr"], "tau": tau, "ntrials": 5})
    for tau in _TAUS
]

CONFIGS["fig3_pubmed_tau_sweep"] = [
    _cfg(**{**CONFIGS["pubmed_sr"], "tau": tau, "ntrials": 5})
    for tau in _TAUS
]

# ---------------------------------------------------------------------------
# Fig 5 — Robustness (edge perturbation sweep)
# Cora only, see paper
# ---------------------------------------------------------------------------

def build_robustness_config(
    perturb: str = "delete",
    rate: float = 0.2,
    ntrials: int = 5,
    epochs: int = 4000,
) -> dict[str, Any]:
    """ Helper to build a config dict for a single data point in figure 5.

    Args:
        perturb: "delete" or "add"
        rate:    perturbation rate (0.0 – 0.9) - see paper
        ntrials: number of independent seeds, paper specifies 5
        epochs:  training epochs, paper uses 4000
    """
    return _cfg(
        **{**CONFIGS["cora_sr"],
           "perturb": perturb,
           "rate": rate,
           "ntrials": ntrials,
           "epochs": epochs}
    )


_FIG5_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CONFIGS["fig5_robustness"] = [
    build_robustness_config(perturb=mode, rate=r)
    for mode in ("delete", "add")
    for r in _FIG5_RATES
]
