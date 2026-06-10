"""Drop-in CLI for our framework that matches the original main.py interface.

Accepts the same flag names the original SUBLIME `main.py`,
`experiments/curves_runner.py`, `experiments/extra_runner.py`, and
`experiments/robustness_runner.py` accept, maps them onto
`sublime.pipeline.train`, and prints the same human-readable summary lines
(`Best test ACC:`, `Final NMI:`, `Test accuracy: ... +/- ...`, optional
`Epoch X | CL Loss Y`, `CURVE_EVAL val=... test=...`) so notebook log parsers
written against the original continue to work unchanged.

Run via the paper env wrapper:

    /tmp/sublime-env/bin/sublime-python scripts/run.py -dataset cora -ntrials 5 \\
        -gsl_mode structure_inference -type_learner fgp -k 30 ...

Robustness sweep (Figure 5):

    ... scripts/run.py -dataset cora -gsl_mode structure_refinement \\
        -perturb delete -rate 0.3 -perturb_seed 0 ...

Curves emission (Figure 3 / Table 4):

    ... scripts/run.py ... -emit_curves
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("DGLBACKEND", "pytorch")
logging.getLogger("dgl").setLevel(logging.ERROR)
logging.getLogger("ogb").setLevel(logging.ERROR)

# Make `import sublime` work when invoked from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from sublime.data import load_dataset
from sublime.framework.eval import ClsConfig
from sublime.pipeline import TrainConfig, train_one_trial
import sublime.framework.encoder as enc
import sublime.framework.eval as ev
import sublime.pipeline as pl
from configs import CONFIGS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Experimental setting -- original main.py
    p.add_argument("-dataset", type=str, default="cora")
    p.add_argument("-ntrials", type=int, default=5)
    p.add_argument("-sparse", type=int, default=0)
    p.add_argument("-gsl_mode", type=str, default="structure_inference",
                   choices=["structure_inference", "structure_refinement"])
    p.add_argument("-eval_freq", type=int, default=5)
    p.add_argument("-downstream_task", type=str, default="classification",
                   choices=["classification", "clustering"])
    p.add_argument("-gpu", type=int, default=0)
    p.add_argument("-seed", type=int, default=0,
                   help="base seed; trial t uses seed = seed + t")

    # GCL Module
    p.add_argument("-epochs", type=int, default=1000)
    p.add_argument("-lr", type=float, default=0.01)
    p.add_argument("-w_decay", type=float, default=0.0)
    p.add_argument("-hidden_dim", type=int, default=512)
    p.add_argument("-rep_dim", type=int, default=64)
    p.add_argument("-proj_dim", type=int, default=64)
    p.add_argument("-dropout", type=float, default=0.5)
    p.add_argument("-contrast_batch_size", type=int, default=0)
    p.add_argument("-nlayers", type=int, default=2)

    # Augmentation
    p.add_argument("-maskfeat_rate_learner", type=float, default=0.2)
    p.add_argument("-maskfeat_rate_anchor", type=float, default=0.2)
    p.add_argument("-dropedge_rate", type=float, default=0.5)

    # GSL Module
    p.add_argument("-type_learner", type=str, default="fgp",
                   choices=["fgp", "att", "mlp", "gnn"])
    p.add_argument("-k", type=int, default=30)
    p.add_argument("-sim_function", type=str, default="cosine",
                   choices=["cosine", "minkowski"])
    p.add_argument("-gamma", type=float, default=0.9,
                   help="accepted for compatibility; not used by the new FGP learner")
    p.add_argument("-activation_learner", type=str, default="relu",
                   choices=["relu", "tanh"])

    # Evaluation Network (downstream GCN)
    p.add_argument("-epochs_cls", type=int, default=200)
    p.add_argument("-lr_cls", type=float, default=0.001)
    p.add_argument("-w_decay_cls", type=float, default=0.0005)
    p.add_argument("-hidden_dim_cls", type=int, default=32)
    p.add_argument("-dropout_cls", type=float, default=0.5)
    p.add_argument("-dropedge_cls", type=float, default=0.25)
    p.add_argument("-nlayers_cls", type=int, default=2)
    p.add_argument("-patience_cls", type=int, default=10)

    # Structure Bootstrapping
    p.add_argument("-tau", type=float, default=1.0)
    p.add_argument("-c", type=int, default=0)

    # Clustering
    p.add_argument("-n_clu_trials", type=int, default=5)

    # ---- our extensions ----
    # Curves emission (replaces experiments/curves_runner.py)
    p.add_argument("-emit_curves", action="store_true",
                   help="emit per-epoch 'Epoch X | CL Loss Y' and per-eval "
                        "'CURVE_EVAL val=.. test=..' lines")

    # Robustness perturbation (replaces experiments/robustness_runner.py)
    p.add_argument("-perturb", type=str, default=None, choices=["delete", "add"])
    p.add_argument("-rate", type=float, default=0.0)
    p.add_argument("-perturb_seed", type=int, default=0)

    # Pre-baked configs from configs.py. CLI flags still override.
    p.add_argument("-config", type=str, default=None,
                   help="name of an entry in configs.CONFIGS; sets defaults that "
                        "CLI flags can still override")
    p.add_argument("-config_index", type=int, default=None,
                   help="if -config maps to a list (e.g. fig3_*_tau_sweep, "
                        "fig5_robustness), pick this index")
    return p


# ---------------------------------------------------------------------------
# Flag -> dataclass mapping
# ---------------------------------------------------------------------------

def cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        sparse=bool(args.sparse),
        gsl_mode=args.gsl_mode,
        task=args.downstream_task,
        n_layers=args.nlayers,
        hidden_dim=args.hidden_dim,
        emb_dim=args.rep_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        dropout_adj=args.dropedge_rate,
        learner_type=args.type_learner,
        learner_k=args.k,
        learner_activation=args.activation_learner,
        learner_metric=args.sim_function,
        learner_n_layers=2,
        maskfeat_rate_anchor=args.maskfeat_rate_anchor,
        maskfeat_rate_learner=args.maskfeat_rate_learner,
        epochs=args.epochs,
        lr=args.lr,
        w_decay=args.w_decay,
        contrast_batch_size=args.contrast_batch_size,
        tau=args.tau,
        c=args.c,
        eval_freq=args.eval_freq,
        n_clu_trials=args.n_clu_trials,
    )


def cls_cfg_from_args(args: argparse.Namespace) -> ClsConfig:
    return ClsConfig(
        hidden_dim=args.hidden_dim_cls,
        n_layers=args.nlayers_cls,
        dropout=args.dropout_cls,
        dropout_adj=args.dropedge_cls,
        lr=args.lr_cls,
        weight_decay=args.w_decay_cls,
        epochs=args.epochs_cls,
        patience=args.patience_cls,
    )


# ---------------------------------------------------------------------------
# Log / curve emission via monkey-patching (no pipeline changes)
# ---------------------------------------------------------------------------
#
# Two independent hooks:
#   install_loss_logger()    -> per-epoch "Epoch X | CL Loss Y" lines,
#                               throttled by LOG_INTERVAL_SEC (default 0 = every
#                               epoch when curves are requested, or 300s when
#                               curves are NOT requested -- see main()).
#   install_curve_emitter()  -> per-eval  "CURVE_EVAL val=... test=..." lines,
#                               needed for Figure 3 / Table 4.

_curve_state = {"epoch": 0, "trial": 0, "log_interval": 0.0, "last_log": 0.0}


def install_loss_logger(log_interval_sec: float) -> None:
    _curve_state["log_interval"] = float(log_interval_sec)
    _orig_calc = enc.GCL.calc_loss

    @staticmethod
    def _calc(x, x_aug, temperature=0.2, sym=True):
        out = _orig_calc(x, x_aug, temperature=temperature, sym=sym)
        _curve_state["epoch"] += 1
        ep = _curve_state["epoch"]
        loss_v = float(out.detach())
        now = time.monotonic()
        interval = _curve_state["log_interval"]
        last = _curve_state["last_log"]
        # always print first epoch; otherwise every epoch when interval <= 0,
        # or throttled by LOG_INTERVAL_SEC seconds
        if interval <= 0 or ep == 1 or (now - last) >= interval:
            print(f"Epoch {ep:05d} | CL Loss {loss_v:.4f}", flush=True)
            _curve_state["last_log"] = now
        return out

    enc.GCL.calc_loss = _calc


def install_curve_emitter() -> None:
    _orig_eval = ev.evaluate_classification

    def _eval(adj, features, labels, n_classes,
              train_mask, val_mask, test_mask, sparse, cfg):
        val_acc, test_acc = _orig_eval(adj, features, labels, n_classes,
                                       train_mask, val_mask, test_mask, sparse, cfg)
        v = float(val_acc) if hasattr(val_acc, "item") else float(val_acc)
        t = float(test_acc) if hasattr(test_acc, "item") else float(test_acc)
        print(f"CURVE_EVAL val={v:.6f} test={t:.6f}", flush=True)
        return val_acc, test_acc

    ev.evaluate_classification = _eval
    pl.evaluate_classification = _eval


# ---------------------------------------------------------------------------
# Adjacency perturbation (Figure 5)
# ---------------------------------------------------------------------------

def perturb_anchor_adj(dataset, mode: str, rate: float, seed: int, sparse: bool):
    """Symmetric edge delete/add on the raw anchor adjacency (no self-loops)."""
    A = dataset.anchor_adj
    if A.is_sparse:
        A = A.to_dense()
    A = A.clone().cpu().numpy().astype(np.float32)
    n = A.shape[0]

    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    edge_mask = A[iu, ju] > 0
    edges = np.stack([iu[edge_mask], ju[edge_mask]], axis=1)
    m = edges.shape[0]

    if mode == "delete":
        k = int(round(rate * m))
        if k > 0:
            idx = rng.choice(m, size=k, replace=False)
            for i, j in edges[idx]:
                A[i, j] = 0.0
                A[j, i] = 0.0
    elif mode == "add":
        k = int(round(rate * m))
        if k > 0:
            cand_i = iu[~edge_mask]
            cand_j = ju[~edge_mask]
            c = cand_i.shape[0]
            idx = rng.choice(c, size=min(k, c), replace=False)
            for i, j in zip(cand_i[idx], cand_j[idx]):
                A[i, j] = 1.0
                A[j, i] = 1.0
    else:
        raise ValueError(f"unknown perturb mode {mode!r}")

    A_t = torch.from_numpy(A)
    dataset.anchor_adj = A_t.to_sparse().coalesce() if sparse else A_t
    return dataset


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _resolve_config_defaults(parser: argparse.ArgumentParser) -> None:
    """Two-pass: peek for `-config`, pull defaults from `configs.CONFIGS`, then
    let the main parse_args still override anything on the command line."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-config", type=str, default=None)
    pre.add_argument("-config_index", type=int, default=None)
    pre_args, _ = pre.parse_known_args()
    if not pre_args.config:
        return
    if pre_args.config not in CONFIGS:
        parser.error(
            f"unknown -config {pre_args.config!r}; available: "
            f"{sorted(CONFIGS)}"
        )
    entry = CONFIGS[pre_args.config]
    if isinstance(entry, list):
        if pre_args.config_index is None:
            parser.error(
                f"-config {pre_args.config!r} is a list of {len(entry)} "
                f"variants; pass -config_index 0..{len(entry)-1}"
            )
        entry = entry[pre_args.config_index]
    elif pre_args.config_index is not None:
        parser.error(
            f"-config_index given but -config {pre_args.config!r} is not a list"
        )
    known = {a.dest for a in parser._actions}
    unknown = [k for k in entry if k not in known]
    if unknown:
        parser.error(
            f"config {pre_args.config!r} has keys with no matching flag: {unknown}"
        )
    parser.set_defaults(**entry)


def main():
    parser = build_parser()
    _resolve_config_defaults(parser)
    args = parser.parse_args()

    # Per-epoch loss logger.
    #   - With -emit_curves: default to every-epoch logging (interval = 0),
    #     unless LOG_INTERVAL_SEC overrides it. Needed for Figure 3 loss curves.
    #   - Without -emit_curves: only enable when LOG_INTERVAL_SEC is explicitly
    #     set (e.g. =300 for Tables 1/2/3), so unattended long runs still emit
    #     a heartbeat line every N seconds without flooding the log.
    env_interval = os.environ.get("LOG_INTERVAL_SEC")
    if args.emit_curves:
        install_loss_logger(float(env_interval) if env_interval else 0.0)
        install_curve_emitter()
    elif env_interval is not None:
        install_loss_logger(float(env_interval))

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}",
          flush=True)
    print(
        f"CONFIG dataset={args.dataset} ntrials={args.ntrials} sparse={args.sparse} "
        f"gsl_mode={args.gsl_mode} task={args.downstream_task} type_learner={args.type_learner}",
        flush=True,
    )

    dataset = load_dataset(args.dataset, sparse=bool(args.sparse), seed=args.seed)

    if args.perturb is not None:
        dataset = perturb_anchor_adj(
            dataset, args.perturb, args.rate, args.perturb_seed, bool(args.sparse)
        )
        print(
            f"ROBUSTNESS_CONFIG dataset={args.dataset} perturb={args.perturb} "
            f"rate={args.rate} seed={args.perturb_seed} ntrials={args.ntrials}",
            flush=True,
        )

    cfg = cfg_from_args(args)
    cls_cfg = cls_cfg_from_args(args)

    val_accs, test_accs = [], []
    for t in range(args.ntrials):
        _curve_state["trial"] = t
        _curve_state["epoch"] = 0
        _curve_state["last_log"] = 0.0
        result = train_one_trial(dataset, cfg, cls_cfg, seed=args.seed + t)

        if args.downstream_task == "classification":
            print(f"Trial:  {t + 1}", flush=True)
            print(f"Best val ACC:  {result.best_val_acc}", flush=True)
            print(f"Best test ACC:  {result.best_test_acc}", flush=True)
            if result.best_val_acc is not None:
                val_accs.append(float(result.best_val_acc))
            if result.best_test_acc is not None:
                test_accs.append(float(result.best_test_acc))
        else:
            cs = result.cluster_scores
            print(f"Final ACC:  {cs.acc}", flush=True)
            print(f"Final NMI:  {cs.nmi}", flush=True)
            print(f"Final F-score:  {cs.f1}", flush=True)
            print(f"Final ARI:  {cs.ari}", flush=True)

    if args.downstream_task == "classification" and len(test_accs) > 1:
        v_m = statistics.mean(val_accs); v_s = statistics.pstdev(val_accs)
        t_m = statistics.mean(test_accs); t_s = statistics.pstdev(test_accs)
        print(f"Val accuracy: {v_m:.4f} +/- {v_s:.4f}", flush=True)
        print(f"Test accuracy: {t_m:.4f} +/- {t_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
