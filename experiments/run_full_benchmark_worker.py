"""
run_full_benchmark_worker.py — single-method benchmark worker
=============================================================
Evaluates ONE method across all (dataset, shot, outlier-ratio) conditions.

Methods (10 total):
  transductive baselines:
    OSTIM  OSLO  EOL  ROLE

  inductive baselines (this release):
    OPP-I        Overall Positive Prototype — inductive
    OPP-T        Overall Positive Prototype — transductive (OSLO-style refinement)
    MET          Maximum Entropy Test (ProtoNet + entropy threshold)
    Glocal       Glocal Energy-Based (global + local energy)
    TANE         Task-Adaptive Negative Envision simplified (att variant, no semantics)

Datasets: esc50, fsd, urbansound (3 datasets)
Shots   : 1, 5
Ratios  : 20pct, 50pct, 80pct

Usage (single worker):
    python experiments/run_full_benchmark_worker.py --method OPP-I --ratio 20pct

Parallelism: launch one process per (method × ratio) = 10 × 3 = 30 workers.
See scripts/launch_full_benchmark.sh for the parallel launcher.

Results are saved to:
    results/full_benchmark/<method>_<ratio>_<timestamp>.json
"""

import os, sys, json, argparse, traceback, time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

from evaluation.evaluator import FewShotEvaluator
from data.dataset_registry import get_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Outlier-ratio configurations
# n_way_in=5, k_query_in=4 → 20 inlier queries per episode
# ─────────────────────────────────────────────────────────────────────────────
RATIO_CONFIGS = {
    "20pct": dict(n_way_out=5, k_query_out=1,  b=0.20),
    "50pct": dict(n_way_out=5, k_query_out=4,  b=0.50),
    "80pct": dict(n_way_out=5, k_query_out=16, b=0.80),
}

DATASETS = ["esc50", "fsd", "urbansound"]
SHOTS    = [1, 5]
NUM_EPS  = 300
SEEDS    = [0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: softmax
# ─────────────────────────────────────────────────────────────────────────────
def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Method factories
# ─────────────────────────────────────────────────────────────────────────────

def _tim_maxprob_fn():
    from models.tim import TIM
    tim    = TIM(temperature=15.0, loss_weights=(0.1, 1.0, 0.1),
                 num_iter=150, normalize=True)
    timing = defaultdict(list)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        t0  = time.perf_counter()
        preds, logits = tim.predict(sup, lbl, qry, return_logits=True)
        timing["t_total_ms"].append((time.perf_counter() - t0) * 1000)
        probs = _softmax(logits).astype(np.float32)
        return preds, probs, (1.0 - probs.max(axis=1)).astype(np.float32)
    fn._timing_accum = timing
    return fn


def _ostim_fn():
    from models.tim import TIM
    tim    = TIM(temperature=15.0, loss_weights=(0.1, 1.0, 0.1),
                 num_iter=150, normalize=True)
    timing = defaultdict(list)
    def _tbn(sup, qry):
        all_e = np.concatenate([sup, qry], 0)
        mu    = all_e.mean(0, keepdims=True)
        sd    = all_e.std(0,  keepdims=True) + 1e-6
        def n(x):
            bn = (x - mu) / sd
            return (bn / (np.linalg.norm(bn, axis=1, keepdims=True) + 1e-12)
                    ).astype(np.float32)
        return n(sup), n(qry)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        sn, qn = _tbn(sup, qry)
        t0  = time.perf_counter()
        _, logits = tim.predict(sn, lbl, qn, return_logits=True)
        timing["t_total_ms"].append((time.perf_counter() - t0) * 1000)
        dummy = -logits.mean(axis=1, keepdims=True)
        lf    = np.concatenate([logits, dummy], 1)
        pf    = _softmax(lf)
        pi    = pf[:, :-1].astype(np.float32)
        return pi.argmax(1).astype(int), pi, pf[:, -1].astype(np.float32)
    fn._timing_accum = timing
    return fn


def _oslo_fn():
    from experiments.experiment_utils import make_oslo_fn
    return make_oslo_fn()


def _eol_fn(b=0.5):
    from experiments.experiment_utils import make_eol_fn
    return make_eol_fn(b=b)


def _role_fn(b=0.5):
    """Two-phase ROLE: Phase-1 BCD inlierness estimation + Phase-2 xi-gated prototype optimisation."""
    from models.role import ROLE
    role = ROLE(temperature=15.0, lr=1e-3, b=b)
    def fn(sup, lbl, qry, b=None, **kwargs):
        if b is not None:
            role.b = b
        return role.predict(sup, lbl, qry)
    return fn


def _opp_inductive_fn():
    from models.opp import OPPInductive
    model = OPPInductive(temperature=15.0)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        return model.predict(sup, lbl, qry)
    return fn


def _opp_transductive_fn():
    from models.opp import OPPTransductive
    model = OPPTransductive(temperature=15.0, num_iter=8)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        return model.predict(sup, lbl, qry)
    return fn


def _met_fn():
    from models.met import MET
    model = MET(temperature=15.0)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        return model.predict(sup, lbl, qry)
    return fn


def _glocal_fn():
    from models.glocal_energy_inf import GlocalEnergyInf
    model = GlocalEnergyInf(temperature=15.0, alpha=0.5)
    def fn(sup, lbl, qry, b=None,
           support_patches=None, query_patches=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        return model.predict(sup, lbl, qry,
                             support_patches=support_patches,
                             query_patches=query_patches)
    return fn


def _tane_fn():
    from models.tane_simple import TANESimple
    model = TANESimple(temperature=15.0, neg_scale=1.0)
    def fn(sup, lbl, qry, b=None, **kwargs):
        sup = np.asarray(sup, np.float32)
        lbl = np.asarray(lbl, np.int64)
        qry = np.asarray(qry, np.float32)
        return model.predict(sup, lbl, qry)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
METHOD_FACTORIES = {
    # ── existing transductive baselines ─────────────────────────────────────
    "TIM+MaxProb": _tim_maxprob_fn,
    "OSTIM"      : _ostim_fn,
    "OSLO"       : _oslo_fn,
    "EOL"        : lambda b=0.5: _eol_fn(b=b),
    "ROLE"       : lambda b=0.5: _role_fn(b=b),
    # ── new baselines (non-transductive / lightweight) ───────────────────────
    "OPP-I"      : _opp_inductive_fn,
    "OPP-T"      : _opp_transductive_fn,
    "MET"        : _met_fn,
    "Glocal"     : _glocal_fn,
    "TANE"       : _tane_fn,
}

# Methods that need the outlier ratio b passed to their factory
B_METHODS = {"EOL", "ROLE"}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_one(method_name, ds_name, n_shot, ratio_name, ratio_cfg):
    print(f"\n>>> {method_name} | {ds_name} | {n_shot}-shot | {ratio_name}", flush=True)
    b         = ratio_cfg["b"]
    n_way_out = ratio_cfg["n_way_out"]
    k_qout    = ratio_cfg["k_query_out"]

    if method_name in B_METHODS:
        fn = METHOD_FACTORIES[method_name](b=b)
    else:
        fn = METHOD_FACTORIES[method_name]()

    evaluator = FewShotEvaluator(
        num_episodes=NUM_EPS, seeds=SEEDS,
        verbose=True, dataset=ds_name)

    res = evaluator.evaluate_openset_method(
        method_fn=fn,
        n_way_in=5, n_way_out=n_way_out,
        n_shot=n_shot,
        k_query_in=4, k_query_out=k_qout,
        folds=[1, 2, 3, 4, 5], exclude_fold=True,
        method_name=f"{method_name} ({ds_name} {n_shot}shot {ratio_name})",
    )
    return res


def _cvt(o):
    if isinstance(o, float):     return float(o)
    if isinstance(o, int):       return int(o)
    if hasattr(o, "tolist"):     return o.tolist()
    if isinstance(o, dict):      return {k: _cvt(v) for k, v in o.items()}
    if isinstance(o, list):      return [_cvt(v) for v in o]
    return o


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full benchmark worker")
    parser.add_argument("--method", required=True,
                        choices=list(METHOD_FACTORIES.keys()))
    parser.add_argument("--ratio",  default="all",
                        choices=list(RATIO_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    method_name = args.method
    ratios_to_run = (list(RATIO_CONFIGS.keys())
                     if args.ratio == "all" else [args.ratio])

    OUT_DIR = os.path.join("results", "full_benchmark")
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    print("=" * 70, flush=True)
    print(f" FULL BENCHMARK — method={method_name}  ratios={ratios_to_run}", flush=True)
    print(f" Datasets: {DATASETS}  Shots: {SHOTS}  Episodes: {NUM_EPS}", flush=True)
    print("=" * 70, flush=True)

    for ratio_name in ratios_to_run:
        ratio_cfg = RATIO_CONFIGS[ratio_name]
        for ds_name in DATASETS:
            get_dataset(ds_name)
            for n_shot in SHOTS:
                key = f"{ds_name}_{n_shot}shot_{ratio_name}"
                try:
                    res = run_one(method_name, ds_name, n_shot, ratio_name, ratio_cfg)
                    all_results[key] = res
                    ia = res.get("inlier_accuracy_mean",
                                 res.get("accuracy_mean", 0))
                    au = res.get("auroc_mean", 0)
                    ap = res.get("aupr_mean",  0)
                    tm = res.get("inference_time_mean", 0) * 1000
                    print(f"  v {key}: inAcc={ia:.4f} AUROC={au:.4f} "
                          f"AUPR={ap:.4f} t={tm:.2f}ms", flush=True)
                except Exception as e:
                    print(f"  x {key}: {e}", flush=True)
                    traceback.print_exc()
                    all_results[key] = {"error": str(e)}

    safe = (method_name.replace("+", "plus").replace("(", "")
                       .replace(")", "").replace(" ", "_").replace("-", ""))
    ratio_tag = args.ratio
    out_file = os.path.join(OUT_DIR, f"{safe}_{ratio_tag}_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(_cvt(all_results), f, indent=2)
    print(f"\nDONE [{method_name}]. Saved -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
