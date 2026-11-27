#!/usr/bin/env python3
"""
Measure what ``ModelConfig.standardize_features`` does to the real training matrix.

Run this before enabling the flag in production. It answers the two questions that
decide whether the change is free or consequential:

1. **Conditioning.** How many L-BFGS-B iterations does each path need, and what
   condition number is it fighting? This is the "free speed" half.

2. **Fit quality.** With the regulariser off, both paths optimise the same objective and
   must agree — any gap is the raw path failing to converge. With the regulariser on they
   *will* differ, because equalising the L2 penalty across columns is the point. The size
   of that difference tells you how much accidental feature selection ``l2_lambda`` was
   doing, and therefore how badly the tuned hyperparameters need to be redone.

If held-out log-loss is flat and iterations collapse, enabling the flag is close to free
(still re-tune ``l2_lambda``, since its meaning changed). If held-out log-loss moves, the
regularisation was load-bearing and a Phase-3 sweep is mandatory before shipping.

Usage:
    python scripts/dev/benchmark_feature_scaling.py --data-dir ./data
    python scripts/dev/benchmark_feature_scaling.py --data-dir ./data --holdout-start 2023-01-01
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.loader import filter_tier1_post_era  # noqa: E402
from src.matchup.interactions import FEATURE_NAMES  # noqa: E402
from src.model.regression import (  # noqa: E402
    MultinomialLogisticModel,
    column_scales,
)
from src.pipeline import MMAPredictor  # noqa: E402


def _log_loss(model: MultinomialLogisticModel, X: np.ndarray, y: np.ndarray) -> float:
    probs = model.predict_proba(X)
    picked = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(picked)))


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    *,
    standardize: bool,
    l2_lambda: float,
) -> Tuple[MultinomialLogisticModel, float]:
    m = cfg.model
    model = MultinomialLogisticModel(
        n_features=X.shape[1], delta=m.huber_delta, l2_lambda=l2_lambda
    )
    t0 = time.perf_counter()
    model.fit(
        X,
        y,
        max_iter=m.lbfgs_max_iter,
        ftol=m.lbfgs_ftol,
        gtol=m.lbfgs_gtol,
        standardize=standardize,
    )
    return model, time.perf_counter() - t0


def _row(label: str, model: MultinomialLogisticModel, secs: float, tr: float, te: Optional[float]) -> str:
    te_s = f"{te:12.6f}" if te is not None else f"{'—':>12}"
    return (
        f"  {label:<12} {str(model.n_iter):>8} {secs:>9.1f}s "
        f"{model.final_loss:14.6f} {tr:12.6f} {te_s}  {str(model.converged):>5}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument(
        "--holdout-start",
        type=str,
        default=None,
        help="ISO date; rows on/after it become the held-out set (default: config value).",
    )
    args = ap.parse_args()

    cfg = Config()
    if args.holdout_start:
        cfg.holdout_start_date = date.fromisoformat(args.holdout_start)

    print("Loading data and building ELO ...", flush=True)
    predictor = MMAPredictor(cfg)
    predictor.load_data(args.data_dir)
    predictor.build_elo()

    fights = filter_tier1_post_era(predictor.fights, cfg.master_start_year)
    hsd = cfg.holdout_start_date
    if hsd is None:
        train_f, test_f = fights, []
    else:
        train_f = [f for f in fights if f.fight_date < hsd]
        test_f = [f for f in fights if f.fight_date >= hsd]

    print(f"Building feature matrices ({len(train_f):,} train / {len(test_f):,} holdout) ...", flush=True)
    X, y, _, _ = predictor.build_xyw_for_fights(train_f, matrix_progress_every=0)
    if test_f:
        X_te, y_te, _, _ = predictor.build_xyw_for_fights(test_f, matrix_progress_every=0)
    else:
        X_te = y_te = None

    scale = column_scales(X)
    print()
    print("=" * 78)
    print("COLUMN SCALES  (a shared l2_lambda penalises a column ~1/scale^2 as hard)")
    print("=" * 78)
    widest = max(scale) / min(scale)
    for name, sc in sorted(zip(FEATURE_NAMES, scale), key=lambda t: -t[1]):
        print(f"  {name:<28} std={sc:12.4f}   relative penalty x{(max(scale) / sc) ** 2:>12.2e}")
    print(f"\n  widest scale ratio: {widest:.3e}")
    print(f"  cond(X'X)  raw     = {np.linalg.cond(X.T @ X):.4e}")
    print(f"  cond(X'X)  scaled  = {np.linalg.cond((X / scale).T @ (X / scale)):.4e}")

    header = (
        f"  {'path':<12} {'n_iter':>8} {'wall':>10} "
        f"{'robust NLL':>14} {'train LL':>12} {'holdout LL':>12}  {'conv':>5}"
    )

    print()
    print("=" * 78)
    print("A. REGULARISER OFF (l2_lambda = 0) — same objective, so these must agree")
    print("=" * 78)
    print(header)
    for label, std in (("raw", False), ("scaled", True)):
        model, secs = _fit(X, y, cfg, standardize=std, l2_lambda=0.0)
        tr = _log_loss(model, X, y)
        te = _log_loss(model, X_te, y_te) if X_te is not None else None
        print(_row(label, model, secs, tr, te))
    print("\n  Any gap here is the raw path failing to converge, not a model difference.")

    print()
    print("=" * 78)
    print(f"B. REGULARISER ON (l2_lambda = {cfg.model.l2_lambda:g}) — these SHOULD differ")
    print("=" * 78)
    print(header)
    results = {}
    for label, std in (("raw", False), ("scaled", True)):
        model, secs = _fit(X, y, cfg, standardize=std, l2_lambda=cfg.model.l2_lambda)
        tr = _log_loss(model, X, y)
        te = _log_loss(model, X_te, y_te) if X_te is not None else None
        results[label] = (model, tr, te)
        print(_row(label, model, secs, tr, te))

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    raw_m, _, raw_te = results["raw"]
    std_m, _, std_te = results["scaled"]
    if raw_m.n_iter and std_m.n_iter:
        print(f"  Iterations: {raw_m.n_iter} -> {std_m.n_iter} ({raw_m.n_iter / max(std_m.n_iter, 1):.1f}x fewer)")
    if raw_te is not None and std_te is not None:
        delta = std_te - raw_te
        print(f"  Holdout log-loss: {raw_te:.6f} -> {std_te:.6f}  (delta {delta:+.6f})")
        if abs(delta) < 1e-3:
            print("\n  Held-out performance is unchanged. Enabling standardize_features is")
            print("  essentially free speed. Still re-tune l2_lambda: its meaning changed,")
            print("  even if the current value happens to land in the same place.")
        else:
            print("\n  Held-out performance MOVED. l2_lambda was doing real (accidental)")
            print("  feature selection via the scale disparity. Do NOT enable the flag")
            print("  until a Phase-3 sweep has re-tuned huber_delta and l2_lambda:")
            print("    python -m src.cli.run_phase3_tuning --help")
    print()


if __name__ == "__main__":
    main()
