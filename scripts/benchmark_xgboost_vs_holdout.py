"""
Train a multiclass XGBoost classifier on the same Tabular features and time split
as Phase 3 / ``first_run_report.json`` by default: **train** on Tier‑1 rows with
``fight_date < holdout_start``, **evaluate** only on **pristine calendar years**
(2023--2025), not on every future fight in the CSV.

That matches the bespoke model's published pristine block (same *n* as the report
when data match). Use ``--eval-mode expanding`` only if you want all fights with
``fight_date >= holdout_start`` (grows as you append 2026+ cards).

Requires: pip install xgboost
  (or: pip install -r requirements-benchmark.txt)

Example:
  python scripts/benchmark_xgboost_vs_holdout.py --data-dir ./data --elo-cache ./data/elo_cache.pkl

Add --fit-logistic for an on-the-shelf multinomial head on identical (X_train, X_test).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _print_score(title: str, s) -> None:
    print(f"\n=== {title}  (n={s.n}) ===")
    if s.n == 0:
        print("  (no rows)")
        return
    print(f"  mean log-loss:  {s.mean_log_loss:.4f}  (uniform 6-way: {math.log(6):.4f})")
    print(f"  mean Brier:     {s.mean_brier:.4f}")
    print(f"  accuracy:       {s.accuracy:.2%}  (random class: {100 / 6:.2f}%)")
    print(f"  macro F1:       {s.macro_f1:.4f}")
    print(f"  WL log-loss:    {s.mean_wl_log_loss:.4f}  (50/50: {math.log(2):.4f})")
    print(f"  WL accuracy:    {s.wl_accuracy:.2%}")
    print(f"  finish F1:      {s.finish_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--holdout-start",
        type=str,
        default="2023-01-01",
        help="Train on Tier-1 rows with fight_date < this (regression cutoff).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("pristine", "expanding"),
        default="pristine",
        help="pristine = calendar years from --eval-years only (default; matches Phase 3 report). "
        "expanding = every Tier-1 fight with fight_date >= holdout_start (includes 2026+ as data grows).",
    )
    parser.add_argument(
        "--eval-years",
        type=str,
        default="2023,2024,2025",
        help="Comma-separated calendar years scored when --eval-mode pristine (default matches first_run_report.json).",
    )
    parser.add_argument("--elo-cache", type=Path, default=None, help="Reuse PIT ELO cache if valid.")
    parser.add_argument(
        "--sample-weight",
        choices=("none", "recency"),
        default="none",
        help="recency = same 1/(1+days/365) as matrix build (point L-BFGS fit is unweighted).",
    )
    parser.add_argument("--fit-logistic", action="store_true", help="Also fit MultinomialLogisticModel on same X.")
    parser.add_argument("--matrix-progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print(
            "Missing dependency: xgboost\n"
            "  pip install xgboost\n"
            "  or: pip install -r requirements-benchmark.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.cli.common import resolve_date
    from src.config import Config
    from src.data.loader import filter_tier1_post_era
    from src.eval.fight_scoring import (
        filter_tier1_fights_in_calendar_year,
        tier1_slice_score_from_probs,
    )
    from src.model.regression import MultinomialLogisticModel
    from src.pipeline import MMAPredictor

    hsd = resolve_date(args.holdout_start)
    cfg = Config()
    cfg.holdout_start_date = hsd
    p = MMAPredictor(cfg)

    t0 = time.perf_counter()
    print(f"Loading data from {args.data_dir} ...")
    p.load_data(args.data_dir)

    cache = Path(args.elo_cache).resolve() if args.elo_cache else None
    if cache and p.try_load_elo_from_cache(cache):
        print(f"ELO from cache {cache.name}")
    else:
        print("Building ELO (full history) ...")
        p.build_elo()
        if cache:
            p.save_elo_cache(cache)
            print(f"Wrote ELO cache -> {cache}")

    tier1 = filter_tier1_post_era(p.fights, cfg.master_start_year)
    train_f = [f for f in tier1 if f.fight_date < hsd]

    if args.eval_mode == "pristine":
        years = [int(x.strip()) for x in args.eval_years.split(",") if x.strip()]
        test_f: list = []
        for y in years:
            test_f.extend(
                filter_tier1_fights_in_calendar_year(tier1, cfg.master_start_year, y)
            )
        test_f.sort(key=lambda f: (f.fight_date, f.fight_id))
        print(
            f"Eval mode: pristine  years={years}  (matches Phase 3 / first_run_report.json cohort)",
            flush=True,
        )
    else:
        test_f = [f for f in tier1 if f.fight_date >= hsd]
        test_f.sort(key=lambda f: (f.fight_date, f.fight_id))
        print(
            "Eval mode: expanding  (all fight_date >= holdout_start; n grows with new calendar years in CSV)",
            flush=True,
        )

    print(
        f"Tier-1 post-{cfg.master_start_year}: train fight rows {len(train_f):,}  |  "
        f"test fight rows {len(test_f):,}",
        flush=True,
    )

    print("Feature matrix (train) ...")
    X_train, y_train, w_train, _ = p.build_xyw_for_fights(
        train_f,
        matrix_progress_every=args.matrix_progress_every,
        progress_prefix="  [train]",
    )
    print(f"  shape {X_train.shape}")

    print("Feature matrix (test) ...")
    X_test, y_test, _, test_included = p.build_xyw_for_fights(
        test_f,
        matrix_progress_every=args.matrix_progress_every,
        progress_prefix="  [test]",
    )
    print(f"  shape {X_test.shape}")

    t_data = time.perf_counter()

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    print("\nFitting XGBoost ...")
    fit_kw: dict = {}
    if args.sample_weight == "recency":
        fit_kw["sample_weight"] = w_train
        print("  using recency sample weights on train rows")
    clf.fit(X_train, y_train, **fit_kw)
    probs_xgb = clf.predict_proba(X_test)
    s_xgb = tier1_slice_score_from_probs(y_test, probs_xgb, fights=test_included)
    eval_label = f"pristine {args.eval_years}" if args.eval_mode == "pristine" else "expanding holdout"
    _print_score(f"XGBoost ({eval_label})", s_xgb)

    t_xgb = time.perf_counter()

    if args.fit_logistic:
        print("\nFitting multinomial logistic (same X_train; unweighted L-BFGS) ...")
        m = cfg.model
        log_model = MultinomialLogisticModel(
            n_features=X_train.shape[1],
            delta=m.huber_delta,
            l2_lambda=m.l2_lambda,
        )
        log_model.fit(
            X_train,
            y_train,
            verbose=True,
            max_iter=m.lbfgs_max_iter,
            ftol=m.lbfgs_ftol,
            gtol=m.lbfgs_gtol,
        )
        probs_log = log_model.predict_proba(X_test)
        s_log = tier1_slice_score_from_probs(y_test, probs_log, fights=test_included)
        _print_score(f"Multinomial logistic ({eval_label})", s_log)

    t_end = time.perf_counter()
    print(
        f"\nTimings:  data+ELO+features {t_data - t0:.1f}s  |  "
        f"XGBoost fit+score {t_xgb - t_data:.1f}s"
        + (f"  |  total {t_end - t0:.1f}s" if not args.fit_logistic else f"  |  through XGB {t_xgb - t_data:.1f}s  |  total {t_end - t0:.1f}s")
    )


if __name__ == "__main__":
    main()
