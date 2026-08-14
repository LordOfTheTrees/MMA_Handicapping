#!/usr/bin/env python3
"""
Recompute the headline comparison table in docs/model-efficacy-vs-baselines.md section 4.

Scores all three systems — bespoke multinomial, generative ELO-only (section 3), and
XGBoost — on one cohort with one ELO build, so the table is internally consistent.

Cohort defaults to the **pristine** calendar years 2023-2025 used by the published
table (n = 1,529 on the current CSV), NOT ``fight_date >= holdout_start``, which also
sweeps in 2026 cards and yields a different n.

With ``--ab`` every system is scored twice: once with the current point-in-time
``get_state`` and once with the pre-fix implementation that leaked terminal ELO into
historical queries. The leaky column reproduces the previously published numbers and
is retained as provenance for why the table changed.

Bootstrap resamples are disabled: they feed prediction CIs, while scoring uses
``predict_proba_point_only``, so metrics are unchanged and the run is far faster.

Run from repo root::

    python scripts/dev/rebuild_efficacy_table.py --ab
    python scripts/dev/rebuild_efficacy_table.py --eval-mode expanding
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import resolve_date  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.loader import filter_tier1_post_era  # noqa: E402
from src.data.schema import FightRecord  # noqa: E402
from src.elo.elo import ELOModel  # noqa: E402
from src.eval.fight_scoring import (  # noqa: E402
    Tier1SliceScore,
    filter_tier1_fights_in_calendar_year,
    score_tier1_fight_slice,
    tier1_slice_score_from_probs,
)
from src.pipeline import MMAPredictor  # noqa: E402

from baseline_elo_only import (  # noqa: E402  (same directory)
    FIXED_GET_STATE,
    estimate_method_priors,
    leaky_get_state,
    score_elo_only,
)

#: Kept identical to scripts/dev/benchmark_xgboost_vs_holdout.py defaults so the
#: XGBoost row here matches the script the docs cite as its source.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    seed=42,
)


def _cohort(
    predictor: MMAPredictor, cfg: Config, mode: str, years: List[int]
) -> Tuple[List[FightRecord], List[FightRecord]]:
    tier1 = filter_tier1_post_era(predictor.fights, cfg.master_start_year)
    hsd = cfg.holdout_start_date
    train_f = [f for f in tier1 if f.fight_date < hsd]
    if mode == "pristine":
        test_f: List[FightRecord] = []
        for y in years:
            test_f.extend(filter_tier1_fights_in_calendar_year(tier1, cfg.master_start_year, y))
    else:
        test_f = [f for f in tier1 if f.fight_date >= hsd]
    test_f.sort(key=lambda f: (f.fight_date, f.fight_id))
    return train_f, test_f


def _xgb_score(
    predictor: MMAPredictor, train_f: List[FightRecord], test_f: List[FightRecord]
) -> Optional[Tier1SliceScore]:
    try:
        import xgboost as xgb
    except ImportError:
        return None
    X_train, y_train, _, _ = predictor.build_xyw_for_fights(train_f, matrix_progress_every=0)
    X_test, y_test, _, test_included = predictor.build_xyw_for_fights(test_f, matrix_progress_every=0)
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        max_depth=XGB_PARAMS["max_depth"],
        n_estimators=XGB_PARAMS["n_estimators"],
        learning_rate=XGB_PARAMS["learning_rate"],
        subsample=XGB_PARAMS["subsample"],
        colsample_bytree=XGB_PARAMS["colsample_bytree"],
        tree_method="hist",
        random_state=XGB_PARAMS["seed"],
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    return tier1_slice_score_from_probs(list(y_test), probs, fights=test_included)


def run_variant(
    data_dir: Path, holdout_start: str, mode: str, years: List[int], quiet: bool
) -> Dict[str, Tier1SliceScore]:
    cfg = Config()
    cfg.model.n_bootstrap = 0
    cfg.holdout_start_date = resolve_date(holdout_start)
    predictor = MMAPredictor(cfg)

    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        predictor.load_data(data_dir)
        predictor.build_elo()
        train_f, test_f = _cohort(predictor, cfg, mode, years)
        predictor.train_regression()
        out: Dict[str, Tier1SliceScore] = {}
        out["bespoke"] = score_tier1_fight_slice(predictor, test_f)
        pi_win, pi_loss, _, _ = estimate_method_priors(train_f)
        out["elo_only"] = score_elo_only(predictor, test_f, pi_win, pi_loss)
        xs = _xgb_score(predictor, train_f, test_f)
    if xs is not None:
        out["xgboost"] = xs
    return out


def _row(label: str, s: Optional[Tier1SliceScore]) -> str:
    if s is None:
        return f"{label:<28} {'(xgboost not installed)':>50}"
    return (
        f"{label:<28} {s.n:>6,} {s.mean_log_loss:>10.4f} {s.mean_brier:>8.4f} "
        f"{s.accuracy:>8.2%} {s.macro_f1:>8.4f} {s.mean_wl_log_loss:>9.4f} {s.wl_accuracy:>8.2%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--holdout-start", default="2023-01-01")
    ap.add_argument("--eval-mode", choices=("pristine", "expanding"), default="pristine")
    ap.add_argument("--eval-years", default="2023,2024,2025")
    ap.add_argument("--ab", action="store_true", help="Also compute the pre-fix leaky column.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    years = [int(x.strip()) for x in args.eval_years.split(",") if x.strip()]
    variants = (("leaky", leaky_get_state), ("fixed", FIXED_GET_STATE)) if args.ab else (("fixed", FIXED_GET_STATE),)

    results: Dict[str, Dict[str, Tier1SliceScore]] = {}
    for label, impl in variants:
        ELOModel.get_state = impl  # type: ignore[method-assign]
        try:
            results[label] = run_variant(
                args.data_dir, args.holdout_start, args.eval_mode, years, not args.verbose
            )
        finally:
            ELOModel.get_state = FIXED_GET_STATE  # type: ignore[method-assign]

    cohort = f"pristine {','.join(str(y) for y in years)}" if args.eval_mode == "pristine" else f">= {args.holdout_start}"
    print("\n" + "=" * 100)
    print(f"SECTION 4 HEADLINE COMPARISON — cohort: {cohort}")
    print("=" * 100)
    hdr = (
        f"{'system':<28} {'n':>6} {'log-loss':>10} {'Brier':>8} "
        f"{'6-way':>8} {'macroF1':>8} {'WL LL':>9} {'W/L acc':>8}"
    )
    for label, _ in variants:
        print(f"\n--- {label} " + "-" * (96 - len(label)))
        print(hdr)
        print(f"{'Uniform random (6-way)':<28} {'':>6} {math.log(6):>10.4f} {'':>8} {1/6:>8.2%} {'~0':>8} {math.log(2):>9.4f} {0.5:>8.2%}")
        for key, name in (
            ("bespoke", "Full model (bespoke)"),
            ("xgboost", "XGBoost"),
            ("elo_only", "ELO-only (section 3)"),
        ):
            print(_row(name, results[label].get(key)))

    if args.ab:
        print("\n" + "-" * 100)
        print("LEAK EFFECT  (leaky minus fixed; negative log-loss = the leak flattered the system)")
        print(f"{'system':<28} {'d log-loss':>12} {'d Brier':>10} {'d 6-way':>10} {'d W/L acc':>11}")
        for key, name in (
            ("bespoke", "Full model (bespoke)"),
            ("xgboost", "XGBoost"),
            ("elo_only", "ELO-only (section 3)"),
        ):
            a, b = results["leaky"].get(key), results["fixed"].get(key)
            if a is None or b is None:
                continue
            print(
                f"{name:<28} {a.mean_log_loss - b.mean_log_loss:>+12.4f} "
                f"{a.mean_brier - b.mean_brier:>+10.4f} {a.accuracy - b.accuracy:>+10.2%} "
                f"{a.wl_accuracy - b.wl_accuracy:>+11.2%}"
            )

    fx = results["fixed"]
    if "bespoke" in fx and "elo_only" in fx:
        gap = fx["elo_only"].mean_log_loss - fx["bespoke"].mean_log_loss
        u = math.log(6)
        print(
            f"\nHonest gain over uniform:  bespoke {u - fx['bespoke'].mean_log_loss:+.4f} nats  |  "
            f"ELO-only {u - fx['elo_only'].mean_log_loss:+.4f} nats"
        )
        print(f"Honest bespoke advantage over ELO-only: {gap:+.4f} nats/fight")


if __name__ == "__main__":
    main()
