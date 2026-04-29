"""
Holdout evaluation: log-loss, Brier, accuracy on Tier-1 decisive fights
with ``fight_date >= Config.holdout_start_date`` (fighter A perspective).

Baselines shown in :func:`print_holdout_baseline_report` compare the model to
uninformed random baselines (uniform 6-class probabilities, coin-flip binary W/L
and finish/decision).
"""
from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING, Tuple

from ..data.loader import filter_tier1_post_era
from .fight_scoring import Tier1SliceScore, score_tier1_fight_slice

if TYPE_CHECKING:
    from ..pipeline import MMAPredictor

#: Six-way uniform log-loss baseline  :math:`\\log 6`.
LOG_LOSS_UNIFORM_SIXWAY = math.log(6.0)

#: Calibrated coin-flip for binary outcomes (fair 50/50 in log-loss).
LOG_LOSS_COIN_BINARY = math.log(2.0)


def holdout_tier1_slice(
    predictor: "MMAPredictor",
    *,
    eps: float = 1e-15,
) -> Tier1SliceScore:
    """
    Full metric slice on the holdout Tier-1 rows (same cohort as ``run_holdout_eval``).
    """
    hsd = predictor.config.holdout_start_date
    if hsd is None:
        raise ValueError("predictor.config.holdout_start_date must be set")

    if predictor.regression is None:
        raise RuntimeError("Trained predictor required")

    cand = filter_tier1_post_era(predictor.fights, predictor.config.master_start_year)
    holdout_fights = [f for f in cand if f.fight_date >= hsd]
    return score_tier1_fight_slice(predictor, holdout_fights, eps=eps)


def print_holdout_baseline_report(s: Tier1SliceScore, holdout_date: date) -> None:
    """Print model metrics next to uninformed noise / random baselines (same *n*)."""
    if s.n == 0:
        return

    rnd6 = LOG_LOSS_UNIFORM_SIXWAY
    rnd_bin = LOG_LOSS_COIN_BINARY

    print(
        f"\n  Relative to random / noise  (holdout start {holdout_date}; n={s.n:,})\n"
        "\n  Six-way outcome (KO / sub / decision x win or loss for A)\n"
        f"    Mean log-loss:     {s.mean_log_loss:.4f}"
        f"  |  uniform prior  {rnd6:.4f}"
        f"  |  gain  {rnd6 - s.mean_log_loss:+.4f}  (positive = better than noise)\n"
        f"    Accuracy:          {s.accuracy:.2%}"
        f"  |  random class  {100.0 / 6.0:.2f}%"
        f"  |  gain  {s.accuracy - 1.0 / 6.0:+.2%}  (absolute points)\n"
        f"    Macro F1:          {s.macro_f1:.4f}  (uniform random labels -> ~0 on balanced data)\n"
    )
    print(
        "  Binary W/L (fighter A)\n"
        f"    Mean log-loss:     {s.mean_wl_log_loss:.4f}"
        f"  |  50/50 baseline  {rnd_bin:.4f}"
        f"  |  gain  {rnd_bin - s.mean_wl_log_loss:+.4f}\n"
        f"    Accuracy:          {s.wl_accuracy:.2%}"
        f"  |  coin flip  50.00%"
        f"  |  gain  {s.wl_accuracy - 0.5:+.2%}  (absolute points)\n"
        f"    Binary F1:         {s.wl_f1:.4f}\n"
    )
    print(
        "  Finish vs decision (binary; collapse finishes from both sides)\n"
        f"    Binary F1:         {s.finish_f1:.4f}  (vs ~0 F1 for i.i.d. fair coin labels on balanced data)\n"
    )

    if s.by_weight_class:
        print("  Per weight class (mean log-loss / macro F1 / WL acc):")
        for wck in sorted(s.by_weight_class.keys()):
            w = s.by_weight_class[wck]
            print(
                f"    {wck:<22s}  n={w.n:4d}  "
                f"LL={w.mean_log_loss:.3f}  macroF1={w.macro_f1:.3f}  wl_acc={w.wl_accuracy:.1%}"
            )
    print()


def run_holdout_eval(
    predictor: "MMAPredictor",
    *,
    eps: float = 1e-15,
) -> Tuple[int, float, float, float]:
    """
    Score all holdout rows using ``predict_proba_point_only`` (no CI cost).

    Returns:
        n_fights, mean_log_loss, mean_brier, accuracy
    """
    s = holdout_tier1_slice(predictor, eps=eps)
    n = s.n
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return n, s.mean_log_loss, s.mean_brier, s.accuracy
