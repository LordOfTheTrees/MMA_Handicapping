"""
Holdout evaluation: log-loss, Brier, accuracy on Tier-1 decisive fights
with ``fight_date >= Config.holdout_start_date`` (fighter A perspective).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..data.loader import filter_tier1_post_era
from .fight_scoring import score_tier1_fight_slice

if TYPE_CHECKING:
    from ..pipeline import MMAPredictor


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
    hsd = predictor.config.holdout_start_date
    if hsd is None:
        raise ValueError("predictor.config.holdout_start_date must be set")

    if predictor.regression is None:
        raise RuntimeError("Trained predictor required")

    cand = filter_tier1_post_era(predictor.fights, predictor.config.master_start_year)
    holdout_fights = [f for f in cand if f.fight_date >= hsd]
    s = score_tier1_fight_slice(predictor, holdout_fights, eps=eps)
    n = s.n
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return n, s.mean_log_loss, s.mean_brier, s.accuracy
