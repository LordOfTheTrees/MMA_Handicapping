"""
Holdout evaluation: log-loss, Brier, accuracy on Tier-1 decisive fights
with ``fight_date >= Config.holdout_start_date`` (fighter A perspective).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from ..data.loader import filter_tier1_post_era
from ..model.regression import N_CLASSES, encode_outcome

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

    log_losses = []
    briers = []
    correct = 0

    for fight in holdout_fights:
        a_id, b_id = fight.fighter_a_id, fight.fighter_b_id
        wc, fdate = fight.weight_class, fight.fight_date
        y = encode_outcome(fight, a_id)
        if y is None:
            continue
        p = predictor.predict_proba_point_only(a_id, b_id, wc, fdate)
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        log_losses.append(-float(np.log(p[y])))
        oh = np.zeros(N_CLASSES, dtype=float)
        oh[y] = 1.0
        briers.append(float(np.sum((oh - p) ** 2)))
        if int(np.argmax(p)) == y:
            correct += 1

    n = len(log_losses)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")

    return (
        n,
        float(np.mean(log_losses)),
        float(np.mean(briers)),
        correct / n,
    )
