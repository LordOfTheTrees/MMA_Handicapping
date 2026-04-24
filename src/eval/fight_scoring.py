"""
Shared Tier-1 fight scoring: log-loss, Brier, accuracy, macro F1, per–weight-class slices.

Used by holdout eval and walk-forward / tuning harness. No CI cost — point probabilities only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from ..data.schema import DataTier, FightRecord, WeightClass
from ..model.regression import N_CLASSES, encode_outcome

if TYPE_CHECKING:
    from ..pipeline import MMAPredictor


def multiclass_macro_f1(y_true: List[int], y_pred: List[int], n_classes: int = N_CLASSES) -> float:
    """
    Unweighted (macro) F1: mean over classes of per-class F1. Classes with no true support
    are skipped in the mean (if no support, contribution 0) — same spirit as zero_division=0 in sklearn.
    """
    f1s: List[float] = []
    n = len(y_true)
    if n == 0:
        return float("nan")
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
        else:
            p_prec = tp / (tp + fp)
            p_rec = tp / (tp + fn)
            if p_prec + p_rec == 0:
                f1s.append(0.0)
            else:
                f1s.append(2.0 * p_prec * p_rec / (p_prec + p_rec))
    if not f1s:
        return float("nan")
    return float(np.mean(f1s))


@dataclass
class WeightClassScoreSlice:
    n: int
    mean_log_loss: float
    mean_brier: float
    accuracy: float
    macro_f1: float


@dataclass
class Tier1SliceScore:
    n: int
    mean_log_loss: float
    mean_brier: float
    accuracy: float
    macro_f1: float
    by_weight_class: Dict[str, WeightClassScoreSlice] = field(default_factory=dict)


def score_tier1_fight_slice(
    predictor: "MMAPredictor",
    fights: List[FightRecord],
    *,
    eps: float = 1e-15,
) -> Tier1SliceScore:
    """
    Score a list of fights (e.g. one calendar year, one holdout) using PIT predict.

    Fights with no valid decisive label (draw/NC) are skipped.
    """
    if predictor.regression is None:
        raise RuntimeError("Trained predictor required")

    log_losses: List[float] = []
    briers: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    by_wc: Dict[str, Dict[str, List[Any]]] = {}

    for fight in fights:
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
        pred = int(np.argmax(p))
        y_true.append(y)
        y_pred.append(pred)

        wck = wc.value
        if wck not in by_wc:
            by_wc[wck] = {"ll": [], "br": [], "yt": [], "yp": []}
        by_wc[wck]["ll"].append(log_losses[-1])
        by_wc[wck]["br"].append(briers[-1])
        by_wc[wck]["yt"].append(y)
        by_wc[wck]["yp"].append(pred)

    n = len(log_losses)
    if n == 0:
        return Tier1SliceScore(0, float("nan"), float("nan"), float("nan"), float("nan"), {})

    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / n)
    f1 = multiclass_macro_f1(y_true, y_pred)
    w_slices: Dict[str, WeightClassScoreSlice] = {}
    for wck, b in by_wc.items():
        m = len(b["ll"])
        if m == 0:
            continue
        w_acc = float(sum(1 for t, p in zip(b["yt"], b["yp"]) if t == p) / m)
        w_f1 = multiclass_macro_f1(b["yt"], b["yp"])
        w_slices[wck] = WeightClassScoreSlice(
            n=m,
            mean_log_loss=float(np.mean(b["ll"])),
            mean_brier=float(np.mean(b["br"])),
            accuracy=w_acc,
            macro_f1=w_f1,
        )

    return Tier1SliceScore(
        n=n,
        mean_log_loss=float(np.mean(log_losses)),
        mean_brier=float(np.mean(briers)),
        accuracy=acc,
        macro_f1=f1,
        by_weight_class=w_slices,
    )


def filter_tier1_fights_in_calendar_year(
    fights: List[FightRecord],
    master_start_year: int,
    year: int,
) -> List[FightRecord]:
    return [
        f
        for f in fights
        if f.tier == DataTier.TIER_1
        and f.fight_date.year == year
        and f.fight_date.year >= master_start_year
    ]
