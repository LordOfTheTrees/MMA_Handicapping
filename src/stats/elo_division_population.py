"""ELO samples per weight class for fighters that appear on recorded fight cards.

Used by distribution charts and ``reference_distributions.json`` export so pedigree-only
state keys do not inflate division populations.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from src.data.schema import WeightClass

if TYPE_CHECKING:
    from src.pipeline import MMAPredictor


def fight_pairs_for_elo_charts(predictor: "MMAPredictor") -> set[tuple[str, WeightClass]]:
    pairs: set[tuple[str, WeightClass]] = set()
    for f in predictor.fights:
        if f.weight_class == WeightClass.UNKNOWN:
            continue
        pairs.add((f.fighter_a_id, f.weight_class))
        pairs.add((f.fighter_b_id, f.weight_class))
    return pairs


def division_order_public() -> list[WeightClass]:
    return [wc for wc in WeightClass if wc != WeightClass.UNKNOWN]


def collect_elos_by_division(
    predictor: "MMAPredictor",
    as_of: date,
    pairs: set[tuple[str, WeightClass]],
) -> dict[WeightClass, list[float]]:
    by_wc: dict[WeightClass, list[float]] = defaultdict(list)
    for fid, wc in pairs:
        e = predictor.elo_model.get_elo(fid, wc, as_of)
        by_wc[wc].append(float(e))
    return by_wc


def collect_uncertainty_by_division(
    predictor: "MMAPredictor",
    as_of: date,
    pairs: set[tuple[str, WeightClass]],
) -> dict[WeightClass, list[float]]:
    by_wc: dict[WeightClass, list[float]] = defaultdict(list)
    for fid, wc in pairs:
        st = predictor.elo_model.get_state(fid, wc, as_of)
        by_wc[wc].append(float(st.uncertainty))
    return by_wc
