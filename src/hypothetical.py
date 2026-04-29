"""
Hypothetical matchups with optional assumed idle periods.

``MMAPredictor.predict(..., hypothetical_days_idle_a=b)`` substitutes **calendar days idle**
when computing Cauchy–ELO Monte Carlo **γ** (confidence-interval width only). Points, ELO state
applied in features (Kalman), and style axes still come from observations through ``fight_date``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .data.schema import PredictionResult, WeightClass
from .pipeline import MMAPredictor

# Matches ``src/cli/plot_prediction_three_viz.py`` defaults (Holloway vs Volkanovski examples).
DEFAULT_DEMO_FIGHTER_A = "150ff4cc642270b9"
DEFAULT_DEMO_FIGHTER_B = "e1248941344b3288"
DEFAULT_DEMO_DAYS_IDLE_A = 120
DEFAULT_DEMO_DAYS_IDLE_B = 90


@dataclass(frozen=True)
class HypotheticalFightSpec:
    """Two corners, booked division and date; optional idle overrides for Cauchy γ."""

    fighter_a_id: str
    fighter_b_id: str
    weight_class: WeightClass
    fight_date: date
    days_idle_a: Optional[int] = None
    days_idle_b: Optional[int] = None


def predict_hypothetical(
    predictor: MMAPredictor,
    spec: HypotheticalFightSpec,
    *,
    verbose: bool = True,
) -> PredictionResult:
    """``predict()`` with idle overrides routed through kwargs (γ only — see module docstring)."""
    return predictor.predict(
        spec.fighter_a_id,
        spec.fighter_b_id,
        spec.weight_class,
        spec.fight_date,
        verbose=verbose,
        hypothetical_days_idle_a=spec.days_idle_a,
        hypothetical_days_idle_b=spec.days_idle_b,
    )


def predict_hypothetical_default_pair(
    predictor: MMAPredictor,
    *,
    fight_date: Optional[date] = None,
    weight_class: WeightClass = WeightClass.FEATHERWEIGHT,
    days_idle_a: int = DEFAULT_DEMO_DAYS_IDLE_A,
    days_idle_b: int = DEFAULT_DEMO_DAYS_IDLE_B,
    verbose: bool = True,
) -> PredictionResult:
    """
    Built-in demo matchup (default Holloway-style vs Volkanovski-style ids) at ``days_idle_*``.
    Prefer :func:`predict_hypothetical` with explicit ids when you care about realism.
    """
    fd = fight_date if fight_date is not None else date.today()
    spec = HypotheticalFightSpec(
        fighter_a_id=DEFAULT_DEMO_FIGHTER_A,
        fighter_b_id=DEFAULT_DEMO_FIGHTER_B,
        weight_class=weight_class,
        fight_date=fd,
        days_idle_a=days_idle_a,
        days_idle_b=days_idle_b,
    )
    return predict_hypothetical(predictor, spec, verbose=verbose)
