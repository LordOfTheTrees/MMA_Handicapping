"""
Matchup interaction terms and full feature vector assembly.

Interaction terms are specified from the causal theory of fight outcomes
(architecture Section 6) — they are NOT learned by the regression model but
constructed explicitly before the model sees the data.

    striking_matchup  = striker_score_A  × (1 - striker_score_B)
    grappling_matchup = grappler_score_A × (1 - grappler_score_B)
    finish_matchup    = finish_threat_A  × finish_vulnerability_B

Antisymmetry guarantee: swapping A↔B negates all signed differences,
reverses the interaction terms, and flips predicted win↔lose probabilities.
"""
import numpy as np
from datetime import date
from typing import List, Optional

from ..data.schema import (
    ELOState, FighterProfile, MatchupFeatures, Stance, StyleAxes,
)


# ---------------------------------------------------------------------------
# Interaction term functions
# ---------------------------------------------------------------------------

def striking_matchup(striker_score_a: float, striker_score_b: float) -> float:
    """
    A's striking advantage enabled by B's striking weakness.
    High when A is a strong striker and B is not.
    """
    return striker_score_a * (1.0 - striker_score_b)


def grappling_matchup(grappler_score_a: float, grappler_score_b: float) -> float:
    """
    A's grappling advantage enabled by B's grappling weakness.
    High when A is a strong grappler and B is not.
    """
    return grappler_score_a * (1.0 - grappler_score_b)


def finish_matchup(finish_threat_a: float, finish_vulnerability_b: float) -> float:
    """
    A's finishing ability enabled by B's durability weakness.
    High when A finishes opponents and B gets finished.
    """
    return finish_threat_a * finish_vulnerability_b


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

#: Canonical feature name ordering — must match features_to_array() output.
FEATURE_NAMES: List[str] = [
    "elo_differential",
    "striker_score_diff",
    "grappler_score_diff",
    "finish_threat_diff",
    "finish_vulnerability_diff",
    "striking_matchup",
    "grappling_matchup",
    "finish_matchup",
    "reach_diff_cm",
    "height_diff_cm",
    "stance_mismatch",
    "age_diff_days",
]

N_FEATURES = len(FEATURE_NAMES)


def build_matchup_features(
    elo_a: ELOState,
    elo_b: ELOState,
    axes_a: StyleAxes,
    axes_b: StyleAxes,
    profile_a: FighterProfile,
    profile_b: FighterProfile,
    fight_date: date,
) -> MatchupFeatures:
    """
    Construct the full MatchupFeatures dataclass from Fighter A's perspective.

    Physical features (reach, height, age) are set to None when either
    fighter's profile is missing the required attribute; they are imputed
    to 0.0 (mean-centered) when the numpy array is built in features_to_array().
    """
    reach_diff = (
        profile_a.reach_cm - profile_b.reach_cm
        if profile_a.reach_cm is not None and profile_b.reach_cm is not None
        else None
    )
    height_diff = (
        profile_a.height_cm - profile_b.height_cm
        if profile_a.height_cm is not None and profile_b.height_cm is not None
        else None
    )

    # Stance mismatch: orthodox vs. southpaw is the meaningful asymmetry
    stance_values = {profile_a.stance, profile_b.stance}
    mismatch = int(
        Stance.ORTHODOX in stance_values and Stance.SOUTHPAW in stance_values
    )

    # Age diff: positive = Fighter A is older on fight day
    age_diff = None
    if profile_a.date_of_birth is not None and profile_b.date_of_birth is not None:
        age_diff = float((profile_a.date_of_birth - profile_b.date_of_birth).days)

    return MatchupFeatures(
        elo_differential=elo_a.elo - elo_b.elo,
        striker_score_diff=axes_a.striker_score - axes_b.striker_score,
        grappler_score_diff=axes_a.grappler_score - axes_b.grappler_score,
        finish_threat_diff=axes_a.finish_threat - axes_b.finish_threat,
        finish_vulnerability_diff=axes_a.finish_vulnerability - axes_b.finish_vulnerability,
        striking_matchup=striking_matchup(axes_a.striker_score, axes_b.striker_score),
        grappling_matchup=grappling_matchup(axes_a.grappler_score, axes_b.grappler_score),
        finish_matchup=finish_matchup(axes_a.finish_threat, axes_b.finish_vulnerability),
        reach_diff_cm=reach_diff,
        height_diff_cm=height_diff,
        stance_mismatch=mismatch,
        age_diff_days=age_diff,
    )


def features_to_array(features: MatchupFeatures) -> np.ndarray:
    """
    Convert a MatchupFeatures dataclass to a 1-D numpy array.

    Missing physical attributes (reach, height, age) are imputed with 0.0,
    which corresponds to no measurable advantage — a conservative default.
    """
    return np.array([
        features.elo_differential,
        features.striker_score_diff,
        features.grappler_score_diff,
        features.finish_threat_diff,
        features.finish_vulnerability_diff,
        features.striking_matchup,
        features.grappling_matchup,
        features.finish_matchup,
        features.reach_diff_cm    if features.reach_diff_cm    is not None else 0.0,
        features.height_diff_cm   if features.height_diff_cm   is not None else 0.0,
        float(features.stance_mismatch),
        features.age_diff_days    if features.age_diff_days    is not None else 0.0,
    ], dtype=float)
