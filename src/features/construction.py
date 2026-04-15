"""
ELO-weighted, recency-decayed style axis construction.

Each axis score is a weighted average over a fighter's fight history where
each fight's contribution = ELO quality weight × recency weight.

  quality_weight  = f(opponent ELO at fight time)   → better wins count more
  recency_weight  = exp(-λ × fights_ago)             → recent fights count more

Architecture responsibilities (Section 5):
  - Striker score: net significant strike rate, Tier 1 only
  - Grappler score: takedown rate + control share + sub attempts, Tier 1 only
  - Finish threat: weighted finish win rate, all tiers
  - Finish vulnerability: weighted finish loss rate, all tiers
  - Cold-start pedigree blending when effective sample < threshold
"""
import math
from datetime import date
from typing import List, Optional

from ..config import FeatureConfig
from ..data.schema import (
    DataTier, FightRecord, FightStats, FighterProfile,
    ResultMethod, StyleAxes, WeightClass,
)
from ..elo.elo import ELOModel


# Finish-result methods for finish threat / vulnerability
_FINISH_METHODS = {ResultMethod.KO_TKO, ResultMethod.SUBMISSION}

# Days assumed per "one fight" for recency decay unit conversion
_DAYS_PER_FIGHT = 122.0   # ≈ 3 fights/year


# ---------------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------------

def _recency_weight(fight_date: date, reference_date: date, decay_rate: float) -> float:
    """
    Exponential weight: exp(-λ × fights_ago).
    fights_ago is estimated from calendar days / average inter-fight interval.
    """
    days = max(0, (reference_date - fight_date).days)
    fights_ago = days / _DAYS_PER_FIGHT
    return math.exp(-decay_rate * fights_ago)


def _quality_weight(opponent_elo: float, baseline: float = 1500.0) -> float:
    """
    Linear quality weight centred at baseline ELO, soft-clipped to [0.1, 3.0].
    Each 400 ELO points above/below baseline shifts weight by ±1.0.
    """
    raw = 1.0 + (opponent_elo - baseline) / 400.0
    return max(0.1, min(raw, 3.0))


# ---------------------------------------------------------------------------
# Per-fight signal functions
# ---------------------------------------------------------------------------

def _soft_normalize(value: float, center: float, scale: float) -> float:
    """Logistic squash to (0, 1) centred at center with steepness scale."""
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def _striking_signal(stats: FightStats) -> Optional[float]:
    """
    Normalised net significant strike rate for the fighter.

    Net strikes per minute = (landed - absorbed) / fight_minutes.
    Squashed through a logistic centred at 0 net strikes/min.
    Returns None when required fields are missing.
    """
    if (
        stats.significant_strikes_landed is None
        or stats.significant_strikes_absorbed is None
        or stats.total_fight_time_seconds is None
        or stats.total_fight_time_seconds == 0
    ):
        return None

    fight_min = stats.total_fight_time_seconds / 60.0
    net_per_min = (stats.significant_strikes_landed - stats.significant_strikes_absorbed) / fight_min
    return _soft_normalize(net_per_min, center=0.0, scale=3.0)


def _grappling_signal(stats: FightStats) -> Optional[float]:
    """
    Composite grappling domain signal: average of available sub-components.

    Components (each in [0, 1]):
      - Takedown landing rate (TDs landed / fight-minute)
      - Control time share (control seconds / total fight seconds)
      - Submission attempt rate (attempts / fight-minute)

    Returns None when no component can be computed.
    """
    if stats.total_fight_time_seconds is None or stats.total_fight_time_seconds == 0:
        return None

    fight_min = stats.total_fight_time_seconds / 60.0
    components = []

    if stats.takedowns_landed is not None:
        td_rate = stats.takedowns_landed / fight_min
        components.append(_soft_normalize(td_rate, center=0.5, scale=1.0))

    if stats.control_time_seconds is not None:
        ctrl_share = stats.control_time_seconds / stats.total_fight_time_seconds
        components.append(max(0.0, min(ctrl_share, 1.0)))

    if stats.submission_attempts is not None:
        sub_rate = stats.submission_attempts / fight_min
        components.append(_soft_normalize(sub_rate, center=0.3, scale=0.5))

    if not components:
        return None
    return sum(components) / len(components)


# ---------------------------------------------------------------------------
# Main axis computation
# ---------------------------------------------------------------------------

def compute_style_axes(
    fighter_id: str,
    wc: WeightClass,
    fight_history: List[FightRecord],
    elo_model: ELOModel,
    reference_date: date,
    config: FeatureConfig,
) -> StyleAxes:
    """
    Compute all four style axis scores for fighter_id in wc as of reference_date.

    fight_history should contain ALL fights the fighter has been in across any
    weight class and tier — filtering to wc happens internally.
    No fight on or after reference_date is used (strict no-lookahead).
    """
    striker_num = striker_den = 0.0
    grappler_num = grappler_den = 0.0
    finish_threat_num = finish_threat_den = 0.0
    finish_vuln_num = finish_vuln_den = 0.0
    effective_n = 0.0

    for fight in fight_history:
        if fight.weight_class != wc:
            continue
        if fight.fight_date >= reference_date:
            continue  # no lookahead

        # Determine perspective
        if fight.fighter_a_id == fighter_id:
            my_stats = fight.fighter_a_stats
            opponent_id = fight.fighter_b_id
        elif fight.fighter_b_id == fighter_id:
            my_stats = fight.fighter_b_stats
            opponent_id = fight.fighter_a_id
        else:
            continue

        is_decisive = fight.winner_id is not None
        i_won = fight.winner_id == fighter_id
        i_finished_opponent = i_won and fight.result_method in _FINISH_METHODS
        i_was_finished = (not i_won) and is_decisive and fight.result_method in _FINISH_METHODS

        # Weights
        opp_elo = elo_model.get_elo(opponent_id, wc, as_of_date=fight.fight_date)
        q_w = _quality_weight(opp_elo)
        r_w = _recency_weight(fight.fight_date, reference_date, config.recency_decay_rate)
        w = q_w * r_w
        effective_n += w

        # Striker / grappler scores: Tier 1 only (full stats required)
        if fight.tier == DataTier.TIER_1 and my_stats is not None:
            strike_sig = _striking_signal(my_stats)
            if strike_sig is not None:
                striker_num += w * strike_sig
                striker_den += w

            grap_sig = _grappling_signal(my_stats)
            if grap_sig is not None:
                grappler_num += w * grap_sig
                grappler_den += w

        # Finish threat / vulnerability: all tiers
        if is_decisive:
            finish_threat_num += w * float(i_finished_opponent)
            finish_threat_den += w
            finish_vuln_num += w * float(i_was_finished)
            finish_vuln_den += w

    # Weighted averages with weight-class-mean fallbacks on zero denominators
    striker_score = striker_num / striker_den if striker_den > 0.0 else 0.50
    grappler_score = grappler_num / grappler_den if grappler_den > 0.0 else 0.40
    finish_threat = finish_threat_num / finish_threat_den if finish_threat_den > 0.0 else 0.30
    finish_vuln = finish_vuln_num / finish_vuln_den if finish_vuln_den > 0.0 else 0.20

    # Uncertainty: shrinks as effective sample grows
    uncertainty = 1.0 / (1.0 + effective_n)

    return StyleAxes(
        fighter_id=fighter_id,
        weight_class=wc,
        striker_score=striker_score,
        grappler_score=grappler_score,
        finish_threat=finish_threat,
        finish_vulnerability=finish_vuln,
        striker_uncertainty=uncertainty,
        grappler_uncertainty=uncertainty,
        n_quality_fights=effective_n,
    )


# ---------------------------------------------------------------------------
# Cold-start pedigree blending (architecture Section 7.4)
# ---------------------------------------------------------------------------

def apply_cold_start_prior(
    axes: StyleAxes,
    profile: FighterProfile,
    config: FeatureConfig,
) -> StyleAxes:
    """
    Blend computed axes toward pedigree priors when effective sample is low.

    When n_quality_fights < min_fights_style_estimate the axes are interpolated
    linearly between the data estimate and the pedigree-derived prior.
    Above the threshold the data estimate is returned unchanged.
    """
    if axes.n_quality_fights >= config.min_fights_style_estimate:
        return axes

    data_weight = axes.n_quality_fights / config.min_fights_style_estimate

    grappler_prior = max(profile.wrestling_pedigree, profile.bjj_pedigree)
    striker_prior = profile.boxing_pedigree

    blended_grappler = (
        data_weight * axes.grappler_score
        + (1.0 - data_weight) * (grappler_prior if grappler_prior > 0 else 0.40)
    )
    blended_striker = (
        data_weight * axes.striker_score
        + (1.0 - data_weight) * (striker_prior if striker_prior > 0 else 0.50)
    )

    return StyleAxes(
        fighter_id=axes.fighter_id,
        weight_class=axes.weight_class,
        striker_score=blended_striker,
        grappler_score=blended_grappler,
        finish_threat=axes.finish_threat,
        finish_vulnerability=axes.finish_vulnerability,
        striker_uncertainty=axes.striker_uncertainty,
        grappler_uncertainty=axes.grappler_uncertainty,
        n_quality_fights=axes.n_quality_fights,
    )
