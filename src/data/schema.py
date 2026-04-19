"""
Canonical in-memory data schemas for fights, fighters, and derived states.

All pipeline stages pass these dataclasses across their boundaries.
No external dependencies — pure Python stdlib.
"""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WeightClass(Enum):
    STRAWWEIGHT = "strawweight"
    FLYWEIGHT = "flyweight"
    BANTAMWEIGHT = "bantamweight"
    FEATHERWEIGHT = "featherweight"
    LIGHTWEIGHT = "lightweight"
    WELTERWEIGHT = "welterweight"
    MIDDLEWEIGHT = "middleweight"
    LIGHT_HEAVYWEIGHT = "light_heavyweight"
    HEAVYWEIGHT = "heavyweight"
    W_STRAWWEIGHT = "w_strawweight"
    W_FLYWEIGHT = "w_flyweight"
    W_BANTAMWEIGHT = "w_bantamweight"
    W_FEATHERWEIGHT = "w_featherweight"
    CATCH_WEIGHT = "catch_weight"
    UNKNOWN = "unknown"


class ResultMethod(Enum):
    KO_TKO = "ko_tko"
    SUBMISSION = "submission"
    UNANIMOUS_DECISION = "unanimous_decision"
    SPLIT_DECISION = "split_decision"
    MAJORITY_DECISION = "majority_decision"
    DRAW = "draw"
    NO_CONTEST = "no_contest"
    DQ = "dq"


class DataTier(Enum):
    TIER_1 = 1   # UFCStats — full per-fight stats
    TIER_2 = 2   # Major promotions — outcomes + partial stats
    TIER_3 = 3   # Sherdog regional — outcomes only
    TIER_4 = 4   # Combat sports background — signals only


class Stance(Enum):
    ORTHODOX = "orthodox"
    SOUTHPAW = "southpaw"
    SWITCH = "switch"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Raw source records
# ---------------------------------------------------------------------------

@dataclass
class FightStats:
    """Per-fight striking and grappling statistics. Tier 1 only."""
    significant_strikes_landed: Optional[int] = None
    significant_strikes_attempted: Optional[int] = None
    significant_strikes_absorbed: Optional[int] = None
    takedowns_landed: Optional[int] = None
    takedowns_attempted: Optional[int] = None
    control_time_seconds: Optional[int] = None
    submission_attempts: Optional[int] = None
    total_fight_time_seconds: Optional[int] = None


@dataclass
class FightRecord:
    """A single fight, promotion-agnostic."""
    fight_id: str
    fighter_a_id: str
    fighter_b_id: str
    winner_id: Optional[str]       # None for draw / NC / DQ
    result_method: ResultMethod
    weight_class: WeightClass
    fight_date: date
    promotion: str
    tier: DataTier
    fighter_a_stats: Optional[FightStats] = None
    fighter_b_stats: Optional[FightStats] = None
    #: UFCStats (etc.) label when ``weight_class`` is ``UNKNOWN`` (catch weight, interim wording, …).
    weight_class_raw: Optional[str] = None


@dataclass
class FighterProfile:
    """Static fighter attributes and combat sports pedigree signals."""
    fighter_id: str
    name: str
    reach_cm: Optional[float] = None
    height_cm: Optional[float] = None
    date_of_birth: Optional[date] = None
    stance: Stance = Stance.UNKNOWN
    # Pedigree signals in [0, 1] for cold-start ELO and style axis priors
    wrestling_pedigree: float = 0.0
    boxing_pedigree: float = 0.0
    bjj_pedigree: float = 0.0


# ---------------------------------------------------------------------------
# Derived state objects
# ---------------------------------------------------------------------------

@dataclass
class ELOState:
    """Current ELO estimate for a fighter in one weight class."""
    fighter_id: str
    weight_class: WeightClass
    elo: float
    uncertainty: float              # Kalman variance
    #: Last bout **in this weight class**. Kalman time-updates use last bout **any** class (see ADR-15).
    last_fight_date: Optional[date] = None
    n_fights: int = 0
    primary_tier: DataTier = DataTier.TIER_3


@dataclass
class StyleAxes:
    """ELO-weighted, recency-decayed style axis scores for one weight class."""
    fighter_id: str
    weight_class: WeightClass
    striker_score: float            # [0, 1]
    grappler_score: float           # [0, 1]
    finish_threat: float            # [0, 1]
    finish_vulnerability: float     # [0, 1]
    striker_uncertainty: float
    grappler_uncertainty: float
    n_quality_fights: float         # ELO-weighted effective sample size


@dataclass
class MatchupFeatures:
    """
    Full 12-feature vector for a matchup from Fighter A's perspective.

    Antisymmetry guarantee: swapping A↔B negates all signed differences
    and flips the interaction terms accordingly, so predicted win probs
    become lose probs and vice versa.
    """
    elo_differential: float
    striker_score_diff: float
    grappler_score_diff: float
    finish_threat_diff: float
    finish_vulnerability_diff: float
    striking_matchup: float
    grappling_matchup: float
    finish_matchup: float
    reach_diff_cm: Optional[float]
    height_diff_cm: Optional[float]
    stance_mismatch: int            # 1 if orthodox vs. southpaw, else 0
    age_diff_days: Optional[float]  # positive = Fighter A is older


@dataclass
class PredictionResult:
    """6-class prediction with confidence intervals and metadata."""
    fighter_a_id: str
    fighter_b_id: str
    weight_class: WeightClass
    fight_date: date
    # Point estimates (sum to 1.0)
    p_win_ko_tko: float
    p_win_submission: float
    p_win_decision: float
    p_lose_decision: float
    p_lose_ko_tko: float
    p_lose_submission: float
    # 95% CI tuples (lower, upper)
    ci_win_ko_tko: tuple
    ci_win_submission: tuple
    ci_win_decision: tuple
    ci_lose_decision: tuple
    ci_lose_ko_tko: tuple
    ci_lose_submission: tuple
    # Metadata
    ci_method: str                  # bootstrap | bootstrap_elo_mc | elo_mc | cauchy | cauchy_wc_debut
    effective_n: float
    pct_post_era: float
    features: Optional[MatchupFeatures] = None

    # ---------------------------------------------------------------------------
    # Derived aggregates (architecture Section 1.2)
    # ---------------------------------------------------------------------------

    @property
    def total_win(self) -> float:
        return self.p_win_ko_tko + self.p_win_submission + self.p_win_decision

    @property
    def finish_win(self) -> float:
        return self.p_win_ko_tko + self.p_win_submission

    @property
    def finish_lose(self) -> float:
        return self.p_lose_ko_tko + self.p_lose_submission

    @property
    def go_to_decision(self) -> float:
        return self.p_win_decision + self.p_lose_decision
