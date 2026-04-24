"""
Tunable model parameters.

All values here are empirically adjustable against holdout prediction
performance. Defaults are principled starting points, not final specs.
The regression calendar floor is ``Config.master_start_year`` (single source of truth).
See docs/architecture.md Section 10 and docs/todo.md §3.3 for the full tuning inventory.
"""
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional


#: Default time-based holdout for Tier-1 **regression** training (fights on/after this
#: date are excluded from the multinomial fit; ELO still uses full history). Matches the
#: Phase-3 / ``eval-holdout`` protocol in docs. Override via ``--holdout-start``; disable
#: only for shipping / special runs with ``--no-holdout`` on the train command.
DEFAULT_HOLDOUT_START_DATE = date(2023, 1, 1)


@dataclass
class ELOConfig:
    # ELO tuning: k_base and logistic_divisor (see docs/elo-tuning-knobs.md).
    k_base: float = 100.0
    logistic_divisor: float = 300.0
    initial_elo: float = 1500.0

    # Cross-promotion ELO discount by tier (fraction of delta above baseline retained).
    # tier_discount[tier_int] -> float in [0, 1]
    tier_discount: Dict[int, float] = field(default_factory=lambda: {
        1: 1.00,   # UFC — no discount
        2: 0.85,   # Major promotions (Bellator, ONE, PFL, RIZIN)
        3: 0.65,   # Regional/minor promotions (Sherdog)
        4: 0.00,   # Combat sports background — no ELO transfer, prior only
    })

    # Kalman process noise: ELO variance added per day of inactivity
    kalman_process_noise: float = 0.01

    # Kalman measurement noise: variance scale for each fight observation
    kalman_measurement_noise: float = 1.0


@dataclass
class FeatureConfig:
    # Exponential recency decay rate (lambda), applied per fight-equivalent unit.
    # Higher = more weight on recent fights.
    recency_decay_rate: float = 0.10

    # Minimum ELO-weighted effective fights before style axes leave cold-start blending.
    min_fights_style_estimate: int = 3


@dataclass
class ModelConfig:
    # Bootstrap resamples for CI computation (each refits on the training set).
    # Refits run once at ``train`` time; ``predict`` applies stored draws only.
    # Lower default keeps training time reasonable; raise for tighter CIs.
    n_bootstrap: int = 200

    # RNG seed for weighted bootstrap resamples (reproducible CIs after retrain).
    bootstrap_seed: int = 42

    # Two-sided confidence level for intervals (0.10 → 90% CI).
    ci_alpha: float = 0.10

    # Effective sample size below which Cauchy fallback triggers instead of bootstrap.
    cauchy_fallback_threshold: int = 20

    # Cauchy scale parameter used in the fallback CI.
    cauchy_scale: float = 0.15

    # --- Cauchy ELO Monte Carlo (prediction-time; independent ε_a, ε_b per draw) ---
    # For each corner, ε ~ Cauchy(0, γ) in **ELO points**, then elo_draw = μ + ε.
    # γ grows with calendar idle (global last fight → predict date); cap at elo_mc_gamma_max.
    # Formula (idle_years = max(0, days_idle) / 365.25):
    #   γ = min(elo_mc_gamma_max,
    #           elo_mc_gamma_min + elo_mc_gamma_slope_sqrt_year * sqrt(idle_years))
    # Wider intervals when idle is long: larger γ ⇒ fatter Cauchy tails. Not the same as
    # training row recency weights (see ADR-18 / ADR-19).
    elo_mc_n_draws: int = 200
    elo_mc_gamma_min: float = 5.0
    elo_mc_gamma_slope_sqrt_year: float = 25.0
    elo_mc_gamma_max: float = 120.0

    # Huber delta for the robust loss function (transition from quadratic to linear).
    huber_delta: float = 1.35

    # L2 regularization weight on regression coefficients.
    l2_lambda: float = 1e-4

    def elo_mc_gamma_for_days_idle(self, days_idle: int) -> float:
        """
        Cauchy scale **γ** for one corner given calendar days since last fight (any division).

        ``idle_years = max(0, days_idle) / 365.25``;
        ``γ = min(elo_mc_gamma_max, elo_mc_gamma_min + elo_mc_gamma_slope_sqrt_year * sqrt(idle_years))``.
        """
        idle_years = max(0.0, float(days_idle)) / 365.25
        g = self.elo_mc_gamma_min + self.elo_mc_gamma_slope_sqrt_year * math.sqrt(idle_years)
        return min(self.elo_mc_gamma_max, g)


@dataclass
class Config:
    elo: ELOConfig = field(default_factory=ELOConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    #: Single calendar floor for **Tier-1 regression training** (inclusive) and for the
    #: first **expanding walk-forward** training year. Change here only — do not duplicate
    #: year cutoffs elsewhere. ELO construction still uses full fight history.
    master_start_year: int = 2005

    #: ``None`` = no date cut (train on all Tier-1 post-era rows; use sparingly). Default is
    #: :data:`DEFAULT_HOLDOUT_START_DATE`. ``main.py train --holdout-start`` / ``--no-holdout``
    #: override. ELO is always built on full history.
    holdout_start_date: Optional[date] = field(default=DEFAULT_HOLDOUT_START_DATE)
