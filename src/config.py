"""
Tunable model parameters.

All values here are empirically adjustable against holdout prediction
performance. Defaults are principled starting points, not final specs.
See docs/architecture.md Section 10 for the full list of open design questions.
"""
from dataclasses import dataclass, field
from typing import Dict


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
    kalman_process_noise: float = 0.0025

    # Kalman measurement noise: variance scale for each fight observation
    kalman_measurement_noise: float = 1.0


@dataclass
class FeatureConfig:
    # Exponential recency decay rate (lambda), applied per fight-equivalent unit.
    # Higher = more weight on recent fights.
    recency_decay_rate: float = 0.10

    # Minimum ELO-weighted effective fights before style axes leave cold-start blending.
    min_fights_style_estimate: int = 3

    # Only Tier 1 fights from this year onward enter the regression training set.
    era_cutoff_year: int = 2013


@dataclass
class ModelConfig:
    # Bootstrap resamples for CI computation (each refits on the training set).
    # Refits run once at ``train`` time; ``predict`` applies stored draws only.
    # Lower default keeps training time reasonable; raise for tighter CIs.
    n_bootstrap: int = 200

    # RNG seed for weighted bootstrap resamples (reproducible CIs after retrain).
    bootstrap_seed: int = 42

    # Two-sided confidence level for intervals (0.05 → 95% CI).
    ci_alpha: float = 0.05

    # Effective sample size below which Cauchy fallback triggers instead of bootstrap.
    cauchy_fallback_threshold: int = 20

    # Cauchy scale parameter used in the fallback CI.
    cauchy_scale: float = 0.15

    # Huber delta for the robust loss function (transition from quadratic to linear).
    huber_delta: float = 1.35

    # L2 regularization weight on regression coefficients.
    l2_lambda: float = 1e-4


@dataclass
class Config:
    elo: ELOConfig = field(default_factory=ELOConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
