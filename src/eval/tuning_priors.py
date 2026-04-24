"""
Random Config sampling for Phase-3 random search (i.i.d. trials).

Keep priors **explicit** here; see ``docs/hyperparameter-tuning.md``. Does not
sample ``tier_discount`` (interaction-heavy); can extend later.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config


def sample_random_config(rng: np.random.Generator, base: "Config") -> "Config":
    """
    One i.i.d. draw: copies *base* and perturbs a subset of ELO / feature / model scalars.
    """
    c = copy.deepcopy(base)
    c.elo.k_base = float(rng.uniform(70.0, 130.0))
    c.elo.logistic_divisor = float(rng.uniform(220.0, 380.0))
    c.elo.kalman_process_noise = float(rng.uniform(0.005, 0.02))
    c.elo.kalman_measurement_noise = float(rng.uniform(0.5, 2.0))
    c.features.recency_decay_rate = float(rng.uniform(0.05, 0.18))
    c.features.min_fights_style_estimate = int(rng.integers(2, 6))
    c.model.l2_lambda = float(10 ** rng.uniform(-5.0, -2.5))
    c.model.huber_delta = float(rng.uniform(1.1, 1.6))
    return c
