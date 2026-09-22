"""
Real per-feature contribution and percentile reference — the authoritative "why these numbers".

The deploy site renders a per-factor breakdown (percentage contribution, division-average share,
marginal percentile) from hand-tuned coefficients ranked against a synthetically sampled
population, i.e. fabricated statistics presented as model output. Everything that breakdown
needs is derivable from the trained ``W`` and the real training matrix, so it is exported here
once, upstream, rather than re-derived (differently) by each consumer.

Definitions, all in the model's own logit units and on raw, unscaled features — the regression
applies ``X @ W.T`` with no intercept and no standardisation, so a coefficient is already
per-raw-unit:

``marginal_beta[j]``
    :math:`\\lVert W_{\\cdot j} \\rVert_2`, the Euclidean norm of feature *j*'s coefficients
    across the six class rows (:func:`~src.model.regression.coefficient_column_norms`). One
    comparable scalar per feature, replacing a hand-tuned constant.

``marginal magnitude``
    :math:`m_j = \\text{marginal_beta}[j] \\cdot \\lvert x_j \\rvert` for a matchup's feature
    vector — how much scale that factor puts on the logits for *this* bout.

``share percent``
    :math:`100 \\cdot m_j / \\sum_k m_k`; the twelve values sum to 100.

``population.marginal_magnitude_quantiles``
    101-point empirical quantile grid of :math:`m_j` over the real training rows, the same grid
    shape as ``reference_distributions.json``. Interpolating a bout's :math:`m_j` into this grid
    gives its marginal percentile against fights that actually happened.

``by_division[wc].average_share_percent``
    Mean share percent over that division's training rows — an actual weight-class baseline.

For the exact signed, per-class decomposition of a single prediction see
:meth:`~src.model.regression.MultinomialLogisticModel.predict_with_decomposition`; the
magnitudes here are its class-aggregated, sign-free summary.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.schema import WeightClass
from src.export.reference_distributions_export import (
    QUANTILE_PERCENT_LEVELS,
    ensure_training_matrix,
    quantile_grid_block_from_sample,
)
from src.matchup.interactions import FEATURE_GROUPS, FEATURE_NAMES
from src.model.regression import (
    CLASS_LABELS,
    N_CLASSES,
    coefficient_column_norms,
    relative_feature_importance,
)
from src.pipeline import MMAPredictor

FEATURE_INTERPRETABILITY_FILENAME = "feature_interpretability.json"

#: How ``marginal_beta`` is derived from ``W``; consumers should assert on this tag.
MARGINAL_BETA_DEFINITION = "l2_norm_over_class_rows"

#: Below this many training rows a division's average share is too thin to show on its own.
MIN_DIVISION_ROWS = 50


def marginal_betas(W: np.ndarray) -> np.ndarray:
    """Per-feature scalar magnitude ``||W[:, j]||_2``, aligned with :data:`FEATURE_NAMES`."""
    W_arr = np.asarray(W, dtype=float)
    if W_arr.shape != (N_CLASSES, len(FEATURE_NAMES)):
        raise ValueError(f"Unexpected W shape {W_arr.shape}; expected ({N_CLASSES}, {len(FEATURE_NAMES)})")
    return coefficient_column_norms(W_arr)


def marginal_magnitude_matrix(X: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """``|X| * betas`` broadcast over rows — one marginal magnitude per (row, feature)."""
    X_arr = np.asarray(X, dtype=float)
    b = np.asarray(betas, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[1] != b.size:
        raise ValueError(f"X shape {X_arr.shape} incompatible with {b.size} betas")
    return np.abs(X_arr) * b[None, :]


def share_percent_rows(M: np.ndarray) -> np.ndarray:
    """
    Row-normalised percentages of *M*; each row sums to 100.

    A row whose magnitudes are all ~0 (every feature at its neutral value) carries no
    information about relative weight, so it is spread uniformly rather than left as NaN.
    """
    M_arr = np.asarray(M, dtype=float)
    totals = M_arr.sum(axis=1)
    out = np.empty_like(M_arr)
    degenerate = totals <= 1e-12
    safe = np.where(degenerate, 1.0, totals)
    out[:] = 100.0 * M_arr / safe[:, None]
    if np.any(degenerate):
        out[degenerate, :] = 100.0 / M_arr.shape[1]
    return out


def average_share_percent(M: np.ndarray) -> np.ndarray:
    """Mean over rows of :func:`share_percent_rows` — the long-run share of each factor."""
    if np.asarray(M).shape[0] == 0:
        return np.full(len(FEATURE_NAMES), 100.0 / len(FEATURE_NAMES), dtype=float)
    return share_percent_rows(M).mean(axis=0)


def _named(values: Sequence[float]) -> dict[str, float]:
    return {name: float(values[j]) for j, name in enumerate(FEATURE_NAMES)}


def _division_row_indices(included: Sequence[Any]) -> dict[WeightClass, list[int]]:
    by_wc: dict[WeightClass, list[int]] = {}
    for i, fight in enumerate(included):
        by_wc.setdefault(fight.weight_class, []).append(i)
    return by_wc


def build_feature_interpretability_document(
    predictor: MMAPredictor,
    as_of: date,
    manifest: Mapping[str, Any],
    *,
    export_schema_version: str,
) -> dict[str, Any]:
    """
    Full document for :data:`FEATURE_INTERPRETABILITY_FILENAME`.

    Requires trained regression weights and the training matrix (built on demand by
    ``train_regression(fit_model=False)`` when the pickle did not retain it).
    """
    reg = predictor.regression
    if reg is None or reg.W is None:
        raise RuntimeError("Predictor must have trained regression weights (train_regression completed).")

    W = np.asarray(reg.W, dtype=float)
    betas = marginal_betas(W)

    X = ensure_training_matrix(predictor)
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"X columns {X.shape[1]} vs FEATURE_NAMES {len(FEATURE_NAMES)}")
    included = getattr(predictor, "_train_included_fights", None) or []
    if included and len(included) != X.shape[0]:
        raise ValueError(f"included fights {len(included)} != training rows {X.shape[0]}")

    M = marginal_magnitude_matrix(X, betas)
    std = np.std(X, axis=0, ddof=0) if X.shape[0] > 1 else np.zeros(len(FEATURE_NAMES))
    rel, group_fraction = relative_feature_importance(W, FEATURE_NAMES, groups=FEATURE_GROUPS)
    rel_std, group_fraction_std = relative_feature_importance(
        W, FEATURE_NAMES, scales=std, groups=FEATURE_GROUPS
    )

    divisions: dict[str, Any] = {}
    for wc, idx in sorted(_division_row_indices(included).items(), key=lambda kv: kv[0].value):
        rows = M[idx, :]
        divisions[wc.value] = {
            "n_rows": len(idx),
            "sufficient_rows": len(idx) >= MIN_DIVISION_ROWS,
            "average_share_percent": _named(average_share_percent(rows)),
        }

    cfg = predictor.config
    hsd = cfg.holdout_start_date

    return {
        "export_manifest": dict(manifest),
        "export_schema_version": export_schema_version,
        "as_of_date": as_of.isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(CLASS_LABELS),
        "marginal_beta_definition": MARGINAL_BETA_DEFINITION,
        "marginal_beta": _named(betas),
        "class_coefficients": {
            name: [float(W[k, j]) for k in range(N_CLASSES)] for j, name in enumerate(FEATURE_NAMES)
        },
        "cohort": {
            "description": "same decisive Tier-1 training rows as reference_distributions.matchup_features",
            "n_rows": int(X.shape[0]),
            "master_start_year": int(cfg.master_start_year),
            "holdout_start_date": hsd.isoformat() if hsd is not None else None,
            "min_division_rows": MIN_DIVISION_ROWS,
        },
        "population": {
            "marginal_magnitude_quantiles": {
                name: quantile_grid_block_from_sample(M[:, j])
                for j, name in enumerate(FEATURE_NAMES)
            },
            "average_share_percent": _named(average_share_percent(M)),
            "feature_std_training": _named(std),
            "coefficient_fraction": _named(rel),
            "coefficient_fraction_std_scaled": _named(rel_std),
            "group_fraction": dict(group_fraction),
            "group_fraction_std_scaled": dict(group_fraction_std),
        },
        "by_division": divisions,
        "percentile_levels": list(QUANTILE_PERCENT_LEVELS),
        "notes": (
            "marginal_beta[j] = ||W[:, j]||_2 over the six class rows; a bout's marginal "
            "magnitude is marginal_beta[j] * |x_j| on the same raw feature scale as "
            "reference_distributions.matchup_features. Percentage contribution is that "
            "magnitude normalised across the twelve features (sums to 100). Marginal "
            "percentile interpolates the magnitude into "
            "population.marginal_magnitude_quantiles (101 points, 0..100), which is the real "
            "training population, not a synthetic draw. Division-average share comes from "
            "by_division[<weight_class>].average_share_percent; fall back to "
            "population.average_share_percent when sufficient_rows is false."
        ),
    }
