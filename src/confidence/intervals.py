"""
Confidence interval computation for 6-class fight predictions.

Two methods (architecture Section 8):

  Bootstrap (primary):
    Resample training observations with recency weights, refit coefficients,
    propagate through softmax, report percentile intervals.
    Used when effective sample size >= cauchy_fallback_threshold.
    Bootstrap refits run once during ``train_regression``; coefficient draws are
    stored on the predictor so ``predict`` only evaluates softmax(W @ x) per
    draw (no per-prediction refit). Old pickles without stored draws fall back
    to the legacy per-predict bootstrap path if training arrays are present.

  Cauchy (fallback):
    Heavy-tailed conservative interval making no distributional assumptions.
    Used when reference class data is sparse or fighter is poorly observed.

The choice is automatic based on effective sample size after weighting.
"""
import numpy as np
from scipy.stats import cauchy
from typing import List, Optional, Tuple

from ..config import ModelConfig
from ..model.regression import MultinomialLogisticModel, N_CLASSES, CLASS_LABELS


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish effective sample size: ESS = (Σw)² / Σ(w²).
    Measures how many equally-weighted observations the weighted sample
    is worth. Smaller than n when weights are unequal.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total == 0.0:
        return 0.0
    return total ** 2 / np.sum(w ** 2)


# ---------------------------------------------------------------------------
# Train-time bootstrap coefficient storage + fast prediction CI
# ---------------------------------------------------------------------------

def fit_bootstrap_coefficients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: Optional[np.ndarray],
    config: ModelConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    progress_every: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Weighted bootstrap: refit ``n_bootstrap`` multinomial models on resamples.

    Returns ``(W_stack, n_valid)`` where ``W_stack`` has shape
    ``(n_valid, N_CLASSES, n_features)`` — one coefficient matrix per successful
    resample. Used for fast CIs at prediction time without refitting.

    If fewer than 10 successful fits, returns an empty stack ``(0, K, F)`` and
    ``n_valid`` so callers can fall back to Cauchy or legacy bootstrap.
    """
    if rng is None:
        seed = getattr(config, "bootstrap_seed", 42)
        rng = np.random.default_rng(seed)

    n = len(y_train)
    n_features = X_train.shape[1]
    W_list: List[np.ndarray] = []

    probs_w = None
    if train_weights is not None:
        s = train_weights.sum()
        probs_w = train_weights / s if s > 0 else None

    for b in range(config.n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=probs_w)
        X_b, y_b = X_train[idx], y_train[idx]

        if len(np.unique(y_b)) < N_CLASSES:
            continue

        model_b = MultinomialLogisticModel(
            n_features=n_features,
            delta=config.huber_delta,
            l2_lambda=config.l2_lambda,
        )
        try:
            model_b.fit(X_b, y_b, max_iter=500)
            if model_b.W is not None:
                W_list.append(np.asarray(model_b.W, dtype=float).copy())
        except Exception:
            pass

        if progress_every > 0 and (b + 1) % progress_every == 0:
            print(
                f"  [train bootstrap] resample {b + 1:,} / {config.n_bootstrap:,}",
                flush=True,
            )

    n_valid = len(W_list)
    if n_valid < 10:
        return np.empty((0, N_CLASSES, n_features), dtype=float), n_valid

    return np.stack(W_list, axis=0), n_valid


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for (n_samples, n_classes)."""
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=1, keepdims=True)


def bootstrap_ci_from_stored_W(
    x: np.ndarray,
    W_stack: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentile CIs from precomputed bootstrap coefficient matrices.

    ``W_stack`` shape ``(n_draws, N_CLASSES, n_features)``;
    ``x`` shape ``(n_features,)``.
    """
    logits = np.matmul(W_stack, x)  # (n_draws, N_CLASSES)
    probs = _softmax_rows(logits)
    lower = np.percentile(probs, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(probs, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper


# ---------------------------------------------------------------------------
# Legacy per-predict bootstrap CI (fallback for old pickles without stored W)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: Optional[np.ndarray],
    config: ModelConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Percentile bootstrap confidence intervals for a single prediction vector x.

    Procedure:
      1. Weighted resample training observations with replacement.
      2. Refit regression coefficients on resample.
      3. Predict x with refitted model.
      4. Report alpha/2 and 1-alpha/2 percentiles across resamples.

    Returns:
        lower      : (N_CLASSES,) lower bounds
        upper      : (N_CLASSES,) upper bounds
        n_valid    : number of successful resamples (failed fits are dropped)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y_train)
    n_features = x.shape[0]
    all_probs = np.full((config.n_bootstrap, N_CLASSES), np.nan)

    probs_w = None
    if train_weights is not None:
        s = train_weights.sum()
        probs_w = train_weights / s if s > 0 else None

    for b in range(config.n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=probs_w)
        X_b, y_b = X_train[idx], y_train[idx]

        # Skip degenerate resamples (fewer than N_CLASSES unique classes)
        if len(np.unique(y_b)) < N_CLASSES:
            continue

        model_b = MultinomialLogisticModel(
            n_features=n_features,
            delta=config.huber_delta,
            l2_lambda=config.l2_lambda,
        )
        try:
            model_b.fit(X_b, y_b, max_iter=500)
            all_probs[b] = model_b.predict_proba(x)
        except Exception:
            pass  # leave as NaN — filtered below

        if progress_every > 0 and (b + 1) % progress_every == 0:
            print(
                f"  [bootstrap CI] resample {b + 1:,} / {config.n_bootstrap:,}",
                flush=True,
            )

    valid = all_probs[~np.any(np.isnan(all_probs), axis=1)]
    n_valid = len(valid)

    if n_valid < 10:
        # Too few valid resamples — fall back to Cauchy instead
        return np.zeros(N_CLASSES), np.ones(N_CLASSES), n_valid

    alpha = config.ci_alpha
    lower = np.percentile(valid, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(valid, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper, n_valid


# ---------------------------------------------------------------------------
# Cauchy fallback CI
# ---------------------------------------------------------------------------

def cauchy_ci(
    point_estimate: np.ndarray,
    scale: float,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cauchy-distribution conservative CI centred on the point estimate.

    The Cauchy has no finite variance — it reflects genuine ignorance rather
    than a specific assumed error distribution. Each class is treated
    independently (does not enforce that CI bounds sum to 1; intentional
    for the sparse case where honest uncertainty can exceed the simplex).

    Bounds are clipped to [0, 1] to remain valid probabilities.
    """
    half_alpha = alpha / 2.0
    # ppf of Cauchy at 1 - half_alpha gives the half-width
    half_width = cauchy.ppf(1.0 - half_alpha, loc=0.0, scale=scale)
    lower = np.clip(point_estimate - half_width, 0.0, 1.0)
    upper = np.clip(point_estimate + half_width, 0.0, 1.0)
    return lower, upper


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def compute_prediction_ci(
    x: np.ndarray,
    point_estimate: np.ndarray,
    X_train: Optional[np.ndarray],
    y_train: Optional[np.ndarray],
    train_weights: Optional[np.ndarray],
    effective_n: float,
    config: ModelConfig,
    *,
    bootstrap_W: Optional[np.ndarray] = None,
    bootstrap_progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Route to bootstrap or Cauchy based on effective sample size.

    If ``bootstrap_W`` is provided (from train-time bootstrap) with at least 10
    draws, intervals are computed via ``bootstrap_ci_from_stored_W`` — no refit.

    If ``bootstrap_W`` is missing but training arrays are present, falls back
    to legacy ``bootstrap_ci`` (slow) for old pickles.

    Returns:
        lower   : (N_CLASSES,) lower CI bounds
        upper   : (N_CLASSES,) upper CI bounds
        method  : "bootstrap" or "cauchy"
    """
    use_bootstrap = effective_n >= config.cauchy_fallback_threshold

    if use_bootstrap:
        if (
            bootstrap_W is not None
            and bootstrap_W.size > 0
            and bootstrap_W.shape[0] >= 10
        ):
            lower, upper = bootstrap_ci_from_stored_W(
                x, bootstrap_W, config.ci_alpha,
            )
            return lower, upper, "bootstrap"

        if X_train is not None and y_train is not None:
            lower, upper, _ = bootstrap_ci(
                x=x,
                X_train=X_train,
                y_train=y_train,
                train_weights=train_weights,
                config=config,
                progress_every=bootstrap_progress_every,
            )
            return lower, upper, "bootstrap"

    lower, upper = cauchy_ci(point_estimate, config.cauchy_scale, config.ci_alpha)
    return lower, upper, "cauchy"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_prediction_table(
    point_estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ci_method: str,
    effective_n: float,
    pct_post_era: float,
    era_cutoff_year: int,
    alpha: float = 0.05,
) -> str:
    """
    Format the 6-class prediction as the output table shown in architecture Section 8.5.
    """
    ci_pct = int((1.0 - alpha) * 100)
    method_str = (
        f"Bootstrap n={int(effective_n)}"
        if ci_method == "bootstrap"
        else "Cauchy — sparse reference class"
    )
    era_note = f"{pct_post_era:.0%} post-{era_cutoff_year}"

    col_w = [25, 11, 16]
    header = (
        f"{'Outcome':<{col_w[0]}} {'Point Est.':>{col_w[1]}}  "
        f"{f'{ci_pct}% CI':>{col_w[2]}}  Method"
    )
    sep = "-" * (col_w[0] + col_w[1] + col_w[2] + 12)

    lines = [header, sep]
    for k, label in enumerate(CLASS_LABELS):
        lines.append(
            f"{label:<{col_w[0]}} {point_estimate[k]:>{col_w[1]}.2f}  "
            f"[{lower[k]:.2f}, {upper[k]:.2f}]  {method_str}"
        )

    lines.append("")
    if ci_method == "cauchy":
        lines.append(
            f"Reference: {int(effective_n)} similar fights | Mixed eras | "
            f"⚠ Interpret with caution"
        )
    else:
        lines.append(
            f"Reference: {int(effective_n)} similar fights | {era_note} | Era: Modern"
        )

    return "\n".join(lines)
