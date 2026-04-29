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
    Used when reference class data is sparse, bootstrap is unavailable, or
    ``force_cauchy_wc_debut`` applies (see ``compute_prediction_ci``).

  Cauchy ELO Monte Carlo (ADR-19):
    Independent Cauchy shocks in ELO points per corner perturb ``elo_differential``
    before ``softmax(Wx)``. When stored bootstrap ``W`` exists, each MC draw
    cycles coefficient rows so coefficient and ELO uncertainty both appear.

The choice is automatic based on effective sample size after weighting and
optional weight-class debut routing in ``MMAPredictor.predict``.
"""
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import cauchy

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

_TR_POOL_X: Optional[np.ndarray] = None
_TR_POOL_Y: Optional[np.ndarray] = None
_TR_POOL_CFG: Optional[ModelConfig] = None


def _bootstrap_env_threads_for_workers() -> None:
    """Limit BLAS/OpenMP threading per subprocess to reduce oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _bootstrap_train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    idx: np.ndarray,
    config: ModelConfig,
) -> Optional[np.ndarray]:
    y_b = y_train[idx]
    if len(np.unique(y_b)) < N_CLASSES:
        return None
    X_b = X_train[idx]
    n_features = X_train.shape[1]
    model_b = MultinomialLogisticModel(
        n_features=n_features,
        delta=config.huber_delta,
        l2_lambda=config.l2_lambda,
    )
    m = config
    try:
        model_b.fit(
            X_b, y_b,
            max_iter=getattr(m, "lbfgs_max_iter", 10_000),
            ftol=getattr(m, "lbfgs_ftol", 1e-12),
            gtol=getattr(m, "lbfgs_gtol", 1e-7),
        )
        if model_b.W is not None:
            return np.asarray(model_b.W, dtype=float).copy()
    except Exception:
        pass
    return None


def _init_train_process_pool(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> None:
    global _TR_POOL_X, _TR_POOL_Y, _TR_POOL_CFG
    _bootstrap_env_threads_for_workers()
    _TR_POOL_X = X
    _TR_POOL_Y = y
    _TR_POOL_CFG = cfg


def _train_process_task(args: Tuple[int, np.ndarray]) -> Tuple[int, Optional[np.ndarray]]:
    b, idx = args
    assert _TR_POOL_X is not None and _TR_POOL_Y is not None and _TR_POOL_CFG is not None
    W = _bootstrap_train_one(_TR_POOL_X, _TR_POOL_Y, idx, _TR_POOL_CFG)
    return b, W


_CI_POOL_X: Optional[np.ndarray] = None
_CI_POOL_Y: Optional[np.ndarray] = None
_CI_POOL_X_VEC: Optional[np.ndarray] = None
_CI_POOL_CFG: Optional[ModelConfig] = None


def _bootstrap_ci_predict_one(
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    idx: np.ndarray,
    config: ModelConfig,
) -> Optional[np.ndarray]:
    """Refit on resample, return ``predict_proba(x)`` or None if skipped/failed."""
    y_b = y_train[idx]
    if len(np.unique(y_b)) < N_CLASSES:
        return None
    X_b = X_train[idx]
    n_features = x.shape[0]
    model_b = MultinomialLogisticModel(
        n_features=n_features,
        delta=config.huber_delta,
        l2_lambda=config.l2_lambda,
    )
    m = config
    try:
        model_b.fit(
            X_b, y_b,
            max_iter=getattr(m, "lbfgs_max_iter", 10_000),
            ftol=getattr(m, "lbfgs_ftol", 1e-12),
            gtol=getattr(m, "lbfgs_gtol", 1e-7),
        )
        return model_b.predict_proba(np.asarray(x, dtype=float))
    except Exception:
        return None


def _init_ci_process_pool(X: np.ndarray, y: np.ndarray, x_vec: np.ndarray, cfg: ModelConfig) -> None:
    global _CI_POOL_X, _CI_POOL_Y, _CI_POOL_X_VEC, _CI_POOL_CFG
    _bootstrap_env_threads_for_workers()
    _CI_POOL_X = X
    _CI_POOL_Y = y
    _CI_POOL_X_VEC = np.asarray(x_vec, dtype=float)
    _CI_POOL_CFG = cfg


def _ci_process_task(args: Tuple[int, np.ndarray]) -> Tuple[int, Optional[np.ndarray]]:
    b, idx = args
    assert (
        _CI_POOL_X is not None
        and _CI_POOL_Y is not None
        and _CI_POOL_X_VEC is not None
        and _CI_POOL_CFG is not None
    )
    p = _bootstrap_ci_predict_one(
        _CI_POOL_X_VEC,
        _CI_POOL_X,
        _CI_POOL_Y,
        idx,
        _CI_POOL_CFG,
    )
    return b, p


def _precompute_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    train_weights: Optional[np.ndarray],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    probs_w = None
    if train_weights is not None:
        s = train_weights.sum()
        probs_w = train_weights / s if s > 0 else None
    return [
        rng.choice(n, size=n, replace=True, p=probs_w)
        for _ in range(n_bootstrap)
    ]


def _resolve_bootstrap_max_workers(
    config: ModelConfig,
    n_bootstrap: int,
    override: Optional[int],
) -> int:
    if override is not None:
        return max(1, min(int(override), n_bootstrap))
    raw = getattr(config, "bootstrap_max_workers", None)
    if raw is not None:
        return max(1, min(int(raw), n_bootstrap))
    cpu = os.cpu_count() or 2
    return max(1, min(n_bootstrap, max(1, cpu - 1)))


def fit_bootstrap_coefficients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: Optional[np.ndarray],
    config: ModelConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    progress_every: int = 0,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Weighted bootstrap: refit ``n_bootstrap`` multinomial models on resamples.

    Returns ``(W_stack, n_valid)`` where ``W_stack`` has shape
    ``(n_valid, N_CLASSES, n_features)`` — one coefficient matrix per successful
    resample. Used for fast CIs at prediction time without refitting.

    Index draws use the same sequential RNG steps as legacy single-threaded runs.

    If fewer than 10 successful fits, returns an empty stack ``(0, K, F)`` and
    ``n_valid`` so callers can fall back to Cauchy or legacy bootstrap.

    Parallelism uses ``multiprocessing`` worker processes when
    ``_resolve_bootstrap_max_workers(...) > 1``. Set ``OMP_NUM_THREADS=1``
    etc. via environment variables to reduce BLAS oversubscription (defaults
    applied in worker initializers).
    """
    if rng is None:
        seed = getattr(config, "bootstrap_seed", 42)
        rng = np.random.default_rng(seed)

    n = len(y_train)
    n_features = X_train.shape[1]
    n_bootstrap = config.n_bootstrap
    indices = _precompute_bootstrap_indices(n, n_bootstrap, train_weights, rng)
    workers = _resolve_bootstrap_max_workers(config, n_bootstrap, max_workers)

    if workers <= 1:
        W_list: List[np.ndarray] = []
        for b, idx in enumerate(indices):
            W = _bootstrap_train_one(X_train, y_train, idx, config)
            if W is not None:
                W_list.append(W)
            if progress_every > 0 and (b + 1) % progress_every == 0:
                print(
                    f"  [train bootstrap] resample {b + 1:,} / {n_bootstrap:,}",
                    flush=True,
                )
    else:
        tasks = [(b, indices[b]) for b in range(n_bootstrap)]
        done = 0
        results_map: Dict[int, Optional[np.ndarray]] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_train_process_pool,
            initargs=(
                np.asarray(X_train, dtype=float),
                np.asarray(y_train, dtype=np.int_),
                copy.deepcopy(config),
            ),
        ) as exe:
            futures = [exe.submit(_train_process_task, t) for t in tasks]
            for fut in as_completed(futures):
                b, W = fut.result()
                results_map[b] = W
                done += 1
                if progress_every > 0 and done % progress_every == 0:
                    print(
                        f"  [train bootstrap] completed {done:,} / {n_bootstrap:,} resamples",
                        flush=True,
                    )
        W_list = []
        for b in range(n_bootstrap):
            W = results_map.get(b)
            if W is not None:
                W_list.append(W)

    n_valid = len(W_list)
    if n_valid < 10:
        return np.empty((0, N_CLASSES, n_features), dtype=float), n_valid

    return np.stack(W_list, axis=0), n_valid


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for (n_samples, n_classes)."""
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _softmax_vec(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for a single logit vector (N_CLASSES,)."""
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    s = float(ex.sum())
    return ex / s if s > 0 else np.full_like(logits, 1.0 / len(logits))


def elo_mc_percentile_ci(
    x: np.ndarray,
    W_stack: np.ndarray,
    gamma_a: float,
    gamma_b: float,
    n_mc: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentile CIs from ``n_mc`` draws: cycle rows of ``W_stack``, independent
    ``Cauchy(0, γ_a)`` / ``Cauchy(0, γ_b)`` shocks on ELO (feature 0 only).
    """
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")
    B = W_stack.shape[0]
    probs = np.empty((n_mc, N_CLASSES), dtype=float)
    for k in range(n_mc):
        W = W_stack[k % B]
        ea = float(rng.standard_cauchy() * gamma_a)
        eb = float(rng.standard_cauchy() * gamma_b)
        x2 = x.copy()
        x2[0] += ea - eb
        logits = W @ x2
        probs[k] = _softmax_vec(logits)
    lower = np.percentile(probs, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(probs, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper


def point_W_elo_mc_ci(
    x: np.ndarray,
    W_point: np.ndarray,
    gamma_a: float,
    gamma_b: float,
    n_mc: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """ELO MC only (fixed point estimate coefficients ``W_point``)."""
    W_stack = W_point.reshape(1, *W_point.shape)
    return elo_mc_percentile_ci(
        x, W_stack, gamma_a, gamma_b, n_mc, alpha, rng,
    )


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
    elo_mc_gamma_a: Optional[float] = None,
    elo_mc_gamma_b: Optional[float] = None,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Percentile bootstrap confidence intervals for a single prediction vector x.

    Procedure:
      1. Weighted resample training observations with replacement.
      2. Refit regression coefficients on resample.
      3. Predict x with refitted model.
      4. Report alpha/2 and 1-alpha/2 percentiles across resamples.

    When Cauchy ELO shocks are used (both gammas set), refits run sequentially
    so RNG usage matches the legacy path. Otherwise processes may run in parallel.

    Returns:
        lower      : (N_CLASSES,) lower bounds
        upper      : (N_CLASSES,) upper bounds
        n_valid    : number of successful resamples (failed fits are dropped)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y_train)
    n_features = x.shape[0]
    n_bootstrap = config.n_bootstrap
    all_probs = np.full((n_bootstrap, N_CLASSES), np.nan)

    use_elo_mc = (
        elo_mc_gamma_a is not None
        and elo_mc_gamma_b is not None
    )

    probs_w = None
    if train_weights is not None:
        s = train_weights.sum()
        probs_w = train_weights / s if s > 0 else None

    if use_elo_mc:
        for b in range(n_bootstrap):
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
                model_b.fit(
                    X_b, y_b,
                    max_iter=getattr(config, "lbfgs_max_iter", 10_000),
                    ftol=getattr(config, "lbfgs_ftol", 1e-12),
                    gtol=getattr(config, "lbfgs_gtol", 1e-7),
                )
                x_pred = x.copy()
                x_pred[0] += (
                    float(rng.standard_cauchy() * elo_mc_gamma_a)
                    - float(rng.standard_cauchy() * elo_mc_gamma_b)
                )
                all_probs[b] = model_b.predict_proba(x_pred)
            except Exception:
                pass

            if progress_every > 0 and (b + 1) % progress_every == 0:
                print(
                    f"  [bootstrap CI] resample {b + 1:,} / {n_bootstrap:,}",
                    flush=True,
                )
    else:
        indices = _precompute_bootstrap_indices(n, n_bootstrap, train_weights, rng)
        workers = _resolve_bootstrap_max_workers(config, n_bootstrap, max_workers)
        if workers <= 1:
            for b, idx in enumerate(indices):
                pb = _bootstrap_ci_predict_one(x, X_train, y_train, idx, config)
                if pb is not None:
                    all_probs[b] = pb
                if progress_every > 0 and (b + 1) % progress_every == 0:
                    print(
                        f"  [bootstrap CI] resample {b + 1:,} / {n_bootstrap:,}",
                        flush=True,
                    )
        else:
            tasks = [(b, indices[b]) for b in range(n_bootstrap)]
            done = 0
            results_map: Dict[int, Optional[np.ndarray]] = {}
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_ci_process_pool,
                initargs=(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train, dtype=np.int_),
                    np.asarray(x, dtype=float),
                    copy.deepcopy(config),
                ),
            ) as exe:
                futures = [exe.submit(_ci_process_task, t) for t in tasks]
                for fut in as_completed(futures):
                    b, pb = fut.result()
                    results_map[b] = pb
                    done += 1
                    if progress_every > 0 and done % progress_every == 0:
                        print(
                            f"  [bootstrap CI] completed {done:,} / {n_bootstrap:,} resamples",
                            flush=True,
                        )
            for b in range(n_bootstrap):
                pb = results_map.get(b)
                if pb is not None:
                    all_probs[b] = pb

    valid = all_probs[~np.any(np.isnan(all_probs), axis=1)]
    n_valid = len(valid)

    if n_valid < 10:
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
    force_cauchy_wc_debut: bool = False,
    elo_mc_gamma_a: Optional[float] = None,
    elo_mc_gamma_b: Optional[float] = None,
    W_point: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    bootstrap_max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Route to bootstrap or Cauchy based on effective sample size.

    If ``force_cauchy_wc_debut`` is True (at least one corner has no prior bout
    in this weight class before ``fight_date``), returns Cauchy intervals for
    all six classes and method tag ``cauchy_wc_debut`` — bootstrap is
    skipped regardless of effective sample size.

    If ``bootstrap_W`` is provided (from train-time bootstrap) with at least 10
    draws, intervals are computed via ``bootstrap_ci_from_stored_W`` — no refit.

    If ``bootstrap_W`` is missing but training arrays are present, falls back
    to legacy ``bootstrap_ci`` (slow) for old pickles.

    Returns:
        lower   : (N_CLASSES,) lower CI bounds
        upper   : (N_CLASSES,) upper CI bounds
        method  : ``bootstrap``, ``bootstrap_elo_mc``, ``elo_mc``, ``cauchy``, or ``cauchy_wc_debut``
    """
    if rng is None:
        rng = np.random.default_rng(getattr(config, "bootstrap_seed", 42) + 90210)

    use_elo_mc = (
        config.elo_mc_n_draws > 0
        and elo_mc_gamma_a is not None
        and elo_mc_gamma_b is not None
    )

    if force_cauchy_wc_debut:
        lower, upper = cauchy_ci(point_estimate, config.cauchy_scale, config.ci_alpha)
        return lower, upper, "cauchy_wc_debut"

    use_bootstrap = effective_n >= config.cauchy_fallback_threshold

    if use_bootstrap:
        if (
            bootstrap_W is not None
            and bootstrap_W.size > 0
            and bootstrap_W.shape[0] >= 10
        ):
            if use_elo_mc:
                lower, upper = elo_mc_percentile_ci(
                    x,
                    bootstrap_W,
                    elo_mc_gamma_a,
                    elo_mc_gamma_b,
                    config.elo_mc_n_draws,
                    config.ci_alpha,
                    rng,
                )
                return lower, upper, "bootstrap_elo_mc"
            lower, upper = bootstrap_ci_from_stored_W(
                x, bootstrap_W, config.ci_alpha,
            )
            return lower, upper, "bootstrap"

        if X_train is not None and y_train is not None:
            elo_a = elo_mc_gamma_a if use_elo_mc else None
            elo_b = elo_mc_gamma_b if use_elo_mc else None
            lower, upper, _ = bootstrap_ci(
                x=x,
                X_train=X_train,
                y_train=y_train,
                train_weights=train_weights,
                config=config,
                rng=rng,
                progress_every=bootstrap_progress_every,
                elo_mc_gamma_a=elo_a,
                elo_mc_gamma_b=elo_b,
                max_workers=bootstrap_max_workers,
            )
            return lower, upper, ("bootstrap_elo_mc" if use_elo_mc else "bootstrap")

    if use_elo_mc and W_point is not None:
        lower, upper = point_W_elo_mc_ci(
            x,
            W_point,
            elo_mc_gamma_a,
            elo_mc_gamma_b,
            config.elo_mc_n_draws,
            config.ci_alpha,
            rng,
        )
        return lower, upper, "elo_mc"

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
    master_start_year: int,
    alpha: float = 0.10,
) -> str:
    """
    Format the 6-class prediction as the output table shown in architecture Section 8.5.
    """
    ci_pct = int((1.0 - alpha) * 100)
    if ci_method == "bootstrap":
        method_str = f"Bootstrap n={int(effective_n)}"
    elif ci_method == "bootstrap_elo_mc":
        method_str = f"Bootstrap x ELO MC n={int(effective_n)}"
    elif ci_method == "elo_mc":
        method_str = "ELO MC (point W)"
    elif ci_method == "cauchy_wc_debut":
        method_str = "Cauchy -- weight-class debut"
    else:
        method_str = "Cauchy -- sparse reference class"
    era_note = f"{pct_post_era:.0%} from {master_start_year}"

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
    if ci_method in ("bootstrap", "bootstrap_elo_mc"):
        lines.append(
            f"Reference: {int(effective_n)} similar fights | {era_note} | Era: Modern"
        )
        if ci_method == "bootstrap_elo_mc":
            lines.append("CIs include Cauchy ELO shocks (gamma per corner; ADR-19).")
    elif ci_method == "elo_mc":
        lines.append(
            f"Reference: training ESS ~ {int(effective_n)} | {era_note} | "
            "CIs from Cauchy ELO MC (point coefficients; ADR-19)"
        )
    elif ci_method == "cauchy_wc_debut":
        lines.append(
            f"Reference: training ESS ~ {int(effective_n)} | {era_note} | "
            "(!) Weight-class debut (Cauchy CIs; bootstrap skipped for this bout)"
        )
    else:
        lines.append(
            f"Reference: {int(effective_n)} similar fights | Mixed eras | "
            "(!) Interpret with caution"
        )

    return "\n".join(lines)
