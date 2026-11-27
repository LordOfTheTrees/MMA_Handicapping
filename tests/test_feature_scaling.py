"""Scale normalisation must be a reconditioning, not a model change.

``fit(standardize=True)`` optimises over ``X / column_std`` and then divides the
coefficients back, so ``W`` is always in raw-feature space and nothing downstream —
``predict_proba``, the additive decomposition, the JSON export, the deployed service —
has to know which path was taken.

The invariant these tests defend: with the regulariser switched off, both paths must
reach the *same model*. If they do not, the un-scaling algebra is wrong and every
prediction shifts. With the regulariser on they legitimately differ, because making the
L2 penalty scale-free is the entire point.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import ModelConfig
from src.model.regression import (
    N_CLASSES,
    MultinomialLogisticModel,
    column_scales,
)


def _wide_scale_problem(n: int = 400, seed: int = 3):
    """
    Columns spanning four orders of magnitude, like the real 12-feature matrix:
    ELO differentials in the hundreds, age gaps in the thousands, style axes near 0.3,
    and a 0/1 indicator.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(0.0, 180.0, n),      # elo_differential
            rng.normal(0.0, 0.25, n),       # striker_score_diff
            rng.normal(0.0, 0.30, n),       # grappler_score_diff
            rng.uniform(0.0, 1.0, n),       # finish_matchup
            rng.normal(0.0, 3000.0, n),     # age_diff_days
            rng.integers(0, 2, n).astype(float),  # stance_mismatch
        ]
    )
    y = np.tile(np.arange(N_CLASSES), n // N_CLASSES + 1)[:n].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# column_scales
# ---------------------------------------------------------------------------

def test_column_scales_are_the_column_stds() -> None:
    X, _ = _wide_scale_problem()
    np.testing.assert_allclose(column_scales(X), np.std(X, axis=0))


def test_constant_column_gets_unit_scale() -> None:
    """Dividing by ~0 would explode the column and contract it back — guard it."""
    X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    scales = column_scales(X)
    assert scales[0] == 1.0
    assert scales[1] == pytest.approx(np.std(X[:, 1]))


def test_non_finite_scale_falls_back_to_one() -> None:
    X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    assert column_scales(X)[0] == 1.0


# ---------------------------------------------------------------------------
# The invariant: same model, without the regulariser
# ---------------------------------------------------------------------------

def test_unscaling_reproduces_the_scaled_models_logits_exactly() -> None:
    """
    The algebra, isolated from the optimiser.

    Fitting on ``X / scale`` gives ``W_fit``; the model stores ``W_fit / scale``. For that
    to be a faithful change of basis, ``(X / scale) @ W_fit.T`` must equal
    ``X @ (W_fit / scale).T`` for every row. This is an identity, so it holds to floating
    point — no convergence involved.
    """
    X, _ = _wide_scale_problem()
    scale = column_scales(X)
    rng = np.random.default_rng(0)
    W_fit = rng.normal(0.0, 0.5, size=(N_CLASSES, X.shape[1]))

    logits_scaled_space = (X / scale) @ W_fit.T
    logits_raw_space = X @ (W_fit / scale).T
    np.testing.assert_allclose(logits_raw_space, logits_scaled_space, rtol=1e-12, atol=1e-12)


def test_unregularised_fits_agree_on_predictions() -> None:
    """
    With l2_lambda = 0 the objective is scale-invariant, so both paths optimise the same
    function and must reach the same predictions.

    The tolerance is loose because the *raw* path is the imprecise one: on this input it
    needs thousands of L-BFGS iterations against a Hessian with condition number ~1e8 and
    still stops at a marginally higher loss than the scaled path reaches in ~20. The
    residual disagreement is the raw fit's incomplete convergence, which is the problem
    scaling exists to fix — see ``test_scaled_fit_reaches_a_lower_or_equal_loss``.
    """
    X, y = _wide_scale_problem()
    kw = dict(max_iter=20_000, ftol=1e-15, gtol=1e-10)

    raw = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(X, y, **kw)
    std = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(
        X, y, standardize=True, **kw
    )

    assert raw.W is not None and std.W is not None
    for x in X[:40]:
        np.testing.assert_allclose(
            std.predict_proba(x), raw.predict_proba(x), rtol=0.0, atol=1e-4
        )


def test_scaling_collapses_the_condition_number() -> None:
    """The mechanism behind the iteration reduction, asserted directly."""
    X, _ = _wide_scale_problem()
    scale = column_scales(X)
    cond_raw = np.linalg.cond(X.T @ X)
    cond_scaled = np.linalg.cond((X / scale).T @ (X / scale))
    assert cond_raw > 1e6
    assert cond_scaled < 1e3


def test_scaled_fit_reaches_a_lower_or_equal_loss() -> None:
    """Better conditioning should not make the optimiser do worse on the same objective."""
    X, y = _wide_scale_problem()
    kw = dict(max_iter=20_000, ftol=1e-15, gtol=1e-10)
    raw = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(X, y, **kw)
    std = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(
        X, y, standardize=True, **kw
    )
    assert std.final_loss is not None and raw.final_loss is not None
    assert std.final_loss <= raw.final_loss + 1e-6


def test_scaling_reduces_iterations_on_badly_conditioned_input() -> None:
    """The practical payoff: fewer L-BFGS steps to the same place."""
    X, y = _wide_scale_problem()
    kw = dict(max_iter=20_000, ftol=1e-15, gtol=1e-10)
    raw = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(X, y, **kw)
    std = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=0.0).fit(
        X, y, standardize=True, **kw
    )
    assert std.n_iter is not None and raw.n_iter is not None
    assert std.n_iter < raw.n_iter


# ---------------------------------------------------------------------------
# W stays in raw space
# ---------------------------------------------------------------------------

def test_scaled_fit_leaves_W_in_raw_feature_space() -> None:
    """Callers pass raw x; nothing downstream applies a transform."""
    X, y = _wide_scale_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y, standardize=True)
    assert m.W is not None
    manual = np.exp(X[0] @ m.W.T)
    manual = manual / manual.sum()
    np.testing.assert_allclose(m.predict_proba(X[0]), manual, rtol=1e-10)


def test_feature_scale_records_the_path_taken() -> None:
    X, y = _wide_scale_problem()
    plain = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y)
    assert plain.feature_scale is None

    scaled = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y, standardize=True)
    assert scaled.feature_scale is not None
    np.testing.assert_allclose(scaled.feature_scale, column_scales(X))


def test_regularised_fits_differ_because_the_penalty_is_now_scale_free() -> None:
    """
    Not a bug — the reason to do this. With a shared l2_lambda on raw columns, the
    small-scale style features are penalised far harder than ELO; scaling equalises it,
    which necessarily moves the solution.
    """
    X, y = _wide_scale_problem()
    kw = dict(max_iter=20_000, ftol=1e-15, gtol=1e-10)
    raw = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=1e-2).fit(X, y, **kw)
    std = MultinomialLogisticModel(n_features=X.shape[1], l2_lambda=1e-2).fit(
        X, y, standardize=True, **kw
    )
    assert raw.W is not None and std.W is not None
    assert not np.allclose(raw.W, std.W, rtol=1e-3)


# ---------------------------------------------------------------------------
# Config plumbing and pickles
# ---------------------------------------------------------------------------

def test_standardize_is_off_by_default() -> None:
    """Enabling it invalidates the tuned l2_lambda, so it must be an explicit choice."""
    assert ModelConfig().standardize_features is False


def test_old_pickle_without_feature_scale_loads() -> None:
    X, y = _wide_scale_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y)
    state = dict(m.__dict__)
    state.pop("feature_scale")
    restored = MultinomialLogisticModel.__new__(MultinomialLogisticModel)
    restored.__setstate__(state)
    assert restored.feature_scale is None
    assert restored.is_fitted is True
