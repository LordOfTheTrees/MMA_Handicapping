"""Non-convergence must be visible, not silently shipped.

``fit`` previously set ``is_fitted = True`` unconditionally, so an L-BFGS-B run that hit
its iteration cap or failed a line search produced a model indistinguishable from a good
one. The monthly retrain workflow runs unattended for up to twelve hours and exports
whatever comes out, so that failure mode reaches production with nobody notified.

These tests pin the three places that now stop it: the fit itself, the pipeline stage that
ships coefficients, and the artifact export.
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

from src.model.regression import (
    N_CLASSES,
    ConvergenceError,
    MultinomialLogisticModel,
)


def _separable_problem(n: int = 240, n_features: int = 4, seed: int = 7):
    """A small well-posed problem with every class represented."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, n_features))
    y = np.tile(np.arange(N_CLASSES), n // N_CLASSES + 1)[:n]
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# Convergence is recorded on every fit
# ---------------------------------------------------------------------------

def test_fresh_model_reports_unknown_convergence() -> None:
    m = MultinomialLogisticModel(n_features=4)
    assert m.converged is None
    assert m.n_iter is None
    assert m.final_loss is None
    assert m.is_fitted is False


def test_successful_fit_records_convergence_metadata() -> None:
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y)
    assert m.converged is True
    assert m.is_fitted is True
    assert isinstance(m.n_iter, int) and m.n_iter > 0
    assert isinstance(m.final_loss, float) and np.isfinite(m.final_loss)
    assert m.convergence_message


def test_strict_fit_succeeds_on_a_well_posed_problem() -> None:
    """strict=True must not make ordinary training brittle."""
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y, strict=True)
    assert m.converged is True


# ---------------------------------------------------------------------------
# Non-convergence
# ---------------------------------------------------------------------------

def test_non_converged_fit_is_flagged_but_kept_when_not_strict() -> None:
    """Bootstrap draws tolerate a struggling fit — but it is still marked."""
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y, max_iter=1)
    assert m.converged is False
    assert m.is_fitted is True
    assert m.W is not None


def test_strict_fit_raises_on_non_convergence() -> None:
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1])
    with pytest.raises(ConvergenceError) as exc:
        m.fit(X, y, max_iter=1, strict=True)
    assert "did not converge" in str(exc.value)
    assert exc.value.n_iter is not None
    assert np.isfinite(exc.value.final_loss)


def test_strict_failure_leaves_model_unfitted() -> None:
    """The caller must not be able to use a model whose fit was rejected."""
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1])
    with pytest.raises(ConvergenceError):
        m.fit(X, y, max_iter=1, strict=True)
    assert m.is_fitted is False
    assert m.W is None
    assert m.converged is False


def test_non_converged_fit_warns_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    X, y = _separable_problem()
    MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y, max_iter=1)
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Pickle migration
# ---------------------------------------------------------------------------

def test_old_pickle_without_convergence_fields_loads() -> None:
    """Models trained before this change must still unpickle, reporting unknown."""
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y)
    state = dict(m.__dict__)
    for key in ("converged", "convergence_message", "n_iter", "final_loss"):
        state.pop(key)

    restored = MultinomialLogisticModel.__new__(MultinomialLogisticModel)
    restored.__setstate__(state)
    assert restored.converged is None
    assert restored.convergence_message == ""
    assert restored.is_fitted is True
    assert restored.W is not None


def test_current_pickle_roundtrip_preserves_convergence() -> None:
    X, y = _separable_problem()
    m = MultinomialLogisticModel(n_features=X.shape[1]).fit(X, y)
    restored = pickle.loads(pickle.dumps(m))
    assert restored.converged is True
    assert restored.n_iter == m.n_iter
    assert restored.final_loss == pytest.approx(m.final_loss)


# ---------------------------------------------------------------------------
# Export refuses known-bad coefficients
# ---------------------------------------------------------------------------

class _StubPredictor:
    """Minimal stand-in for MMAPredictor's export surface."""

    def __init__(self, reg: MultinomialLogisticModel) -> None:
        self.regression = reg
        self.config = None
        self._bootstrap_W = None


def test_export_refuses_non_converged_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import export_artifacts

    X, y = _separable_problem(n_features=12)
    reg = MultinomialLogisticModel(n_features=12).fit(X, y, max_iter=1)
    assert reg.converged is False

    with pytest.raises(RuntimeError, match="non-converged"):
        export_artifacts._export_model_weights(_StubPredictor(reg), {})


def test_export_allows_converged_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import export_artifacts

    X, y = _separable_problem(n_features=12)
    reg = MultinomialLogisticModel(n_features=12).fit(X, y)
    assert reg.converged is True

    monkeypatch.setattr(export_artifacts, "_config_snapshot", lambda p: {})
    doc: dict[str, Any] = export_artifacts._export_model_weights(_StubPredictor(reg), {})
    assert doc["regression"]["converged"] is True
    assert doc["regression"]["n_iter"] == reg.n_iter
    assert doc["regression"]["final_loss"] == pytest.approx(reg.final_loss)


def test_export_allows_legacy_pickle_with_unknown_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``converged is None`` means "trained before we tracked it", not "known bad"."""
    from scripts import export_artifacts

    X, y = _separable_problem(n_features=12)
    reg = MultinomialLogisticModel(n_features=12).fit(X, y)
    reg.converged = None

    monkeypatch.setattr(export_artifacts, "_config_snapshot", lambda p: {})
    doc = export_artifacts._export_model_weights(_StubPredictor(reg), {})
    assert doc["regression"]["converged"] is None
