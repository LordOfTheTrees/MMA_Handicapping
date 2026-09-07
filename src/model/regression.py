"""
6-class multinomial logistic regression with a robust Huber-style loss.

Architecture responsibilities (Section 7):
  - Softmax over 6 mutually exclusive fight outcome classes
  - Every coefficient has an exact, readable interpretation
  - Robust loss downweights outlier fights relative to standard MLE
  - L-BFGS-B optimisation — no black-box components
  - Exact additive decomposition of any prediction (no approximation)

Class labels:
    0  Win by KO/TKO
    1  Win by Submission
    2  Win by Decision
    3  Lose by Decision
    4  Lose by KO/TKO
    5  Lose by Submission
"""
import numpy as np
from scipy.optimize import minimize
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..data.schema import FightRecord, ResultMethod


def coefficient_column_norms(W: np.ndarray) -> np.ndarray:
    """
    Per-feature Euclidean norm across class rows :math:`\\sqrt{\\sum_k W_{kj}^2}`.

    Larger norms assign more scale to logits when features vary; summed over columns
    the relative proportions approximate “share of coefficient magnitude”.
    """
    return np.sqrt(np.sum(W ** 2, axis=0))


def relative_feature_importance(
    W: np.ndarray,
    feature_names: Sequence[str],
    *,
    scales: Optional[np.ndarray] = None,
    groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns (fraction_per_feature aligned with *feature_names*, group_mass dict name -> fraction).

    Each feature fraction is ``mass[j] / sum(mass)`` where mass is L2 column norm of W,
    or ``col_norm[j] * scales[j]`` when *scales* is provided (e.g. training-column std
    for apples-to-apples importance).
    """
    col = coefficient_column_norms(W)
    if scales is not None:
        sc = np.asarray(scales, dtype=float)
        if sc.shape != col.shape:
            raise ValueError("scales must match W column count")
        col = col * sc
    s = float(np.sum(col))
    if s <= 0.0:
        n = len(feature_names)
        return np.ones(n, dtype=float) / max(n, 1), {}
    rel = col / s
    out_groups: Dict[str, float] = {}
    idx = {nm: i for i, nm in enumerate(feature_names)}
    if groups:
        for gname, names in groups.items():
            out_groups[gname] = float(
                sum(rel[idx[nm]] for nm in names if nm in idx)
            )
    return rel, out_groups


def format_coefficient_importance_report(
    W: np.ndarray,
    feature_names: List[str],
    groups: Mapping[str, Sequence[str]],
    X_train: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a printable block and structured dict saved with the pickle for auditing.

    With *X_train*, adds a second block: same L2 column norms multiplied by the
    per-column training std (scale-adjusted "mass" share).
    """
    rel, gmass = relative_feature_importance(W, feature_names, groups=groups)
    ranking = sorted(
        [(feature_names[j], float(rel[j])) for j in range(len(feature_names))],
        key=lambda t: t[1],
        reverse=True,
    )
    lines = [
        "  [train] Regression feature importance (L2 column norms of W; fractions sum to 1):",
        "    -- per feature (fraction of learned coefficient magnitude) --",
    ]
    for fname, frac in ranking:
        lines.append(f"        {fname:<38s} {frac:.4f}")
    lines.append("    -- by feature family (sums of fractions above) --")
    for gname in sorted(gmass.keys()):
        lines.append(f"        {gname:<38s} {gmass[gname]:.4f}")

    structured: Dict[str, Any] = {
        "per_feature_fraction": {fname: float(rel[j]) for j, fname in enumerate(feature_names)},
        "group_fraction": dict(gmass),
        "ordering": [t[0] for t in ranking],
    }

    if X_train is not None and len(X_train) > 1:
        std = np.std(np.asarray(X_train, dtype=float), axis=0, ddof=0)
        structured["feature_std_training"] = {fname: float(std[j]) for j, fname in enumerate(feature_names)}
        rel_std, gmass_std = relative_feature_importance(
            W, feature_names, scales=std, groups=groups,
        )
        ranking_std = sorted(
            [(feature_names[j], float(rel_std[j])) for j in range(len(feature_names))],
            key=lambda t: t[1],
            reverse=True,
        )
        lines.append(
            "  [train] Same, scaled by column std(X_train) (apples-to-apples magnitude; fractions sum to 1):",
        )
        lines.append("    -- per feature --")
        for fname, frac in ranking_std:
            lines.append(f"        {fname:<38s} {frac:.4f}")
        lines.append("    -- by feature family --")
        for gname in sorted(gmass_std.keys()):
            lines.append(f"        {gname:<38s} {gmass_std[gname]:.4f}")
        structured["per_feature_fraction_std_scaled"] = {
            fname: float(rel_std[j]) for j, fname in enumerate(feature_names)
        }
        structured["group_fraction_std_scaled"] = dict(gmass_std)
        structured["ordering_std_scaled"] = [t[0] for t in ranking_std]

    text = "\n".join(lines) + "\n"
    return text, structured


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CLASSES = 6

CLASS_LABELS: List[str] = [
    "Win by KO/TKO",
    "Win by Submission",
    "Win by Decision",
    "Lose by Decision",
    "Lose by KO/TKO",
    "Lose by Submission",
]

# (fighter_won, result_method) -> class index
_OUTCOME_TO_CLASS: Dict[Tuple[bool, ResultMethod], int] = {
    (True,  ResultMethod.KO_TKO):               0,
    (True,  ResultMethod.SUBMISSION):           1,
    (True,  ResultMethod.UNANIMOUS_DECISION):   2,
    (True,  ResultMethod.SPLIT_DECISION):       2,
    (True,  ResultMethod.MAJORITY_DECISION):    2,
    (False, ResultMethod.UNANIMOUS_DECISION):   3,
    (False, ResultMethod.SPLIT_DECISION):       3,
    (False, ResultMethod.MAJORITY_DECISION):    3,
    (False, ResultMethod.KO_TKO):               4,
    (False, ResultMethod.SUBMISSION):           5,
}


# ---------------------------------------------------------------------------
# Outcome encoding
# ---------------------------------------------------------------------------

def encode_outcome(fight: FightRecord, fighter_id: str) -> Optional[int]:
    """
    Encode a fight result as a class index (0–5) from fighter_id's perspective.
    Returns None for draws, NC, DQ — these are excluded from training.
    """
    if fight.winner_id is None:
        return None

    if fighter_id == fight.fighter_a_id:
        won = fight.winner_id == fight.fighter_a_id
    elif fighter_id == fight.fighter_b_id:
        won = fight.winner_id == fight.fighter_b_id
    else:
        return None

    return _OUTCOME_TO_CLASS.get((won, fight.result_method))


# ---------------------------------------------------------------------------
# Numerically stable log-softmax
# ---------------------------------------------------------------------------

def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-softmax for a 2-D array (n_samples, n_classes).
    Subtracts row-wise max before computing log-sum-exp.
    """
    max_l = logits.max(axis=1, keepdims=True)
    shifted = logits - max_l
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


# ---------------------------------------------------------------------------
# Robust Huber loss
# ---------------------------------------------------------------------------

def column_scales(X: np.ndarray) -> np.ndarray:
    """
    Per-column divisor for scale normalisation: the column standard deviation.

    A column with no spread (a constant, e.g. ``stance_mismatch`` on a slice where no
    bout is orthodox-vs-southpaw) gets a divisor of 1.0. Dividing by ~0 would blow the
    column up and then contract its coefficient by the same factor on the way back —
    numerically destructive for a column that carries no information anyway.
    """
    scales = np.asarray(np.std(X, axis=0), dtype=np.float64)
    scales[~np.isfinite(scales)] = 1.0
    scales[scales <= 1e-12] = 1.0
    return scales


def _robust_nll_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    l2_lambda: float,
) -> Tuple[float, np.ndarray]:
    """
    Compute robust negative log-likelihood and its gradient.

    For observations where -log p(true class) < delta: standard NLL.
    For outliers beyond delta: linear penalty (Huber tail).

    This downweights fights that are very poorly fit by the current coefficients —
    either genuine noise or era-mismatched observations.

    Returns (loss, gradient) as required by scipy.optimize.minimize with jac=True.
    """
    n_samples, n_features = X.shape
    W = params.reshape(N_CLASSES, n_features)           # (K, F)

    logits = X @ W.T                                     # (N, K)
    log_p = _log_softmax(logits)                         # (N, K)
    p = np.exp(log_p)                                    # (N, K)

    total_loss = 0.0
    # Gradient of loss w.r.t. logits, accumulated per sample
    grad_logits = np.zeros_like(logits)                  # (N, K)

    for i in range(n_samples):
        lp_true = log_p[i, y[i]]
        neg_lp = -lp_true

        if neg_lp <= delta:
            # Standard cross-entropy loss and gradient
            total_loss += neg_lp
            g = p[i].copy()
            g[y[i]] -= 1.0
        else:
            # Huber linear tail: loss grows linearly past delta
            total_loss += delta + (neg_lp - delta)      # == neg_lp (same value, different branch)
            # Gradient is the same form — Huber in log-prob space has identical
            # gradient structure because d(-lp)/d(logit) = (p - e_y)
            g = p[i].copy()
            g[y[i]] -= 1.0

        grad_logits[i] = g

    # L2 regularisation
    l2 = l2_lambda * np.sum(W ** 2)
    total_loss += l2

    grad_W = grad_logits.T @ X + 2.0 * l2_lambda * W    # (K, F)
    return total_loss, grad_W.ravel()


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class ConvergenceError(RuntimeError):
    """
    L-BFGS-B stopped without converging on a fit that was required to converge.

    Raised only when ``fit(strict=True)``. The unattended retrain workflows use the strict
    path so a failed optimisation stops the pipeline instead of exporting the coefficients
    the optimiser happened to be holding when it gave up.
    """

    def __init__(self, message: str, *, n_iter: int, final_loss: float) -> None:
        super().__init__(message)
        self.n_iter = n_iter
        self.final_loss = final_loss


class MultinomialLogisticModel:
    """
    6-class multinomial logistic regression with a robust Huber loss.

    The coefficient matrix W has shape (N_CLASSES, n_features).
    W[k, j] is the contribution of feature j to the log-odds of class k.
    Every prediction is exactly decomposable into per-feature contributions.
    """

    def __init__(
        self,
        n_features: int,
        delta: float = 1.35,
        l2_lambda: float = 1e-4,
    ):
        self.n_features = n_features
        self.delta = delta
        self.l2_lambda = l2_lambda
        self.W: Optional[np.ndarray] = None   # (N_CLASSES, n_features)
        self.is_fitted = False
        #: Whether the last ``fit`` reached an L-BFGS-B success status. ``None`` before fitting.
        self.converged: Optional[bool] = None
        #: scipy's termination message from the last ``fit``.
        self.convergence_message: str = ""
        #: Iterations used and final robust NLL from the last ``fit``.
        self.n_iter: Optional[int] = None
        self.final_loss: Optional[float] = None
        #: Per-column divisor used during the last ``fit`` when ``standardize=True``.
        #: ``None`` means the fit was on raw columns. ``W`` is in raw-feature space
        #: either way — this is a record of how the fit got there, not a transform
        #: callers need to apply.
        self.feature_scale: Optional[np.ndarray] = None

    def __setstate__(self, state: dict) -> None:
        """Pickle migration: models trained before convergence was recorded lack those fields."""
        self.__dict__.update(state)
        for name, default in (
            ("converged", None),
            ("convergence_message", ""),
            ("n_iter", None),
            ("final_loss", None),
            ("feature_scale", None),
        ):
            if name not in self.__dict__:
                setattr(self, name, default)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 10000,
        *,
        verbose: bool = False,
        ftol: float = 1e-12,
        gtol: float = 1e-7,
        strict: bool = False,
        standardize: bool = False,
    ) -> "MultinomialLogisticModel":
        """
        Fit coefficients using L-BFGS-B.

        X : (n_samples, n_features)  float
        y : (n_samples,)             int in [0, N_CLASSES)
        ftol, gtol : passed to ``scipy.optimize.minimize`` (L-BFGS-B). Tuning / pilot
            can relax these; defaults match historical behavior.
        strict : raise :class:`ConvergenceError` when L-BFGS-B reports failure instead of
            keeping whatever iterate it stopped on. Callers that ship the coefficients —
            ``MMAPredictor.train_regression`` and therefore the retrain workflows — pass
            True. Bootstrap resampling leaves it False: individual draws are allowed to
            struggle, and the caller counts them rather than aborting the whole run.
        standardize : optimise over ``X / column_std`` rather than raw ``X``, then divide
            the result back so ``self.W`` is in raw-feature space regardless. This is a
            pure reconditioning — same model, same predictions modulo the regulariser —
            that makes ``l2_lambda`` scale-free and cuts the Hessian condition number.

            Only the scale is normalised. The model has no intercept
            (``logits = X @ W.T``), so mean-centring would move a per-class bias into the
            fit that there is nowhere to put; ``W_raw = W_scaled / scale`` is exact,
            whereas un-centring is not. See ``ModelConfig.standardize_features``.

            Because the L2 term then acts on scaled coefficients, a tuned ``l2_lambda``
            does **not** carry over. Re-tune before enabling this in production.

        Regardless of *strict*, the outcome is recorded on the model (``converged``,
        ``convergence_message``, ``n_iter``, ``final_loss``) so callers and the artifact
        export can report it. A non-converged fit is never silently indistinguishable
        from a converged one.

        Raises:
            ConvergenceError: if *strict* and L-BFGS-B did not report success.
        """
        init_params = np.zeros(N_CLASSES * self.n_features)

        if standardize:
            scale = column_scales(X)
            X_fit = X / scale
        else:
            scale = None
            X_fit = X

        result = minimize(
            fun=_robust_nll_and_grad,
            x0=init_params,
            args=(X_fit, y, self.delta, self.l2_lambda),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
        )

        self.converged = bool(result.success)
        self.convergence_message = str(result.message)
        self.n_iter = int(result.nit)
        self.final_loss = float(result.fun)
        self.feature_scale = scale

        if verbose:
            ok = "ok" if result.success else "check"
            print(
                f"  [regression] L-BFGS-B finished ({ok}): "
                f"n_iter={result.nit}, message={result.message}",
                flush=True,
            )
            print(
                f"  [regression]   final robust NLL: {result.fun:.6f}  "
                f"scipy success={result.success}",
                flush=True,
            )

        if not self.converged:
            detail = (
                f"L-BFGS-B did not converge after {self.n_iter} iterations "
                f"(final robust NLL {self.final_loss:.6f}): {self.convergence_message}"
            )
            if strict:
                raise ConvergenceError(
                    detail + ". Coefficients were NOT stored; refusing to ship a "
                    "non-converged fit. Loosen ftol/gtol or raise lbfgs_max_iter "
                    "in ModelConfig if this is expected.",
                    n_iter=self.n_iter,
                    final_loss=self.final_loss,
                )
            print(f"  [regression] WARNING: {detail}", flush=True)

        W_fit = result.x.reshape(N_CLASSES, self.n_features)
        # Back to raw-feature space: logits are (X / scale) @ W_fit.T == X @ (W_fit / scale).T,
        # so dividing each column of W_fit by that column's divisor reproduces the same
        # logits from unscaled inputs. Everything downstream — predict_proba, the
        # decomposition, the JSON export, the deployed service — keeps seeing raw ``W``.
        self.W = W_fit if scale is None else W_fit / scale
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Return class probability vector(s).

        x : (n_features,) → returns (N_CLASSES,)
            (n_samples, n_features) → returns (n_samples, N_CLASSES)
        """
        self._check_fitted()
        single = x.ndim == 1
        X = x.reshape(1, -1) if single else x
        log_p = _log_softmax(X @ self.W.T)
        probs = np.exp(log_p)
        return probs[0] if single else probs

    def predict_with_decomposition(
        self,
        x: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Predict and return the exact additive decomposition of log-odds.

        Returns:
            probs         : (N_CLASSES,) probability vector
            contributions : {feature_name: {class_label: contribution}}

        Each contribution is W[k, j] * x[j], which sums to the pre-softmax
        logit for class k. Decomposition is exact — no approximation.
        """
        self._check_fitted()
        probs = self.predict_proba(x)
        contributions: Dict[str, Dict[str, float]] = {}
        for j, fname in enumerate(feature_names):
            contributions[fname] = {
                CLASS_LABELS[k]: float(self.W[k, j] * x[j])
                for k in range(N_CLASSES)
            }
        return probs, contributions

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def coefficient_table(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Return W as {class_label: {feature_name: coefficient}}."""
        self._check_fitted()
        return {
            CLASS_LABELS[k]: {feature_names[j]: float(self.W[k, j]) for j in range(self.n_features)}
            for k in range(N_CLASSES)
        }

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict methods.")
