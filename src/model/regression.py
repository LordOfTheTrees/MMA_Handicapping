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
from typing import Dict, List, Optional, Tuple

from ..data.schema import FightRecord, ResultMethod


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

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 2000,
    ) -> "MultinomialLogisticModel":
        """
        Fit coefficients using L-BFGS-B.

        X : (n_samples, n_features)  float
        y : (n_samples,)             int in [0, N_CLASSES)
        """
        init_params = np.zeros(N_CLASSES * self.n_features)

        result = minimize(
            fun=_robust_nll_and_grad,
            x0=init_params,
            args=(X, y, self.delta, self.l2_lambda),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7},
        )

        self.W = result.x.reshape(N_CLASSES, self.n_features)
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
