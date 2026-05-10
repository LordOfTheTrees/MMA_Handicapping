"""Strict equality check: pickle point probs vs JSON snapshot path (harness)."""
from __future__ import annotations

import numpy as np

from src.model.regression import CLASS_LABELS


def assert_point_probs_match_pkl(
    p_pkl: np.ndarray,
    p_json: np.ndarray,
    *,
    context: str = "",
) -> None:
    """
    Assert ``(6,)`` float64 vectors are **bitwise equal** after JSON round-trip.

    On failure, emit per-class table (pickle, json, abs diff) and ``max_abs_delta``.
    """
    pa = np.asarray(p_pkl, dtype=np.float64).reshape(6)
    pj = np.asarray(p_json, dtype=np.float64).reshape(6)
    if np.array_equal(pa, pj):
        return

    diff = np.abs(pa - pj)
    lines = [
        "pickle vs JSON snapshot point probabilities differ.",
        f"  context: {context or '(none)'}",
        f"  max_abs_delta: {float(np.max(diff)):.17g}",
        "  per class:",
    ]
    for k, label in enumerate(CLASS_LABELS):
        lines.append(
            f"    [{k}] {label!s}: pkl={pa[k]:.17g} json={pj[k]:.17g} |diff|={diff[k]:.17g}",
        )

    raise AssertionError("\n".join(lines))


__all__ = ["assert_point_probs_match_pkl"]