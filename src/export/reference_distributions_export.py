"""Build ``reference_distributions.json`` in the shape ``mma.ai`` expects (`api/reference_distributions.py`).

Core contract: ``matchup_features`` with 101-point empirical quantile grids (percentiles 0…100),
optional ``division_elo``, optional ``global_days_idle``. Schema version is the same as other
deploy JSONs: ``mma-handicapping-export-v1``.

Extra top-level keys (e.g. ``chart_histograms``) are preserved by mma.ai after validation.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np

from src.matchup.interactions import FEATURE_NAMES
from src.pipeline import MMAPredictor
from src.stats.elo_division_population import (
    collect_elos_by_division,
    collect_uncertainty_by_division,
    division_order_public,
    fight_pairs_for_elo_charts,
)

# Must match ``mma.ai`` ``api/reference_distributions.QUANTILE_PERCENT_LEVELS`` and ``export_constants``.
QUANTILE_PERCENT_LEVELS: tuple[int, ...] = tuple(range(0, 101))
N_QUANTILE_POINTS = len(QUANTILE_PERCENT_LEVELS)

# Sparse percentiles for compact diagnostics inside ``chart_histograms`` (optional UI).
CHART_PERCENTILE_LEVELS: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99)

DEFAULT_TRAINING_BINS = 60
DEFAULT_ELO_DIVISION_BINS = 28
REFERENCE_DISTRIBUTIONS_FILENAME = "reference_distributions.json"


def _finite_array(xs: np.ndarray) -> tuple[np.ndarray, int]:
    a = np.asarray(xs, dtype=float).ravel()
    mask = np.isfinite(a)
    return a[mask], int(a.size - np.count_nonzero(mask))


def quantile_grid_block_from_sample(xs: np.ndarray) -> dict[str, Any]:
    """
    One block: ``percentile_levels`` 0…100 and ``values`` (101 non-decreasing floats),
    matching ``mma.ai`` ``_validate_quantile_block``.
    """
    a, _ = _finite_array(xs)
    levels = list(QUANTILE_PERCENT_LEVELS)
    if a.size == 0:
        vals = [0.0] * N_QUANTILE_POINTS
    else:
        pts = np.linspace(0.0, 100.0, N_QUANTILE_POINTS)
        vals_arr = np.percentile(a, pts, method="linear")
        vals = [float(x) for x in vals_arr]
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
    return {"percentile_levels": levels, "values": vals}


def _percentile_map_sparse(a: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {}
    return {str(p): float(np.percentile(a, p, method="linear")) for p in CHART_PERCENTILE_LEVELS}


def _histogram_block(
    xs: np.ndarray,
    *,
    n_bins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if bin_edges is None and n_bins is None:
        raise ValueError("pass n_bins or bin_edges")
    a, nan_count = _finite_array(xs)
    base: dict[str, Any] = {
        "n": int(a.size),
        "nan_count": nan_count,
    }
    if a.size == 0:
        base["min"] = None
        base["max"] = None
        base["mean"] = None
        base["std"] = None
        base["percentiles"] = {}
        base["histogram"] = None
        return base
    if bin_edges is not None:
        counts, edges = np.histogram(a, bins=bin_edges)
    else:
        assert n_bins is not None
        counts, edges = np.histogram(a, bins=n_bins)
    base.update(
        {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=0)),
            "percentiles": _percentile_map_sparse(a),
            "histogram": {
                "n_bins": int(counts.size),
                "bin_edges": edges.astype(float).tolist(),
                "counts": counts.astype(int).tolist(),
            },
        }
    )
    return base


def _ensure_training_matrix(predictor: MMAPredictor) -> np.ndarray:
    if predictor._X_train is not None:
        return predictor._X_train
    predictor.train_regression(fit_model=False)
    if predictor._X_train is None:
        raise RuntimeError("train_regression(fit_model=False) did not populate _X_train.")
    return predictor._X_train


def build_matchup_feature_quantile_grids(predictor: MMAPredictor) -> dict[str, Any]:
    X = _ensure_training_matrix(predictor)
    names = list(FEATURE_NAMES)
    if X.shape[1] != len(names):
        raise ValueError(f"X columns {X.shape[1]} vs FEATURE_NAMES {len(names)}")
    return {name: quantile_grid_block_from_sample(X[:, j]) for j, name in enumerate(names)}


def build_division_elo_quantile_grids(predictor: MMAPredictor, as_of: date) -> dict[str, Any]:
    pairs = fight_pairs_for_elo_charts(predictor)
    by_elo = collect_elos_by_division(predictor, as_of, pairs)
    out: dict[str, Any] = {}
    for wc in division_order_public():
        vals = by_elo.get(wc, [])
        if not vals:
            continue
        out[wc.value] = quantile_grid_block_from_sample(np.array(vals, dtype=float))
    return out


def build_training_histogram_extras(
    predictor: MMAPredictor,
    *,
    n_bins: int = DEFAULT_TRAINING_BINS,
) -> dict[str, Any]:
    X = _ensure_training_matrix(predictor)
    cfg = predictor.config
    hsd = cfg.holdout_start_date
    feats: dict[str, Any] = {}
    names = list(FEATURE_NAMES)
    for j, name in enumerate(names):
        feats[name] = _histogram_block(X[:, j], n_bins=n_bins)
    return {
        "cohort": "tier1_post_era_pre_holdout",
        "master_start_year": int(cfg.master_start_year),
        "holdout_start_date": hsd.isoformat() if hsd is not None else None,
        "n_rows": int(X.shape[0]),
        "n_bins": int(n_bins),
        "percentile_levels_sparse": list(CHART_PERCENTILE_LEVELS),
        "features": feats,
    }


def build_elo_division_chart_extras(
    predictor: MMAPredictor,
    as_of: date,
    *,
    n_bins_elo: int = DEFAULT_ELO_DIVISION_BINS,
    n_bins_uncertainty: int = 28,
    elo_margin: float = 30.0,
) -> dict[str, Any]:
    if predictor.elo_model is None:
        raise RuntimeError("Predictor missing elo_model.")
    pairs = fight_pairs_for_elo_charts(predictor)
    by_elo = collect_elos_by_division(predictor, as_of, pairs)
    by_var = collect_uncertainty_by_division(predictor, as_of, pairs)
    order = division_order_public()
    all_elo = [x for wc in order for x in by_elo.get(wc, [])]
    baseline = float(predictor.config.elo.initial_elo)
    if all_elo:
        x_min = min(all_elo) - elo_margin
        x_max = max(all_elo) + elo_margin
    else:
        x_min = baseline - 200.0
        x_max = baseline + 200.0
    elo_bin_edges = np.linspace(x_min, x_max, n_bins_elo + 1)

    divisions_out: dict[str, Any] = {}
    for wc in order:
        key = wc.value
        elo_vals = by_elo.get(wc, [])
        if not elo_vals:
            divisions_out[key] = {
                "n_fighters": 0,
                "elo": _histogram_block(np.array([], dtype=float), n_bins=n_bins_elo),
                "uncertainty": _histogram_block(np.array([], dtype=float), n_bins=n_bins_uncertainty),
            }
            continue
        elo_arr = np.array(elo_vals, dtype=float)
        var_arr = np.array(by_var.get(wc, []), dtype=float)
        divisions_out[key] = {
            "n_fighters": int(elo_arr.size),
            "elo": _histogram_block(elo_arr, bin_edges=elo_bin_edges),
            "uncertainty": _histogram_block(var_arr, n_bins=n_bins_uncertainty),
        }

    return {
        "cohort_note": "one sample per (fighter_id, division) seen on fight records; excludes UNKNOWN",
        "initial_elo_baseline": baseline,
        "n_bins_elo": int(n_bins_elo),
        "elo_bin_edges_shared": elo_bin_edges.astype(float).tolist(),
        "n_bins_uncertainty_default": int(n_bins_uncertainty),
        "percentile_levels_sparse": list(CHART_PERCENTILE_LEVELS),
        "divisions": divisions_out,
    }


def build_reference_distributions_document(
    predictor: MMAPredictor,
    as_of: date,
    manifest: dict[str, Any],
    *,
    export_schema_version: str,
) -> dict[str, Any]:
    """
    Full document for ``reference_distributions.json``.

    *export_schema_version* must be ``mma-handicapping-export-v1`` (same as ``model_weights.json``).
    """
    doc: dict[str, Any] = {
        "export_manifest": manifest,
        "export_schema_version": export_schema_version,
        "as_of_date": as_of.isoformat(),
        "matchup_features": build_matchup_feature_quantile_grids(predictor),
        "division_elo": build_division_elo_quantile_grids(predictor, as_of),
        "chart_histograms": {
            "training_features": build_training_histogram_extras(predictor),
            "elo_by_division": build_elo_division_chart_extras(predictor, as_of),
        },
        "notes": (
            "matchup_features + division_elo: 101-point empirical quantiles for mma.ai "
            "(reference_distributions.json). chart_histograms: optional bin/count payloads for SPA."
        ),
    }
    return doc
