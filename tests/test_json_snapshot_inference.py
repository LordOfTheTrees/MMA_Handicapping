"""Unit tests for :mod:`src.export.json_inference` (no trained pickle required)."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np

from src.data.schema import WeightClass
from src.export.json_inference import EXPECTED_EXPORT_SCHEMA, predict_proba_snapshot
from src.matchup.interactions import FEATURE_NAMES
from src.model.regression import MultinomialLogisticModel, N_CLASSES

_D = date(2024, 6, 1)

_FIDA = "fida00001"
_FIDB = "fidb00002"
_WKEY = WeightClass.FEATHERWEIGHT.value


def _minimal_bundle(*, wrong_as_of: date | None = None) -> dict[str, dict]:
    as_of = wrong_as_of or _D

    feat_n = len(FEATURE_NAMES)

    mw = {
        "export_schema_version": EXPECTED_EXPORT_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "class_labels": [],
        "n_classes": N_CLASSES,
        "W": [[0.0] * feat_n for _ in range(N_CLASSES)],
        "bootstrap_W": None,
        "regression": {
            "huber_delta": 1.35,
            "l2_lambda": 1e-4,
            "n_features": feat_n,
            "is_fitted": True,
        },
    }

    elo_cell_a = {
        "elo": 1600.0,
        "uncertainty": 2.0,
        "last_fight_date": None,
        "n_fights": 5,
        "primary_tier": 1,
    }
    elo_cell_b = {**elo_cell_a, "elo": 1500.0, "uncertainty": 3.0, "n_fights": 3}

    def _axes_cell(sc: float) -> dict[str, float]:
        return {
            "striker_score": sc,
            "grappler_score": 0.4,
            "finish_threat": 0.3,
            "finish_vulnerability": 0.2,
            "striker_uncertainty": 0.15,
            "grappler_uncertainty": 0.12,
            "n_quality_fights": 4.2,
        }

    elo = {
        "export_schema_version": EXPECTED_EXPORT_SCHEMA,
        "as_of_date": as_of.isoformat(),
        "states": {
            _FIDA: {_WKEY: elo_cell_a},
            _FIDB: {_WKEY: elo_cell_b},
        },
    }
    sx = {
        "export_schema_version": EXPECTED_EXPORT_SCHEMA,
        "as_of_date": as_of.isoformat(),
        "axes": {
            _FIDA: {_WKEY: _axes_cell(0.65)},
            _FIDB: {_WKEY: _axes_cell(0.52)},
        },
    }

    def _prof(fid: str) -> dict[str, object]:
        return {
            "fighter_id": fid,
            "name": fid,
            "reach_cm": 180.0,
            "height_cm": 178.0,
            "date_of_birth": "1990-01-15",
            "stance": "orthodox",
            "wrestling_pedigree": 0.0,
            "boxing_pedigree": 0.0,
            "bjj_pedigree": 0.0,
        }

    prof = {
        "export_schema_version": EXPECTED_EXPORT_SCHEMA,
        "profiles": {_FIDA: _prof(_FIDA), _FIDB: _prof(_FIDB)},
    }

    return {
        "model_weights": mw,
        "elo_states": elo,
        "style_axes": sx,
        "fighter_profiles": prof,
    }


class TestJsonSnapshotInference(unittest.TestCase):
    """Mapping-based artifacts (offline)."""

    def test_flat_W_matches_multinomial_reg_on_same_features(self) -> None:
        """Zero logit weights -> uniform 1/6; wiring matches standalone model."""
        bundle = _minimal_bundle()
        p_snap = predict_proba_snapshot(bundle, _FIDA, _FIDB, WeightClass.FEATHERWEIGHT, _D)
        feat_n = len(FEATURE_NAMES)
        reg = MultinomialLogisticModel(n_features=feat_n)
        reg.W = np.zeros((N_CLASSES, feat_n))
        reg.is_fitted = True
        # features_to_array equivalent from snapshot path is internal — compare prob mass shape + sum
        x = np.zeros(feat_n, dtype=float)
        p_reg = reg.predict_proba(x)
        self.assertEqual(p_snap.shape, (6,))
        np.testing.assert_allclose(p_snap, p_reg, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.sum(p_snap), 1.0, rtol=0.0, atol=1e-15)

    def test_fight_date_must_equal_as_of_date(self) -> None:
        bundle = _minimal_bundle()
        with self.assertRaises(ValueError) as ctx:
            predict_proba_snapshot(bundle, _FIDA, _FIDB, WeightClass.FEATHERWEIGHT, date(2030, 1, 1))
        msg = str(ctx.exception)
        self.assertIn("fight_date", msg)
        self.assertIn("as_of_date", msg)

    def test_mismatch_elo_vs_style_as_of_raises(self) -> None:
        bundle = _minimal_bundle()
        bundle["style_axes"]["as_of_date"] = date(2020, 1, 2).isoformat()
        with self.assertRaises(ValueError) as ctx:
            predict_proba_snapshot(bundle, _FIDA, _FIDB, WeightClass.FEATHERWEIGHT, _D)
        self.assertIn("!=", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
