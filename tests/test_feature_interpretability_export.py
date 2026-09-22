"""Unit tests for :mod:`src.export.feature_interpretability` (no trained pickle required)."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import DataTier, FightRecord, ResultMethod, WeightClass  # noqa: E402
from src.export.feature_interpretability import (  # noqa: E402
    MARGINAL_BETA_DEFINITION,
    MIN_DIVISION_ROWS,
    average_share_percent,
    build_feature_interpretability_document,
    marginal_betas,
    marginal_magnitude_matrix,
    share_percent_rows,
)
from src.export.reference_distributions_export import (  # noqa: E402
    N_QUANTILE_POINTS,
    QUANTILE_PERCENT_LEVELS,
)
from src.matchup.interactions import FEATURE_NAMES  # noqa: E402
from src.model.regression import CLASS_LABELS, N_CLASSES  # noqa: E402

SCHEMA = "mma-handicapping-export-v1"
MANIFEST = {"export_schema_version": SCHEMA, "notes": "unit test"}
AS_OF = date(2026, 3, 1)
N_FEATURES = len(FEATURE_NAMES)


def _fight(wc: WeightClass) -> FightRecord:
    return FightRecord(
        fight_id="f" * 16,
        fighter_a_id="a" * 16,
        fighter_b_id="b" * 16,
        winner_id="a" * 16,
        result_method=ResultMethod.KO_TKO,
        weight_class=wc,
        fight_date=date(2025, 6, 1),
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


def _stub_predictor(W: np.ndarray, X: np.ndarray, weight_classes: list[WeightClass]):
    """Minimal duck-typed predictor: ``ensure_training_matrix`` short-circuits on cached X."""
    return SimpleNamespace(
        regression=SimpleNamespace(W=W),
        _X_train=X,
        _train_included_fights=[_fight(wc) for wc in weight_classes],
        config=SimpleNamespace(master_start_year=2008, holdout_start_date=date(2024, 1, 1)),
    )


class TestMarginalBetas(unittest.TestCase):
    def test_beta_is_the_l2_norm_of_the_feature_column_of_W(self) -> None:
        W = np.zeros((N_CLASSES, N_FEATURES))
        W[0, 0] = 3.0
        W[1, 0] = 4.0
        W[2, 1] = -2.0
        betas = marginal_betas(W)
        self.assertAlmostEqual(betas[0], 5.0)
        self.assertAlmostEqual(betas[1], 2.0)
        self.assertTrue(np.allclose(betas[2:], 0.0))

    def test_rejects_a_W_of_the_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            marginal_betas(np.zeros((N_CLASSES, N_FEATURES + 1)))


class TestMarginalMagnitudesAndShares(unittest.TestCase):
    def test_magnitude_is_beta_times_absolute_feature_value(self) -> None:
        X = np.array([[2.0, -3.0]])
        M = marginal_magnitude_matrix(X, np.array([1.5, 2.0]))
        self.assertTrue(np.allclose(M, [[3.0, 6.0]]))

    def test_shares_sum_to_one_hundred_per_row(self) -> None:
        M = np.array([[1.0, 3.0], [4.0, 4.0]])
        S = share_percent_rows(M)
        self.assertTrue(np.allclose(S.sum(axis=1), 100.0))
        self.assertTrue(np.allclose(S[0], [25.0, 75.0]))
        self.assertTrue(np.allclose(S[1], [50.0, 50.0]))

    def test_an_all_zero_row_spreads_uniformly_instead_of_producing_nan(self) -> None:
        S = share_percent_rows(np.zeros((1, 4)))
        self.assertTrue(np.all(np.isfinite(S)))
        self.assertTrue(np.allclose(S, 25.0))

    def test_average_share_is_the_mean_of_per_row_shares(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(average_share_percent(M), [50.0, 50.0]))


class TestBuildFeatureInterpretabilityDocument(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260301)
        self.W = rng.normal(scale=0.4, size=(N_CLASSES, N_FEATURES))
        self.X = rng.normal(scale=2.0, size=(240, N_FEATURES))
        self.weight_classes = (
            [WeightClass.LIGHTWEIGHT] * 200
            + [WeightClass.HEAVYWEIGHT] * 30
            + [WeightClass.W_FLYWEIGHT] * 10
        )
        self.doc = build_feature_interpretability_document(
            _stub_predictor(self.W, self.X, self.weight_classes),
            AS_OF,
            MANIFEST,
            export_schema_version=SCHEMA,
        )

    def test_document_envelope_matches_the_other_exports(self) -> None:
        self.assertEqual(self.doc["export_schema_version"], SCHEMA)
        self.assertEqual(self.doc["as_of_date"], AS_OF.isoformat())
        self.assertEqual(self.doc["export_manifest"], MANIFEST)
        self.assertEqual(list(self.doc["feature_names"]), FEATURE_NAMES)
        self.assertEqual(list(self.doc["class_labels"]), list(CLASS_LABELS))
        self.assertEqual(self.doc["marginal_beta_definition"], MARGINAL_BETA_DEFINITION)
        self.assertEqual(self.doc["cohort"]["n_rows"], self.X.shape[0])

    def test_marginal_beta_and_class_coefficients_come_from_W(self) -> None:
        """Real trained weights, not hand-tuned constants."""
        expected = marginal_betas(self.W)
        for j, name in enumerate(FEATURE_NAMES):
            self.assertAlmostEqual(self.doc["marginal_beta"][name], float(expected[j]))
            self.assertEqual(
                self.doc["class_coefficients"][name],
                [float(self.W[k, j]) for k in range(N_CLASSES)],
            )

    def test_population_quantiles_are_non_negative_and_non_decreasing(self) -> None:
        """Magnitudes are absolute values, so the grid starts at or above zero."""
        for name in FEATURE_NAMES:
            block = self.doc["population"]["marginal_magnitude_quantiles"][name]
            self.assertEqual(block["percentile_levels"], list(QUANTILE_PERCENT_LEVELS))
            self.assertEqual(len(block["values"]), N_QUANTILE_POINTS)
            self.assertGreaterEqual(block["values"][0], 0.0)
            for lo, hi in zip(block["values"], block["values"][1:]):
                self.assertGreaterEqual(hi, lo - 1e-9)

    def test_population_quantiles_rank_against_the_real_training_rows(self) -> None:
        """The median of the exported grid must equal the median of the actual magnitudes."""
        M = marginal_magnitude_matrix(self.X, marginal_betas(self.W))
        for j, name in enumerate(FEATURE_NAMES):
            block = self.doc["population"]["marginal_magnitude_quantiles"][name]
            self.assertAlmostEqual(block["values"][50], float(np.percentile(M[:, j], 50)), places=9)
            self.assertAlmostEqual(block["values"][0], float(M[:, j].min()), places=9)
            self.assertAlmostEqual(block["values"][100], float(M[:, j].max()), places=9)

    def test_average_share_percent_sums_to_one_hundred(self) -> None:
        self.assertAlmostEqual(sum(self.doc["population"]["average_share_percent"].values()), 100.0, places=6)

    def test_by_division_partitions_the_training_rows(self) -> None:
        divisions = self.doc["by_division"]
        self.assertEqual(
            {k: v["n_rows"] for k, v in divisions.items()},
            {"lightweight": 200, "heavyweight": 30, "w_flyweight": 10},
        )
        self.assertEqual(sum(v["n_rows"] for v in divisions.values()), self.X.shape[0])
        for wc, blk in divisions.items():
            self.assertAlmostEqual(sum(blk["average_share_percent"].values()), 100.0, places=6, msg=wc)

    def test_thin_divisions_are_flagged_rather_than_dropped(self) -> None:
        divisions = self.doc["by_division"]
        self.assertTrue(divisions["lightweight"]["sufficient_rows"])
        self.assertFalse(divisions["w_flyweight"]["sufficient_rows"])
        self.assertEqual(self.doc["cohort"]["min_division_rows"], MIN_DIVISION_ROWS)

    def test_division_share_differs_from_the_population_share(self) -> None:
        """A per-division baseline is only worth exporting if it is not the global one."""
        population = self.doc["population"]["average_share_percent"]
        heavyweight = self.doc["by_division"]["heavyweight"]["average_share_percent"]
        self.assertNotEqual(
            [round(population[n], 6) for n in FEATURE_NAMES],
            [round(heavyweight[n], 6) for n in FEATURE_NAMES],
        )

    def test_requires_trained_weights(self) -> None:
        pred = _stub_predictor(self.W, self.X, self.weight_classes)
        pred.regression = None
        with self.assertRaises(RuntimeError):
            build_feature_interpretability_document(
                pred, AS_OF, MANIFEST, export_schema_version=SCHEMA
            )

    def test_rejects_included_fights_misaligned_with_the_matrix(self) -> None:
        pred = _stub_predictor(self.W, self.X, self.weight_classes[:-1])
        with self.assertRaises(ValueError):
            build_feature_interpretability_document(
                pred, AS_OF, MANIFEST, export_schema_version=SCHEMA
            )


if __name__ == "__main__":
    unittest.main()
