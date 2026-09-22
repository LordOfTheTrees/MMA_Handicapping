"""Smoke: ``export_all`` produces structurally valid JSON (requires ``model.pkl``)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_artifacts  # noqa: E402
from src.export.reference_distributions_export import (  # noqa: E402
    CHART_PERCENTILE_LEVELS,
    N_QUANTILE_POINTS,
    QUANTILE_PERCENT_LEVELS,
)
from src.export.feature_interpretability import (  # noqa: E402
    FEATURE_INTERPRETABILITY_FILENAME,
    MARGINAL_BETA_DEFINITION,
)
from src.export.espn_crosswalk_export import ID_SCHEME  # noqa: E402
from src.matchup.interactions import FEATURE_NAMES  # noqa: E402
from src.model.regression import N_CLASSES  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402

from tests.harness_skip import (  # noqa: E402
    HAS_HARNESS_MODEL,
    HARNESS_SKIP_REASON,
    harness_model_path,
    print_harness_integration_preamble,
)


def setUpModule() -> None:
    print_harness_integration_preamble(
        module="tests.test_export_artifacts_smoke",
        description="Smoke: export_all() writes every artifact JSON file, all structurally valid.",
    )


@unittest.skipUnless(HAS_HARNESS_MODEL, HARNESS_SKIP_REASON)
class TestExportArtifactsSmoke(unittest.TestCase):
    def test_export_all_writes_every_artifact_json_file(self) -> None:
        from datetime import date

        model_path = harness_model_path()
        assert model_path is not None
        pred = MMAPredictor.load(model_path)
        if pred.fights:
            d_asof = pred.fights[-1].fight_date
        else:
            d_asof = date.today()

        print(
            f"[export smoke] Loading pickle: {model_path}\n"
            f"[export smoke] Fight rows in model: {len(pred.fights)}  export as_of_date: {d_asof.isoformat()}\n"
            f"[export smoke] Writing temp JSON bundle + validating shapes/schema...",
            flush=True,
            file=sys.stderr,
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            export_artifacts.export_all(pred, out, as_of=d_asof)
            for name in (
                "model_weights",
                "elo_states",
                "style_axes",
                "fighter_profiles",
                "reference_distributions",
                "espn_crosswalk",
                "feature_interpretability",
            ):
                p = out / f"{name}.json"
                self.assertTrue(p.is_file(), msg=f"missing {p}")
                doc = json.loads(p.read_text(encoding="utf-8"))
                if name == "reference_distributions":
                    self.assertEqual(doc.get("export_schema_version"), export_artifacts.EXPORT_SCHEMA_VERSION)
                    self.assertEqual(doc.get("as_of_date"), d_asof.isoformat())
                    mf = doc["matchup_features"]
                    for fn in FEATURE_NAMES:
                        block = mf[fn]
                        self.assertEqual(block["percentile_levels"], list(QUANTILE_PERCENT_LEVELS))
                        self.assertEqual(len(block["values"]), N_QUANTILE_POINTS)
                        prev = block["values"][0]
                        for v in block["values"][1:]:
                            self.assertGreaterEqual(float(v), prev - 1e-9)
                            prev = float(v)
                    gdi = doc["global_days_idle"]
                    self.assertEqual(gdi["percentile_levels"], list(QUANTILE_PERCENT_LEVELS))
                    self.assertEqual(len(gdi["values"]), N_QUANTILE_POINTS)
                    ch = doc["chart_histograms"]
                    tf = ch["training_features"]
                    self.assertGreater(tf["n_rows"], 0)
                    for fn in FEATURE_NAMES:
                        tblock = tf["features"][fn]
                        self.assertIn("histogram", tblock)
                        self.assertIn("percentiles", tblock)
                        self.assertEqual(len(tblock["percentiles"]), len(CHART_PERCENTILE_LEVELS))
                    gid = ch["global_days_idle"]
                    self.assertIn("histogram", gid)
                    self.assertEqual(gid["n"], 2 * tf["n_rows"])
                    divs = ch["elo_by_division"]["divisions"]
                    self.assertIsInstance(divs, dict)
                    self.assertGreater(len(divs), 0)
                else:
                    self.assertEqual(doc.get("export_schema_version"), export_artifacts.EXPORT_SCHEMA_VERSION)

            mw = json.loads((out / "model_weights.json").read_text(encoding="utf-8"))
            W = np.asarray(mw["W"], dtype=float)
            self.assertEqual(W.shape, (N_CLASSES, len(FEATURE_NAMES)))
            self.assertEqual(list(mw["feature_names"]), FEATURE_NAMES)

            elo = json.loads((out / "elo_states.json").read_text(encoding="utf-8"))
            sx = json.loads((out / "style_axes.json").read_text(encoding="utf-8"))
            self.assertEqual(elo.get("as_of_date"), d_asof.isoformat())
            self.assertEqual(sx.get("as_of_date"), d_asof.isoformat())

            cw = json.loads((out / "espn_crosswalk.json").read_text(encoding="utf-8"))
            self.assertEqual(cw.get("as_of_date"), d_asof.isoformat())
            self.assertEqual(cw["id_scheme"], ID_SCHEME)
            self.assertEqual(cw["counts"]["fights"], len(cw["fights"]))
            self.assertEqual(cw["counts"]["fighters"], len(cw["fighters"]))
            for internal_id, espn_id in cw["fights"].items():
                self.assertTrue(espn_id.startswith("espn_"), msg=f"bad espn id {espn_id!r}")
                self.assertEqual(len(espn_id.split("_")), 3, msg=f"bad espn id {espn_id!r}")
                self.assertTrue(internal_id, msg="blank internal fight id")

            profiles = json.loads((out / "fighter_profiles.json").read_text(encoding="utf-8"))["profiles"]
            trajectories = [
                (fid, point)
                for fid, prof in profiles.items()
                for series in prof.get("elo_trajectories", {}).values()
                for point in series
            ]
            self.assertGreater(len(trajectories), 0, msg="no elo_trajectories in this pickle")
            outcomes = 0
            for fid, point in trajectories:
                cls = point["outcome_class"]
                if cls is None:
                    continue
                outcomes += 1
                self.assertIn(cls, range(N_CLASSES))
                self.assertIsNotNone(point["result_method"])
                self.assertIsNotNone(point["fight_id"])
            self.assertGreater(outcomes, 0, msg="no trajectory point carried an outcome")

            fi = json.loads((out / FEATURE_INTERPRETABILITY_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(fi.get("as_of_date"), d_asof.isoformat())
            self.assertEqual(list(fi["feature_names"]), FEATURE_NAMES)
            self.assertEqual(fi["marginal_beta_definition"], MARGINAL_BETA_DEFINITION)
            self.assertEqual(fi["cohort"]["n_rows"], tf["n_rows"])
            for fn in FEATURE_NAMES:
                self.assertGreaterEqual(fi["marginal_beta"][fn], 0.0)
                self.assertEqual(len(fi["class_coefficients"][fn]), N_CLASSES)
                block = fi["population"]["marginal_magnitude_quantiles"][fn]
                self.assertEqual(block["percentile_levels"], list(QUANTILE_PERCENT_LEVELS))
                self.assertEqual(len(block["values"]), N_QUANTILE_POINTS)
                self.assertGreaterEqual(block["values"][0], 0.0)
            self.assertAlmostEqual(
                sum(fi["population"]["average_share_percent"].values()), 100.0, places=6
            )
            self.assertGreater(len(fi["by_division"]), 0)
            for wc, blk in fi["by_division"].items():
                self.assertGreater(blk["n_rows"], 0, msg=f"empty division {wc}")
                self.assertAlmostEqual(
                    sum(blk["average_share_percent"].values()), 100.0, places=6, msg=wc
                )

        print("[export smoke] OK: every artifact JSON file valid for this pickle.", flush=True, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
