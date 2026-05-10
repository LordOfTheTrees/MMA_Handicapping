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
        description="Smoke: export_all() writes five valid JSON inference files.",
    )


@unittest.skipUnless(HAS_HARNESS_MODEL, HARNESS_SKIP_REASON)
class TestExportArtifactsSmoke(unittest.TestCase):
    def test_export_all_writes_five_valid_json_files(self) -> None:
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
                    ch = doc["chart_histograms"]
                    tf = ch["training_features"]
                    self.assertGreater(tf["n_rows"], 0)
                    for fn in FEATURE_NAMES:
                        tblock = tf["features"][fn]
                        self.assertIn("histogram", tblock)
                        self.assertIn("percentiles", tblock)
                        self.assertEqual(len(tblock["percentiles"]), len(CHART_PERCENTILE_LEVELS))
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

        print("[export smoke] OK: all five JSON files valid for this pickle.", flush=True, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
