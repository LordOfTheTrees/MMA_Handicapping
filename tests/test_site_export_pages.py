"""
Website page contracts against committed deploy JSON (``JSON_exports/``).

Mapped from ``docs/website_elements.md`` + ``docs/webpage-data-by-specificity.md``.
Validates that shipped artifacts contain the fields those pages need at a **structural** level
(point-inference parity remains ``tests.test_artifact_parity``).

Override export directory: env ``MMA_SITE_EXPORT_DIR`` (absolute or relative to repo root).

Each test prints one ``[site_pages]`` line on stderr for human scan.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import WeightClass  # noqa: E402
from src.export.json_inference import (  # noqa: E402
    EXPECTED_EXPORT_SCHEMA,
    predict_proba_snapshot,
)
from src.matchup.interactions import FEATURE_NAMES  # noqa: E402
from src.model.regression import CLASS_LABELS, N_CLASSES  # noqa: E402


def _site_export_dir() -> Path:
    raw = os.environ.get("MMA_SITE_EXPORT_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p.resolve()
    return (ROOT / "JSON_exports").resolve()


_UPCOMING_KEYS = (
    "bout_order",
    "fight_id",
    "fighter_a_id",
    "fighter_b_id",
    "weight_class",
)
_PROFILE_KEYS = ("fighter_id", "name", "stance")


def setUpModule() -> None:
    d = _site_export_dir()
    print(
        "\n"
        "================================================================================\n"
        " MODULE: tests.test_site_export_pages\n"
        f" Site JSON dir: {d}\n"
        " Ref: docs/website_elements.md (pages), docs/webpage-data-by-specificity.md\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


class TestSiteExportPages(unittest.TestCase):
    """Page clusters vs committed deploy JSON (``JSON_exports/``)."""

    export_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.export_dir = _site_export_dir()
        need = [
            "upcoming_events.json",
            "model_weights.json",
            "elo_states.json",
            "style_axes.json",
            "fighter_profiles.json",
            "reference_distributions.json",
        ]
        missing = [n for n in need if not (cls.export_dir / n).is_file()]
        if missing:
            raise unittest.SkipTest(
                f"Site export dir missing files {missing} under {cls.export_dir}. "
                "Run export_artifacts (includes reference_distributions.json for mma.ai) + export_upcoming_events or set MMA_SITE_EXPORT_DIR."
            )

    def test_01_home_upcoming_calendar(self) -> None:
        """website_elements.md Main/Home - calendar of upcoming events + card bouts."""
        p = self.export_dir / "upcoming_events.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc.get("export_schema_version"), "mma-handicapping-upcoming-v1")
        events = doc.get("events")
        self.assertIsInstance(events, list)
        bout_count = 0
        for ev in events:
            self.assertIsInstance(ev.get("bouts"), list)
            for b in ev["bouts"]:
                bout_count += 1
                for k in _UPCOMING_KEYS:
                    self.assertIn(k, b, msg=f"bout missing {k!r} in event {ev.get('event_id')}")
        print(
            f"[site_pages] Home / upcoming calendar: OK "
            f"(events={len(events)} bouts_total={bout_count})  ref=website_elements.md #1",
            flush=True,
            file=sys.stderr,
        )

    def test_02_rankings_elo_browser(self) -> None:
        """website_elements.md Rankings - ELO by weight class from snapshot."""
        p = self.export_dir / "elo_states.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc.get("export_schema_version"), EXPECTED_EXPORT_SCHEMA)
        states = doc["states"]
        self.assertIsInstance(states, dict)
        self.assertGreater(len(states), 0, msg="rankings page needs non-empty ELO snapshot")
        fid0, wcs0 = next(iter(states.items()))
        self.assertIsInstance(wcs0, dict)
        wc_key0, cell0 = next(iter(wcs0.items()))
        self.assertIn("elo", cell0)
        float(cell0["elo"])
        print(
            f"[site_pages] Rankings / ELO browser: OK "
            f"(fighters={len(states)} sample_fighter={fid0!r} sample_div={wc_key0!r}) "
            "ref=website_elements.md Rankings",
            flush=True,
            file=sys.stderr,
        )

    def test_03_fighter_profile(self) -> None:
        """website_elements.md Fighter Profile - static profile + cross-links to ELO."""
        prof_path = self.export_dir / "fighter_profiles.json"
        elo_path = self.export_dir / "elo_states.json"
        prof_doc = json.loads(prof_path.read_text(encoding="utf-8"))
        elo_doc = json.loads(elo_path.read_text(encoding="utf-8"))
        profiles = prof_doc["profiles"]
        state_ids = set(elo_doc["states"].keys())
        prof_ids = set(profiles.keys())
        overlap = state_ids & prof_ids
        self.assertGreater(len(overlap), 0, msg="no overlap between elo states and profiles")
        sample = next(iter(overlap))
        block = profiles[sample]
        for k in _PROFILE_KEYS:
            self.assertIn(k, block, msg=f"profile {sample} missing {k}")
        print(
            f"[site_pages] Fighter profile: OK "
            f"(profiles={len(profiles)} overlap_with_elo={len(overlap)}) "
            "ref=website_elements.md Fighter Profile",
            flush=True,
            file=sys.stderr,
        )

    def test_04_hypothetical_and_card_prediction_json(self) -> None:
        """Single bout / hypothetical: four JSONs + snapshot inference (website_elements Single Bout)."""
        as_of = date.fromisoformat(
            str(json.loads((self.export_dir / "elo_states.json").read_text(encoding="utf-8"))["as_of_date"])
        )
        up = json.loads((self.export_dir / "upcoming_events.json").read_text(encoding="utf-8"))
        bout = None
        for ev in up.get("events") or []:
            bouts = ev.get("bouts") or []
            if bouts:
                bout = bouts[0]
                break
        if bout is None:
            print(
                "[site_pages] Hypothetical / card prediction: SKIP (no bouts in upcoming_events.json)",
                flush=True,
                file=sys.stderr,
            )
            self.skipTest("upcoming_events has no bouts to pair with inference JSON")

        a_id = bout["fighter_a_id"]
        b_id = bout["fighter_b_id"]
        wc_s = bout["weight_class"]
        wc = WeightClass(wc_s)

        probs = predict_proba_snapshot(self.export_dir, a_id, b_id, wc, as_of)
        self.assertEqual(probs.shape, (N_CLASSES,))
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(float(np.sum(probs)), 1.0, rtol=0.0, atol=1e-9)
        print(
            f"[site_pages] Hypothetical / scheduled bout (JSON snapshot): OK "
            f"(sample {a_id[:8]}... vs {b_id[:8]}... {wc_s} @ {as_of}) "
            "ref=website_elements.md Single Bout / Hypothetical",
            flush=True,
            file=sys.stderr,
        )

    def test_05_about_model_export(self) -> None:
        """About the model - regression transparency (website_elements ADR / feature list)."""
        p = self.export_dir / "model_weights.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc.get("export_schema_version"), EXPECTED_EXPORT_SCHEMA)
        self.assertEqual(list(doc["feature_names"]), list(FEATURE_NAMES))
        self.assertEqual(len(doc["class_labels"]), N_CLASSES)
        self.assertEqual(doc["class_labels"], list(CLASS_LABELS))
        W = np.asarray(doc["W"], dtype=float)
        self.assertEqual(W.shape, (N_CLASSES, len(FEATURE_NAMES)))
        self.assertIn("training_config", doc)
        self.assertIn("regression", doc)
        print(
            f"[site_pages] About model (weights + training_config): OK "
            f"(features={len(FEATURE_NAMES)} classes={N_CLASSES}) "
            "ref=website_elements.md About the Model",
            flush=True,
            file=sys.stderr,
        )


    def test_06_reference_distributions_export(self) -> None:
        """``reference_distributions.json`` — mma.ai quantile grids + optional chart histograms."""
        from src.export.reference_distributions_export import (  # noqa: E402
            CHART_PERCENTILE_LEVELS,
            N_QUANTILE_POINTS,
            QUANTILE_PERCENT_LEVELS,
        )

        p = self.export_dir / "reference_distributions.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc.get("export_schema_version"), EXPECTED_EXPORT_SCHEMA)
        self.assertEqual(doc.get("as_of_date"), doc["export_manifest"]["as_of_date"])
        mf = doc["matchup_features"]
        for fn in FEATURE_NAMES:
            block = mf[fn]
            self.assertEqual(block["percentile_levels"], list(QUANTILE_PERCENT_LEVELS))
            self.assertEqual(len(block["values"]), N_QUANTILE_POINTS)
        tf = doc["chart_histograms"]["training_features"]
        for fn in FEATURE_NAMES:
            tblock = tf["features"][fn]
            hist = tblock.get("histogram")
            self.assertIsNotNone(hist, msg=f"{fn} missing histogram")
            self.assertEqual(len(hist["counts"]), len(hist["bin_edges"]) - 1)
            self.assertEqual(len(tblock["percentiles"]), len(CHART_PERCENTILE_LEVELS))
        print(
            f"[site_pages] Reference distributions: OK "
            f"(training_rows={tf['n_rows']} "
            f"chart_divisions={len(doc['chart_histograms']['elo_by_division']['divisions'])} "
            f"quantile_divisions={len(doc.get('division_elo') or {})}) "
            "ref=mma.ai api/reference_distributions",
            flush=True,
            file=sys.stderr,
        )


if __name__ == "__main__":
    unittest.main()
