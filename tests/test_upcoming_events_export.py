"""Golden transform: ``upcoming_cards`` -> ``build_upcoming_events_doc`` (deterministic core)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export.upcoming_events_doc import EXPORT_SCHEMA_VERSION, build_upcoming_events_doc  # noqa: E402


class TestUpcomingEventsExport(unittest.TestCase):
    def test_build_matches_expected_events_shape(self) -> None:
        sample = json.loads((ROOT / "tests" / "fixtures" / "upcoming_cards.sample.json").read_text(encoding="utf-8"))
        expected = json.loads((ROOT / "tests" / "fixtures" / "upcoming_events.expected.json").read_text(encoding="utf-8"))

        doc = build_upcoming_events_doc(sample)
        self.assertEqual(doc["export_schema_version"], EXPORT_SCHEMA_VERSION)
        self.assertEqual(doc["export_schema_version"], expected["export_schema_version"])
        self.assertEqual(doc["events"], expected["events"])

        man = doc["export_manifest"]
        self.assertEqual(man["source_upstream_scraped_at"], sample["scraped_at"])
        self.assertEqual(man["source_upstream_schema"], sample["schema_version"])
        self.assertIn("exported_at", man)
        self.assertIn("git_sha_training_repo", man)


if __name__ == "__main__":
    unittest.main()
