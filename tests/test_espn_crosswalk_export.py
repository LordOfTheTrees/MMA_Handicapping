"""Unit tests for :mod:`src.export.espn_crosswalk_export` (no trained pickle required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.espn_crosswalk import CrosswalkStore  # noqa: E402
from src.export.espn_crosswalk_export import (  # noqa: E402
    ID_SCHEME,
    build_espn_crosswalk_document,
    espn_form_fight_id,
)

SCHEMA = "mma-handicapping-export-v1"
MANIFEST = {"export_schema_version": SCHEMA, "notes": "unit test"}
AS_OF = date(2026, 3, 1)


class _CrosswalkFixture:
    """Temp-dir :class:`CrosswalkStore` populated through its real ``record_*`` API."""

    def __enter__(self) -> CrosswalkStore:
        self._td = tempfile.TemporaryDirectory()
        cw = CrosswalkStore(Path(self._td.name))
        cw.record_fight(
            ufcstats_fight_id="f1" * 8,
            espn_competition_id="401700001",
            espn_event_id="600500001",
            event_date=date(2026, 2, 14),
            match_method="unit_test",
        )
        cw.record_fight(
            ufcstats_fight_id="f2" * 8,
            espn_competition_id="401700002",
            espn_event_id="600500001",
            event_date=date(2026, 2, 14),
            match_method="unit_test",
        )
        # A bout absent from UFCStats keeps its ``espn_<competition>`` placeholder fight id.
        cw.record_fight(
            ufcstats_fight_id="espn_401700003",
            espn_competition_id="401700003",
            espn_event_id="600500002",
            event_date=date(2026, 1, 10),
            match_method="espn_new",
        )
        cw.record_fighter(
            ufcstats_fighter_id="aaaaaaaaaaaaaaaa",
            espn_athlete_id="900001",
            fighter_name="Alpha Ant",
            match_method="unit_test",
        )
        cw.record_fighter(
            ufcstats_fighter_id="bbbbbbbbbbbbbbbb",
            espn_athlete_id="900002",
            fighter_name="Beta Bee",
            match_method="unit_test",
        )
        return cw

    def __exit__(self, *exc: object) -> None:
        self._td.cleanup()


class TestEspnFormFightId(unittest.TestCase):
    def test_maps_ufcstats_id_to_event_scoped_espn_id(self) -> None:
        with _CrosswalkFixture() as cw:
            self.assertEqual(espn_form_fight_id(cw, "f1" * 8), "espn_600500001_401700001")

    def test_maps_espn_placeholder_fight_id(self) -> None:
        """``espn_<competition>`` must still resolve to the event-scoped form."""
        with _CrosswalkFixture() as cw:
            self.assertEqual(espn_form_fight_id(cw, "espn_401700003"), "espn_600500002_401700003")

    def test_returns_none_for_unknown_or_blank_ids(self) -> None:
        with _CrosswalkFixture() as cw:
            self.assertIsNone(espn_form_fight_id(cw, "deadbeefdeadbeef"))
            self.assertIsNone(espn_form_fight_id(cw, ""))


class TestBuildEspnCrosswalkDocument(unittest.TestCase):
    def _doc(self) -> dict:
        with _CrosswalkFixture() as cw:
            return build_espn_crosswalk_document(
                cw, manifest=MANIFEST, export_schema_version=SCHEMA, as_of=AS_OF
            )

    def test_document_envelope_matches_the_other_exports(self) -> None:
        doc = self._doc()
        self.assertEqual(doc["export_schema_version"], SCHEMA)
        self.assertEqual(doc["as_of_date"], AS_OF.isoformat())
        self.assertEqual(doc["export_manifest"], MANIFEST)
        self.assertEqual(doc["id_scheme"], ID_SCHEME)

    def test_fight_map_is_internal_id_to_espn_form_id(self) -> None:
        doc = self._doc()
        self.assertEqual(
            doc["fights"],
            {
                "f1f1f1f1f1f1f1f1": "espn_600500001_401700001",
                "f2f2f2f2f2f2f2f2": "espn_600500001_401700002",
                "espn_401700003": "espn_600500002_401700003",
            },
        )
        self.assertEqual(doc["counts"]["fights"], 3)

    def test_fighter_map_is_internal_id_to_athlete_id(self) -> None:
        doc = self._doc()
        self.assertEqual(
            doc["fighters"],
            {"aaaaaaaaaaaaaaaa": "900001", "bbbbbbbbbbbbbbbb": "900002"},
        )
        self.assertEqual(doc["counts"]["fighters"], 2)

    def test_maps_are_sorted_for_stable_diffs(self) -> None:
        doc = self._doc()
        self.assertEqual(list(doc["fights"]), sorted(doc["fights"]))
        self.assertEqual(list(doc["fighters"]), sorted(doc["fighters"]))

    def test_a_fight_without_an_event_id_is_counted_not_half_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cw = CrosswalkStore(Path(td))
            cw.record_fight(
                ufcstats_fight_id="f9" * 8,
                espn_competition_id="401700009",
                espn_event_id="",
                event_date=date(2026, 2, 14),
                match_method="unit_test",
            )
            doc = build_espn_crosswalk_document(
                cw, manifest=MANIFEST, export_schema_version=SCHEMA, as_of=AS_OF
            )
        self.assertEqual(doc["fights"], {})
        self.assertEqual(doc["counts"]["fights_without_event_id"], 1)

    def test_empty_crosswalk_yields_an_empty_but_valid_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            doc = build_espn_crosswalk_document(
                CrosswalkStore(Path(td)),
                manifest=MANIFEST,
                export_schema_version=SCHEMA,
                as_of=AS_OF,
            )
        self.assertEqual(doc["fights"], {})
        self.assertEqual(doc["fighters"], {})
        self.assertEqual(doc["counts"], {"fights": 0, "fighters": 0, "fights_without_event_id": 0})


if __name__ == "__main__":
    unittest.main()
