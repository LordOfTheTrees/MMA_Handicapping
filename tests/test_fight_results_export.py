"""Unit tests for :mod:`src.export.fight_results` (no trained pickle required)."""

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
from src.data.schema import DataTier, FightRecord, ResultMethod, WeightClass  # noqa: E402
from src.export.fight_results import (  # noqa: E402
    ID_SCHEME,
    UNDECIDED_METHODS,
    build_fight_results_document,
    espn_form_fight_id,
)
from src.model.regression import CLASS_LABELS  # noqa: E402

SCHEMA = "mma-handicapping-export-v1"
MANIFEST = {"export_schema_version": SCHEMA, "notes": "unit test"}
AS_OF = date(2026, 3, 1)


def _fight(
    fight_id: str,
    *,
    winner: str | None,
    method: ResultMethod,
    a: str = "aaaaaaaaaaaaaaaa",
    b: str = "bbbbbbbbbbbbbbbb",
    fight_date: date = date(2026, 2, 14),
) -> FightRecord:
    return FightRecord(
        fight_id=fight_id,
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=winner,
        result_method=method,
        weight_class=WeightClass.LIGHTWEIGHT,
        fight_date=fight_date,
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


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


class TestBuildFightResultsDocument(unittest.TestCase):
    def _doc(self, fights: list[FightRecord]) -> dict:
        with _CrosswalkFixture() as cw:
            return build_fight_results_document(
                fights,
                cw,
                manifest=MANIFEST,
                export_schema_version=SCHEMA,
                as_of=AS_OF,
            )

    def test_document_envelope_matches_the_other_exports(self) -> None:
        doc = self._doc([_fight("f1" * 8, winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO)])
        self.assertEqual(doc["export_schema_version"], SCHEMA)
        self.assertEqual(doc["as_of_date"], AS_OF.isoformat())
        self.assertEqual(doc["export_manifest"], MANIFEST)
        self.assertEqual(doc["id_scheme"], ID_SCHEME)
        self.assertEqual(list(doc["class_labels"]), list(CLASS_LABELS))

    def test_rows_are_keyed_by_espn_form_id_and_carry_winner_and_method(self) -> None:
        doc = self._doc([_fight("f1" * 8, winner="bbbbbbbbbbbbbbbb", method=ResultMethod.SUBMISSION)])
        row = doc["results"]["espn_600500001_401700001"]
        self.assertEqual(row["fight_id"], "f1" * 8)
        self.assertEqual(row["espn_event_id"], "600500001")
        self.assertEqual(row["espn_competition_id"], "401700001")
        self.assertEqual(row["winner_id"], "bbbbbbbbbbbbbbbb")
        self.assertEqual(row["winner_corner"], "b")
        self.assertEqual(row["espn_winner_athlete_id"], "900002")
        self.assertEqual(row["espn_athlete_a_id"], "900001")
        self.assertEqual(row["espn_athlete_b_id"], "900002")
        self.assertEqual(row["result_method"], "submission")
        self.assertEqual(row["weight_class"], "lightweight")
        self.assertEqual(row["tier"], 1)

    def test_outcome_class_a_matches_class_labels_for_every_decisive_method(self) -> None:
        """``outcome_class_a`` indexes ``CLASS_LABELS`` from corner A's perspective."""
        cases = [
            (ResultMethod.KO_TKO, True, 0),
            (ResultMethod.SUBMISSION, True, 1),
            (ResultMethod.UNANIMOUS_DECISION, True, 2),
            (ResultMethod.SPLIT_DECISION, True, 2),
            (ResultMethod.MAJORITY_DECISION, True, 2),
            (ResultMethod.UNANIMOUS_DECISION, False, 3),
            (ResultMethod.KO_TKO, False, 4),
            (ResultMethod.SUBMISSION, False, 5),
        ]
        for method, a_wins, expected in cases:
            with self.subTest(method=method, a_wins=a_wins):
                winner = "aaaaaaaaaaaaaaaa" if a_wins else "bbbbbbbbbbbbbbbb"
                doc = self._doc([_fight("f1" * 8, winner=winner, method=method)])
                row = doc["results"]["espn_600500001_401700001"]
                self.assertEqual(row["outcome_class_a"], expected)
                self.assertEqual(row["winner_corner"], "a" if a_wins else "b")
                side = "Win by" if a_wins else "Lose by"
                self.assertTrue(CLASS_LABELS[expected].startswith(side))

    def test_undecided_results_export_with_null_winner_and_class(self) -> None:
        for method in (ResultMethod.DRAW, ResultMethod.NO_CONTEST, ResultMethod.DQ):
            with self.subTest(method=method):
                doc = self._doc([_fight("f1" * 8, winner=None, method=method)])
                row = doc["results"]["espn_600500001_401700001"]
                self.assertIsNone(row["winner_id"])
                self.assertIsNone(row["winner_corner"])
                self.assertIsNone(row["outcome_class_a"])
                self.assertIn(row["result_method"], UNDECIDED_METHODS)
                self.assertEqual(doc["counts"]["undecided_no_winner"], 1)

    def test_fights_without_a_crosswalk_row_are_counted_not_emitted(self) -> None:
        """A consumer that only knows ESPN ids cannot use an unmapped fight."""
        doc = self._doc(
            [
                _fight("f1" * 8, winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO),
                _fight("deadbeefdeadbeef", winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO),
            ]
        )
        self.assertEqual(list(doc["results"]), ["espn_600500001_401700001"])
        self.assertEqual(doc["counts"]["fights_seen"], 2)
        self.assertEqual(doc["counts"]["exported"], 1)
        self.assertEqual(doc["counts"]["unmapped_no_crosswalk"], 1)

    def test_results_are_ordered_by_fight_date_then_key(self) -> None:
        doc = self._doc(
            [
                _fight("f2" * 8, winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO, fight_date=date(2026, 2, 14)),
                _fight("espn_401700003", winner="bbbbbbbbbbbbbbbb", method=ResultMethod.KO_TKO, fight_date=date(2026, 1, 10)),
                _fight("f1" * 8, winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO, fight_date=date(2026, 2, 14)),
            ]
        )
        self.assertEqual(
            list(doc["results"]),
            [
                "espn_600500002_401700003",
                "espn_600500001_401700001",
                "espn_600500001_401700002",
            ],
        )

    def test_empty_crosswalk_yields_an_empty_but_valid_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cw = CrosswalkStore(Path(td))
            doc = build_fight_results_document(
                [_fight("f1" * 8, winner="aaaaaaaaaaaaaaaa", method=ResultMethod.KO_TKO)],
                cw,
                manifest=MANIFEST,
                export_schema_version=SCHEMA,
                as_of=AS_OF,
            )
        self.assertEqual(doc["results"], {})
        self.assertEqual(doc["counts"]["exported"], 0)
        self.assertEqual(doc["counts"]["unmapped_no_crosswalk"], 1)


if __name__ == "__main__":
    unittest.main()
