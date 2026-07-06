"""
Tests for ESPN **upcoming** (announced, non-final) bout parsing (no HTTP).

WHAT RUNS
    Executes :func:`~src.data.espn_upcoming._parse_future_bouts_from_fightcenter`,
    :func:`~src.data.espn_upcoming._resolve_known_fighter_id`, and
    :func:`~src.data.espn_upcoming._collect_future_events` against static JSON shaped
    like ESPN's ``fightcenter`` API response and a minimal fake ``ESPNClient``.

HOW TO RUN (repo root)
    ``python -m unittest tests.test_espn_upcoming_parse -v``

READING FAILURES
    If the good fixture yields zero bouts, ESPN likely changed the ``cards`` /
    ``competitions`` / ``competitors`` shape — compare to
    ``src.data.espn_ingest._iter_competitions`` and
    ``src.data.espn_normalize._athlete_id_from_competitor``, which this module reuses
    verbatim (see ``docs/ufc-com-upcoming-scrape-plan.md`` §0).
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

from src.data.espn_crosswalk import CrosswalkStore
from src.data.espn_upcoming import (
    _collect_future_events,
    _parse_future_bouts_from_fightcenter,
    _resolve_known_fighter_id,
)


def _fightcenter_fixture() -> Dict[str, Any]:
    """One announced (non-final) bout, one already-final bout that must be excluded."""
    return {
        "cards": {
            "main": {
                "competitions": [
                    {
                        "id": "401706999",
                        "note": "Featherweight Bout",
                        "type": {"text": "Featherweight"},
                        "status": {"type": {"completed": False, "state": "pre", "name": "STATUS_SCHEDULED"}},
                        "competitors": [
                            {"athlete": {"id": "1234", "displayName": "Conor McGregor"}},
                            {"athlete": {"id": "5678", "displayName": "Max Holloway"}},
                        ],
                    },
                    {
                        "id": "401706998",
                        "note": "Lightweight Bout",
                        "status": {"type": {"completed": True, "state": "post", "name": "STATUS_FINAL"}},
                        "competitors": [
                            {"athlete": {"id": "111", "displayName": "Fighter A"}},
                            {"athlete": {"id": "222", "displayName": "Fighter B"}},
                        ],
                    },
                ]
            }
        }
    }


def setUpModule() -> None:
    print(
        "\n"
        "================================================================================\n"
        " MODULE: tests.test_espn_upcoming_parse\n"
        " Target: espn_upcoming parsing/resolution (offline; no ESPN HTTP)\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


class TestFightcenterBoutParse(unittest.TestCase):
    def test_01_non_final_bout_yields_expected_dict(self) -> None:
        cw = CrosswalkStore(Path("/nonexistent-for-this-test"))
        rows = _parse_future_bouts_from_fightcenter(
            _fightcenter_fixture(), "evt1", crosswalk=cw, name_index={}
        )
        print(f"    -- Parsed {len(rows)} announced bout(s) (want 1).", flush=True, file=sys.stderr)
        self.assertEqual(
            len(rows), 1, msg="Final competition must be excluded; only the announced bout should remain."
        )
        bout = rows[0]
        checks: Dict[str, tuple[object, object]] = {
            "fight_id": ("espn_evt1_401706999", bout["fight_id"]),
            "fighter_a_name": ("Conor McGregor", bout["fighter_a_name"]),
            "fighter_b_name": ("Max Holloway", bout["fighter_b_name"]),
            "weight_class": ("featherweight", bout["weight_class"]),
            "bout_order": (0, bout["bout_order"]),
            "fighter_a_id": (None, bout["fighter_a_id"]),
            "fighter_b_id": (None, bout["fighter_b_id"]),
        }
        for name, (want, got) in checks.items():
            with self.subTest(field=name):
                self.assertEqual(want, got, msg=f"Field `{name}` mismatch: expected {want!r}, got {got!r}")
        print("    -- test_01: all assertions passed.\n", flush=True, file=sys.stderr)

    def test_02_final_competition_excluded(self) -> None:
        cw = CrosswalkStore(Path("/nonexistent-for-this-test"))
        rows = _parse_future_bouts_from_fightcenter(
            _fightcenter_fixture(), "evt1", crosswalk=cw, name_index={}
        )
        names = {(r["fighter_a_name"], r["fighter_b_name"]) for r in rows}
        self.assertNotIn(
            ("Fighter A", "Fighter B"),
            names,
            msg="A completed (_competition_is_final=True) bout must never appear in upcoming output.",
        )

    def test_03_incomplete_competitor_list_is_skipped(self) -> None:
        fightcenter = {
            "cards": {
                "main": {
                    "competitions": [
                        {
                            "id": "1",
                            "note": "Lightweight Bout",
                            "status": {"type": {"completed": False, "state": "pre"}},
                            "competitors": [{"athlete": {"id": "1", "displayName": "Only One"}}],
                        }
                    ]
                }
            }
        }
        cw = CrosswalkStore(Path("/nonexistent-for-this-test"))
        rows = _parse_future_bouts_from_fightcenter(fightcenter, "evt2", crosswalk=cw, name_index={})
        self.assertEqual(len(rows), 0, msg="A bout missing its second competitor must not be invented.")


class TestFighterIdResolution(unittest.TestCase):
    """Read-only resolution — must never mutate the crosswalk (ADR-23: no new state
    from an announced-but-unplayed bout)."""

    def test_01_crosswalk_hit(self) -> None:
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "espn_crosswalk_fighters.csv").write_text(
                "ufcstats_fighter_id,espn_athlete_id,fighter_name,match_method\n"
                "deadbeef01,1234,Conor McGregor,name\n",
                encoding="utf-8",
            )
            cw = CrosswalkStore(tmp)
            before = dict(cw.athlete_to_fighter)
            fid = _resolve_known_fighter_id("1234", "Conor McGregor", crosswalk=cw, name_index={})
            self.assertEqual(fid, "deadbeef01")
            self.assertEqual(cw.athlete_to_fighter, before, msg="Resolution must not mutate the crosswalk store.")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_02_name_index_fallback(self) -> None:
        cw = CrosswalkStore(Path("/nonexistent-for-this-test"))
        fid = _resolve_known_fighter_id(
            "9999", "Some Debutant", crosswalk=cw, name_index={"somedebutant": "nameidx01"}
        )
        self.assertEqual(fid, "nameidx01")

    def test_03_unknown_fighter_returns_none(self) -> None:
        cw = CrosswalkStore(Path("/nonexistent-for-this-test"))
        fid = _resolve_known_fighter_id("8888", "Totally Unknown", crosswalk=cw, name_index={})
        self.assertIsNone(fid, msg="Unresolved fighters must ship name-only, never a fabricated ID.")


class _FakeESPNClient:
    """Minimal stand-in for ESPNClient — only the two methods _collect_future_events calls."""

    def __init__(self, events_by_year: Dict[int, List[Dict[str, Any]]]) -> None:
        self._events_by_year = events_by_year

    def list_event_refs(self, year: int) -> List[str]:
        return [f"ref-{e['id']}" for e in self._events_by_year.get(year, [])]

    def fetch_event(self, event_ref: str) -> Dict[str, Any]:
        event_id = event_ref.split("-", 1)[1]
        for events in self._events_by_year.values():
            for e in events:
                if e["id"] == event_id:
                    return e
        raise AssertionError(f"unknown ref in test fixture: {event_ref}")


class TestCollectFutureEvents(unittest.TestCase):
    def test_01_keeps_future_drops_past(self) -> None:
        today = date(2026, 7, 6)
        past = today - timedelta(days=7)
        future = today + timedelta(days=14)
        client = _FakeESPNClient(
            {
                2026: [
                    {"id": "1", "date": f"{past.isoformat()}T00:00Z", "name": "Past Event"},
                    {"id": "2", "date": f"{future.isoformat()}T00:00Z", "name": "Future Event"},
                    {"id": "3", "date": f"{today.isoformat()}T00:00Z", "name": "Today Event"},
                ]
            }
        )
        pending = _collect_future_events(client, [2026], today=today)
        ids = [p.event_id for p in pending]
        print(f"    -- Resolved future event ids: {ids} (want ['3', '2'] in date order)", flush=True, file=sys.stderr)
        self.assertEqual(ids, ["3", "2"], msg="Must keep today-or-later events, sorted by date, and drop past ones.")


def tearDownModule() -> None:
    print(
        "================================================================================\n"
        " END tests.test_espn_upcoming_parse\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
