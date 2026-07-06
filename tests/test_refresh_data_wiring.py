"""
Smoke tests for :func:`src.data.refresh.refresh_data`'s **control flow** (no real HTTP).

WHAT RUNS
    Every ESPN/UFCStats network-touching dependency is mocked; these tests verify
    ``refresh_data`` wires branches together correctly — which ``RefreshResult`` fields
    get set in which scenario, that a blocked/failed source is non-fatal, and that the
    one ``ESPNClient`` instance is actually shared across all three ESPN-touching calls
    (fights incremental, profiles, upcoming cards) rather than each creating its own and
    re-hitting the same season/event URLs.

HOW TO RUN (repo root)
    ``python -m unittest tests.test_refresh_data_wiring -v``
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.refresh import DataRefreshError, refresh_data


class _FakeProbe:
    def __init__(self, blocked: bool) -> None:
        self.blocked = blocked
        self.detail = "fake probe"
        self.event_count = 0


def _passing_audit(*args, **kwargs):
    return {"passed": True, "reject_count": 0, "warn_count": 0}, 0


def setUpModule() -> None:
    print(
        "\n"
        "================================================================================\n"
        " MODULE: tests.test_refresh_data_wiring\n"
        " Target: refresh_data() branch wiring (offline; every ESPN/UFCStats call mocked)\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


class TestRefreshDataWiring(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        (self._tmp / "ufcstats_fights.csv").write_text(
            "fight_id,fighter_a_id,fighter_b_id,winner_id,method,weight_class,date\n"
            "deadbeef,aaa111,bbb222,aaa111,ko/tko,lightweight,2024-11-16\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_01_both_sources_fresh_this_run(self) -> None:
        with patch("src.data.refresh.refresh_espn_fights_incremental", return_value=(1, 1)) as m_fights, \
             patch("src.data.refresh.refresh_espn_profiles_incremental") as m_profiles, \
             patch("src.data.refresh.run_espn_ingest_audit", side_effect=_passing_audit), \
             patch("src.data.refresh.probe_completed_events_index", return_value=_FakeProbe(blocked=False)), \
             patch("src.data.refresh.scrape_fighter_profiles_to_csv") as m_ufc_profiles, \
             patch("src.data.refresh.scrape_upcoming_cards_to_path") as m_ufc_upcoming, \
             patch("src.data.refresh.scrape_espn_upcoming_cards_to_path") as m_espn_upcoming:
            result = refresh_data(self._tmp)

        print(f"    -- result: {result}", flush=True, file=sys.stderr)
        self.assertTrue(result.upcoming_cards_scraped, msg="UFCStats not blocked -> should be True")
        self.assertTrue(result.espn_upcoming_cards_scraped, msg="ESPN scrape mocked to succeed -> should be True")
        m_ufc_profiles.assert_called_once()
        m_ufc_upcoming.assert_called_once()
        m_espn_upcoming.assert_called_once()

        # All three ESPN-touching calls must share the exact same ESPNClient instance —
        # otherwise the upcoming pass re-hits the same season/event URLs the fights pass
        # already resolved instead of reusing ESPNClient's in-memory/disk cache.
        client_from_fights = m_fights.call_args.kwargs["client"]
        client_from_profiles = m_profiles.call_args.kwargs["client"]
        client_from_espn_upcoming = m_espn_upcoming.call_args.kwargs["client"]
        self.assertIs(client_from_fights, client_from_profiles, msg="fights/profiles must share one ESPNClient")
        self.assertIs(
            client_from_fights, client_from_espn_upcoming, msg="fights/upcoming must share one ESPNClient"
        )
        print("    -- test_01: all assertions passed (shared client confirmed).\n", flush=True, file=sys.stderr)

    def test_02_ufcstats_blocked_espn_upcoming_still_attempted(self) -> None:
        with patch("src.data.refresh.refresh_espn_fights_incremental", return_value=(1, 1)), \
             patch("src.data.refresh.refresh_espn_profiles_incremental"), \
             patch("src.data.refresh.run_espn_ingest_audit", side_effect=_passing_audit), \
             patch("src.data.refresh.probe_completed_events_index", return_value=_FakeProbe(blocked=True)), \
             patch("src.data.refresh.scrape_fighter_profiles_to_csv") as m_ufc_profiles, \
             patch("src.data.refresh.scrape_upcoming_cards_to_path") as m_ufc_upcoming, \
             patch("src.data.refresh.scrape_espn_upcoming_cards_to_path") as m_espn_upcoming:
            result = refresh_data(self._tmp)

        self.assertFalse(result.upcoming_cards_scraped, msg="UFCStats blocked -> False")
        self.assertTrue(result.espn_upcoming_cards_scraped, msg="ESPN attempt is independent of UFCStats' block")
        m_espn_upcoming.assert_called_once()
        m_ufc_profiles.assert_not_called()
        m_ufc_upcoming.assert_not_called()

    def test_03_espn_upcoming_failure_is_non_fatal(self) -> None:
        with patch("src.data.refresh.refresh_espn_fights_incremental", return_value=(1, 1)), \
             patch("src.data.refresh.refresh_espn_profiles_incremental"), \
             patch("src.data.refresh.run_espn_ingest_audit", side_effect=_passing_audit), \
             patch("src.data.refresh.probe_completed_events_index", return_value=_FakeProbe(blocked=False)), \
             patch("src.data.refresh.scrape_fighter_profiles_to_csv"), \
             patch("src.data.refresh.scrape_upcoming_cards_to_path"), \
             patch(
                 "src.data.refresh.scrape_espn_upcoming_cards_to_path",
                 side_effect=RuntimeError("simulated ESPN failure"),
             ):
            result = refresh_data(self._tmp)  # must not raise

        self.assertFalse(result.espn_upcoming_cards_scraped)
        self.assertTrue(result.upcoming_cards_scraped, msg="UFCStats path must be unaffected by ESPN failure")

    def test_04_require_fight_updates_still_raises(self) -> None:
        """Regression check: sharing espn_client across calls must not disturb this
        pre-existing fatal-error path."""
        with patch("src.data.refresh.refresh_espn_fights_incremental", return_value=(1, 0)), \
             patch("src.data.refresh.refresh_espn_profiles_incremental"), \
             patch("src.data.refresh.run_espn_ingest_audit", side_effect=_passing_audit), \
             patch("src.data.refresh.probe_completed_events_index", return_value=_FakeProbe(blocked=True)), \
             patch("src.data.refresh.scrape_espn_upcoming_cards_to_path"):
            with self.assertRaises(DataRefreshError):
                refresh_data(self._tmp, require_fight_updates=True)


def tearDownModule() -> None:
    print(
        "================================================================================\n"
        " END tests.test_refresh_data_wiring\n"
        "================================================================================\n",
        flush=True,
        file=sys.stderr,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
