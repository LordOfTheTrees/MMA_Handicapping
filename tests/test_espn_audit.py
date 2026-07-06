"""ESPN ingest audit (no network)."""
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data.espn_audit import (
    audit_fight_method5,
    audit_fighter_method1,
    count_ufc_bouts_in_eventlog,
    name_similarity,
    parse_record_string,
    run_espn_ingest_audit,
)


class TestEspnAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_name_similarity(self):
        assert name_similarity("Jon Jones", "Jon Jones") == 1.0
        assert name_similarity("Jon Jones", "Alexander Volkanovski") < 0.5

    def test_parse_record_and_eventlog(self):
        assert parse_record_string("5-1-0") == (5, 1, 0)
        payload = {
            "events": [
                {"league": {"slug": "ufc"}},
                {"league": {"name": "Bellator"}},
                {"league": {"abbreviation": "UFC"}},
            ]
        }
        assert count_ufc_bouts_in_eventlog(payload) == 2

    def test_count_ufc_core_v2_eventlog(self):
        core = {
            "events": {
                "count": 2,
                "items": [
                    {"played": True, "$ref": "http://example/event/1"},
                    {"played": False, "$ref": "http://example/event/2"},
                ],
            }
        }
        assert count_ufc_bouts_in_eventlog(core) == 1
        site_shell = {"playerSwitcher": {"athletes": []}, "season": {"year": 2026}}
        assert count_ufc_bouts_in_eventlog(site_shell) == 0

    def test_auto_reject_fuzzy_phys(self):
        profiles = {
            "aaa111": {
                "fighter_id": "aaa111",
                "name": "Jonathan Martinez",
                "height_cm": "170.18",
                "reach_cm": "177.8",
            }
        }
        entry = {
            "fighter_id": "espn_999",
            "espn_athlete_id": "999",
            "display_name": "Jonathan Martínez",
            "match_method": "espn_new",
            "height_cm": 170.18,
            "reach_cm": 177.8,
        }
        rep, findings = audit_fighter_method1(
            entry,
            profiles,
            None,
            fetch_rookie=False,
        )
        assert any(f.code == "duplicate_fuzzy_phys" for f in findings)
        assert rep["phys_identical"] is True

    def test_method5_both_hex_espn_fight_id(self):
        fights = {
            "espn_401": {
                "fight_id": "espn_401",
                "fighter_a_id": "aaa111",
                "fighter_b_id": "bbb222",
                "date": "2025-06-01",
            }
        }
        findings = audit_fight_method5(
            {"fight_id": "espn_401", "match_method": "espn_new"},
            fights,
            {"aaa111": "Jon", "bbb222": "Alex"},
            max_date=date(2025, 6, 15),
        )
        assert any(f.code == "fight_espn_id_both_fighters_matched" for f in findings)

    def test_run_audit_from_last_run(self):
        tmp_path = self.tmp_path
        state = tmp_path / "espn_ingest_state.json"
        state.write_text(
            """{
  "last_run": {
    "new_fighters": [{
      "fighter_id": "espn_1",
      "espn_athlete_id": "1",
      "display_name": "Unique Debut",
      "match_method": "espn_new"
    }],
    "new_fights": []
  }
}""",
            encoding="utf-8",
        )
        (tmp_path / "fighter_profiles.csv").write_text(
            "fighter_id,name,height_cm,reach_cm\nhex99,Other Guy,,\n",
            encoding="utf-8",
        )
        audit, code = run_espn_ingest_audit(
            tmp_path,
            fetch_rookie=False,
            fail_on_reject=False,
        )
        assert audit["new_fighters_count"] == 1
        assert code == 0
        assert (tmp_path / "espn_ingest_audit.json").is_file()


if __name__ == "__main__":
    unittest.main(verbosity=2)
