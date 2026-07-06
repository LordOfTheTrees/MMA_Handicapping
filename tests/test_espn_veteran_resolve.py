"""Resolve ESPN veterans to provisioned hex ids (mocked eventlog)."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.data.espn_crosswalk import CrosswalkStore, build_name_index_from_profiles, resolve_fighter_id


def _ufc_eventlog(n: int) -> dict:
    return {"events": [{"league": {"slug": "ufc"}} for _ in range(n)]}


class TestEspnVeteranResolve(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_resolve_fighter_id_espn_veteran(self):
        tmp_path = self.tmp_path
        profiles = tmp_path / "fighter_profiles.csv"
        profiles.write_text("fighter_id,name\naaa111,Jon Jones\n", encoding="utf-8")
        cw = CrosswalkStore(tmp_path)
        espn = MagicMock()
        espn.fetch_athlete.return_value = {"displayName": "Luis Felipe Dias", "displayRecord": "18-5-0"}
        espn.fetch_athlete_records.side_effect = RuntimeError("unused")
        espn.fetch_athlete_eventlog.return_value = _ufc_eventlog(23)

        ufc_id, method = resolve_fighter_id(
            "4693161",
            "Luis Felipe Dias",
            crosswalk=cw,
            name_index=build_name_index_from_profiles(profiles),
            profiles_by_id={},
            espn=espn,
            auto_link_fuzzy=False,
        )
        assert method == "espn_veteran"
        assert not ufc_id.startswith("espn_")
        assert len(ufc_id) == 16
        assert cw.athlete_to_fighter["4693161"] == ufc_id

    def test_resolve_fighter_id_still_espn_new_for_debut(self):
        tmp_path = self.tmp_path
        profiles = tmp_path / "fighter_profiles.csv"
        profiles.write_text("fighter_id,name\n", encoding="utf-8")
        cw = CrosswalkStore(tmp_path)
        espn = MagicMock()
        espn.fetch_athlete.return_value = {"displayName": "True Debut", "displayRecord": "1-0-0"}
        espn.fetch_athlete_eventlog.return_value = _ufc_eventlog(1)

        ufc_id, method = resolve_fighter_id(
            "9999999",
            "True Debut",
            crosswalk=cw,
            name_index={},
            profiles_by_id={},
            espn=espn,
            auto_link_fuzzy=False,
        )
        assert method == "espn_new"
        assert ufc_id == "espn_9999999"


if __name__ == "__main__":
    unittest.main(verbosity=2)
