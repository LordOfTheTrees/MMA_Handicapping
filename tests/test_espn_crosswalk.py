"""Crosswalk name/date matching (no network)."""
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data.espn_crosswalk import (
    BoutIdentity,
    CrosswalkStore,
    build_fight_index_from_csv,
    build_name_index_from_profiles,
    provision_ufcstats_fighter_id,
    remap_espn_placeholders_in_fight_rows,
    resolve_fight_id,
    resolve_fighter_id,
)


class TestEspnCrosswalk(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_resolve_fight_id_by_name_date(self):
        tmp_path = self.tmp_path
        fights = tmp_path / "ufcstats_fights.csv"
        profiles = tmp_path / "fighter_profiles.csv"
        fights.write_text(
            "fight_id,fighter_a_id,fighter_b_id,winner_id,method,weight_class,date\n"
            "deadbeef,aaa111,bbb222,aaa111,ko/tko,lightweight,2024-11-16\n",
            encoding="utf-8",
        )
        profiles.write_text(
            "fighter_id,name\naaa111,Jon Jones\nbbb222,Stipe Miocic\n",
            encoding="utf-8",
        )
        index = build_fight_index_from_csv(fights, profiles)
        cw = CrosswalkStore(tmp_path)
        bout = BoutIdentity(
            espn_competition_id="401706833",
            espn_event_id="600049124",
            event_date=date(2024, 11, 16),
            espn_athlete_ids=("1", "2"),
            fighter_names=("Jon Jones", "Stipe Miocic"),
        )
        fid, method = resolve_fight_id(bout, crosswalk=cw, fight_index=index)
        assert fid == "deadbeef"
        assert method == "name_date"
        assert cw.competition_to_fight["401706833"] == "deadbeef"

    def test_resolve_fighter_id_by_name(self):
        tmp_path = self.tmp_path
        profiles = tmp_path / "fighter_profiles.csv"
        profiles.write_text("fighter_id,name\naaa111,Jon Jones\n", encoding="utf-8")

        cw = CrosswalkStore(tmp_path)
        fid, method = resolve_fighter_id(
            "2335639",
            "Jon Jones",
            crosswalk=cw,
            name_index=build_name_index_from_profiles(profiles),
        )
        assert fid == "aaa111"
        assert method == "name"

    def test_provision_and_remap_espn_placeholder(self):
        tmp_path = self.tmp_path
        fid = provision_ufcstats_fighter_id("4693161", {"aaa111"})
        assert len(fid) == 16
        assert fid != "aaa111"
        # Deterministic for a fixed (espn_athlete_id, taken) pair — real call sites
        # (resolve_fighter_id, _repair_espn_veteran_placeholders) only ever provision an
        # athlete not yet in the crosswalk, so `taken` never legitimately contains this
        # athlete's own id; that self-collision scenario isn't a case the function needs
        # to handle, only same-input-same-output.
        assert provision_ufcstats_fighter_id("4693161", {"aaa111"}) == fid

        cw = CrosswalkStore(tmp_path)
        cw.record_fighter(
            ufcstats_fighter_id=fid,
            espn_athlete_id="4693161",
            fighter_name="Luis Felipe Dias",
            match_method="espn_veteran",
        )
        rows = {
            "espn_99": {
                "fight_id": "espn_99",
                "fighter_a_id": "espn_4693161",
                "fighter_b_id": "bbb222",
                "winner_id": "espn_4693161",
            }
        }
        assert remap_espn_placeholders_in_fight_rows(rows, cw) == 2
        assert rows["espn_99"]["fighter_a_id"] == fid
        assert rows["espn_99"]["winner_id"] == fid

    def test_resolve_fight_id_provisioned_when_both_hex(self):
        tmp_path = self.tmp_path
        fights = tmp_path / "ufcstats_fights.csv"
        profiles = tmp_path / "fighter_profiles.csv"
        fights.write_text(
            "fight_id,fighter_a_id,fighter_b_id,winner_id,method,weight_class,date\n",
            encoding="utf-8",
        )
        profiles.write_text(
            "fighter_id,name\naaa111,Jon Jones\nbbb222,Stipe Miocic\n",
            encoding="utf-8",
        )
        index = build_fight_index_from_csv(fights, profiles)
        cw = CrosswalkStore(tmp_path)
        bout = BoutIdentity(
            espn_competition_id="401873935",
            espn_event_id="600058517",
            event_date=date(2026, 5, 30),
            espn_athlete_ids=("1", "2"),
            fighter_names=("Rodrigo Vera", "Zhu Kangjie"),
        )
        fid, method = resolve_fight_id(
            bout,
            crosswalk=cw,
            fight_index=index,
            fighter_a_id="aaa111",
            fighter_b_id="bbb222",
            taken_fight_ids=set(),
        )
        assert method == "espn_provisioned"
        assert not fid.startswith("espn_")
        assert len(fid) == 16
        assert cw.competition_to_fight["401873935"] == fid


if __name__ == "__main__":
    unittest.main(verbosity=2)
