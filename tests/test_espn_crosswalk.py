"""Crosswalk name/date matching (no network)."""
from datetime import date
from pathlib import Path

from src.data.espn_crosswalk import (
    BoutIdentity,
    CrosswalkStore,
    build_fight_index_from_csv,
    provision_ufcstats_fighter_id,
    remap_espn_placeholders_in_fight_rows,
    resolve_fight_id,
    resolve_fighter_id,
)


def test_resolve_fight_id_by_name_date(tmp_path: Path):
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


def test_resolve_fighter_id_by_name(tmp_path: Path):
    profiles = tmp_path / "fighter_profiles.csv"
    profiles.write_text("fighter_id,name\naaa111,Jon Jones\n", encoding="utf-8")
    from src.data.espn_crosswalk import build_name_index_from_profiles

    cw = CrosswalkStore(tmp_path)
    fid, method = resolve_fighter_id(
        "2335639",
        "Jon Jones",
        crosswalk=cw,
        name_index=build_name_index_from_profiles(profiles),
    )
    assert fid == "aaa111"
    assert method == "name"


def test_provision_and_remap_espn_placeholder(tmp_path: Path):
    fid = provision_ufcstats_fighter_id("4693161", {"aaa111"})
    assert len(fid) == 16
    assert fid != "aaa111"
    assert provision_ufcstats_fighter_id("4693161", {fid}) == fid

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


def test_resolve_fight_id_provisioned_when_both_hex(tmp_path: Path):
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
