"""Unit tests for ESPN → CSV normalization (no network)."""
from datetime import date

from src.data.espn_normalize import (
    build_fight_csv_row,
    espn_method_to_csv,
    fight_time_sec_from_status,
    normalize_fighter_name,
    weight_class_from_note,
)


def test_espn_method_mapping():
    assert espn_method_to_csv("decision---unanimous") == "unanimous decision"
    assert espn_method_to_csv("kotko") == "ko/tko"
    assert espn_method_to_csv("submission") == "submission"


def test_weight_class_from_note():
    assert weight_class_from_note("Lightweight - Main Event") == "lightweight"
    assert weight_class_from_note("Women's Strawweight Bout") == "women's strawweight"


def test_fight_time_sec_elapsed_clock():
    status = {"period": 3, "clock": 269.0}
    assert fight_time_sec_from_status(status, round_length_sec=300) == 869


def test_fight_time_sec_full_distance():
    status = {"period": 5, "clock": 300.0}
    assert fight_time_sec_from_status(status, round_length_sec=300) == 1500


def test_normalize_fighter_name_strips_diacritics():
    assert normalize_fighter_name("José Aldo") == normalize_fighter_name("Jose Aldo")


def test_build_fight_csv_row_corner_order():
    row = build_fight_csv_row(
        fight_id="abc",
        event_date=date(2024, 11, 16),
        fighter_a_id="zzz",
        fighter_b_id="aaa",
        winner_id="aaa",
        method="ko/tko",
        weight_class="heavyweight",
        fight_time_sec=100,
        side_a={"sig_landed": 10, "sig_attempted": 20, "td_landed": 1, "td_attempted": 2, "ctrl_sec": 30, "sub_attempts": 0},
        side_b={"sig_landed": 5, "sig_attempted": 8, "td_landed": 0, "td_attempted": 1, "ctrl_sec": 10, "sub_attempts": 1},
    )
    assert row["fighter_a_id"] == "aaa"
    assert row["fighter_b_id"] == "zzz"
    assert row["a_sig_str_landed"] == 5
    assert row["b_sig_str_landed"] == 10
    assert row["a_sig_str_absorbed"] == 10
    assert row["b_sig_str_absorbed"] == 5
