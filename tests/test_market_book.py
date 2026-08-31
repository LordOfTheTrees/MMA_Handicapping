"""Math and alignment tests for the post-hoc market book (no walk-forward, no network)."""
from __future__ import annotations

import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data.schema import DataTier, FightRecord, ResultMethod, WeightClass
from src.eval.market_book import (
    JSON_EXPORTS_DIR,
    PostedLines,
    american_to_decimal,
    assert_out_dir_allowed,
    card_slots_from_billed_index,
    edge,
    fill_method_hits,
    fill_two_way_hits,
    gender_of_weight_class,
    hex_id_from_url,
    implied_log_growth,
    jurek_row_to_posted,
    kelly_fraction,
    median_float,
    pick_max_edge,
    realized_flat_pnl,
    realized_multiplier,
    swap_method_decimals_to_a,
)


def _fight(*, a: str = "aaa", b: str = "bbb", fid: str = "deadbeef") -> FightRecord:
    return FightRecord(
        fight_id=fid,
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=a,
        result_method=ResultMethod.KO_TKO,
        weight_class=WeightClass.LIGHTWEIGHT,
        fight_date=date(2020, 1, 1),
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


class TestAmericanDecimal(unittest.TestCase):
    def test_plus_money(self):
        d = american_to_decimal(150.0)
        assert d is not None
        self.assertAlmostEqual(d, 2.5)

    def test_minus_money(self):
        d = american_to_decimal(-200.0)
        assert d is not None
        self.assertAlmostEqual(d, 1.5)

    def test_zero_rejected(self):
        self.assertIsNone(american_to_decimal(0.0))


class TestAlignAndIds(unittest.TestCase):
    def test_hex_from_fight_url(self):
        self.assertEqual(
            hex_id_from_url("http://ufcstats.com/fight-details/d215c4e6dc1346ae"),
            "d215c4e6dc1346ae",
        )

    def test_swap_method_when_f1_is_b(self):
        # f1 is B: A's KO is f2_ko → class 0
        out = swap_method_decimals_to_a(
            (10.0, 20.0, 30.0),
            (4.0, 5.0, 6.0),
            f1_is_a=False,
        )
        self.assertEqual(out, (4.0, 5.0, 6.0, 30.0, 10.0, 20.0))

    def test_jurek_swaps_when_fighter_1_is_b(self):
        fight = _fight(a="aaa", b="bbb")
        row = {
            "f1_id": "bbb",
            "f2_id": "aaa",
            "odds_1": 3.0,
            "odds_2": 1.4,
            "f1_ko": 10.0,
            "f1_sub": 20.0,
            "f1_dec": 30.0,
            "f2_ko": 4.0,
            "f2_sub": 5.0,
            "f2_dec": 6.0,
        }
        pl = jurek_row_to_posted(row, fight)
        assert pl is not None
        self.assertAlmostEqual(pl.d_a or 0.0, 1.4)
        self.assertAlmostEqual(pl.d_b or 0.0, 3.0)
        self.assertEqual(pl.method[0], 4.0)  # A KO = f2_ko
        self.assertEqual(pl.method[4], 10.0)  # B KO = f1_ko

    def test_median(self):
        self.assertEqual(median_float([1.2, 3.4, 2.0]), 2.0)
        self.assertEqual(median_float([1.0, 2.0]), 1.5)


class TestKellyAndEdge(unittest.TestCase):
    def test_edge_positive(self):
        self.assertAlmostEqual(edge(0.6, 2.0), 0.2)

    def test_kelly_matches_formula_and_clips(self):
        e = edge(0.6, 2.0)
        f = kelly_fraction(e, 2.0)
        self.assertAlmostEqual(f, 0.2)
        self.assertLess(kelly_fraction(10.0, 1.1), 1.0)
        self.assertEqual(kelly_fraction(-0.1, 2.0), 0.0)

    def test_plus_ev_skip_when_all_negative(self):
        pick = pick_max_edge(
            [("A", 0.4, 2.0), ("B", 0.6, 1.5)],
            {"A": True, "B": False},
        )
        # 0.4*2-1 = -0.2; 0.6*1.5-1 = -0.1
        self.assertIsNone(pick)

    def test_picks_largest_edge(self):
        pick = pick_max_edge(
            [("A", 0.7, 2.0), ("B", 0.3, 5.0)],
            {"A": True, "B": False},
        )
        assert pick is not None
        # 0.7*2-1 = 0.4; 0.3*5-1 = 0.5
        self.assertEqual(pick.contract, "B")
        self.assertAlmostEqual(pick.e, 0.5)

    def test_hit_miss_multipliers(self):
        f = 0.1
        d = 3.0
        self.assertAlmostEqual(realized_multiplier(True, f, d), 1.0 + 0.2)
        self.assertAlmostEqual(realized_multiplier(False, f, d), 0.9)
        self.assertAlmostEqual(realized_flat_pnl(True, d), 2.0)
        self.assertAlmostEqual(realized_flat_pnl(False, d), -1.0)

    def test_log_growth_zero_stake(self):
        self.assertEqual(implied_log_growth(0.6, 0.0, 2.0), 0.0)

    def test_log_growth_finite(self):
        g = implied_log_growth(0.6, 0.2, 2.0)
        self.assertTrue(math.isfinite(g))
        self.assertGreater(g, 0.0)

    def test_two_way_hits_follow_y(self):
        self.assertEqual(fill_two_way_hits(0)["A"], True)
        self.assertEqual(fill_two_way_hits(4)["A"], False)
        self.assertTrue(fill_method_hits(1)["A_sub"])
        self.assertFalse(fill_method_hits(1)["A_ko"])


class TestPostedLinesAndOutDir(unittest.TestCase):
    def test_has_method_requires_all_six(self):
        pl = PostedLines(source="jurek", d_a=1.5, d_b=2.5, method=(2.0, 3.0, 4.0, None, 6.0, 7.0))
        self.assertTrue(pl.has_two_way())
        self.assertFalse(pl.has_method())

    def test_refuse_json_exports(self):
        with self.assertRaises(SystemExit):
            assert_out_dir_allowed(JSON_EXPORTS_DIR)
        with self.assertRaises(SystemExit):
            assert_out_dir_allowed(JSON_EXPORTS_DIR / "nested")

    def test_allow_market_eval(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "market_eval"
            got = assert_out_dir_allowed(p)
            self.assertEqual(got, p.resolve())


class TestSlices(unittest.TestCase):
    def test_gender_from_weight_class(self):
        self.assertEqual(gender_of_weight_class(WeightClass.W_FLYWEIGHT), "women")
        self.assertEqual(gender_of_weight_class(WeightClass.WELTERWEIGHT), "men")
        self.assertEqual(gender_of_weight_class(WeightClass.CATCH_WEIGHT), "other")

    def test_card_slots_13_fight_card(self):
        self.assertEqual(
            card_slots_from_billed_index(0, 13, is_title=True),
            ("title", "main_event", "main_card"),
        )
        self.assertEqual(
            card_slots_from_billed_index(4, 13, is_title=False),
            ("main_card",),
        )
        self.assertEqual(
            card_slots_from_billed_index(5, 13, is_title=False),
            ("prelim_main_event",),
        )
        self.assertEqual(
            card_slots_from_billed_index(6, 13, is_title=False),
            ("generic_prelims",),
        )

    def test_short_card_no_prelims(self):
        self.assertEqual(
            card_slots_from_billed_index(0, 5, is_title=False),
            ("main_event", "main_card"),
        )
        self.assertEqual(card_slots_from_billed_index(4, 5, is_title=False), ("main_card",))

    def test_doubleheader_skips_position(self):
        self.assertEqual(card_slots_from_billed_index(0, 21, is_title=True), ("title",))
        self.assertEqual(card_slots_from_billed_index(None, 12, is_title=False), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
