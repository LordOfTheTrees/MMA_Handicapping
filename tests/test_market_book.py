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
    BookAccum,
    FightSliceTags,
    PostedLines,
    StakePick,
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
    pick_model_favorite,
    realized_flat_pnl,
    realized_flat_pnl_simul,
    realized_multiplier,
    realized_multiplier_simul,
    simul_fight_from_candidates,
    simultaneous_kelly_fractions,
    swap_method_decimals_to_a,
    two_way_overround,
    _kelly_path,
)
from src.eval.tuning_plots import _priced_series


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

    def test_favorite_does_not_shop_the_dog(self):
        hits = {"A": True, "B": False}
        pick = pick_model_favorite(0.7, 2.0, 0.3, 5.0, hits)
        assert pick is not None
        self.assertEqual(pick.contract, "A")
        self.assertAlmostEqual(pick.e, 0.4)
        max_e = pick_max_edge([("A", 0.7, 2.0), ("B", 0.3, 5.0)], hits)
        assert max_e is not None
        self.assertEqual(max_e.contract, "B")

    def test_favorite_skips_when_preferred_side_not_plus_ev(self):
        pick = pick_model_favorite(0.55, 1.5, 0.45, 3.0, {"A": True, "B": False})
        self.assertIsNone(pick)

    def test_favorite_tie_is_no_bet(self):
        pick = pick_model_favorite(0.5, 2.1, 0.5, 2.1, {"A": True, "B": False})
        self.assertIsNone(pick)

    def test_underround_flag(self):
        self.assertLess(two_way_overround(3.6, 4.6), 1.0)
        self.assertGreater(two_way_overround(1.91, 1.91), 1.0)

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

    def test_simul_two_way_matches_isolated(self):
        p, d = 0.6, 2.0
        f_iso = kelly_fraction(edge(p, d), d)
        fs = simultaneous_kelly_fractions([p, 1.0 - p], [d, 2.0])
        self.assertAlmostEqual(fs[0], f_iso)
        self.assertEqual(fs[1], 0.0)

    def test_simul_abstains_when_both_negative(self):
        fs = simultaneous_kelly_fractions([0.4, 0.6], [2.0, 1.5])
        self.assertEqual(fs, [0.0, 0.0])

    def test_simul_splits_two_plus_ev_method_classes(self):
        ps = [0.40, 0.25, 0.10, 0.10, 0.10, 0.05]
        qs = [0.33, 0.15, 0.20, 0.20, 0.15, 0.10]
        ds = [1.0 / q for q in qs]
        fs = simultaneous_kelly_fractions(ps, ds)
        self.assertGreater(fs[0], 0.0)
        self.assertGreater(fs[1], 0.0)
        self.assertEqual(sum(1 for f in fs if f > 0.0), 2)
        pick = pick_max_edge(
            [(str(i), ps[i], ds[i]) for i in range(6)],
            {str(i): False for i in range(6)},
        )
        assert pick is not None
        self.assertEqual(pick.contract, "1")

    def test_simul_fight_one_leg_matches_isolated_multiplier(self):
        fight = simul_fight_from_candidates(
            [("A", 0.6, 2.0), ("B", 0.4, 2.0)],
            {"A": True, "B": False},
        )
        assert fight is not None
        self.assertEqual(len(fight.legs), 1)
        f = kelly_fraction(edge(0.6, 2.0), 2.0)
        self.assertAlmostEqual(realized_multiplier_simul(fight, 1.0), realized_multiplier(True, f, 2.0))
        self.assertAlmostEqual(realized_flat_pnl_simul(fight), realized_flat_pnl(True, 2.0))

    def test_quarter_kelly_scales_mean_f(self):
        pk = StakePick(contract="A", p=0.6, decimal_odds=2.0, e=0.2, hit=True)
        full = _kelly_path([pk], 1.0)
        quarter = _kelly_path([pk], 0.25)
        self.assertAlmostEqual(quarter["mean_f"], full["mean_f"] * 0.25)

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


class TestOddsTapes(unittest.TestCase):
    def test_mdabbert_fill_does_not_enter_jurek_buckets(self):
        acc = BookAccum()
        tags = FightSliceTags(gender="men", weight_class="lightweight", card=())
        jurek_pick = StakePick("A", 0.6, 2.0, 0.2, True)
        fill_pick = StakePick("B", 0.4, 3.0, 0.2, False)
        acc.add_fight(tags, jurek_pick, True, None, False, source="jurek")
        acc.add_fight(tags, fill_pick, True, fill_pick, True, source="mdabbert")
        rep = acc.as_report()
        self.assertEqual(rep["odds_tape"], "jurek")
        self.assertEqual(rep["two_way"]["n_priced"], 1)
        self.assertEqual(rep["two_way"]["n_plus_ev"], 1)
        self.assertEqual(rep["two_way_favorite"]["n_priced"], 0)
        self.assertEqual(rep["method"]["n_priced"], 0)
        self.assertEqual(rep["mdabbert_fill"]["two_way"]["n_priced"], 1)
        self.assertEqual(rep["mdabbert_fill"]["method"]["n_priced"], 1)
        self.assertEqual(rep["mdabbert_fill"]["method"]["n_plus_ev"], 1)

    def test_favorite_bucket_separate_from_max_edge(self):
        acc = BookAccum()
        tags = FightSliceTags(gender="men", weight_class="lightweight", card=())
        max_pick = StakePick("B", 0.3, 5.0, 0.5, False)
        fav_pick = StakePick("A", 0.7, 2.0, 0.4, True)
        acc.add_fight(
            tags, max_pick, True, None, False, source="jurek",
            tw_fav=fav_pick, tw_fav_priced=True,
        )
        rep = acc.as_report()
        self.assertEqual(rep["two_way"]["n_plus_ev"], 1)
        self.assertEqual(rep["two_way_favorite"]["n_plus_ev"], 1)
        self.assertEqual(rep["two_way"]["n_priced"], 1)
        self.assertEqual(rep["two_way_favorite"]["n_priced"], 1)
        self.assertAlmostEqual(rep["two_way_favorite"]["mean_model_p"], 0.7)
        self.assertAlmostEqual(rep["two_way"]["mean_model_p"], 0.3)

    def test_priced_series_gaps_unpriced_years(self):
        years_map = {
            "2024": {"method": {"n_priced": 10, "full_kelly": {"realized_log_growth": -1.0}}},
            "2025": {"method": {"n_priced": 0, "full_kelly": {"realized_log_growth": 99.0}}},
        }
        got = _priced_series(years_map, [2024, 2025], "method", "full_kelly", "realized_log_growth")
        self.assertAlmostEqual(got[0], -1.0)
        self.assertTrue(got[1] != got[1])  # NaN: do not plot the 99 fake splice

    def test_priced_series_fill_key_is_separate(self):
        years_map = {
            "2025": {
                "method": {"n_priced": 0},
                "mdabbert_fill": {
                    "method": {"n_priced": 5, "flat_1u": {"realized_roi": 1.0}},
                },
            }
        }
        jurek = _priced_series(years_map, [2025], "method", "flat_1u", "realized_roi")
        fill = _priced_series(
            years_map, [2025], "method", "flat_1u", "realized_roi", fill_key="mdabbert_fill"
        )
        self.assertTrue(jurek[0] != jurek[0])
        self.assertAlmostEqual(fill[0], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
