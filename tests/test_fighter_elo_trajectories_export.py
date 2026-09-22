"""ELO trajectory nesting for ``fighter_profiles.json`` export."""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_artifacts  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.schema import DataTier, FightRecord, FighterProfile, ResultMethod, WeightClass
from src.elo.elo import ELOModel, TrajectoryPoint
from src.export.fighter_elo_trajectories import nested_elo_trajectories_by_fighter
from src.model.regression import CLASS_LABELS
from src.pipeline import MMAPredictor


def _fight(
    i: int,
    a: str,
    b: str,
    d: date,
    *,
    winner_a: bool = True,
    method: ResultMethod = ResultMethod.UNANIMOUS_DECISION,
    decisive: bool = True,
) -> FightRecord:
    return FightRecord(
        fight_id=f"f{i}",
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=(a if winner_a else b) if decisive else None,
        result_method=method,
        weight_class=WeightClass.LIGHTWEIGHT,
        fight_date=d,
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


class TestFighterEloTrajectoriesExport(unittest.TestCase):
    def test_nested_elo_trajectories_shape(self) -> None:
        fa, fb = "id_alpha", "id_beta"
        d0 = date(2022, 1, 1)
        fights = [
            _fight(1, fa, fb, d0),
            _fight(2, fa, fb, d0 + timedelta(days=200), winner_a=False),
        ]
        em = ELOModel(Config().elo)
        em.process_fights(fights, None, record_trajectories=True)
        nested = nested_elo_trajectories_by_fighter(em)
        self.assertIn(fa, nested)
        self.assertIn(WeightClass.LIGHTWEIGHT.value, nested[fa])
        series = nested[fa][WeightClass.LIGHTWEIGHT.value]
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0]["opponent_fighter_id"], fb)
        self.assertEqual(series[0]["fight_date"], d0.isoformat())
        self.assertIsInstance(series[0]["elo"], float)

    def test_export_fighter_profiles_merges_elo_trajectories(self) -> None:
        fa, fb = "id_alpha", "id_beta"
        d0 = date(2023, 6, 1)
        fights = [_fight(1, fa, fb, d0)]
        p = MMAPredictor(Config())
        p.profiles = {
            fa: FighterProfile(fighter_id=fa, name="Alpha"),
            fb: FighterProfile(fighter_id=fb, name="Beta"),
            "lonely_nofights": FighterProfile(fighter_id="lonely_nofights", name="No Fights"),
        }
        p.load_fights_direct(fights)
        p.build_elo(record_trajectories=True)
        doc = export_artifacts._export_fighter_profiles(p, {"export_schema_version": "mma-handicapping-export-v1"})
        profs = doc["profiles"]
        self.assertIn("elo_trajectories", profs[fa])
        self.assertIn("elo_trajectories", profs[fb])
        self.assertIn(WeightClass.LIGHTWEIGHT.value, profs[fa]["elo_trajectories"])
        self.assertEqual(len(profs[fa]["elo_trajectories"][WeightClass.LIGHTWEIGHT.value]), 1)
        self.assertNotIn("elo_trajectories", profs["lonely_nofights"])


class TestTrajectoryOutcomes(unittest.TestCase):
    """Each point carries the bout that produced it, from that fighter's perspective."""

    def _series(self, fights: list[FightRecord], fighter_id: str) -> list[dict]:
        em = ELOModel(Config().elo)
        em.process_fights(fights, None, record_trajectories=True)
        nested = nested_elo_trajectories_by_fighter(em)
        return nested[fighter_id][WeightClass.LIGHTWEIGHT.value]

    def test_point_carries_fight_id_method_and_class(self) -> None:
        fa, fb = "id_alpha", "id_beta"
        fights = [_fight(7, fa, fb, date(2024, 3, 2), method=ResultMethod.KO_TKO)]
        point = self._series(fights, fa)[0]
        self.assertEqual(point["fight_id"], "f7")
        self.assertEqual(point["result_method"], "ko_tko")
        self.assertEqual(point["outcome_class"], 0)
        self.assertEqual(CLASS_LABELS[point["outcome_class"]], "Win by KO/TKO")

    def test_both_corners_record_the_same_bout_from_opposite_sides(self) -> None:
        """The loser's point must not inherit the winner's class."""
        fa, fb = "id_alpha", "id_beta"
        fights = [_fight(8, fa, fb, date(2024, 4, 6), method=ResultMethod.SUBMISSION)]
        winner = self._series(fights, fa)[0]
        loser = self._series(fights, fb)[0]
        self.assertEqual(winner["fight_id"], loser["fight_id"])
        self.assertEqual(winner["result_method"], loser["result_method"])
        self.assertEqual(winner["outcome_class"], 1)
        self.assertEqual(loser["outcome_class"], 5)
        self.assertEqual(CLASS_LABELS[1], "Win by Submission")
        self.assertEqual(CLASS_LABELS[5], "Lose by Submission")

    def test_every_decisive_method_maps_to_the_right_class(self) -> None:
        fa, fb = "id_alpha", "id_beta"
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
                fights = [_fight(9, fa, fb, date(2024, 5, 4), winner_a=a_wins, method=method)]
                self.assertEqual(self._series(fights, fa)[0]["outcome_class"], expected)

    def test_draw_and_no_contest_have_a_method_but_no_class(self) -> None:
        fa, fb = "id_alpha", "id_beta"
        for method in (ResultMethod.DRAW, ResultMethod.NO_CONTEST, ResultMethod.DQ):
            with self.subTest(method=method):
                fights = [_fight(10, fa, fb, date(2024, 6, 1), method=method, decisive=False)]
                point = self._series(fights, fa)[0]
                self.assertEqual(point["result_method"], method.value)
                self.assertIsNone(point["outcome_class"])

    def test_legacy_three_tuple_points_widen_instead_of_raising(self) -> None:
        """A pickle recorded before outcomes existed must still export, with null outcome fields."""
        em = ELOModel(Config().elo)
        key = em._key("id_alpha", WeightClass.LIGHTWEIGHT)
        em._trajectories[key] = [(date(2019, 1, 1), 1500.0, "id_beta")]
        point = nested_elo_trajectories_by_fighter(em)["id_alpha"][WeightClass.LIGHTWEIGHT.value][0]
        self.assertEqual(point["fight_date"], "2019-01-01")
        self.assertEqual(point["elo"], 1500.0)
        self.assertEqual(point["opponent_fighter_id"], "id_beta")
        self.assertIsNone(point["fight_id"])
        self.assertIsNone(point["result_method"])
        self.assertIsNone(point["outcome_class"])

    def test_plot_helpers_still_index_the_first_three_positions(self) -> None:
        """Charts read p[0], p[1], p[2]; appended fields must not shift them."""
        p = TrajectoryPoint(date(2024, 7, 1), 1510.5, "id_beta", "f11", "id_alpha", ResultMethod.KO_TKO)
        self.assertEqual(p[0], date(2024, 7, 1))
        self.assertEqual(p[1], 1510.5)
        self.assertEqual(p[2], "id_beta")


if __name__ == "__main__":
    unittest.main()
