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
from src.elo.elo import ELOModel
from src.export.fighter_elo_trajectories import nested_elo_trajectories_by_fighter
from src.pipeline import MMAPredictor


def _fight(i: int, a: str, b: str, d: date, *, winner_a: bool = True) -> FightRecord:
    return FightRecord(
        fight_id=f"f{i}",
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=a if winner_a else b,
        result_method=ResultMethod.UNANIMOUS_DECISION,
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


if __name__ == "__main__":
    unittest.main()
