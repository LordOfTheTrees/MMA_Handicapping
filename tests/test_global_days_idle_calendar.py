"""Layoff samples for reference export use fight-calendar history, not ELO terminal state."""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import DataTier, FightRecord, ResultMethod, WeightClass
from src.export.reference_distributions_export import (
    _fighter_global_fight_dates_sorted,
    days_idle_global_at_fight_date,
)


def _fight(
    i: int,
    a: str,
    b: str,
    d: date,
    *,
    winner: str | None = "a",
) -> FightRecord:
    win = a if winner == "a" else (b if winner == "b" else None)
    return FightRecord(
        fight_id=f"f{i}",
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=win,
        result_method=ResultMethod.UNANIMOUS_DECISION,
        weight_class=WeightClass.LIGHTWEIGHT,
        fight_date=d,
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


class TestGlobalDaysIdleCalendar(unittest.TestCase):
    def test_idle_is_gap_from_prior_bout_not_terminal_state(self) -> None:
        d0 = date(2020, 1, 1)
        fights = [
            _fight(1, "a", "x", d0),
            _fight(2, "a", "y", d0 + timedelta(days=100)),
            _fight(3, "a", "z", d0 + timedelta(days=400)),
        ]
        cal = _fighter_global_fight_dates_sorted(fights)
        self.assertEqual(days_idle_global_at_fight_date(cal, "a", d0), 0)
        self.assertEqual(days_idle_global_at_fight_date(cal, "a", d0 + timedelta(days=100)), 100)
        self.assertEqual(days_idle_global_at_fight_date(cal, "a", d0 + timedelta(days=400)), 300)


if __name__ == "__main__":
    unittest.main()
