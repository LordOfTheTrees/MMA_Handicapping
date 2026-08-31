"""
Point-in-time correctness for :meth:`ELOModel.get_state`.

Regression guard for the lookahead bug where ``get_state(as_of_date=...)`` returned
the *terminal* rating for every historical query, leaking future fight results into
pre-fight training features. The reference oracle is a second model built only from
fights strictly before the query date — the definition of point-in-time.
"""

from __future__ import annotations

import pickle
import sys
import unittest
from datetime import date

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ELOConfig  # noqa: E402
from src.data.schema import (  # noqa: E402
    DataTier,
    FightRecord,
    FighterProfile,
    ResultMethod,
    WeightClass,
)
from src.elo.elo import ELOModel  # noqa: E402

WC = WeightClass.LIGHTWEIGHT


def _fight(i: int, a: str, b: str, winner: str | None, d: date) -> FightRecord:
    return FightRecord(
        fight_id=f"f{i}",
        fighter_a_id=a,
        fighter_b_id=b,
        winner_id=winner,
        result_method=ResultMethod.KO_TKO if winner else ResultMethod.NO_CONTEST,
        weight_class=WC,
        fight_date=d,
        promotion="UFC",
        tier=DataTier.TIER_1,
    )


def _career() -> list[FightRecord]:
    """'climber' wins three straight against fresh opponents, 2015 -> 2019."""
    return [
        _fight(1, "climber", "opp1", "climber", date(2015, 1, 1)),
        _fight(2, "climber", "opp2", "climber", date(2017, 1, 1)),
        _fight(3, "climber", "opp3", "climber", date(2019, 1, 1)),
    ]


class TestEloPointInTime(unittest.TestCase):
    def test_historical_query_matches_model_built_only_from_prior_fights(self) -> None:
        """The whole ballgame: PIT query == model trained on the strict past."""
        fights = _career()
        full = ELOModel(ELOConfig())
        full.process_fights(fights)

        for as_of in (date(2015, 1, 1), date(2017, 1, 1), date(2019, 1, 1)):
            oracle = ELOModel(ELOConfig())
            oracle.process_fights([f for f in fights if f.fight_date < as_of])
            self.assertAlmostEqual(
                full.get_state("climber", WC, as_of).elo,
                oracle.get_state("climber", WC, as_of).elo,
                places=9,
                msg=f"point-in-time ELO disagrees with strict-past rebuild at {as_of}",
            )

    def test_rating_is_not_frozen_across_history(self) -> None:
        """The original bug's signature: identical ELO at every historical date."""
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        early = full.get_state("climber", WC, date(2014, 1, 1)).elo
        late = full.get_state("climber", WC, date(2019, 1, 1)).elo
        self.assertNotAlmostEqual(early, late, places=6)
        self.assertAlmostEqual(early, ELOConfig().initial_elo, places=9)

    def test_query_before_debut_returns_cold_start_prior(self) -> None:
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        self.assertAlmostEqual(
            full.get_state("climber", WC, date(2010, 1, 1)).elo,
            ELOConfig().initial_elo,
            places=9,
        )
        st = full.get_state("climber", WC, date(2010, 1, 1))
        self.assertEqual(st.n_fights, 0)
        self.assertIsNone(st.last_fight_date)

    def test_pedigree_prior_survives_pre_debut_query(self) -> None:
        """A pre-debut PIT query must return the pedigree cold start, not a flat 1500."""
        profile = FighterProfile(
            fighter_id="climber",
            name="Climber",
            wrestling_pedigree=1.0,
        )
        full = ELOModel(ELOConfig())
        full.process_fights(_career(), {"climber": profile})
        self.assertAlmostEqual(
            full.get_state("climber", WC, date(2010, 1, 1)).elo,
            ELOConfig().initial_elo + 20.0,
            places=9,
        )

    def test_future_query_returns_terminal_state_for_live_inference(self) -> None:
        """Production prediction on an upcoming card must be unchanged by the fix."""
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        terminal = full.get_state("climber", WC).elo
        self.assertAlmostEqual(
            full.get_state("climber", WC, date(2026, 1, 1)).elo, terminal, places=9
        )

    def test_layoff_variance_grows_from_the_prior_bout_not_the_last_one(self) -> None:
        """Kalman inflation uses the gap to the previous bout, PIT — not to career end."""
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        mid_gap = full.get_state("climber", WC, date(2016, 1, 1)).uncertainty
        just_after = full.get_state("climber", WC, date(2015, 1, 2)).uncertainty
        self.assertGreater(mid_gap, just_after)

    def test_n_fights_and_last_fight_date_are_point_in_time(self) -> None:
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        st = full.get_state("climber", WC, date(2018, 1, 1))
        self.assertEqual(st.n_fights, 2)
        self.assertEqual(st.last_fight_date, date(2017, 1, 1))

    def test_no_contest_advances_the_clock_without_moving_rating(self) -> None:
        fights = [
            _fight(1, "climber", "opp1", "climber", date(2015, 1, 1)),
            _fight(2, "climber", "opp2", None, date(2016, 1, 1)),
        ]
        full = ELOModel(ELOConfig())
        full.process_fights(fights)
        before_nc = full.get_state("climber", WC, date(2016, 1, 1))
        after_nc = full.get_state("climber", WC, date(2016, 1, 2))
        self.assertAlmostEqual(before_nc.elo, after_nc.elo, places=9)
        self.assertEqual(after_nc.n_fights, 2)

    def test_legacy_pickle_refuses_historical_queries(self) -> None:
        """A pre-index artifact must fail loudly rather than answer with terminal ELO."""
        full = ELOModel(ELOConfig())
        full.process_fights(_career())

        state = dict(full.__dict__)
        for attr in ("_history", "_history_dates", "_initial_states", "_global_dates",
                     "_has_pit_history"):
            state.pop(attr, None)
        legacy = ELOModel.__new__(ELOModel)
        legacy.__setstate__(state)

        with self.assertRaises(RuntimeError):
            legacy.get_state("climber", WC, date(2016, 1, 1))

        # Forward-looking queries still work: terminal state is the right answer there.
        legacy.get_state("climber", WC, date(2030, 1, 1))

    def test_pickle_round_trip_preserves_point_in_time_answers(self) -> None:
        full = ELOModel(ELOConfig())
        full.process_fights(_career())
        revived = pickle.loads(pickle.dumps(full))
        for as_of in (date(2014, 1, 1), date(2017, 1, 1), date(2026, 1, 1)):
            self.assertAlmostEqual(
                full.get_state("climber", WC, as_of).elo,
                revived.get_state("climber", WC, as_of).elo,
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
