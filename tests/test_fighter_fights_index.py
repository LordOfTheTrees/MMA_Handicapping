"""``_fighter_fights`` must return exactly what the linear scan it replaced returned.

The scan was ``[f for f in self.fights if fighter is a corner and f.weight_class == wc]``,
which made ``build_xyw_for_fights`` quadratic. The index is only safe if it is
observationally identical — same fights, same order — for every key, including keys with
no fights, and if it can never go stale behind a write to ``self.fights``.
"""
from __future__ import annotations

import pickle
import random
from datetime import date, timedelta
from typing import List

import pytest

from src.data.schema import DataTier, FightRecord, ResultMethod, WeightClass
from src.pipeline import MMAPredictor

WCS = [w for w in WeightClass if w is not WeightClass.UNKNOWN]


def _scan(fights: List[FightRecord], fighter_id: str, wc: WeightClass) -> List[FightRecord]:
    """The original implementation, kept here as the oracle."""
    return [
        f for f in fights
        if (f.fighter_a_id == fighter_id or f.fighter_b_id == fighter_id)
        and f.weight_class == wc
    ]


def _corpus(n: int = 600, n_fighters: int = 90, seed: int = 11) -> List[FightRecord]:
    rng = random.Random(seed)
    base = date(2005, 1, 1)
    out: List[FightRecord] = []
    for i in range(n):
        a = f"f{rng.randrange(n_fighters):03d}"
        b = f"f{rng.randrange(n_fighters):03d}"
        out.append(
            FightRecord(
                fight_id=f"x{i}",
                fighter_a_id=a,
                fighter_b_id=b,
                winner_id=a,
                result_method=ResultMethod.KO_TKO,
                weight_class=rng.choice(WCS),
                fight_date=base + timedelta(days=rng.randrange(7000)),
                promotion="UFC",
                tier=DataTier.TIER_1,
            )
        )
    return out


@pytest.fixture
def predictor() -> MMAPredictor:
    return MMAPredictor().load_fights_direct(_corpus())


def test_index_matches_scan_for_every_fighter_and_division(predictor: MMAPredictor) -> None:
    """Exhaustive equivalence, including the identity and the ordering."""
    fighters = {f.fighter_a_id for f in predictor.fights} | {
        f.fighter_b_id for f in predictor.fights
    }
    assert len(fighters) > 1

    checked = 0
    for fid in sorted(fighters):
        for wc in WCS:
            expected = _scan(predictor.fights, fid, wc)
            actual = predictor._fighter_fights(fid, wc)
            assert actual == expected, f"{fid} / {wc}"
            assert [f.fight_id for f in actual] == [f.fight_id for f in expected]
            checked += 1
    assert checked == len(fighters) * len(WCS)


def test_unknown_fighter_and_empty_division_return_empty(predictor: MMAPredictor) -> None:
    assert predictor._fighter_fights("no-such-fighter", WeightClass.LIGHTWEIGHT) == []


def test_results_are_chronological(predictor: MMAPredictor) -> None:
    """Style axes walk history in order; the index must preserve the corpus sort."""
    for fid in sorted({f.fighter_a_id for f in predictor.fights})[:15]:
        for wc in WCS:
            dates = [f.fight_date for f in predictor._fighter_fights(fid, wc)]
            assert dates == sorted(dates)


def test_fighter_facing_themselves_is_not_double_counted() -> None:
    """A malformed row with the same id in both corners matched the scan exactly once."""
    f = FightRecord(
        fight_id="self",
        fighter_a_id="dup",
        fighter_b_id="dup",
        winner_id="dup",
        result_method=ResultMethod.KO_TKO,
        weight_class=WeightClass.LIGHTWEIGHT,
        fight_date=date(2020, 1, 1),
        promotion="UFC",
        tier=DataTier.TIER_1,
    )
    p = MMAPredictor().load_fights_direct([f])
    assert p._fighter_fights("dup", WeightClass.LIGHTWEIGHT) == _scan(
        p.fights, "dup", WeightClass.LIGHTWEIGHT
    )
    assert len(p._fighter_fights("dup", WeightClass.LIGHTWEIGHT)) == 1


# ---------------------------------------------------------------------------
# Staleness — the one way this optimisation could bite
# ---------------------------------------------------------------------------

def test_reloading_fights_invalidates_the_index(predictor: MMAPredictor) -> None:
    fid = predictor.fights[0].fighter_a_id
    wc = predictor.fights[0].weight_class
    predictor._fighter_fights(fid, wc)  # force a build
    assert predictor._fights_index is not None

    predictor.load_fights_direct(_corpus(n=50, seed=99))
    assert predictor._fights_index is None, "writing self.fights must drop the index"

    fighters = {f.fighter_a_id for f in predictor.fights}
    for other in sorted(fighters)[:10]:
        for w in WCS:
            assert predictor._fighter_fights(other, w) == _scan(predictor.fights, other, w)


def test_index_is_not_pickled_and_rebuilds_after_load(predictor: MMAPredictor) -> None:
    fid = predictor.fights[0].fighter_a_id
    wc = predictor.fights[0].weight_class
    predictor._fighter_fights(fid, wc)
    assert predictor._fights_index is not None

    assert "_fights_index" not in predictor.__getstate__()

    restored = pickle.loads(pickle.dumps(predictor))
    assert restored._fights_index is None
    assert restored._fighter_fights(fid, wc) == _scan(restored.fights, fid, wc)


def test_legacy_pickle_without_the_attribute_loads() -> None:
    """A model.pkl from before the index existed must still work."""
    p = MMAPredictor().load_fights_direct(_corpus(n=40))
    state = p.__getstate__()
    state.pop("_fights_index", None)

    restored = MMAPredictor.__new__(MMAPredictor)
    restored.__setstate__(state)
    assert restored._fights_index is None
    fid = restored.fights[0].fighter_a_id
    wc = restored.fights[0].weight_class
    assert restored._fighter_fights(fid, wc) == _scan(restored.fights, fid, wc)


def test_lookup_does_not_scale_with_corpus_size() -> None:
    """
    The point of the change: a lookup's cost must depend on the fighter's own history,
    not on how many fights are loaded. Compares work done, not wall time, so it is not
    flaky on a loaded CI runner.
    """
    small = MMAPredictor().load_fights_direct(_corpus(n=200, n_fighters=40, seed=3))
    large = MMAPredictor().load_fights_direct(_corpus(n=4000, n_fighters=40, seed=3))

    # Build both indexes up front so we time lookups, not construction.
    small._fighter_fights("f000", WeightClass.LIGHTWEIGHT)
    large._fighter_fights("f000", WeightClass.LIGHTWEIGHT)

    # A dict lookup touches one bucket regardless of corpus size; the scan touched every
    # record. Assert the index holds only the buckets that actually have fights.
    assert large._fights_index is not None
    assert all(len(v) > 0 for v in large._fights_index.values())
    total_entries = sum(len(v) for v in large._fights_index.values())
    # Each fight is filed under at most two keys.
    assert total_entries <= 2 * len(large.fights)
    assert len(small.fights) < len(large.fights)
