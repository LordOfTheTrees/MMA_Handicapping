"""
Per-weight-class ELO model with Kalman uncertainty tracking.

Architecture responsibilities (Section 4):
  - K-factor scaled by result certainty
  - Zero ELO movement for Draw / NC / DQ
  - Cross-promotion ELO transfer with tier-based discount
  - Cold start from pedigree priors
  - Time-based uncertainty growth via Kalman predict step (elapsed since last fight **any** division)
  - Dual role: quality weight for feature construction AND regression feature
"""
from collections import defaultdict
from datetime import date
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import ELOConfig
from ..data.schema import (
    DataTier, ELOState, FightRecord, FighterProfile,
    ResultMethod, WeightClass,
)
from .kalman import KalmanState, kalman_predict, kalman_update


# ---------------------------------------------------------------------------
# K-factor scaling by result certainty (architecture Section 4.2)
# ---------------------------------------------------------------------------

_K_SCALE: Dict[ResultMethod, float] = {
    ResultMethod.KO_TKO:               1.5,
    ResultMethod.SUBMISSION:           1.5,
    ResultMethod.UNANIMOUS_DECISION:   1.00,
    ResultMethod.SPLIT_DECISION:       0.50,
    ResultMethod.MAJORITY_DECISION:    0.50,
    ResultMethod.DRAW:                 0.00,
    ResultMethod.NO_CONTEST:           0.00,
    ResultMethod.DQ:                   0.00,
}


def result_k_scale(method: ResultMethod) -> float:
    """Return the K-factor multiplier for a given result method."""
    return _K_SCALE.get(method, 0.0)


# ---------------------------------------------------------------------------
# Core ELO mathematics
# ---------------------------------------------------------------------------

def expected_score(elo_a: float, elo_b: float, *, divisor: float) -> float:
    """Standard ELO expected score for Fighter A."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / divisor))


def _defaultdict_none() -> None:
    """Pickle-safe factory for ``defaultdict`` entries that should start as ``None``."""
    return None


def elo_delta(
    elo_a: float,
    elo_b: float,
    a_won: bool,
    method: ResultMethod,
    k_base: float,
    logistic_divisor: float,
) -> Tuple[float, float]:
    """
    Compute ELO change for both fighters after a decisive fight.

    Returns (delta_a, delta_b).
    For draws / NC / DQ both deltas are 0.0.
    """
    scale = result_k_scale(method)
    if scale == 0.0:
        return 0.0, 0.0

    e_a = expected_score(elo_a, elo_b, divisor=logistic_divisor)
    actual_a = 1.0 if a_won else 0.0
    delta_a = k_base * scale * (actual_a - e_a)
    return delta_a, -delta_a


# ---------------------------------------------------------------------------
# ELO model
# ---------------------------------------------------------------------------

class ELOModel:
    """
    Maintains per-fighter, per-weight-class ELO ratings with Kalman uncertainty.

    Typical usage:
        model = ELOModel(config)
        model.process_fights(fights_sorted_chronologically, profiles)
        state = model.get_state("fighter_id", WeightClass.LIGHTWEIGHT)
        elo   = model.get_elo("fighter_id", WeightClass.LIGHTWEIGHT, as_of_date=today)
    """

    def __init__(self, config: ELOConfig):
        self.config = config
        # (fighter_id, WeightClass) -> KalmanState
        self._states: Dict[Tuple[str, WeightClass], KalmanState] = {}
        self._last_fight: Dict[Tuple[str, WeightClass], Optional[date]] = defaultdict(_defaultdict_none)
        # Last cage date anywhere — drives Kalman time-update for the division being fought (ADR-15).
        self._last_fight_global: Dict[str, Optional[date]] = defaultdict(_defaultdict_none)
        self._n_fights: Dict[Tuple[str, WeightClass], int] = defaultdict(int)
        self._best_tier: Dict[Tuple[str, WeightClass], DataTier] = {}
        # Optional: (fighter_id, wc) -> [(fight_date, elo_after_fight, opponent_id), ...]
        self._trajectories: Dict[Tuple[str, WeightClass], List[Tuple[date, float, str]]] = {}
        self._record_trajectories: bool = False

    def __setstate__(self, state: dict) -> None:
        """Pickle migration: older models lack ``_last_fight_global``; backfill from per-division dates."""
        self.__dict__.update(state)
        if "_trajectories" not in self.__dict__:
            self._trajectories = {}
        if "_record_trajectories" not in self.__dict__:
            self._record_trajectories = False
        if "_last_fight_global" not in self.__dict__:
            g: Dict[str, Optional[date]] = defaultdict(_defaultdict_none)
            for (fid, _wc), d0 in list(self._last_fight.items()):
                if d0 is not None:
                    cur = g[fid]
                    if cur is None or d0 > cur:
                        g[fid] = d0
            self._last_fight_global = g

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(fighter_id: str, wc: WeightClass) -> Tuple[str, WeightClass]:
        return (fighter_id, wc)

    def _get_or_init(self, fighter_id: str, wc: WeightClass) -> KalmanState:
        key = self._key(fighter_id, wc)
        if key not in self._states:
            self._states[key] = KalmanState(
                value=self.config.initial_elo,
                variance=self.config.kalman_measurement_noise * 10.0,
            )
        return self._states[key]

    # ------------------------------------------------------------------
    # Cold start initialisation (architecture Section 4.4)
    # ------------------------------------------------------------------

    def initialize_from_pedigree(
        self,
        fighter_id: str,
        wc: WeightClass,
        profile: FighterProfile,
    ) -> None:
        """
        Adjust initial ELO from combat sports pedigree signals.
        Only applied if the fighter has no fight history yet in this weight class.
        Max boost ≈ 50 ELO points to stay conservative.
        """
        key = self._key(fighter_id, wc)
        if key in self._states:
            return  # already has observed data — don't override

        boost = (
            profile.wrestling_pedigree * 20.0
            + profile.boxing_pedigree  * 15.0
            + profile.bjj_pedigree     * 15.0
        )
        self._states[key] = KalmanState(
            value=self.config.initial_elo + boost,
            variance=self.config.kalman_measurement_noise * 12.0,
        )

    def transfer_from_tier(
        self,
        fighter_id: str,
        wc: WeightClass,
        source_elo: float,
        source_tier: DataTier,
    ) -> None:
        """
        Apply cross-promotion ELO transfer with tier-based discount.
        Only takes effect if this fighter has no existing state in wc.
        """
        key = self._key(fighter_id, wc)
        if key in self._states:
            return

        discount = self.config.tier_discount.get(source_tier.value, 0.65)
        transferred = (
            self.config.initial_elo
            + (source_elo - self.config.initial_elo) * discount
        )
        self._states[key] = KalmanState(
            value=transferred,
            variance=self.config.kalman_measurement_noise * 8.0,
        )

    # ------------------------------------------------------------------
    # Processing fights
    # ------------------------------------------------------------------

    def process_fights(
        self,
        fights: List[FightRecord],
        profiles: Optional[Dict[str, FighterProfile]] = None,
        *,
        progress_every: int = 0,
        record_trajectories: bool = False,
    ) -> None:
        """
        Process all fights in chronological order.

        fights MUST be sorted by fight_date ascending before calling this.
        Profiles are used only for pedigree-based cold starts.

        If *progress_every* > 0, print a line every N fights (and at start/end).

        If *record_trajectories* is True, after each fight append
        ``(fight_date, elo, opponent_fighter_id)`` for both corners in that bout's
        weight class — see :meth:`get_trajectory` (Kalman mean after the update).
        """
        self._record_trajectories = record_trajectories
        if record_trajectories:
            self._trajectories.clear()

        if profiles:
            n_prof = len(profiles)
            print(f"  [ELO] pedigree cold-start pass: {n_prof:,} profiles x divisions ...", flush=True)
            for fighter_id, profile in profiles.items():
                for wc in WeightClass:
                    self.initialize_from_pedigree(fighter_id, wc, profile)

        n = len(fights)
        if n == 0:
            return
        if progress_every > 0:
            print(f"  [ELO] processing {n:,} fights chronologically ...", flush=True)

        for i, fight in enumerate(fights, start=1):
            self._process_one(fight)
            if self._record_trajectories:
                self._append_trajectory_snapshots(fight)
            if progress_every > 0 and (i == 1 or i % progress_every == 0 or i == n):
                pct = 100.0 * i / n
                print(f"  [ELO] {i:,} / {n:,} fights ({pct:.1f}%)", flush=True)

    def _process_one(self, fight: FightRecord) -> None:
        wc = fight.weight_class
        a_id, b_id = fight.fighter_a_id, fight.fighter_b_id

        # Kalman predict step: grow uncertainty for both fighters (layoff vs last fight anywhere)
        for fid in (a_id, b_id):
            key = self._key(fid, wc)
            last = self._last_fight_global[fid]
            state = self._get_or_init(fid, wc)
            if last is not None:
                days = max(0, (fight.fight_date - last).days)
                state = kalman_predict(state, days, self.config.kalman_process_noise)
                self._states[key] = state

        state_a = self._get_or_init(a_id, wc)
        state_b = self._get_or_init(b_id, wc)

        # Draws / NC / DQ: update last fight date only (architecture Section 4.2)
        if fight.winner_id is None:
            for fid in (a_id, b_id):
                key = self._key(fid, wc)
                self._last_fight[key] = fight.fight_date
                self._last_fight_global[fid] = fight.fight_date
                self._n_fights[key] += 1
                self._update_best_tier(key, fight.tier)
            return

        a_won = fight.winner_id == a_id
        d_a, d_b = elo_delta(
            state_a.value,
            state_b.value,
            a_won,
            fight.result_method,
            self.config.k_base,
            self.config.logistic_divisor,
        )

        # Kalman measurement update
        new_a = kalman_update(state_a, state_a.value + d_a, self.config.kalman_measurement_noise)
        new_b = kalman_update(state_b, state_b.value + d_b, self.config.kalman_measurement_noise)

        key_a, key_b = self._key(a_id, wc), self._key(b_id, wc)
        self._states[key_a] = new_a
        self._states[key_b] = new_b
        for key, fid in ((key_a, a_id), (key_b, b_id)):
            self._last_fight[key] = fight.fight_date
            self._last_fight_global[fid] = fight.fight_date
            self._n_fights[key] += 1
            self._update_best_tier(key, fight.tier)

    def _append_trajectory_snapshots(self, fight: FightRecord) -> None:
        """Record post-fight Kalman ELO mean for both corners in this bout's weight class."""
        wc = fight.weight_class
        a_id, b_id = fight.fighter_a_id, fight.fighter_b_id
        for fid, opponent_id in ((a_id, b_id), (b_id, a_id)):
            if not fid:
                continue
            key = self._key(fid, wc)
            st = self._states.get(key)
            if st is None:
                continue
            if key not in self._trajectories:
                self._trajectories[key] = []
            self._trajectories[key].append((fight.fight_date, float(st.value), opponent_id or ""))

    def get_trajectory(self, fighter_id: str, wc: WeightClass) -> List[Tuple[date, float, str]]:
        """
        Return chronological points for one fighter in one division.

        Populated only if the last :meth:`process_fights` used ``record_trajectories=True``.
        Each tuple is ``(fight_date, elo_after_fight, opponent_fighter_id)`` — ELO is the Kalman
        mean immediately after that fight is processed.
        """
        key = self._key(fighter_id, wc)
        return list(self._trajectories.get(key, ()))

    def iter_trajectory_keys(self) -> Iterator[Tuple[str, WeightClass]]:
        """Yield (fighter_id, weight_class) keys that have recorded trajectory points."""
        yield from self._trajectories.keys()

    def _update_best_tier(self, key: Tuple[str, WeightClass], tier: DataTier) -> None:
        current = self._best_tier.get(key)
        if current is None or tier.value < current.value:
            self._best_tier[key] = tier

    # ------------------------------------------------------------------
    # Querying state
    # ------------------------------------------------------------------

    def get_state(
        self,
        fighter_id: str,
        wc: WeightClass,
        as_of_date: Optional[date] = None,
    ) -> ELOState:
        """
        Return ELO state for a fighter in a weight class.

        If as_of_date is given, applies the Kalman predict step for days
        elapsed since last fight WITHOUT modifying stored state.
        This ensures lookahead-free queries during training data construction.
        """
        key = self._key(fighter_id, wc)
        state = self._get_or_init(fighter_id, wc)

        if as_of_date is not None:
            last = self._last_fight_global[fighter_id]
            if last is not None:
                days = max(0, (as_of_date - last).days)
                if days > 0:
                    state = kalman_predict(state, days, self.config.kalman_process_noise)

        return ELOState(
            fighter_id=fighter_id,
            weight_class=wc,
            elo=state.value,
            uncertainty=state.variance,
            last_fight_date=self._last_fight[key],
            n_fights=self._n_fights[key],
            primary_tier=self._best_tier.get(key, DataTier.TIER_3),
        )

    def get_elo(
        self,
        fighter_id: str,
        wc: WeightClass,
        as_of_date: Optional[date] = None,
    ) -> float:
        """Return the ELO point estimate. Convenience wrapper over get_state()."""
        return self.get_state(fighter_id, wc, as_of_date).elo
