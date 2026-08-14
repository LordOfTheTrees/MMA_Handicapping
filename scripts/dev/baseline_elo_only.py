#!/usr/bin/env python3
"""
Generative ELO-only baseline (docs/model-efficacy-vs-baselines.md section 3).

Section 3 defined this baseline but no implementation was committed, so its published
numbers could not be reproduced or re-run after the point-in-time ELO fix. This script
implements the documented specification:

1. Point-in-time ELO for A and B at ``fight_date`` (same Kalman build as the pipeline,
   frozen ``logistic_divisor`` from config).
2. Binary win probability ``p_win = expected_score(E_a, E_b)``.
3. Method priors independent of opponent given win/loss: global empirical six-way label
   frequencies conditional on A winning vs A losing, estimated on Tier-1 decisive fights
   with ``fight_date < holdout_start`` (post-era filter), pooled across divisions.

Then ``P(y | A wins) = p_win * pi_win(y)`` normalized within classes {0,1,2} and
``P(y | A loses) = (1 - p_win) * pi_lose(y)`` normalized within {3,4,5}.

With ``--ab`` the baseline is scored twice, once with the current point-in-time
``get_state`` and once with the pre-fix implementation, isolating how much of the
baseline's published strength came from the lookahead leak.

Run from repo root::

    python scripts/dev/baseline_elo_only.py --ab
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.loader import filter_tier1_post_era  # noqa: E402
from src.data.schema import DataTier, ELOState, FightRecord  # noqa: E402
from src.elo.elo import ELOModel, expected_score  # noqa: E402
from src.elo.kalman import kalman_predict  # noqa: E402
from src.eval.fight_scoring import Tier1SliceScore, tier1_slice_score_from_probs  # noqa: E402
from src.model.regression import encode_outcome  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402

WIN_CLASSES = (0, 1, 2)
LOSS_CLASSES = (3, 4, 5)
FIXED_GET_STATE = ELOModel.get_state


def leaky_get_state(self, fighter_id, wc, as_of_date=None):  # type: ignore[no-untyped-def]
    """Pre-fix implementation: terminal state regardless of ``as_of_date``."""
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


def estimate_method_priors(
    train_fights: List[FightRecord],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Empirical six-way label frequencies given A wins and given A loses."""
    win_counts = np.zeros(3, dtype=float)
    loss_counts = np.zeros(3, dtype=float)
    for f in train_fights:
        y = encode_outcome(f, f.fighter_a_id)
        if y is None:
            continue
        if y in WIN_CLASSES:
            win_counts[y] += 1.0
        else:
            loss_counts[y - 3] += 1.0
    n_win, n_loss = int(win_counts.sum()), int(loss_counts.sum())
    if n_win == 0 or n_loss == 0:
        raise RuntimeError("No decisive training fights for method priors")
    return win_counts / n_win, loss_counts / n_loss, n_win, n_loss


def score_elo_only(
    predictor: MMAPredictor,
    holdout_fights: List[FightRecord],
    pi_win: np.ndarray,
    pi_loss: np.ndarray,
) -> Tier1SliceScore:
    divisor = predictor.config.elo.logistic_divisor
    rows: List[np.ndarray] = []
    y_true: List[int] = []
    kept: List[FightRecord] = []
    for f in holdout_fights:
        y = encode_outcome(f, f.fighter_a_id)
        if y is None:
            continue
        ea = predictor.elo_model.get_elo(f.fighter_a_id, f.weight_class, as_of_date=f.fight_date)
        eb = predictor.elo_model.get_elo(f.fighter_b_id, f.weight_class, as_of_date=f.fight_date)
        p_win = expected_score(ea, eb, divisor=divisor)
        p = np.empty(6, dtype=float)
        p[0:3] = p_win * pi_win
        p[3:6] = (1.0 - p_win) * pi_loss
        rows.append(p)
        y_true.append(y)
        kept.append(f)
    return tier1_slice_score_from_probs(y_true, np.vstack(rows), fights=kept)


def build(data_dir: Path, holdout_start: Optional[str]) -> Tuple[MMAPredictor, List[FightRecord], List[FightRecord]]:
    cfg = Config()
    if holdout_start:
        from src.cli.common import resolve_date

        cfg.holdout_start_date = resolve_date(holdout_start)
    predictor = MMAPredictor(cfg)
    predictor.load_data(data_dir)
    predictor.build_elo()
    hsd = cfg.holdout_start_date
    cand = filter_tier1_post_era(predictor.fights, cfg.master_start_year)
    return predictor, [f for f in cand if f.fight_date < hsd], [f for f in cand if f.fight_date >= hsd]


def _print(label: str, s: Tier1SliceScore) -> None:
    print(
        f"{label:<18} {s.n:>6,} {s.mean_log_loss:>10.4f} {s.mean_brier:>8.4f} "
        f"{s.accuracy:>7.2%} {s.mean_wl_log_loss:>9.4f} {s.wl_accuracy:>8.2%} {s.macro_f1:>8.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--holdout-start", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--ab", action="store_true", help="Also score with the pre-fix leaky get_state.")
    args = ap.parse_args()

    variants = (("leaky", leaky_get_state), ("fixed", FIXED_GET_STATE)) if args.ab else (("fixed", FIXED_GET_STATE),)
    results: Dict[str, Tier1SliceScore] = {}
    priors_note = ""
    for label, impl in variants:
        ELOModel.get_state = impl  # type: ignore[method-assign]
        try:
            predictor, train_fights, holdout_fights = build(args.data_dir, args.holdout_start)
            pi_win, pi_loss, n_win, n_loss = estimate_method_priors(train_fights)
            results[label] = score_elo_only(predictor, holdout_fights, pi_win, pi_loss)
        finally:
            ELOModel.get_state = FIXED_GET_STATE  # type: ignore[method-assign]
        priors_note = (
            f"  priors from {n_win + n_loss:,} decisive pre-holdout fights "
            f"(A wins {n_win:,} / A loses {n_loss:,})\n"
            f"  P(KO|win)={pi_win[0]:.4f}  P(sub|win)={pi_win[1]:.4f}  P(dec|win)={pi_win[2]:.4f}\n"
            f"  P(dec|loss)={pi_loss[0]:.4f}  P(KO|loss)={pi_loss[1]:.4f}  P(sub|loss)={pi_loss[2]:.4f}"
        )

    print("\n" + "=" * 88)
    print("GENERATIVE ELO-ONLY BASELINE (docs section 3)")
    print("=" * 88)
    print(priors_note)
    print(
        f"\n{'variant':<18} {'n':>6} {'log-loss':>10} {'Brier':>8} {'6-way':>8} "
        f"{'WL LL':>9} {'W/L acc':>8} {'macroF1':>8}"
    )
    for label, _ in variants:
        _print(label, results[label])
    print(f"{'uniform 6-way':<18} {'':>6} {math.log(6):>10.4f} {'':>8} {1/6:>7.2%} {math.log(2):>9.4f} {0.5:>8.2%}")
    if args.ab:
        lo, hi = results["leaky"], results["fixed"]
        print("-" * 88)
        print(
            f"{'leak effect':<18} {'':>6} {lo.mean_log_loss - hi.mean_log_loss:>+10.4f} "
            f"{lo.mean_brier - hi.mean_brier:>+8.4f} {lo.accuracy - hi.accuracy:>+7.2%} "
            f"{lo.mean_wl_log_loss - hi.mean_wl_log_loss:>+9.4f} "
            f"{lo.wl_accuracy - hi.wl_accuracy:>+8.2%} {lo.macro_f1 - hi.macro_f1:>+8.4f}"
        )


if __name__ == "__main__":
    main()
