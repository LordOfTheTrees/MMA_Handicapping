#!/usr/bin/env python3
"""
Attribute holdout performance to the ELO lookahead leak fixed in ``ELOModel.get_state``.

Trains the model twice on identical data — once with the current point-in-time
``get_state``, once with the pre-fix implementation that returned terminal ELO for
every historical query — and prints the metric and coefficient-mass deltas. Because
both arms share data, config, and code, any difference is attributable to the leak
alone; data gaps (missing profiles, cohort size) cancel out.

Run from repo root::

    python scripts/dev/ab_elo_leak_attribution.py
    python scripts/dev/ab_elo_leak_attribution.py --data-dir ./data --holdout-start 2023-01-01

Bootstrap resamples are disabled: they feed prediction CIs via ``compute_prediction_ci``,
while scoring uses ``predict_proba_point_only``. Metrics are identical and the run is
~50x faster.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.schema import DataTier, ELOState  # noqa: E402
from src.elo.elo import ELOModel  # noqa: E402
from src.elo.kalman import kalman_predict  # noqa: E402
from src.eval.holdout_metrics import holdout_tier1_slice  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402

FIXED_GET_STATE = ELOModel.get_state


def leaky_get_state(self, fighter_id, wc, as_of_date=None):  # type: ignore[no-untyped-def]
    """The pre-fix implementation: reads terminal state regardless of ``as_of_date``."""
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


def _train_and_score(data_dir: Path, holdout_start: Optional[str], quiet: bool):
    cfg = Config()
    cfg.model.n_bootstrap = 0
    if holdout_start:
        from src.cli.common import resolve_date

        cfg.holdout_start_date = resolve_date(holdout_start)
    predictor = MMAPredictor(cfg)
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        predictor.load_data(data_dir)
        predictor.build_elo()
        predictor.train_regression()
    return holdout_tier1_slice(predictor), getattr(predictor, "training_regression_audit", None)


def _fractions(audit: Any, group: bool) -> Dict[str, float]:
    if not isinstance(audit, dict):
        return {}
    if group:
        out = audit.get("group_fraction_std_scaled") or audit.get("group_fraction") or {}
    else:
        out = (
            audit.get("per_feature_fraction_std_scaled")
            or audit.get("per_feature_fraction")
            or {}
        )
    if not isinstance(out, dict):
        from src.features.construction import FEATURE_NAMES

        out = dict(zip(FEATURE_NAMES, out))
    return {str(k): float(v) for k, v in out.items()}


def _print_delta_table(title: str, a: Dict[str, float], b: Dict[str, float], width: int) -> None:
    if not a and not b:
        return
    print(f"\n{title}")
    print(f"{'name':<{width}} {'leaky':>10} {'fixed':>10} {'change':>10}")
    for k in sorted(set(a) | set(b), key=lambda k: -b.get(k, 0.0)):
        va, vb = a.get(k, 0.0), b.get(k, 0.0)
        print(f"{k:<{width}} {va:>10.4f} {vb:>10.4f} {vb - va:>+10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--holdout-start", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--verbose", action="store_true", help="Show training logs from both arms.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results: Dict[str, Tuple[Any, Any, float]] = {}
    for label, impl in (("leaky", leaky_get_state), ("fixed", FIXED_GET_STATE)):
        ELOModel.get_state = impl  # type: ignore[method-assign]
        t0 = time.time()
        try:
            score, audit = _train_and_score(data_dir, args.holdout_start, not args.verbose)
        finally:
            ELOModel.get_state = FIXED_GET_STATE  # type: ignore[method-assign]
        results[label] = (score, audit, time.time() - t0)

    lo, hi = results["leaky"][0], results["fixed"][0]
    print("\n" + "=" * 80)
    print("ELO LEAK ATTRIBUTION — identical data and code, leak toggled")
    print("=" * 80)
    hdr = f"{'variant':<18} {'n':>6} {'log-loss':>10} {'Brier':>8} {'6-way':>8} {'W/L acc':>9} {'macroF1':>8}"
    print(hdr)
    for label in ("leaky", "fixed"):
        s = results[label][0]
        print(
            f"{label:<18} {s.n:>6,} {s.mean_log_loss:>10.4f} {s.mean_brier:>8.4f} "
            f"{s.accuracy:>7.2%} {s.wl_accuracy:>8.2%} {s.macro_f1:>8.4f}"
        )
    print("-" * 80)
    print(
        f"{'leak effect':<18} {'':>6} {lo.mean_log_loss - hi.mean_log_loss:>+10.4f} "
        f"{lo.mean_brier - hi.mean_brier:>+8.4f} {lo.accuracy - hi.accuracy:>+7.2%} "
        f"{lo.wl_accuracy - hi.wl_accuracy:>+8.2%} {lo.macro_f1 - hi.macro_f1:>+8.4f}"
    )
    print("(negative log-loss/Brier = the leak flattered the model)")

    _print_delta_table(
        "FEATURE FAMILY SHARE OF COEFFICIENT MASS",
        _fractions(results["leaky"][1], group=True),
        _fractions(results["fixed"][1], group=True),
        width=28,
    )
    _print_delta_table(
        "PER-FEATURE SHARE (std-scaled), sorted by fixed-model importance",
        _fractions(results["leaky"][1], group=False),
        _fractions(results["fixed"][1], group=False),
        width=32,
    )


if __name__ == "__main__":
    main()
