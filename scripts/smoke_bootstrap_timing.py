#!/usr/bin/env python3
"""
Estimate wall time for a full train: point fit + 200 bootstrap refits.

Runs one real point fit with n_bootstrap=0, then times ``n_probe`` bootstrap
resamples on the same (X, y, weights) and extrapolates linearly to 200.

Usage (from repo root)::

    python scripts/smoke_bootstrap_timing.py --data-dir ./data --elo-cache ./data/elo_cache.pkl

Options match ``python -m src.cli.train`` / ``main.py train`` where useful; does not write a model file.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import resolve_date  # noqa: E402
from src.config import Config, DEFAULT_HOLDOUT_START_DATE  # noqa: E402
from src.confidence.intervals import (  # noqa: E402
    _resolve_bootstrap_max_workers,
    fit_bootstrap_coefficients,
)
from src.pipeline import MMAPredictor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Time point fit + bootstrap; extrapolate to n_bootstrap.")
    p.add_argument("--data-dir", default="./data", type=Path)
    p.add_argument(
        "--elo-cache",
        default=None,
        metavar="PATH",
        help="Same as ``python -m src.cli.train``: skip ELO rebuild when cache matches.",
    )
    p.add_argument("--no-holdout", action="store_true")
    p.add_argument("--holdout-start", default=None, metavar="YYYY-MM-DD")
    p.add_argument(
        "--n-probe",
        type=int,
        default=5,
        help="Number of bootstrap resamples to time after point fit (default: 5).",
    )
    p.add_argument(
        "--n-target",
        type=int,
        default=200,
        help="Extrapolate bootstrap cost to this many resamples (default: 200).",
    )
    p.add_argument(
        "--bootstrap-max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Overrides ModelConfig.bootstrap_max_workers for the probe (same as train).",
    )
    args = p.parse_args()

    if args.n_probe < 1:
        print("--n-probe must be >= 1", file=sys.stderr)
        sys.exit(2)

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    config = Config()
    if args.no_holdout:
        config.holdout_start_date = None
    elif args.holdout_start:
        config.holdout_start_date = resolve_date(args.holdout_start)

    config.model.n_bootstrap = 0

    predictor = MMAPredictor(config)
    print(f"Loading data from {data_dir} ...")
    predictor.load_data(data_dir)
    print(f"  {len(predictor.fights):,} fights, {len(predictor.profiles):,} profiles")

    elo_path = Path(args.elo_cache).resolve() if args.elo_cache else None
    if elo_path and predictor.try_load_elo_from_cache(elo_path):
        print(f"ELO from cache: {elo_path.name}")
    else:
        if elo_path and elo_path.exists():
            print("ELO cache stale or mismatch; rebuilding ELO ...")
        else:
            print("Building ELO (no usable cache) ...")
        predictor.build_elo()

    hsd = config.holdout_start_date
    if hsd is None:
        print("Holdout: none (--no-holdout)")
    else:
        print(f"Holdout: fight_date >= {hsd} excluded (default {DEFAULT_HOLDOUT_START_DATE})")

    print("\n--- Timing point fit (n_bootstrap=0) ---")
    t0 = time.perf_counter()
    predictor.train_regression(matrix_progress_every=0)
    t_point = time.perf_counter() - t0
    print(f"Point fit + matrix build: {t_point:.1f}s\n")

    if predictor._X_train is None or predictor._y_train is None:
        print("No training matrix; aborting.", file=sys.stderr)
        sys.exit(1)

    shp = predictor._X_train.shape
    print(f"Training matrix: {shp[0]:,} rows x {shp[1]} features")

    model_probe = copy.deepcopy(predictor.config.model)
    model_probe.n_bootstrap = args.n_probe
    if args.bootstrap_max_workers is not None:
        model_probe.bootstrap_max_workers = int(args.bootstrap_max_workers)

    mw_eff = _resolve_bootstrap_max_workers(model_probe, args.n_probe, None)
    print(
        f"--- Timing {args.n_probe} bootstrap resample(s); "
        f"effective_workers={mw_eff} (progress every 1) ---",
    )
    t1 = time.perf_counter()
    W_stack, n_valid = fit_bootstrap_coefficients(
        predictor._X_train,
        predictor._y_train,
        predictor._train_weights,
        model_probe,
        progress_every=1,
        max_workers=args.bootstrap_max_workers,
    )
    t_probe = time.perf_counter() - t1
    print(f"Probe bootstrap block: {t_probe:.1f}s  (valid fits: {n_valid} / {args.n_probe} attempts)")

    if n_valid < 1:
        print("No successful bootstrap fits in probe; cannot extrapolate per-resample time.", file=sys.stderr)
        sys.exit(1)

    # Valid fits may be fewer than n_probe if resamples lack all classes; scale by attempts for fairness.
    per_attempt = t_probe / float(args.n_probe)
    per_valid = t_probe / float(n_valid)
    target = args.n_target
    est_boot_total_attempts = per_attempt * target
    est_boot_total_valid = per_valid * target

    print("\n=== Extrapolation (linear; assumes similar mix of skip vs fit as probe) ===")
    print(
        f"  Time per bootstrap *attempt* (probe): ~{per_attempt:.2f}s  ->  "
        f"{target} attempts ~ {est_boot_total_attempts / 60:.1f} min"
    )
    print(
        f"  Time per *successful* fit (probe):     ~{per_valid:.2f}s  ->  "
        f"{target} valid fits ~ {est_boot_total_valid / 60:.1f} min"
    )
    print(
        f"  Estimated total train (point + bootstrap attempts): "
        f"~{t_point + est_boot_total_attempts:.0f}s  ({(t_point + est_boot_total_attempts) / 60:.1f} min)"
    )
    print(
        f"  (If every attempt must fit: ~{t_point + est_boot_total_valid:.0f}s; "
        f"skips are faster so actual is often between the two.)"
    )


if __name__ == "__main__":
    main()
