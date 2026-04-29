"""
Training command: ``main.py train`` and ``python -m src.cli.train`` (same flags; module adds top-level ``--model-path``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..config import Config, DEFAULT_HOLDOUT_START_DATE
from ..pipeline import MMAPredictor
from .common import resolve_date


def register_train_arguments(p: argparse.ArgumentParser) -> None:
    """All train flags (``--data-dir`` through ``--holdout-start``)."""
    p.add_argument(
        "--data-dir", default="./data",
        help="Directory containing data CSVs (default: ./data)",
    )
    p.add_argument(
        "--full-rebuild",
        action="store_true",
        help="With default scrape: call refresh_data() to repopulate CSVs, then train (see src/data/refresh.py)",
    )
    p.add_argument(
        "--no-scrape",
        action="store_true",
        help="With --full-rebuild: do not run refresh_data (use existing ufcstats CSVs in --data-dir)",
    )
    p.add_argument(
        "--skip-refresh-if-present",
        action="store_true",
        help="With --full-rebuild: if ufcstats_fights.csv already exists in --data-dir, skip network refresh",
    )
    p.add_argument(
        "--elo-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a pickle cache of the ELO model. If the file exists and matches loaded fight count "
        "and ELOConfig, skip ELO rebuild; otherwise rebuild and write the cache. Example: data/elo_cache.pkl",
    )
    p.add_argument(
        "--no-holdout",
        action="store_true",
        help="Do not hold out a recent window: train regression on all Tier-1 post-era rows. "
        "For shipping / special only; the default is a time holdout for eval-holdout.",
    )
    p.add_argument(
        "--holdout-start",
        default=None,
        metavar="YYYY-MM-DD",
        help=f"Exclude Tier-1 fights on/after this date from regression training. "
        f"Default in Config: {DEFAULT_HOLDOUT_START_DATE} (if --no-holdout is not set).",
    )
    p.add_argument(
        "--bootstrap-max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap L-BFGS worker processes (default from ModelConfig.bootstrap_max_workers; "
        "unset means min(n_bootstrap, cpu_count-1); 1 forces serial).",
    )


def build_train_parser() -> argparse.ArgumentParser:
    """
    Parser for ``python -m src.cli.train``: same train flags as ``main.py train`` plus top-level ``--model-path``.
    """
    p = argparse.ArgumentParser(
        prog="python -m src.cli.train",
        description="Train the MMA model. Same flags as: python main.py train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        default="model.pkl",
        help="Path to the saved model file (default: model.pkl)",
    )
    register_train_arguments(p)
    return p


def cmd_train(args: argparse.Namespace) -> None:
    """Run full load → ELO (or cache) → regression → save. Expects *args* from register_train_arguments + model_path."""
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    if args.full_rebuild and not args.no_scrape:
        from ..data.refresh import refresh_data
        from ..data.ufcstats_scraper import DEFAULT_UFCSTATS_FIGHTS_CSV

        fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
        if args.skip_refresh_if_present and fights_path.exists():
            print(
                f"Skipping refresh_data ({fights_path.name} already in {data_dir}). "
                "Omit --skip-refresh-if-present to re-scrape; use --no-scrape to always skip a network refresh.",
            )
        else:
            print(f"Refreshing data -> {data_dir} ...")
            refresh_data(data_dir)
    elif args.full_rebuild and args.no_scrape:
        print("Skipping refresh_data (--no-scrape).")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    config = Config()
    if getattr(args, "no_holdout", False):
        config.holdout_start_date = None
    elif getattr(args, "holdout_start", None) is not None:
        config.holdout_start_date = resolve_date(args.holdout_start)
    hsd = config.holdout_start_date
    if hsd is None:
        print("  Holdout: none — regression uses all Tier-1 post-era rows (--no-holdout).")
    else:
        print(
            f"  Holdout: exclude Tier-1 training rows with fight_date >= {hsd} "
            f"(default {DEFAULT_HOLDOUT_START_DATE}; override with --holdout-start).",
        )
    if getattr(args, "bootstrap_max_workers", None) is not None:
        config.model.bootstrap_max_workers = int(args.bootstrap_max_workers)

    predictor = MMAPredictor(config)

    print(f"Stage 1: Loading data from {data_dir} ...")
    predictor.load_data(data_dir)
    print(f"  {len(predictor.fights):,} fight records loaded.")
    print(f"  {len(predictor.profiles):,} fighter profiles loaded.")

    elo_cache_path: Optional[Path] = None
    if getattr(args, "elo_cache", None):
        elo_cache_path = Path(args.elo_cache).resolve()

    if elo_cache_path and predictor.try_load_elo_from_cache(elo_cache_path):
        print(
            f"Stage 2: Loaded ELO from cache ({elo_cache_path.name}) — "
            "skipping full ELO rebuild (same fight count and ELOConfig as this run).",
        )
    else:
        if elo_cache_path and elo_cache_path.exists():
            print(
                "  ELO cache present but stale (data length or ELOConfig changed); "
                "rebuilding ELO ...",
            )
        print("Stage 2: Building ELO on full fight history ...")
        predictor.build_elo()
        print("  ELO construction complete.")
        if elo_cache_path:
            predictor.save_elo_cache(elo_cache_path)
            print(f"  Wrote ELO cache -> {elo_cache_path}")

    print("Stages 3–5: Style features (per row) + multinomial regression ...")
    predictor.train_regression()
    n_train = len(predictor._y_train) if predictor._y_train is not None else 0
    print(f"  Training rows used: {n_train:,} decisive post-era fights.")

    print(f"Saving model -> {model_path}")
    predictor.save(model_path)
    print("Done.\n")


def main() -> None:
    args = build_train_parser().parse_args()
    cmd_train(args)


if __name__ == "__main__":
    main()
