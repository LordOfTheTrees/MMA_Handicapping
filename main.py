"""
MMA Pre-Fight Prediction Model — Master Script

Commands
--------
train   Build ELO, construct features, fit regression. Saves model to disk.
        Implementation: ``src/cli/train.py`` (same behavior as ``scripts/train_model.py``).
        By default reads CSVs in --data-dir (no network refresh). Regression uses
        a time holdout (default 2023-01-01) unless you pass --no-holdout. Use
        --full-rebuild to re-scrape first, unless --no-scrape or
        --skip-refresh-if-present. --elo-cache loads/saves a PIT ELO cache.
predict Produce a calibrated 6-class probability distribution for a matchup.
explain Show the exact additive decomposition of a prediction's log-odds.

Usage
-----
    python main.py train --data-dir ./data [--model-path model.pkl]
    python scripts/train_model.py
    python main.py train ... --no-holdout
    python main.py train ... --holdout-start 2022-06-01
    python main.py train --data-dir ./data --full-rebuild
    python main.py eval-holdout
    python main.py predict <fighter_a> <fighter_b> <weight_class> [--date YYYY-MM-DD]
    python main.py explain <fighter_a> <fighter_b> <weight_class> [--date YYYY-MM-DD]

Weight class aliases (case-insensitive)
---------------------------------------
    strawweight, flyweight, bantamweight, featherweight,
    lightweight, welterweight, middleweight, lhw, heavyweight
    w_strawweight, w_flyweight, w_bantamweight, w_featherweight

Examples
--------
    python main.py train --data-dir ./data
    python main.py predict fighter_001 fighter_002 lightweight
    python main.py predict fighter_001 fighter_002 lightweight --date 2025-06-01
    python main.py explain fighter_001 fighter_002 lightweight
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.cli.common import resolve_date, resolve_weight_class
from src.cli.train import cmd_train, register_train_arguments
from src.pipeline import MMAPredictor


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_eval_holdout(args: argparse.Namespace) -> None:
    mp = getattr(args, "eval_model_path", None) or args.model_path
    predictor = _load_or_exit(Path(mp))
    if predictor.config.holdout_start_date is None:
        print(
            "This model has holdout_start_date=None (trained with --no-holdout).\n"
            "Re-train with a time holdout (default) or:  "
            "python main.py train --data-dir ./data --holdout-start YYYY-MM-DD",
        )
        sys.exit(1)
    from src.eval.holdout_metrics import run_holdout_eval

    n, ll, brier, acc = run_holdout_eval(predictor)
    hsd = predictor.config.holdout_start_date
    print(f"\nHoldout evaluation  (fight_date >= {hsd})")
    print(f"  Tier-1 decisive fights (A perspective): {n:,}")
    if n == 0:
        print("  No rows to score. Check holdout_start_date vs data.")
        return
    print(f"  Mean log-loss: {ll:.4f}")
    print(f"  Mean Brier:    {brier:.4f}")
    print(f"  Accuracy:      {acc:.2%}\n")


def cmd_predict(args: argparse.Namespace) -> None:
    predictor = _load_or_exit(Path(args.model_path))
    wc = resolve_weight_class(args.weight_class)
    fdate = resolve_date(args.date)

    print(f"\n{args.fighter_a}  vs  {args.fighter_b}  |  {wc.value}  |  {fdate}\n")
    result = predictor.predict(args.fighter_a, args.fighter_b, wc, fdate, verbose=True)

    print(f"\nDerived:")
    print(f"  Total win %    {result.total_win:.2f}")
    print(f"  Finish win %   {result.finish_win:.2f}")
    print(f"  Finish lose %  {result.finish_lose:.2f}")
    print(f"  Decision %     {result.go_to_decision:.2f}\n")


def cmd_explain(args: argparse.Namespace) -> None:
    predictor = _load_or_exit(Path(args.model_path))
    wc = resolve_weight_class(args.weight_class)
    fdate = resolve_date(args.date)
    predictor.explain(args.fighter_a, args.fighter_b, wc, fdate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_exit(model_path: Path) -> MMAPredictor:
    if not model_path.exists():
        print(f"No trained model found at {model_path}.")
        print("Run:  python main.py train --data-dir ./data  or  python scripts/train_model.py")
        sys.exit(1)
    return MMAPredictor.load(model_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MMA Pre-Fight Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        default="model.pkl",
        help="Path to the saved model file (default: model.pkl)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train",
        help="Train from CSVs in --data-dir. Default: time holdout (see --holdout-start / --no-holdout).",
    )
    register_train_arguments(p_train)

    p_eval = sub.add_parser(
        "eval-holdout",
        help="Log-loss, Brier, accuracy on the holdout slice (model must have a time holdout unless legacy).",
    )
    p_eval.add_argument(
        "--model-path",
        dest="eval_model_path",
        default=None,
        metavar="PATH",
        help="Model pickle (overrides top-level --model-path if set)",
    )

    p_pred = sub.add_parser("predict", help="Predict fight outcome probabilities")
    p_pred.add_argument("fighter_a", help="Fighter A identifier")
    p_pred.add_argument("fighter_b", help="Fighter B identifier")
    p_pred.add_argument("weight_class", help="Weight class name or alias")
    p_pred.add_argument(
        "--date", default=None,
        help="Fight date as YYYY-MM-DD (default: today)",
    )

    p_exp = sub.add_parser("explain", help="Print log-odds decomposition for a matchup")
    p_exp.add_argument("fighter_a", help="Fighter A identifier")
    p_exp.add_argument("fighter_b", help="Fighter B identifier")
    p_exp.add_argument("weight_class", help="Weight class name or alias")
    p_exp.add_argument(
        "--date", default=None,
        help="Fight date as YYYY-MM-DD (default: today)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train":        cmd_train,
        "eval-holdout": cmd_eval_holdout,
        "predict":      cmd_predict,
        "explain":      cmd_explain,
    }
    dispatch[args.command](args)
