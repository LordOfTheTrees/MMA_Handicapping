"""
MMA Pre-Fight Prediction Model — Master Script

Commands
--------
train   Build ELO, construct features, fit regression. Saves model to disk.
        By default reads CSVs already in --data-dir (no refresh). Use
        --full-rebuild to run refresh_data() first (implement in src/data/refresh.py).
predict Produce a calibrated 6-class probability distribution for a matchup.
explain Show the exact additive decomposition of a prediction's log-odds.

Usage
-----
    python main.py train --data-dir ./data [--model-path model.pkl]
    python main.py train --data-dir ./data --full-rebuild
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
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from src.config import Config
from src.data.schema import WeightClass
from src.pipeline import MMAPredictor


# ---------------------------------------------------------------------------
# Weight class name resolution
# ---------------------------------------------------------------------------

_WC_ALIASES = {
    "strawweight":       WeightClass.STRAWWEIGHT,
    "straw":             WeightClass.STRAWWEIGHT,
    "flyweight":         WeightClass.FLYWEIGHT,
    "fly":               WeightClass.FLYWEIGHT,
    "bantamweight":      WeightClass.BANTAMWEIGHT,
    "bantam":            WeightClass.BANTAMWEIGHT,
    "featherweight":     WeightClass.FEATHERWEIGHT,
    "feather":           WeightClass.FEATHERWEIGHT,
    "lightweight":       WeightClass.LIGHTWEIGHT,
    "light":             WeightClass.LIGHTWEIGHT,
    "welterweight":      WeightClass.WELTERWEIGHT,
    "welter":            WeightClass.WELTERWEIGHT,
    "middleweight":      WeightClass.MIDDLEWEIGHT,
    "middle":            WeightClass.MIDDLEWEIGHT,
    "light_heavyweight": WeightClass.LIGHT_HEAVYWEIGHT,
    "lhw":               WeightClass.LIGHT_HEAVYWEIGHT,
    "heavyweight":       WeightClass.HEAVYWEIGHT,
    "heavy":             WeightClass.HEAVYWEIGHT,
    "hw":                WeightClass.HEAVYWEIGHT,
    "w_strawweight":     WeightClass.W_STRAWWEIGHT,
    "w_straw":           WeightClass.W_STRAWWEIGHT,
    "w_flyweight":       WeightClass.W_FLYWEIGHT,
    "w_fly":             WeightClass.W_FLYWEIGHT,
    "w_bantamweight":    WeightClass.W_BANTAMWEIGHT,
    "w_bantam":          WeightClass.W_BANTAMWEIGHT,
    "w_featherweight":   WeightClass.W_FEATHERWEIGHT,
    "w_feather":         WeightClass.W_FEATHERWEIGHT,
}


def resolve_weight_class(raw: str) -> WeightClass:
    wc = _WC_ALIASES.get(raw.strip().lower().replace("-", "_").replace(" ", "_"))
    if wc is None:
        valid = sorted(set(_WC_ALIASES.keys()))
        print(f"Unknown weight class: {raw!r}")
        print(f"Valid options: {', '.join(valid)}")
        sys.exit(1)
    return wc


def resolve_date(raw=None) -> date:
    if raw is None:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {raw!r}  (expected YYYY-MM-DD)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    if args.full_rebuild:
        from src.data.refresh import refresh_data

        print(f"Refreshing data -> {data_dir} ...")
        refresh_data(data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    config = Config()
    predictor = MMAPredictor(config)

    print(f"Stage 1: Loading data from {data_dir} ...")
    predictor.load_data(data_dir)
    print(f"  {len(predictor.fights):,} fight records loaded.")
    print(f"  {len(predictor.profiles):,} fighter profiles loaded.")

    print("Stage 2: Building ELO on full fight history ...")
    predictor.build_elo()
    print("  ELO construction complete.")

    print("Stages 3–5: Style features (per row) + multinomial regression ...")
    predictor.train_regression()
    n_train = len(predictor._y_train) if predictor._y_train is not None else 0
    print(f"  Training rows used: {n_train:,} decisive post-era fights.")

    print(f"Saving model -> {model_path}")
    predictor.save(model_path)
    print("Done.\n")


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
        print("Run:  python main.py train --data-dir ./data")
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

    # train
    p_train = sub.add_parser(
        "train",
        help="Train from CSVs in --data-dir (default: use existing files; no refresh)",
    )
    p_train.add_argument(
        "--data-dir", default="./data",
        help="Directory containing data CSVs (default: ./data)",
    )
    p_train.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Call refresh_data() first to repopulate CSVs, then train (see src/data/refresh.py)",
    )

    # predict
    p_pred = sub.add_parser("predict", help="Predict fight outcome probabilities")
    p_pred.add_argument("fighter_a", help="Fighter A identifier")
    p_pred.add_argument("fighter_b", help="Fighter B identifier")
    p_pred.add_argument("weight_class", help="Weight class name or alias")
    p_pred.add_argument(
        "--date", default=None,
        help="Fight date as YYYY-MM-DD (default: today)",
    )

    # explain
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
        "train":   cmd_train,
        "predict": cmd_predict,
        "explain": cmd_explain,
    }
    dispatch[args.command](args)
