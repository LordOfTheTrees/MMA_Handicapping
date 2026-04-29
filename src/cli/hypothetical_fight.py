#!/usr/bin/env python3
"""
Hypothetical matchup with optional assumed idle days (Cauchy gamma for CI width only).

Usage (repo root)::

    python -m src.cli.hypothetical_fight --model-path ./data/model.pkl --default-demo
    python -m src.cli.hypothetical_fight --model-path ./data/model.pkl \\
        --fighter-a-id ... --fighter-b-id ... --weight-class featherweight --date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
from datetime import date as date_cls
from pathlib import Path

from src.cli.common import resolve_date, resolve_weight_class
from src.hypothetical import (
    DEFAULT_DEMO_DAYS_IDLE_A,
    DEFAULT_DEMO_DAYS_IDLE_B,
    HypotheticalFightSpec,
    predict_hypothetical,
    predict_hypothetical_default_pair,
)
from src.pipeline import MMAPredictor


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Hypothetical fight prediction. --idle-a / --idle-b set calendar days idle for "
            "Cauchy gamma (CI width); point odds still use history through --date."
        )
    )
    p.add_argument("--model-path", type=Path, required=True)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--default-demo",
        action="store_true",
        help=(
            "Use built-in pair + default idle days (same defaults as src.cli.plot_prediction_three_viz; "
            "featherweight)."
        ),
    )
    p.add_argument("--fighter-a-id", default=None)
    p.add_argument("--fighter-b-id", default=None)
    p.add_argument("--weight-class", default="featherweight")
    p.add_argument("--date", default=None, help="Scheduled fight date YYYY-MM-DD (default: today)")
    p.add_argument("--idle-a", type=int, default=None, help="Hypothetical idle days, corner A (gamma only)")
    p.add_argument("--idle-b", type=int, default=None, help="Hypothetical idle days, corner B (gamma only)")
    args = p.parse_args()

    pred = MMAPredictor.load(args.model_path.resolve())
    fdate = resolve_date(args.date) if args.date else None

    if args.default_demo:
        predict_hypothetical_default_pair(
            pred,
            fight_date=fdate,
            days_idle_a=args.idle_a if args.idle_a is not None else DEFAULT_DEMO_DAYS_IDLE_A,
            days_idle_b=args.idle_b if args.idle_b is not None else DEFAULT_DEMO_DAYS_IDLE_B,
            verbose=True,
        )
        return

    if not args.fighter_a_id or not args.fighter_b_id:
        p.error("Give --fighter-a-id and --fighter-b-id, or use --default-demo.")
    wc = resolve_weight_class(args.weight_class)
    fd = fdate if fdate is not None else date_cls.today()
    spec = HypotheticalFightSpec(
        fighter_a_id=args.fighter_a_id,
        fighter_b_id=args.fighter_b_id,
        weight_class=wc,
        fight_date=fd,
        days_idle_a=args.idle_a,
        days_idle_b=args.idle_b,
    )
    predict_hypothetical(pred, spec, verbose=True)


if __name__ == "__main__":
    main()
