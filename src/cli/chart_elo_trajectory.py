#!/usr/bin/env python3
"""
Build ELO with trajectory recording and plot ELO vs time for one fighter or export all.

Each point is labeled with the **opponent's** name (from ``fighter_profiles.csv`` when
available); the chart title identifies the fighter whose ELO is plotted.

Requires matplotlib. Trajectory points are the Kalman mean **after** each fight in that
weight class (same as :meth:`ELOModel.get_elo` at fight day without lookahead).

Usage (repo root)::

    python -m src.cli.chart_elo_trajectory --data-dir ./data --name "Jon Jones"
    python -m src.cli.chart_elo_trajectory --data-dir ./data --fighter-id <uuid> --weight-class lightweight
    python -m src.cli.chart_elo_trajectory --data-dir ./data --export-all ./data/elo_trajectories --max-files 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.data.fighter_names import require_fighter_id
from src.data.schema import WeightClass
from src.elo.trajectory_charts import (
    TrajectoryPoints,
    export_all_trajectory_charts,
    plot_elo_trajectories_overlay,
    plot_elo_trajectory,
    save_trajectory_figure,
)
from src.pipeline import MMAPredictor

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Reuse the same aliases as main.py (avoid importing main as a module).
_WC_ALIASES = {
    "strawweight": WeightClass.STRAWWEIGHT,
    "straw": WeightClass.STRAWWEIGHT,
    "flyweight": WeightClass.FLYWEIGHT,
    "fly": WeightClass.FLYWEIGHT,
    "bantamweight": WeightClass.BANTAMWEIGHT,
    "bantam": WeightClass.BANTAMWEIGHT,
    "featherweight": WeightClass.FEATHERWEIGHT,
    "feather": WeightClass.FEATHERWEIGHT,
    "lightweight": WeightClass.LIGHTWEIGHT,
    "light": WeightClass.LIGHTWEIGHT,
    "welterweight": WeightClass.WELTERWEIGHT,
    "welter": WeightClass.WELTERWEIGHT,
    "middleweight": WeightClass.MIDDLEWEIGHT,
    "middle": WeightClass.MIDDLEWEIGHT,
    "light_heavyweight": WeightClass.LIGHT_HEAVYWEIGHT,
    "lhw": WeightClass.LIGHT_HEAVYWEIGHT,
    "heavyweight": WeightClass.HEAVYWEIGHT,
    "heavy": WeightClass.HEAVYWEIGHT,
    "hw": WeightClass.HEAVYWEIGHT,
    "catch_weight": WeightClass.CATCH_WEIGHT,
    "catch": WeightClass.CATCH_WEIGHT,
    "w_strawweight": WeightClass.W_STRAWWEIGHT,
    "w_straw": WeightClass.W_STRAWWEIGHT,
    "w_flyweight": WeightClass.W_FLYWEIGHT,
    "w_fly": WeightClass.W_FLYWEIGHT,
    "w_bantamweight": WeightClass.W_BANTAMWEIGHT,
    "w_bantam": WeightClass.W_BANTAMWEIGHT,
    "w_featherweight": WeightClass.W_FEATHERWEIGHT,
    "w_feather": WeightClass.W_FEATHERWEIGHT,
}


def _parse_weight_class(raw: str) -> WeightClass:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    wc = _WC_ALIASES.get(key)
    if wc is None:
        print(f"Unknown weight class: {raw!r}", file=sys.stderr)
        print(f"Valid options: {', '.join(sorted(_WC_ALIASES))}", file=sys.stderr)
        raise SystemExit(2)
    return wc


def _series_for_fighter(
    predictor: MMAPredictor,
    fighter_id: str,
    only_wc: WeightClass | None,
) -> list[tuple[str, TrajectoryPoints]]:
    em = predictor.elo_model
    assert em is not None
    series: list[tuple[str, TrajectoryPoints]] = []
    if only_wc is not None:
        pts = em.get_trajectory(fighter_id, only_wc)
        if pts:
            series.append((only_wc.value, pts))
        return series

    for wc in WeightClass:
        if wc == WeightClass.UNKNOWN:
            continue
        pts = em.get_trajectory(fighter_id, wc)
        if pts:
            series.append((wc.value, pts))
    return series


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot ELO trajectory (requires trajectory recording during ELO build)")
    ap.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "data" / "elo_trajectory.png")
    ap.add_argument("--fighter-id", type=str, default=None)
    ap.add_argument("--name", type=str, default=None, help="Exact profile name (case-insensitive); needs fighter_profiles.csv")
    ap.add_argument(
        "--weight-class",
        type=str,
        default=None,
        help="Single division (alias e.g. lightweight). Omit to overlay every division with points.",
    )
    ap.add_argument("--elo-progress-every", type=int, default=2000)
    ap.add_argument(
        "--export-all",
        type=Path,
        default=None,
        help="Write one PNG per (fighter, division) with trajectory; use --max-files to cap",
    )
    ap.add_argument("--max-files", type=int, default=None, help="With --export-all, stop after this many PNGs")
    ap.add_argument("--min-points", type=int, default=1, help="With --export-all, skip trajectories shorter than this")
    ap.add_argument(
        "--no-opponent-labels",
        action="store_true",
        help="Draw the line only (no per-fight opponent name annotations)",
    )
    ap.add_argument(
        "--opponent-label-fontsize",
        type=float,
        default=6.0,
        help="Font size for opponent labels (default: 6)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        return 2

    p = MMAPredictor(Config())
    p.load_data(data_dir)
    print(f"Building ELO with trajectory recording ({len(p.fights):,} fights) ...", flush=True)
    p.build_elo(elo_progress_every=args.elo_progress_every, record_trajectories=True)
    assert p.elo_model is not None

    if args.export_all is not None:
        n = export_all_trajectory_charts(
            p.elo_model,
            p.profiles,
            args.export_all,
            min_points=args.min_points,
            max_files=args.max_files,
            label_opponents=not args.no_opponent_labels,
            opponent_label_fontsize=args.opponent_label_fontsize,
        )
        print(f"Wrote {n} chart(s) under {args.export_all.resolve()}", flush=True)
        return 0

    fid: str | None = args.fighter_id
    if args.name:
        if fid:
            print("Use only one of --fighter-id or --name", file=sys.stderr)
            return 2
        if not p.profiles:
            print("No fighter_profiles.csv loaded; cannot resolve --name", file=sys.stderr)
            return 2
        try:
            fid = require_fighter_id(args.name, p.profiles)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2

    if not fid:
        print("Provide --fighter-id or --name (or use --export-all)", file=sys.stderr)
        return 2

    only_wc = _parse_weight_class(args.weight_class) if args.weight_class else None
    series = _series_for_fighter(p, fid, only_wc)
    if not series:
        print(f"No trajectory points for fighter {fid} (check division / spelling).", file=sys.stderr)
        return 1

    pr = p.profiles.get(fid)
    display = pr.name if pr else fid
    title = f"{display} ({fid[:8]}…)" if len(fid) > 12 else f"{display} ({fid})"

    label_kw = dict(
        profiles=p.profiles if p.profiles else None,
        label_opponents=not args.no_opponent_labels,
        opponent_label_fontsize=args.opponent_label_fontsize,
    )
    if len(series) == 1:
        fig, _ = plot_elo_trajectory(series[0][1], title=f"{title} — {series[0][0]}", **label_kw)
    else:
        fig, _ = plot_elo_trajectories_overlay(series, title=title, **label_kw)

    out = Path(args.out)
    save_trajectory_figure(fig, out_path=out)
    print(f"Wrote {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
