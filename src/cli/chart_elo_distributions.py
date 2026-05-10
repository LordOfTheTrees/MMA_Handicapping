#!/usr/bin/env python3
"""
Build ELO from data/ CSVs and plot per-weight-class ELO distributions (histograms).

Uses the same pipeline as training but stops after ``build_elo``. Point ELO is
``get_elo(..., as_of_date)`` so inactive fighters get Kalman growth from last fight.

Usage (repo root)::

    python -m src.cli.chart_elo_distributions --data-dir ./data --top-n 15

Default composite chart is ``data/elo_by_division.png``. Any existing file at that path is **removed**
before writing so ``data/`` never keeps a stale chart (copy the PNG aside first if you want
to keep it). Override with ``--out``.

Per-division PNGs default to ``data/figures/division_elo_histograms/`` (disable with
``--no-per-division-charts``).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from src.config import Config
from src.data.schema import WeightClass
from src.pipeline import MMAPredictor
from src.stats.elo_division_population import (
    collect_elos_by_division,
    division_order_public,
    fight_pairs_for_elo_charts,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _division_png_stem_clean(wc: WeightClass) -> str:
    """File stem: ``elo_hist_Heavyweight`` style (spaces -> underscores)."""
    safe = wc.value.replace("/", "-").replace(" ", "_")
    return f"elo_hist_{safe}"


def _print_top_by_division(
    predictor: MMAPredictor,
    as_of: date,
    pairs: set[tuple[str, WeightClass]],
    top_n: int,
) -> None:
    if top_n <= 0:
        return
    order = division_order_public()
    profiles = predictor.profiles
    print(f"\nTop {top_n} by ELO per division (as_of={as_of}):", flush=True)
    for wc in order:
        rows = [
            (fid, float(predictor.elo_model.get_elo(fid, wcc, as_of)))
            for fid, wcc in pairs
            if wcc == wc
        ]
        if not rows:
            continue
        rows.sort(key=lambda x: x[1], reverse=True)
        take = rows[:top_n]
        print(f"\n--- {wc.value} ---", flush=True)
        print(f"{'#':>3} {'elo':>8}  {'fighter_id':<36} name", flush=True)
        for i, (fid, elo) in enumerate(take, start=1):
            pr = profiles.get(fid)
            name = pr.name if pr else ""
            print(f"{i:>3} {elo:>8.1f}  {fid:<36} {name}", flush=True)


def main() -> int:
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Histogram ELO by weight class")
    ap.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "elo_by_division.png",
        help="Combined grid PNG path (default: data/elo_by_division.png)",
    )
    ap.add_argument(
        "--per-division-out-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "figures" / "division_elo_histograms",
        help="Directory for one PNG per division (default: data/figures/division_elo_histograms)",
    )
    ap.add_argument(
        "--no-per-division-charts",
        action="store_true",
        help="Skip writing individual division histograms",
    )
    ap.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--elo-progress-every", type=int, default=2000)
    ap.add_argument("--min-fighters", type=int, default=2, help="Skip panels with fewer fighters")
    ap.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Print top N fighters per division by ELO to stdout (0 to disable)",
    )
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        return 2

    print(f"Loading {data_dir} ...", flush=True)
    p = MMAPredictor(Config())
    p.load_data(data_dir)
    print(f"  {len(p.fights):,} fights, {len(p.profiles):,} profiles", flush=True)

    print("Building ELO (full history) ...", flush=True)
    p.build_elo(elo_progress_every=args.elo_progress_every)

    order = division_order_public()
    fight_pairs = fight_pairs_for_elo_charts(p)
    by_wc = collect_elos_by_division(p, as_of, fight_pairs)

    # Terminal summary
    print(f"\nELO distribution summary (as_of={as_of}, Kalman time growth applied):", flush=True)
    print(f"{'division':<22} {'n':>6} {'mean':>8} {'std':>8} {'p10':>8} {'p50':>8} {'p90':>8}", flush=True)
    print("-" * 78, flush=True)
    for wc in order:
        vals = by_wc.get(wc, [])
        if not vals:
            continue
        a = np.array(vals, dtype=float)
        print(
            f"{wc.value:<22} {len(vals):>6} {a.mean():>8.1f} {a.std():>8.1f} "
            f"{np.percentile(a, 10):>8.1f} {np.percentile(a, 50):>8.1f} {np.percentile(a, 90):>8.1f}",
            flush=True,
        )

    print(
        "\nKalman posterior variance (uncertainty) for same fighters "
        "(low => filter trusts ELO tightly):",
        flush=True,
    )
    print(f"{'division':<22} {'n':>6} {'var_mean':>10} {'var_p50':>10}", flush=True)
    print("-" * 52, flush=True)
    var_by_wc: dict[WeightClass, list[float]] = {wc: [] for wc in order}
    for fid, wc in fight_pairs:
        st = p.elo_model.get_state(fid, wc, as_of)
        var_by_wc[wc].append(float(st.uncertainty))
    for wc in order:
        vv = var_by_wc.get(wc, [])
        if not vv:
            continue
        v = np.array(vv, dtype=float)
        print(
            f"{wc.value:<22} {len(vv):>6} {v.mean():>10.2f} {np.median(v):>10.2f}",
            flush=True,
        )

    _print_top_by_division(p, as_of, fight_pairs, args.top_n)

    baseline = p.config.elo.initial_elo
    all_vals = [x for wc in order for x in by_wc.get(wc, [])]
    x_min = min(all_vals) - 30 if all_vals else baseline - 200
    x_max = max(all_vals) + 30 if all_vals else baseline + 200

    def _draw_one_axis(ax, vals: list[float], title: str) -> None:
        ax.hist(vals, bins=28, range=(x_min, x_max), color="steelblue", edgecolor="white", alpha=0.88)
        ax.axvline(baseline, color="coral", linestyle="--", linewidth=1.2, label="initial 1500")
        a = np.array(vals)
        ax.axvline(float(np.median(a)), color="darkgreen", linestyle=":", linewidth=1.0, label="median")
        ax.set_xlim(x_min, x_max)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("fighters")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    if not args.no_per_division_charts:
        per_dir = Path(args.per_division_out_dir)
        per_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for wc in order:
            vals = by_wc.get(wc, [])
            if len(vals) < args.min_fighters:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            _draw_one_axis(
                ax,
                vals,
                f"{wc.value}\n"
                f"n={len(vals)}, mean={np.mean(vals):.0f}, std={np.std(vals):.1f}",
            )
            fig.suptitle(
                f"ELO distribution (as_of={as_of})",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
            fp = per_dir / f"{_division_png_stem_clean(wc)}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  division chart -> {fp.resolve()}", flush=True)
            n_written += 1
        print(f"\nWrote {n_written} per-division histogram(s) under {per_dir.resolve()}", flush=True)

    # Figure: grid of histograms
    n_cols = 3
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), sharex=False)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, wc in zip(axes_flat, order):
        vals = by_wc.get(wc, [])
        if len(vals) < args.min_fighters:
            ax.set_visible(False)
            continue
        a = np.array(vals)
        _draw_one_axis(
            ax,
            vals,
            f"{wc.value}\n"
            f"n={len(vals)}, mean={a.mean():.0f}, std={a.std():.1f}",
        )

    for ax in axes_flat[len(order) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"ELO distribution by division (as_of={as_of})\n"
        "Wider spread => more separation; tight clustering => ratings similar within division",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    had_existing = out.is_file()
    if had_existing:
        try:
            out.unlink()
        except OSError as exc:
            print(f"  [warn] could not remove existing file (may be open elsewhere): {exc}", flush=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    verb = "Overwrote" if had_existing else "Wrote"
    print(f"\n{verb} {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
