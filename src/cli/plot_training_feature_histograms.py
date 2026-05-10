#!/usr/bin/env python3
"""
Plot per-feature histograms for the Tier-1 **regression training** matrix rows
(the same cohort as multinomial fit: post-era Tier-1, fight_date strictly before holdout).

**Output:** all PNGs are written under **this repo** only (default ``<repo>/data/figures/feature_histograms``).
Deploy copies to a sibling site (e.g. ``mma.ai`` ``public/model-viz/``) are a **manual** step — no adjacent-repo paths here.

Usage (from repo root)::

    python -m src.cli.plot_training_feature_histograms --data-dir ./data
    python -m src.cli.plot_training_feature_histograms --model-path ./data/model.pkl

    # Same training cohort + ``global_days_idle`` as ``scripts/export_artifacts.py``:

    python -m src.cli.plot_training_feature_histograms --model-path ./data/model.pkl \\
        --out-dir ./data/figures/feature_histograms

    python -m src.cli.plot_training_feature_histograms --data-dir ./data \\
        --elo-cache ./data/elo_cache.pkl --out-dir ./data/figures/feature_histograms

Requires matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.config import Config
from src.export.reference_distributions_export import (
    collect_global_days_idle_training_corners,
)
from src.matchup.interactions import FEATURE_NAMES
from src.pipeline import MMAPredictor

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _plot_global_days_idle_histogram(
    ax: Axes,
    idle: np.ndarray,
    *,
    x_max_days: float,
) -> None:
    """
    Histogram of layoff days; x-axis is fixed ``[0, x_max_days]`` so the view is comparable
    across runs. Values above ``x_max_days`` are omitted from the bars but counted in the subtitle.
    """
    a = np.asarray(idle, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
        return

    x_cap = float(x_max_days)
    if x_cap <= 0:
        raise ValueError("x_max_days must be positive")

    in_win = a[a <= x_cap]
    tail_n = int(n - in_win.size)
    tail_pct = 100.0 * tail_n / n

    bins = np.linspace(0.0, x_cap, 61)
    ax.hist(in_win, bins=bins, density=False, alpha=0.85, edgecolor="black", linewidth=0.3)

    m_full = float(np.mean(a))
    med_full = float(np.median(a))
    if m_full <= x_cap:
        ax.axvline(m_full, color="orange", linestyle="--", linewidth=1, label=f"mean={m_full:.4g}")
    if med_full <= x_cap:
        ax.axvline(med_full, color="purple", linestyle=":", linewidth=1, label=f"median={med_full:.4g}")

    ax.set_xlim(0.0, x_cap)
    ax.set_xlabel("days since last fight (global)")
    ax.set_ylabel("count")

    off = []
    if m_full > x_cap:
        off.append(f"mean={m_full:.4g}")
    if med_full > x_cap:
        off.append(f"median={med_full:.4g}")

    ax.legend(fontsize=8, loc="best")
    title = (
        f"global_days_idle (A and B per training bout; n={n:,})\n"
        f"full min={float(np.min(a)):.4g}, max={float(np.max(a)):.4g}, std={float(np.std(a, ddof=0)):.4g}"
    )
    if off:
        title += "\n" + ", ".join(off) + " (past axis — see full max)"
    title += (
        f"\nx-axis 0–{x_cap:.0f} d ({tail_n:,} corners = {tail_pct:.3g}% > {x_cap:.0f} d, not in bars)"
    )
    ax.set_title(title, fontsize=10)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Histogram each matchup feature column for regression training rows.",
    )
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "figures" / "feature_histograms",
        help="Directory for PNG histograms plus combined grid PNG",
    )
    p.add_argument(
        "--elo-cache",
        type=Path,
        default=None,
        help="Reuse PIT ELO cache if compatible (see train --elo-cache).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Load trained pickle (matches export cohort); bypasses --data-dir when set.",
    )
    p.add_argument(
        "--idle-x-max-days",
        type=float,
        default=1000.0,
        metavar="D",
        help=(
            "For global_days_idle PNG: right edge of histogram in days (default 1000); "
            "values above D omitted from bars but counted in subtitle."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        mp = Path(args.model_path).resolve()
        if not mp.is_file():
            raise SystemExit(f"model not found: {mp}")
        pred = MMAPredictor.load(mp)
        print(f"Loaded predictor {mp}", flush=True)
        if pred.elo_model is None:
            raise SystemExit("Pickle missing elo_model — train or use --data-dir")
    else:
        data_dir = args.data_dir
        config = Config()
        pred = MMAPredictor(config)
        print(f"Loading {data_dir} ...", flush=True)
        pred.load_data(data_dir)
        elo_cache = Path(args.elo_cache).resolve() if args.elo_cache else None

        if elo_cache and pred.try_load_elo_from_cache(elo_cache):
            print(f"Loaded ELO from {elo_cache.name}", flush=True)
        else:
            if elo_cache and elo_cache.exists():
                print("ELO cache stale or missing - rebuilding ...", flush=True)
            print("Building ELO ...", flush=True)
            pred.build_elo()
            if elo_cache:
                pred.save_elo_cache(elo_cache)
                print(f"Wrote {elo_cache}", flush=True)

    print("Building training matrix (fit_model=False) ...", flush=True)
    pred.train_regression(fit_model=False)
    X = pred._X_train
    if X is None:
        raise SystemExit("No training matrix.")

    print(f"Matrix shape {X.shape}  (same rows as multinomial regression would use).\n")

    names = list(FEATURE_NAMES)
    if X.shape[1] != len(names):
        raise SystemExit(
            f"Column count mismatch: X has {X.shape[1]}, FEATURE_NAMES has {len(names)}"
        )

    for j, name in enumerate(names):
        xs = np.asarray(X[:, j], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(xs, bins=60, density=False, alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.axvline(float(np.nanmean(xs)), color="orange", linestyle="--", linewidth=1, label=f"mean={np.nanmean(xs):.4g}")
        ax.axvline(float(np.nanmedian(xs)), color="purple", linestyle=":", linewidth=1, label=f"median={np.nanmedian(xs):.4g}")
        ax.set_title(f"{name}\n(min={np.nanmin(xs):.4g}, max={np.nanmax(xs):.4g}, std={np.nanstd(xs, ddof=0):.4g})")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="best")
        fp = out_dir / f"histogram_{name}.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        print(f"  wrote {fp.name}", flush=True)

    idle = collect_global_days_idle_training_corners(pred)
    fig_i, ax_i = plt.subplots(figsize=(7.0, 4.5))
    _plot_global_days_idle_histogram(ax_i, idle, x_max_days=args.idle_x_max_days)
    fp_idle = out_dir / "histogram_global_days_idle.png"
    fig_i.tight_layout()
    fig_i.savefig(fp_idle, dpi=120)
    plt.close(fig_i)
    print(f"  wrote {fp_idle.name}", flush=True)

    ncol = min(4, len(names))
    nrow = max(1, (len(names) + ncol - 1) // ncol)
    fig_all, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.2 * nrow), squeeze=False)
    flat_axes = axes.ravel().tolist()

    for j in range(len(names)):
        ax = flat_axes[j]
        ax.hist(X[:, j].astype(float), bins=35, density=True, alpha=0.8, edgecolor="none")
        ax.set_title(names[j], fontsize=8)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(len(names), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig_all.suptitle(f"Regression training features (n={len(X):,})", fontsize=11)
    fig_all.tight_layout()
    combo = out_dir / "histogram_all_grid.png"
    fig_all.savefig(combo, dpi=140)
    plt.close(fig_all)
    print(f"\nCombined grid -> {combo}", flush=True)


if __name__ == "__main__":
    main()
