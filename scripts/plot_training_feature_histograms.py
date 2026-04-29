#!/usr/bin/env python3
"""
Plot per-feature histograms for the Tier-1 **regression training** matrix rows
(the same cohort as multinomial fit: post-era Tier-1, fight_date strictly before holdout).

Usage (from repo root)::

    python scripts/plot_training_feature_histograms.py --data-dir ./data
    python scripts/plot_training_feature_histograms.py --data-dir ./data \\
        --elo-cache ./data/elo_cache.pkl --out-dir ./data/figures/feature_histograms

Requires matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.matchup.interactions import FEATURE_NAMES
from src.pipeline import MMAPredictor


def main() -> None:
    p = argparse.ArgumentParser(
        description="Histogram each matchup feature column for regression training rows.",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/figures/feature_histograms"),
        help="Directory for PNG histograms plus combined grid PNG",
    )
    p.add_argument(
        "--elo-cache",
        type=Path,
        default=None,
        help="Reuse PIT ELO cache if compatible (see train --elo-cache).",
    )
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
