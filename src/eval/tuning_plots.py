"""
Matplotlib figures for Phase-3 walk-forward / pristine evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

# Deferred import so ``import src.eval.tuning_harness`` works without matplotlib in minimal envs
def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_pristine_yoy_bars(
    years: Sequence[int],
    log_loss: Sequence[float],
    brier: Sequence[float],
    f1: Sequence[float],
    out_path: Path,
    *,
    title: str = "Pristine test years (fixed config): YoY metrics",
) -> None:
    """Three bar charts: mean log-loss, Brier, macro F1 for 2023–2025 (or any years)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    x = list(range(len(years)))
    labels = [str(y) for y in years]
    ax0, ax1, ax2 = axes
    b0 = ax0.bar(x, list(log_loss), color="#1f77b4", edgecolor="white")
    ax0.set_ylabel("Mean log-loss")
    ax0.set_xticks(x, labels)
    ax0.set_title("Log-loss (lower is better)")
    for i, p in enumerate(b0.patches):
        h = float(p.get_height())
        ax0.annotate(
            f"{log_loss[i]:.3f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    b1 = ax1.bar(x, list(brier), color="#2ca02c", edgecolor="white")
    ax1.set_ylabel("Mean Brier")
    ax1.set_xticks(x, labels)
    ax1.set_title("Brier (lower is better)")
    for i, p in enumerate(b1.patches):
        h = float(p.get_height())
        ax1.annotate(
            f"{brier[i]:.3f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    b2 = ax2.bar(x, list(f1), color="#ff7f0e", edgecolor="white")
    ax2.set_ylabel("Macro F1")
    ax2.set_xticks(x, labels)
    ax2.set_title("Macro F1 (higher is better)")
    for i, p in enumerate(b2.patches):
        h = float(p.get_height())
        ax2.annotate(
            f"{f1[i]:.3f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_combined_log_loss_trajectory(
    selection_years: Optional[Sequence[int]],
    selection_ll: Optional[Sequence[float]],
    pristine_years: Sequence[int],
    pristine_ll: Sequence[float],
    out_path: Path,
    *,
    selection_legend: str = "Selection (single baseline Config per year)",
    pristine_legend: str = "Pristine test (2023–2025, same frozen default Config)",
    title: str = "Walk-forward: mean log-loss (Tier-1, fighter A) — selection + pristine",
) -> None:
    """
    Line + markers. *Selection* and *pristine* segments use different colors.
    If *selection* is None or empty, only pristine is plotted.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if (
        selection_years is not None
        and len(selection_years) > 0
        and selection_ll is not None
        and len(list(selection_ll)) > 0
    ):
        ax.plot(
            list(selection_years),
            list(selection_ll),
            "o-",
            color="#6baed6",
            label=selection_legend,
            markersize=5,
        )
    if len(pristine_years) > 0:
        ax.plot(
            list(pristine_years),
            list(pristine_ll),
            "s-",
            color="#fd8d3c",
            label=pristine_legend,
            markersize=7,
        )
    ax.set_xlabel("Calendar year (eval)")
    ax.set_ylabel("Mean log-loss")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
