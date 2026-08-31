"""
Matplotlib figures for Phase-3 walk-forward / pristine evaluation
and the post-hoc market book (``src.eval.market_book``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Deferred import so ``import src.eval.tuning_harness`` works without matplotlib in minimal envs
def _plt():
    import matplotlib.pyplot as plt

    return plt


def _priced_series(
    years_map: Mapping[str, Any],
    years: Sequence[int],
    book: str,
    *keys: str,
    fill_key: Optional[str] = None,
) -> List[float]:
    """
    Yearly metric, NaN when that tape has no priced fights.
    ``fill_key`` reads ``years[y][fill_key][book]`` (mdabbert fill); default is primary jurek.
    """
    out: List[float] = []
    for year in years:
        node: Any = years_map.get(str(year)) or {}
        if fill_key:
            node = node.get(fill_key) or {}
        cur: Any = node.get(book) or {}
        if int(cur.get("n_priced") or 0) <= 0:
            out.append(float("nan"))
            continue
        try:
            for k in keys:
                cur = cur[k]
            out.append(float(cur))
        except (KeyError, TypeError, ValueError):
            out.append(float("nan"))
    return out


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


def plot_market_book_yoy(
    report: Mapping[str, Any],
    out_path: Path,
    *,
    title: str = "Jurek tape only: projected vs realized (mdabbert fill not shown)",
) -> None:
    """
    Two panels: Kelly log-growth and 1u ROI on the **jurek** tape.
    Years with no jurek method (or two-way) are a gap, not a fill splice.
    """
    years_map = report.get("years") or {}
    years = sorted(int(y) for y in years_map.keys())
    if not years:
        return

    tw_k_r = _priced_series(years_map, years, "two_way", "full_kelly", "realized_log_growth")
    tw_k_p = _priced_series(years_map, years, "two_way", "full_kelly", "projected_log_growth")
    mh_k_r = _priced_series(years_map, years, "method", "full_kelly", "realized_log_growth")
    mh_k_p = _priced_series(years_map, years, "method", "full_kelly", "projected_log_growth")
    tw_h_r = _priced_series(years_map, years, "two_way", "half_kelly", "realized_log_growth")
    mh_h_r = _priced_series(years_map, years, "method", "half_kelly", "realized_log_growth")
    tw_q_r = _priced_series(years_map, years, "two_way", "quarter_kelly", "realized_log_growth")
    mh_q_r = _priced_series(years_map, years, "method", "quarter_kelly", "realized_log_growth")
    tw_1_r = _priced_series(years_map, years, "two_way", "flat_1u", "realized_roi")
    tw_1_p = _priced_series(years_map, years, "two_way", "flat_1u", "projected_roi")
    mh_1_r = _priced_series(years_map, years, "method", "flat_1u", "realized_roi")
    mh_1_p = _priced_series(years_map, years, "method", "flat_1u", "projected_roi")
    tw_cov = _priced_series(years_map, years, "two_way", "coverage")
    mh_cov = _priced_series(years_map, years, "method", "coverage")

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), constrained_layout=True, sharex=True)
    ax0, ax1 = axes
    ax0.plot(years, tw_k_p, "--", color="#9ecae1", label="two-way Kelly projected", linewidth=1.5)
    ax0.plot(years, mh_k_p, "--", color="#fdae6b", label="method Kelly projected", linewidth=1.5)
    ax0.plot(years, tw_k_r, "o-", color="#3182bd", label="two-way Kelly realized", markersize=5)
    ax0.plot(years, mh_k_r, "s-", color="#e6550d", label="method Kelly realized", markersize=5)
    ax0.plot(years, tw_h_r, "o--", color="#6baed6", label="two-way half Kelly realized", markersize=4, alpha=0.85)
    ax0.plot(years, mh_h_r, "s--", color="#fd8d3c", label="method half Kelly realized", markersize=4, alpha=0.85)
    ax0.plot(years, tw_q_r, "o:", color="#08519c", label="two-way quarter Kelly realized", markersize=4, alpha=0.9)
    ax0.plot(years, mh_q_r, "s:", color="#a63603", label="method quarter Kelly realized", markersize=4, alpha=0.9)
    ax0.axhline(0.0, color="#888", linewidth=0.6)
    ax0.set_ylabel("Log-growth")
    ax0.legend(loc="best", fontsize=6, ncol=2)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Full / half / quarter Kelly (coverage annotated on 1u panel)")

    ax1.plot(years, tw_1_p, "--", color="#9ecae1", label="two-way 1u projected ROI", linewidth=1.5)
    ax1.plot(years, mh_1_p, "--", color="#fdae6b", label="method 1u projected ROI", linewidth=1.5)
    ax1.plot(years, tw_1_r, "o-", color="#3182bd", label="two-way 1u realized ROI", markersize=5)
    ax1.plot(years, mh_1_r, "s-", color="#e6550d", label="method 1u realized ROI", markersize=5)
    ax1.axhline(0.0, color="#888", linewidth=0.6)
    ax1.set_xlabel("Calendar year (eval)")
    ax1.set_ylabel("1u ROI (profit / 100u bank)")
    ax1.legend(loc="best", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    def _pct(x: float) -> str:
        return "n/a" if x != x else f"{x:.0%}"

    labels = [f"{y}\n{_pct(tw_cov[i])}/{_pct(mh_cov[i])}" for i, y in enumerate(years)]
    ax1.set_xticks(list(years), labels, fontsize=8)
    ax1.set_title("Flat 1u ROI; gaps = no jurek tape that year (not filled)")

    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_market_book_slices(
    report: Mapping[str, Any],
    out_path: Path,
    *,
    title: str = "Pooled +EV slices, jurek tape only (one-way; not crossed)",
) -> None:
    """
    Three panels of pooled realized vs projected 1u ROI: card slot, gender, weight class.
    Two-way vs method. Year-by-slice series are in JSON only, not this figure.
    """
    pooled = report.get("slices_pooled") or {}
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)

    def _bars(ax, labels: Sequence[str], group_key: str) -> None:
        grp = pooled.get(group_key) or {}
        xs = list(range(len(labels)))
        tw_r, tw_p, mh_r, mh_p, ns = [], [], [], [], []
        for lab in labels:
            cell = grp.get(lab) or {}
            tw = cell.get("two_way") or {}
            mh = cell.get("method") or {}
            tw_n = int(tw.get("n_plus_ev") or 0)
            mh_n = int(mh.get("n_plus_ev") or 0)
            tw_r.append(float((tw.get("flat_1u") or {}).get("realized_roi") or 0.0) if tw_n else float("nan"))
            tw_p.append(float((tw.get("flat_1u") or {}).get("projected_roi") or 0.0) if tw_n else float("nan"))
            mh_r.append(float((mh.get("flat_1u") or {}).get("realized_roi") or 0.0) if mh_n else float("nan"))
            mh_p.append(float((mh.get("flat_1u") or {}).get("projected_roi") or 0.0) if mh_n else float("nan"))
            ns.append(tw_n)
        w = 0.2
        ax.bar([x - 1.5 * w for x in xs], tw_p, width=w, color="#9ecae1", label="two-way projected")
        ax.bar([x - 0.5 * w for x in xs], tw_r, width=w, color="#3182bd", label="two-way realized")
        ax.bar([x + 0.5 * w for x in xs], mh_p, width=w, color="#fdae6b", label="method projected")
        ax.bar([x + 1.5 * w for x in xs], mh_r, width=w, color="#e6550d", label="method realized")
        ax.axhline(0.0, color="#888", linewidth=0.6)
        tick = [f"{lab}\nn={n}" for lab, n in zip(labels, ns)]
        ax.set_xticks(xs, tick, fontsize=8)
        ax.set_ylabel("1u ROI")
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)

    card_labels = ["title", "main_event", "main_card", "prelim_main_event", "generic_prelims"]
    _bars(axes[0], card_labels, "by_card")
    axes[0].set_title("Card slot (overlapping; main_card includes main_event)")
    _bars(axes[1], ["men", "women", "other"], "by_gender")
    axes[1].set_title("Gender (other = catch/unknown)")
    wc_labels = sorted((pooled.get("by_weight_class") or {}).keys())
    _bars(axes[2], wc_labels, "by_weight_class")
    axes[2].set_title("Weight class")
    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_market_book_simul_compare(
    report: Mapping[str, Any],
    out_path: Path,
    *,
    title: str = "Isolated vs simultaneous Kelly, jurek tape only",
) -> None:
    """
    Realized Kelly log-growth (full and quarter) and 1u ROI on the **jurek** tape.
    Years with no jurek method are a gap, not an mdabbert splice.
    """
    years_map = report.get("years") or {}
    years = sorted(int(y) for y in years_map.keys())
    if not years:
        return

    tw_i = _priced_series(years_map, years, "two_way", "full_kelly", "realized_log_growth")
    tw_s = _priced_series(years_map, years, "two_way_simul", "full_kelly", "realized_log_growth")
    mh_i = _priced_series(years_map, years, "method", "full_kelly", "realized_log_growth")
    mh_s = _priced_series(years_map, years, "method_simul", "full_kelly", "realized_log_growth")
    tw_qi = _priced_series(years_map, years, "two_way", "quarter_kelly", "realized_log_growth")
    tw_qs = _priced_series(years_map, years, "two_way_simul", "quarter_kelly", "realized_log_growth")
    mh_qi = _priced_series(years_map, years, "method", "quarter_kelly", "realized_log_growth")
    mh_qs = _priced_series(years_map, years, "method_simul", "quarter_kelly", "realized_log_growth")
    tw_1i = _priced_series(years_map, years, "two_way", "flat_1u", "realized_roi")
    tw_1s = _priced_series(years_map, years, "two_way_simul", "flat_1u", "realized_roi")
    mh_1i = _priced_series(years_map, years, "method", "flat_1u", "realized_roi")
    mh_1s = _priced_series(years_map, years, "method_simul", "flat_1u", "realized_roi")
    n_multi = _priced_series(years_map, years, "method_simul", "n_multi_leg")

    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), constrained_layout=True, sharex=True)
    ax0, axq, ax1 = axes
    ax0.plot(years, tw_i, "o--", color="#9ecae1", label="two-way isolated", markersize=4)
    ax0.plot(years, tw_s, "s-", color="#3182bd", label="two-way simultaneous", markersize=5)
    ax0.plot(years, mh_i, "o--", color="#fdae6b", label="method isolated", markersize=4)
    ax0.plot(years, mh_s, "s-", color="#e6550d", label="method simultaneous", markersize=5)
    ax0.axhline(0.0, color="#888", linewidth=0.6)
    ax0.set_ylabel("Full Kelly log-growth")
    ax0.legend(loc="best", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Full Kelly; two-way isolated and simultaneous should nearly match")

    axq.plot(years, tw_qi, "o--", color="#9ecae1", label="two-way isolated 1/4", markersize=4)
    axq.plot(years, tw_qs, "s-", color="#3182bd", label="two-way simultaneous 1/4", markersize=5)
    axq.plot(years, mh_qi, "o--", color="#fdae6b", label="method isolated 1/4", markersize=4)
    axq.plot(years, mh_qs, "s-", color="#e6550d", label="method simultaneous 1/4", markersize=5)
    axq.axhline(0.0, color="#888", linewidth=0.6)
    axq.set_ylabel("Quarter Kelly log-growth")
    axq.legend(loc="best", fontsize=7, ncol=2)
    axq.grid(True, alpha=0.3)
    axq.set_title("Quarter Kelly (same tickets, 0.25 x f*)")

    ax1.plot(years, tw_1i, "o--", color="#9ecae1", label="two-way isolated 1u", markersize=4)
    ax1.plot(years, tw_1s, "s-", color="#3182bd", label="two-way simultaneous 1u", markersize=5)
    ax1.plot(years, mh_1i, "o--", color="#fdae6b", label="method isolated 1u", markersize=4)
    ax1.plot(years, mh_1s, "s-", color="#e6550d", label="method simultaneous 1u", markersize=5)
    ax1.axhline(0.0, color="#888", linewidth=0.6)
    ax1.set_xlabel("Calendar year (eval)")
    ax1.set_ylabel("1u ROI (profit / 100u bank)")
    ax1.legend(loc="best", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    labels = [
        f"{y}\n{int(n_multi[i]) if n_multi[i] == n_multi[i] else 'n/a'} split"
        for i, y in enumerate(years)
    ]
    ax1.set_xticks(list(years), labels, fontsize=8)
    ax1.set_title("Flat 1u; gaps = no jurek method tape")

    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_market_book_fill_tape(
    report: Mapping[str, Any],
    out_path: Path,
    *,
    title: str = "Mdabbert fill tape (not comparable to jurek YoY)",
) -> None:
    """
    Separate figure for the fill sidecar. Do not overlay on jurek charts.
    Only years with priced fill fights are drawn.
    """
    years_map = report.get("years") or {}
    years = sorted(int(y) for y in years_map.keys())
    if not years:
        return
    tw_1 = _priced_series(
        years_map, years, "two_way", "flat_1u", "realized_roi", fill_key="mdabbert_fill"
    )
    mh_1 = _priced_series(
        years_map, years, "method", "flat_1u", "realized_roi", fill_key="mdabbert_fill"
    )
    mh_k = _priced_series(
        years_map, years, "method", "full_kelly", "realized_log_growth", fill_key="mdabbert_fill"
    )
    mh_n = _priced_series(years_map, years, "method", "n_priced", fill_key="mdabbert_fill")
    if all(x != x for x in tw_1 + mh_1):
        return

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True, sharex=True)
    ax0, ax1 = axes
    ax0.plot(years, mh_k, "s-", color="#7a0177", label="method Kelly realized", markersize=5)
    ax0.axhline(0.0, color="#888", linewidth=0.6)
    ax0.set_ylabel("Method Kelly log-growth")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Fill tape only; 6-way overround and missing legs differ from jurek")

    ax1.plot(years, tw_1, "o-", color="#3182bd", label="two-way 1u realized", markersize=5)
    ax1.plot(years, mh_1, "s-", color="#7a0177", label="method 1u realized", markersize=5)
    ax1.axhline(0.0, color="#888", linewidth=0.6)
    ax1.set_xlabel("Calendar year (eval)")
    ax1.set_ylabel("1u ROI")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    labels = [
        f"{y}\nn={int(mh_n[i]) if mh_n[i] == mh_n[i] else 0}"
        for i, y in enumerate(years)
    ]
    ax1.set_xticks(list(years), labels, fontsize=8)
    ax1.set_title("n = mdabbert method fights priced")

    fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
