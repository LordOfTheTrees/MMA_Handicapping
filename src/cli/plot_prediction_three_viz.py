#!/usr/bin/env python3
"""
Single figure from one ``predict()`` call:

  • Top: stacked split bar + total-win line (whole-number **%** labels, no decimals).
  • Middle: combined sums of marginal CI bounds for each fighter’s three win-side classes.
  • Bottom: bar chart (point mass) and optional per-class CI whiskers.

Point CIs are **marginal** per class.

Usage (repo root)::

    python -m src.cli.plot_prediction_three_viz \\
        --model-path ./data/Saved_Runs/First-bootstrap-phase3.pkl \\
        --fighter-a-id 150ff4cc642270b9 --fighter-b-id e1248941344b3288 \\
        --weight-class featherweight --date 2022-07-02

Repeat for other dates::

    python -m src.cli.plot_prediction_three_viz ... --date 2020-07-11
"""
from __future__ import annotations

import argparse
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.text import Text

from src.cli.common import resolve_date, resolve_weight_class
from src.data.schema import PredictionResult
from src.pipeline import MMAPredictor


def _prediction_vectors(r: PredictionResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.array(
        [
            r.p_win_ko_tko,
            r.p_win_submission,
            r.p_win_decision,
            r.p_lose_decision,
            r.p_lose_ko_tko,
            r.p_lose_submission,
        ],
        dtype=float,
    )
    lo = np.array(
        [
            r.ci_win_ko_tko[0],
            r.ci_win_submission[0],
            r.ci_win_decision[0],
            r.ci_lose_decision[0],
            r.ci_lose_ko_tko[0],
            r.ci_lose_submission[0],
        ]
    )
    hi = np.array(
        [
            r.ci_win_ko_tko[1],
            r.ci_win_submission[1],
            r.ci_win_decision[1],
            r.ci_lose_decision[1],
            r.ci_lose_ko_tko[1],
            r.ci_lose_submission[1],
        ]
    )
    return p, lo, hi


def _display_name(predictor: MMAPredictor, fighter_id: str) -> str:
    prof = predictor.profiles.get(fighter_id)
    if prof is not None and getattr(prof, "name", None):
        return str(prof.name).strip()
    return fighter_id[:12] + ("…" if len(fighter_id) > 12 else "")


def _short_title(s: str, n: int = 42) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _outcome_sentence(corner_a_name: str, corner_b_name: str, outcome_index: int) -> str:
    """
    Label for stacked-bar segment order / probability index 0..5.

    Indices 0-2 are corner-A wins by KO/TKO, Submission, Decision.
    Indices 3-5 are corner-B wins matching lose_decision, lose_ko_tko, lose_submission.
    """
    method_a = ("KO/TKO", "Submission", "Decision")
    if outcome_index < 3:
        return f"{corner_a_name} by {method_a[outcome_index]}"
    j = outcome_index - 3
    return f"{corner_b_name} by {method_a[j]}"


def _fighters_no_prior_in_wc(
    predictor: MMAPredictor,
    fighter_a_id: str,
    fighter_b_id: str,
    wc,
    fdate: date,
) -> list[str]:
    """Display names for corners with no prior bout in ``wc`` before ``fdate`` (WC-debut path)."""
    out: list[str] = []
    if predictor._n_prior_bouts_in_wc(fighter_a_id, wc, fdate) < 1:
        out.append(_display_name(predictor, fighter_a_id))
    if predictor._n_prior_bouts_in_wc(fighter_b_id, wc, fdate) < 1:
        out.append(_display_name(predictor, fighter_b_id))
    return out


def _segment_centers(p: np.ndarray) -> np.ndarray:
    left = 0.0
    cx = []
    for i in range(len(p)):
        cx.append(left + float(p[i]) / 2.0)
        left += float(p[i])
    return np.asarray(cx, dtype=float)


def _floored_mass_for_display(p: np.ndarray, floor: float = 0.03) -> np.ndarray:
    """Segment widths that sum to 1, each at least ``floor``, for readable labels on thin slices."""
    x = np.maximum(np.asarray(p, dtype=float), floor)
    return x / np.sum(x)


def _pct_int_from_probability(p_in_0_1: float) -> int:
    """Whole-number percent label (no decimals): 0–100 from a probability in [0, 1]."""
    p = max(0.0, min(1.0, float(p_in_0_1)))
    return int(round(100.0 * p))


def _pct_int_from_optional_mass(x: float) -> int:
    """Whole percent from arbitrary mass (e.g. marginal CI sums — may exceed 1.0)."""
    return int(round(100.0 * float(x)))


def _last_name(display: str) -> str:
    parts = display.strip().split()
    return parts[-1] if parts else ""


def _thin_ci_caption(corner_a_name: str, corner_b_name: str, outcome_index: int, lo_pct: int, hi_pct: int) -> str:
    """Short line for marginal CI when a segment is visually tiny (shown off-bars)."""
    m = ("KO/TKO", "Sub", "Dec")[outcome_index % 3]
    who = _last_name(corner_a_name if outcome_index < 3 else corner_b_name)
    return f"{who} · {m}  [{lo_pct}–{hi_pct}%]"


def _bbox_data_x_extent(ax, text_obj: Text, renderer) -> tuple[float, float]:
    bbox = text_obj.get_window_extent(renderer=renderer)
    y_mid = float(0.5 * (bbox.y0 + bbox.y1))
    inv = ax.transData.inverted()
    x0 = float(inv.transform((bbox.x0, y_mid))[0])
    x1 = float(inv.transform((bbox.x1, y_mid))[0])
    return (min(x0, x1), max(x0, x1))


def _place_corner_name_above_bar(
    fig: plt.Figure,
    ax,
    *,
    xc: float,
    x_bounds: tuple[float, float],
    y: float,
    display_name: str,
    fontsize: float,
    color: str,
) -> Text:
    """
    Draw a fighter label centered above their side of the bar. If ``display_name`` is too wide,
    replace with surname only when that fits the horizontal bounds (marginal widths in data coords).
    """
    x_min, x_max = x_bounds
    span = float(x_max - x_min)
    margin = 0.02 * span if span > 1e-9 else 0.0
    lo_adj, hi_adj = x_min + margin, x_max - margin

    surname = _last_name(display_name)
    candidates = [display_name]
    if surname and surname != display_name:
        candidates.append(surname)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    chosen = candidates[-1]

    for cand in candidates:
        probe = ax.text(
            xc,
            y,
            cand,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            alpha=0.0,
        )
        fig.canvas.draw()
        lx, rx = _bbox_data_x_extent(ax, probe, renderer)
        probe.remove()
        if lx >= lo_adj - 1e-9 and rx <= hi_adj + 1e-9:
            chosen = cand
            break

    return ax.text(xc, y, chosen, ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=color)


def fig_split_barrier_with_ci(
    *,
    fig: plt.Figure,
    p: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    tw: float,
    corner_a_name: str,
    corner_b_name: str,
    ci_alpha: float,
    margin_note: str | None,
    show_whiskers: bool,
) -> None:
    """
    Top: full [0,1] stack + total-win line; middle: combined sums of marginal bounds for win-side buckets;
    bottom: colored bars + literal ``[lo[i], hi[i]]`` when whiskers are on.

    Whiskers use ``lo[i]…hi[i]`` (probability units), matching ``PredictionResult.ci_*``.
    """
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.54, 1.28], hspace=0.11)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)

    labels = [_outcome_sentence(corner_a_name, corner_b_name, i) for i in range(6)]
    blues = plt.cm.Blues(np.linspace(0.45, 0.88, 3))
    reds = plt.cm.Reds(np.linspace(0.45, 0.88, 3))
    seg_colors: list[np.ndarray | tuple[float, ...]] = [
        blues[0],
        blues[1],
        blues[2],
        reds[0],
        reds[1],
        reds[2],
    ]

    # Wider floor for drawing so sub-3% slices stay legible (truth % still shown inside).
    p_vis = _floored_mass_for_display(p, floor=0.03)
    tw_line = float(np.sum(p_vis[:3]))

    left = 0.0
    y = 0.0
    for i in range(3):
        ax_top.barh(
            y,
            p_vis[i],
            height=0.45,
            left=left,
            color=blues[i],
            edgecolor="white",
            linewidth=0.8,
            label=labels[i],
        )
        left += p_vis[i]
    for i in range(3, 6):
        ax_top.barh(
            y,
            p_vis[i],
            height=0.45,
            left=left,
            color=reds[i - 3],
            edgecolor="white",
            linewidth=0.8,
            label=labels[i],
        )
        left += p_vis[i]

    seg_centers_top = _segment_centers(p_vis)
    for i in range(6):
        pct = _pct_int_from_probability(float(p[i]))
        fz = float(np.clip(6.8 + float(p_vis[i]) * 135.0, 6.0, 13.0))
        pct_txt = str(pct) if pct < 5 else f"{pct}%"
        t = ax_top.text(
            seg_centers_top[i],
            y,
            pct_txt,
            ha="center",
            va="center",
            fontsize=fz,
            fontweight="semibold",
            color="#141414",
        )
        t.set_path_effects(
            [
                peffects.Stroke(linewidth=3.25, foreground="white", alpha=0.93),
                peffects.Normal(),
            ]
        )

    centers = _segment_centers(p_vis)

    ax_top.axvline(tw_line, color="black", linewidth=3, ymin=0.02, ymax=0.98)
    tw_pct = _pct_int_from_probability(tw)
    ax_top.text(
        tw_line,
        -0.34,
        f"total win\n{tw_pct}%",
        rotation=0,
        va="top",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.86),
    )

    name_y_top = 0.52
    _place_corner_name_above_bar(
        fig,
        ax_top,
        xc=tw_line / 2.0,
        x_bounds=(0.0, tw_line),
        y=name_y_top,
        display_name=corner_a_name,
        fontsize=11,
        color="#0d47a1",
    )
    _place_corner_name_above_bar(
        fig,
        ax_top,
        xc=tw_line + (1.0 - tw_line) / 2.0,
        x_bounds=(tw_line, 1.0),
        y=name_y_top,
        display_name=corner_b_name,
        fontsize=11,
        color="#b71c1c",
    )

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(-0.58, 0.68)
    ax_top.set_yticks([])
    ax_top.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.set_title(_short_title(title))
    ax_top.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6.5,
        title="Outcome",
        title_fontsize=7,
    )

    nominal_pct = int(round((1.0 - ci_alpha) * 100.0))

    lo_np = np.asarray(lo, dtype=float).flatten()
    hi_np = np.asarray(hi, dtype=float).flatten()
    hi_a_best = float(np.sum(hi_np[:3]))
    lo_a_worst = float(np.sum(lo_np[:3]))
    lo_b_worst = float(np.sum(lo_np[3:]))
    hi_b_best = float(np.sum(hi_np[3:]))

    ax_mid.axis("off")
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    lx, rx = 0.32, 0.68
    tax = ax_mid.transAxes
    ax_mid.text(lx, 0.94, "Best", ha="center", va="bottom", fontsize=9, fontweight="bold", transform=tax)
    ax_mid.text(rx, 0.94, "Worst", ha="center", va="bottom", fontsize=9, fontweight="bold", transform=tax)
    ax_mid.text(
        0.5,
        0.81,
        f"{corner_a_name} wins (marginal CI sum)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#0d47a1",
        transform=tax,
    )
    ax_mid.text(lx, 0.69, f"{_pct_int_from_optional_mass(hi_a_best)}%", ha="center", va="center", fontsize=12, transform=tax)
    ax_mid.text(rx, 0.69, f"{_pct_int_from_optional_mass(lo_a_worst)}%", ha="center", va="center", fontsize=12, transform=tax)

    ax_mid.text(lx, 0.44, "Worst", ha="center", va="bottom", fontsize=9, fontweight="bold", transform=tax)
    ax_mid.text(rx, 0.44, "Best", ha="center", va="bottom", fontsize=9, fontweight="bold", transform=tax)
    ax_mid.text(
        0.5,
        0.31,
        f"{corner_b_name} wins (marginal CI sum)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#b71c1c",
        transform=tax,
    )
    ax_mid.text(lx, 0.24, f"{_pct_int_from_optional_mass(lo_b_worst)}%", ha="center", va="center", fontsize=12, transform=tax)
    ax_mid.text(rx, 0.24, f"{_pct_int_from_optional_mass(hi_b_best)}%", ha="center", va="center", fontsize=12, transform=tax)
    ax_mid.text(
        0.5,
        0.03,
        "Sums over marginal bounds only — not joint; may exceed 100%.",
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#555555",
        transform=tax,
    )

    # Bottom panel: optional minimum bar width when true mass is under ~3% so labels do not collide.
    min_bar_frac = 0.03
    pv = np.asarray(p, dtype=float)
    raw_w = np.clip(0.82 * pv, 0.012, None)
    widths = np.where(pv < 0.03, np.maximum(raw_w, min_bar_frac), raw_w)
    for i in range(6):
        ax_bot.bar(
            centers[i],
            p[i],
            width=widths[i],
            align="center",
            color=seg_colors[i],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    whisker_gray = "#1f1f1f"
    cap_half = np.clip(widths * 0.42, 0.0048, 0.02)
    ci_side_notes: list[str] = []
    has_above_bar_ci_text = False
    thin_ci_threshold = 0.05

    for i in range(6):
        x = float(centers[i])
        li = float(lo_np[i])
        hi_i = float(hi_np[i])
        w = float(cap_half[i])
        if show_whiskers:
            ax_bot.plot(
                [x, x],
                [li, hi_i],
                color=whisker_gray,
                lw=1.45,
                solid_capstyle="round",
                zorder=5,
            )
            ax_bot.plot([x - w, x + w], [li, li], color=whisker_gray, lw=1.15, zorder=5)
            ax_bot.plot([x - w, x + w], [hi_i, hi_i], color=whisker_gray, lw=1.15, zorder=5)
        ec = seg_colors[i]
        ax_bot.scatter(
            x,
            p[i],
            s=54,
            zorder=7,
            facecolors="white",
            edgecolors=ec,
            linewidths=1.85,
            clip_on=False,
        )
        if show_whiskers:
            lo_pct = int(round(100.0 * li))
            hi_pct = int(round(100.0 * hi_i))
            lbl = f"[{lo_pct}%, {hi_pct}%]"
            is_thin_segment = float(pv[i]) < thin_ci_threshold
            if is_thin_segment:
                ci_side_notes.append(
                    _thin_ci_caption(corner_a_name, corner_b_name, i, lo_pct, hi_pct)
                )
            else:
                y_text = hi_i + 0.018
                ax_bot.text(
                    x,
                    y_text,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=5.9,
                    color="#333333",
                )
                has_above_bar_ci_text = True

    if ci_side_notes and show_whiskers:
        if margin_note is not None:
            ci_xy, ci_ha = (0.012, 0.982), "left"
        else:
            ci_xy, ci_ha = (0.995, 0.982), "right"
        ax_bot.text(
            ci_xy[0],
            ci_xy[1],
            "\n".join(ci_side_notes),
            transform=ax_bot.transAxes,
            ha=ci_ha,
            va="top",
            fontsize=5.35,
            color="#333333",
            linespacing=1.22,
            zorder=12,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#ffffff",
                edgecolor="#e0e0e0",
                alpha=0.94,
                linewidth=0.55,
            ),
        )

    cap_y_base = float(np.max(np.concatenate([hi_np.flatten(), np.asarray(p).flatten()])) * 1.1 + 0.02)
    cap_extra = 0.055 if (show_whiskers and has_above_bar_ci_text) else 0.02
    cap_y = min(1.02, cap_y_base + cap_extra)

    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, max(0.12, cap_y))
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax_bot.set_ylabel("Probability")
    ax_bot.set_xlabel("Outcome")

    ax_bot.set_title(f"Confidence Ranges ({nominal_pct}%)", fontsize=10, color="#333333", pad=5)

    ax_bot.set_xticks(centers)
    ax_bot.set_xticklabels(
        labels,
        rotation=48,
        ha="right",
        fontsize=7,
    )

    if margin_note is not None:
        wrapped = textwrap.fill(margin_note, width=40)
        ax_bot.text(
            1.02,
            0.5,
            wrapped,
            transform=ax_bot.transAxes,
            fontsize=6.5,
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#fff8f0",
                edgecolor="#c9a87c",
                linewidth=0.85,
            ),
            color="#2a2108",
            clip_on=False,
        )


def run_one_fight(
    *,
    predictor: MMAPredictor,
    fighter_a: str,
    fighter_b: str,
    wc,
    fdate: date,
    out_dir: Path,
    tag: str,
) -> None:
    res = predictor.predict(fighter_a, fighter_b, wc, fdate, verbose=False)
    p, lo, hi = _prediction_vectors(res)
    tw = float(res.total_win)
    na = _display_name(predictor, fighter_a)
    nb = _display_name(predictor, fighter_b)
    fight_title = f"{na} vs {nb}"

    show_whiskers = res.ci_method != "cauchy_wc_debut"
    margin_note: str | None = None
    if not show_whiskers:
        insufficient = _fighters_no_prior_in_wc(predictor, fighter_a, fighter_b, wc, fdate)
        who = " and ".join(insufficient) if insufficient else "one or both fighters"
        margin_note = (
            f"Note — insufficient in-division bout history in {wc.value} before this card "
            f"for {who}, so probability ranges are unstable."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 7.25), layout="constrained")
    fig_split_barrier_with_ci(
        fig=fig,
        p=p,
        lo=lo,
        hi=hi,
        title=fight_title,
        tw=tw,
        corner_a_name=na,
        corner_b_name=nb,
        ci_alpha=float(predictor.config.model.ci_alpha),
        margin_note=margin_note,
        show_whiskers=show_whiskers,
    )
    p3 = out_dir / f"{tag}_split_barrier.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p3}")

    s = float(p.sum())
    print(f"  Check: sum(p)={s:.6f}  total_win={tw:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Split-barrier + bars/CIs PNG from one predict() call.")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--fighter-a-id", default="150ff4cc642270b9")
    p.add_argument("--fighter-b-id", default="e1248941344b3288")
    p.add_argument("--weight-class", default="featherweight")
    p.add_argument("--date", default="2022-07-02", help="Fight date YYYY-MM-DD")
    p.add_argument(
        "--all-volk-trilogy",
        action="store_true",
        help="Also generate figures for 2020-07-11 and 2019-12-14 (same pair).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/figures/single_predictions/holloway_volkanovski"),
    )
    args = p.parse_args()

    pred = MMAPredictor.load(args.model_path.resolve())
    wc = resolve_weight_class(args.weight_class)
    out = args.out_dir.resolve()

    dates = [resolve_date(args.date)]
    if args.all_volk_trilogy:
        dates = [
            resolve_date("2019-12-14"),
            resolve_date("2020-07-11"),
            resolve_date("2022-07-02"),
        ]

    for fd in dates:
        tag = str(fd)
        run_one_fight(
            predictor=pred,
            fighter_a=args.fighter_a_id,
            fighter_b=args.fighter_b_id,
            wc=wc,
            fdate=fd,
            out_dir=out,
            tag=tag,
        )


if __name__ == "__main__":
    main()
