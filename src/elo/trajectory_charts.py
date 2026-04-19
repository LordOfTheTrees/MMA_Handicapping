"""
Plot ELO vs time from :meth:`ELOModel.get_trajectory` points.

Requires matplotlib. Trajectory data is produced only when ``process_fights(..., record_trajectories=True)``
(or :meth:`MMAPredictor.build_elo` with ``record_trajectories=True``).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..data.schema import FighterProfile, WeightClass

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .elo import ELOModel

TrajectoryPoints = List[Tuple[date, float, str]]

_DEFAULT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")


def opponent_display_name(opponent_id: str, profiles: Optional[Dict[str, FighterProfile]]) -> str:
    """Map opponent ``fighter_id`` to profile name, or a short id fallback."""
    if not opponent_id:
        return "?"
    if profiles and opponent_id in profiles:
        return profiles[opponent_id].name
    oid = opponent_id.strip()
    return oid[:12] + "…" if len(oid) > 14 else oid


def _annotate_opponent_labels(
    ax: "Axes",
    points: TrajectoryPoints,
    profiles: Optional[Dict[str, FighterProfile]],
    *,
    fontsize: float = 6.0,
) -> None:
    """Label each point with the opponent's name (chart title carries the focal fighter)."""
    for p in points:
        if len(p) < 3:
            continue
        xd, yv, opp_id = p[0], p[1], p[2]
        label = opponent_display_name(opp_id, profiles)
        ax.annotate(
            label,
            xy=(xd, yv),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left",
            va="bottom",
            alpha=0.88,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.75", lw=0.4, alpha=0.92),
        )


def plot_elo_trajectory(
    points: TrajectoryPoints,
    *,
    title: str = "",
    ylabel: str = "ELO (Kalman mean)",
    ax: Optional["Axes"] = None,
    profiles: Optional[Dict[str, FighterProfile]] = None,
    label_opponents: bool = True,
    opponent_label_fontsize: float = 6.0,
) -> Tuple["Figure", "Axes"]:
    """
    Line chart of ELO after each recorded fight.

    When *label_opponents* is True and *profiles* is provided, each point is annotated with
    the **opponent's** display name (the focal fighter is expected in the chart *title*).

    Returns ``(fig, ax)``. If *ax* is ``None``, creates a new figure with ``figsize=(11, 5)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    if not points:
        ax.text(0.5, 0.5, "no trajectory points", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or "ELO trajectory")
        return fig, ax

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color="steelblue", linewidth=1.6, marker="o", markersize=3, alpha=0.9, zorder=2)
    if label_opponents:
        _annotate_opponent_labels(ax, points, profiles, fontsize=opponent_label_fontsize)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, zorder=1)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_elo_trajectories_overlay(
    series: Sequence[Tuple[str, TrajectoryPoints]],
    *,
    title: str = "",
    ax: Optional["Axes"] = None,
    profiles: Optional[Dict[str, FighterProfile]] = None,
    label_opponents: bool = True,
    opponent_label_fontsize: float = 6.0,
) -> Tuple["Figure", "Axes"]:
    """
    Overlay multiple trajectories on one axes. *series* is ``(label, points)`` pairs.

    Opponent labels use the same rules as :func:`plot_elo_trajectory` (division in legend;
    each point shows who they fought).
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
    else:
        fig = ax.figure

    if not series:
        ax.text(0.5, 0.5, "no series", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    try:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", list(_DEFAULT_COLORS))
    except Exception:
        colors = list(_DEFAULT_COLORS)
    if not colors:
        colors = list(_DEFAULT_COLORS)
    for i, (label, points) in enumerate(series):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = colors[i % len(colors)]
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=2,
            alpha=0.9,
            zorder=2 + i,
        )
        if label_opponents:
            _annotate_opponent_labels(ax, points, profiles, fontsize=opponent_label_fontsize)

    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("ELO (Kalman mean)")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def fighter_trajectories_by_division(
    fighter_id: str,
    weight_classes: Iterable[WeightClass],
    get_trajectory,
) -> Dict[WeightClass, TrajectoryPoints]:
    """
    Collect trajectory points for *fighter_id* across divisions.

    *get_trajectory* is typically ``elo_model.get_trajectory`` (bound method or lambda).
    """
    out: Dict[WeightClass, TrajectoryPoints] = {}
    for wc in weight_classes:
        pts = get_trajectory(fighter_id, wc)
        if pts:
            out[wc] = pts
    return out


def save_trajectory_figure(
    fig: "Figure",
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Write *fig* to *out_path* (parent dirs created)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_all_trajectory_charts(
    elo_model: "ELOModel",
    profiles: Dict[str, FighterProfile],
    out_dir: Path,
    *,
    min_points: int = 1,
    max_files: Optional[int] = None,
    dpi: int = 150,
    label_opponents: bool = True,
    opponent_label_fontsize: float = 6.0,
) -> int:
    """
    Write one PNG per ``(fighter_id, weight_class)`` that has recorded trajectory points.

    Filenames: ``{fighter_id}_{weight_class}.png``. Use ``max_files`` to cap volume when
    exploring (full UFC history can produce tens of thousands of charts).
    """
    out_dir = Path(out_dir)
    n = 0
    for fid, wc in elo_model.iter_trajectory_keys():
        pts = elo_model.get_trajectory(fid, wc)
        if len(pts) < min_points:
            continue
        pr = profiles.get(fid)
        label = pr.name if pr else fid
        title = f"{label} — {wc.value}"
        fig, _ = plot_elo_trajectory(
            pts,
            title=title,
            profiles=profiles if profiles else None,
            label_opponents=label_opponents,
            opponent_label_fontsize=opponent_label_fontsize,
        )
        safe_wc = wc.value.replace("/", "-")
        save_trajectory_figure(fig, out_path=out_dir / f"{fid}_{safe_wc}.png", dpi=dpi)
        n += 1
        if max_files is not None and n >= max_files:
            break
    return n
