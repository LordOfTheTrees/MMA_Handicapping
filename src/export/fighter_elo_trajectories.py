"""Serialize fight-by-fight ELO trajectories from :class:`~src.elo.elo.ELOModel` for JSON export."""
from __future__ import annotations

from typing import Any

from src.elo.elo import ELOModel, TrajectoryPoint
from src.model.regression import outcome_class_for


def _point_row(fighter_id: str, p: TrajectoryPoint) -> dict[str, Any]:
    """
    One exported trajectory point, from *fighter_id*'s perspective.

    ``outcome_class`` indexes :data:`~src.model.regression.CLASS_LABELS` for this fighter — 0-2
    are wins, 3-5 losses — and is ``None`` for a draw, no contest or DQ. It is derived here
    rather than recomputed downstream so the ``(won, method) -> class`` encoding keeps exactly
    one definition; ``result_method`` alone would force every consumer to restate it.
    """
    method = p.result_method
    outcome_class = (
        outcome_class_for(p.winner_id == fighter_id, method)
        if method is not None and p.winner_id
        else None
    )
    return {
        "fight_date": p.fight_date.isoformat(),
        "elo": float(p.elo),
        "opponent_fighter_id": p.opponent_fighter_id or None,
        "fight_id": p.fight_id or None,
        "result_method": method.value if method is not None else None,
        "outcome_class": outcome_class,
    }


def nested_elo_trajectories_by_fighter(em: ELOModel) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Map ``fighter_id -> weight_class_value -> [point, ...]``.

    Points exist only when the model was built with ``record_trajectories=True``. Each point is
    the post-fight Kalman mean ELO in that division plus the bout that produced it: its
    ``fight_id``, ``opponent_fighter_id``, ``result_method`` and ``outcome_class``. Together
    these make the trajectory a complete per-fighter fight log — a consumer reading a rating
    curve can annotate every point with what happened, and one holding a fighter, division and
    date can resolve a result without a separate lookup.

    Fields are ``null`` where unknown: ``opponent_fighter_id`` when the bout recorded no
    opponent, and the outcome fields for points restored from a pickle predating outcome
    recording.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fid, wc in em.iter_trajectory_keys():
        pts = em.get_trajectory(fid, wc)
        if not pts:
            continue
        out.setdefault(fid, {})
        out[fid][wc.value] = [_point_row(fid, p) for p in pts]
    return out
