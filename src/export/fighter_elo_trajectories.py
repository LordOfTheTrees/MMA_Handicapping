"""Serialize fight-by-fight ELO trajectories from :class:`~src.elo.elo.ELOModel` for JSON export."""
from __future__ import annotations

from typing import Any

from src.elo.elo import ELOModel


def nested_elo_trajectories_by_fighter(em: ELOModel) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Map ``fighter_id -> weight_class_value -> [{fight_date, elo, opponent_fighter_id}, ...]``.

    Points exist only when the model was built with ``record_trajectories=True``.
    Each row is post-fight Kalman mean ELO in that division; ``opponent_fighter_id`` may be
    ``null`` if unknown.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fid, wc in em.iter_trajectory_keys():
        pts = em.get_trajectory(fid, wc)
        if not pts:
            continue
        wc_key = wc.value
        out.setdefault(fid, {})
        out[fid][wc_key] = [
            {
                "fight_date": d.isoformat(),
                "elo": float(elo),
                "opponent_fighter_id": oid if oid else None,
            }
            for d, elo, oid in pts
        ]
    return out
