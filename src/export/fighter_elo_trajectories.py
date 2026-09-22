"""Serialize fight-by-fight ELO trajectories from :class:`~src.elo.elo.ELOModel` for JSON export."""
from __future__ import annotations

from typing import Any, Mapping

from src.elo.elo import ELOModel, TrajectoryPoint
from src.model.regression import outcome_class_for


class MissingTrajectoryOutcomes(RuntimeError):
    """An export would ship ``fighter_profiles.json`` with no fight outcomes on its points."""


def _point_row(fighter_id: str, p: TrajectoryPoint) -> dict[str, Any]:
    """
    One exported trajectory point, from *fighter_id*'s perspective.

    ``outcome_class`` indexes :data:`~src.model.regression.CLASS_LABELS` for this fighter — 0-2
    are wins, 3-5 losses — and is ``None`` for a draw, no contest or DQ. It is derived here
    rather than recomputed downstream so the ``(won, method) -> class`` encoding keeps exactly
    one definition.

    ``result_method`` is emitted **only when** ``outcome_class`` is ``None``. For a decisive
    bout the class already names the method (0/4 KO/TKO, 1/5 submission, 2/3 decision), so
    carrying it on every point cost ~0.89 MB to repeat what the reader can infer; the one thing
    it alone could express — unanimous vs split vs majority — is not worth that. For a draw, no
    contest or DQ there is no class, and the method is the only description of what happened, so
    the key appears. Its absence therefore means "decisive; read outcome_class".
    """
    method = p.result_method
    outcome_class = (
        outcome_class_for(p.winner_id == fighter_id, method)
        if method is not None and p.winner_id
        else None
    )
    row = {
        "fight_date": p.fight_date.isoformat(),
        "elo": float(p.elo),
        "opponent_fighter_id": p.opponent_fighter_id or None,
        "fight_id": p.fight_id or None,
        "outcome_class": outcome_class,
    }
    if outcome_class is None and method is not None:
        row["result_method"] = method.value
    return row


def assert_trajectories_carry_outcomes(
    by_fighter: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> None:
    """
    Raise :class:`MissingTrajectoryOutcomes` unless the nested trajectories carry fight outcomes.

    Outcomes ride on these points and nothing else exports them, so an export that silently
    omits them ships a site whose result-dependent pages have no results. That failure is
    invisible in a green pipeline, which is why this is an exception and not a warning.
    """
    if not by_fighter:
        raise MissingTrajectoryOutcomes(
            "No ELO trajectories recorded, so fighter_profiles.json would ship no fight outcomes "
            "at all. Rebuild ELO with record_trajectories=True "
            "(export_artifacts.py --rebuild-elo-for-trajectories), or pass "
            "--allow-missing-trajectories to export a knowingly incomplete bundle."
        )

    n_points = 0
    n_identified = 0
    for by_wc in by_fighter.values():
        for series in by_wc.values():
            for point in series:
                n_points += 1
                if point.get("fight_id"):
                    n_identified += 1

    if n_identified == 0:
        raise MissingTrajectoryOutcomes(
            f"All {n_points:,} ELO trajectory points lack a fight_id, so this model predates "
            "outcome recording and fighter_profiles.json would ship ratings with no results. "
            "Rebuild ELO with record_trajectories=True "
            "(export_artifacts.py --rebuild-elo-for-trajectories), or pass "
            "--allow-missing-trajectories to export a knowingly incomplete bundle."
        )


def nested_elo_trajectories_by_fighter(em: ELOModel) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Map ``fighter_id -> weight_class_value -> [point, ...]``.

    Points exist only when the model was built with ``record_trajectories=True``. Each point is
    the post-fight Kalman mean ELO in that division plus the bout that produced it: its
    ``fight_id``, ``opponent_fighter_id`` and ``outcome_class``, with ``result_method`` present
    only on non-decisive bouts (see :func:`_point_row`). Together these make the trajectory a
    complete per-fighter fight log — a consumer reading a rating curve can annotate every point
    with what happened, and one holding a fighter, division and date can resolve a result
    without a separate lookup.

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
