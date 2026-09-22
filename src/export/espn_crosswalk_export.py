"""
UFCStats ↔ ESPN id mappings, exported so consumers can join on their own ids.

The crosswalk lives only in this repo's ``data/espn_crosswalk_*.csv``, so anything downstream
holding ESPN ids — which is what the deploy repo has — could not resolve them to the internal
hex ids every other artifact is keyed by. This publishes the mapping itself rather than
pre-joining it into each artifact, so one small file unblocks every join instead of each
export baking in one.

Fight ids resolve to ``espn_<event_id>_<competition_id>``, the shape
:mod:`src.data.espn_upcoming` mints for scheduled bouts and therefore the id the deploy repo
already holds for a card.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional

from src.data.espn_crosswalk import CrosswalkStore, espn_placeholder_competition_id

ESPN_CROSSWALK_FILENAME = "espn_crosswalk.json"

ID_SCHEME = "espn_<event_id>_<competition_id>"


def espn_form_fight_id(crosswalk: CrosswalkStore, fight_id: str) -> Optional[str]:
    """
    ``espn_<event_id>_<competition_id>`` for an internal *fight_id*, or ``None`` when unmapped.

    Handles both a UFCStats hex id carried in the crosswalk and an ``espn_<competition_id>``
    placeholder minted for a bout absent from UFCStats. Returns ``None`` when the fight has no
    crosswalk row, or when that row never recorded an event id.
    """
    fid = (fight_id or "").strip()
    if not fid:
        return None
    competition = crosswalk.fight_to_competition.get(fid) or espn_placeholder_competition_id(fid)
    if not competition:
        return None
    event = crosswalk.espn_event_id_for_competition(competition)
    if not event:
        return None
    return f"espn_{event}_{competition}"


def build_espn_crosswalk_document(
    crosswalk: CrosswalkStore,
    *,
    manifest: Mapping[str, Any],
    export_schema_version: str,
    as_of: date,
) -> dict[str, Any]:
    """
    Full document for :data:`ESPN_CROSSWALK_FILENAME`.

    ``fights`` maps internal fight id to ESPN-form fight id; ``fighters`` maps internal fighter
    id to ESPN athlete id. Both are sorted for stable diffs. A fight whose crosswalk row never
    recorded an event id cannot be expressed in the ESPN form, so it is counted under
    ``counts.fights_without_event_id`` rather than emitted half-resolved.
    """
    fights: dict[str, str] = {}
    n_without_event = 0
    for internal_id in crosswalk.fight_to_competition:
        espn_id = espn_form_fight_id(crosswalk, internal_id)
        if espn_id is None:
            n_without_event += 1
            continue
        fights[internal_id] = espn_id

    fighters = {
        internal_id: athlete_id
        for internal_id, athlete_id in crosswalk.fighter_to_athlete.items()
        if internal_id and athlete_id
    }

    return {
        "export_manifest": dict(manifest),
        "export_schema_version": export_schema_version,
        "as_of_date": as_of.isoformat(),
        "id_scheme": ID_SCHEME,
        "counts": {
            "fights": len(fights),
            "fighters": len(fighters),
            "fights_without_event_id": n_without_event,
        },
        "fights": dict(sorted(fights.items())),
        "fighters": dict(sorted(fighters.items())),
        "notes": (
            "Internal UFCStats-style ids -> ESPN ids. fights values use "
            f"{ID_SCHEME}; fighters values are bare ESPN athlete ids. Fight outcomes live on "
            "the per-fighter points in fighter_profiles.elo_trajectories (fight_id, "
            "result_method, outcome_class), which this file resolves to ESPN ids."
        ),
    }
