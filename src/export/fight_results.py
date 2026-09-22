"""
Settled fight outcomes — winner and finishing method — keyed by ESPN-form fight id.

:class:`~src.data.schema.FightRecord` already carries ``winner_id`` and ``result_method``, but
no other export emits them, so a consumer can at best infer win/loss from the sign of an ELO
delta and never the method. This module closes that gap: it is the authoritative source for
"what actually happened", the other half of a scored prediction.

Rows are keyed by ``espn_<event_id>_<competition_id>`` — the same shape
``src.data.espn_upcoming`` mints for scheduled bouts and therefore the id the deploy repo
already holds. Exporting the internal UFCStats hex id instead would force every consumer to
re-derive the crosswalk. Fights with no ESPN mapping are counted, not emitted, since a consumer
that only knows ESPN ids cannot use them.

Each row also carries ``outcome_class_a``: the 0-5 class index from corner A's perspective
(:data:`~src.model.regression.CLASS_LABELS`), so scoring a prediction is a direct index into
the exported six-way probability vector rather than a re-derivation of the encoding.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping, Optional

from src.data.espn_crosswalk import CrosswalkStore, espn_placeholder_competition_id
from src.data.schema import FightRecord
from src.model.regression import CLASS_LABELS, encode_outcome

FIGHT_RESULTS_FILENAME = "fight_results.json"

#: ``result_method`` values that leave no winner; ``outcome_class_a`` is ``None`` for these.
UNDECIDED_METHODS = ("draw", "no_contest", "dq")

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


def _espn_athlete(crosswalk: CrosswalkStore, fighter_id: str) -> Optional[str]:
    return crosswalk.fighter_to_athlete.get((fighter_id or "").strip()) or None


def _winner_corner(fight: FightRecord) -> Optional[str]:
    if fight.winner_id is None:
        return None
    if fight.winner_id == fight.fighter_a_id:
        return "a"
    if fight.winner_id == fight.fighter_b_id:
        return "b"
    return None


def fight_result_row(
    fight: FightRecord,
    crosswalk: CrosswalkStore,
    espn_fight_id: str,
) -> dict[str, Any]:
    """One exported row for *fight*, already known to map to *espn_fight_id*."""
    competition = espn_fight_id.split("_", 2)[2]
    event = espn_fight_id.split("_", 2)[1]
    return {
        "espn_fight_id": espn_fight_id,
        "espn_event_id": event,
        "espn_competition_id": competition,
        "fight_id": fight.fight_id,
        "fight_date": fight.fight_date.isoformat(),
        "weight_class": fight.weight_class.value,
        "promotion": fight.promotion,
        "tier": int(fight.tier.value),
        "fighter_a_id": fight.fighter_a_id,
        "fighter_b_id": fight.fighter_b_id,
        "espn_athlete_a_id": _espn_athlete(crosswalk, fight.fighter_a_id),
        "espn_athlete_b_id": _espn_athlete(crosswalk, fight.fighter_b_id),
        "winner_id": fight.winner_id,
        "winner_corner": _winner_corner(fight),
        "espn_winner_athlete_id": (
            _espn_athlete(crosswalk, fight.winner_id) if fight.winner_id else None
        ),
        "result_method": fight.result_method.value,
        "outcome_class_a": encode_outcome(fight, fight.fighter_a_id),
    }


def build_fight_results_document(
    fights: Iterable[FightRecord],
    crosswalk: CrosswalkStore,
    *,
    manifest: Mapping[str, Any],
    export_schema_version: str,
    as_of: date,
) -> dict[str, Any]:
    """
    Full document for :data:`FIGHT_RESULTS_FILENAME`.

    Rows land under ``results``, keyed by ESPN-form fight id and ordered by ``fight_date`` then
    key so re-exports diff cleanly. ``counts`` reports what was dropped: ``unmapped`` fights
    carry no ESPN crosswalk row, ``undecided`` ones (draw / no contest / DQ) are still exported
    but have a ``null`` ``winner_id`` and ``outcome_class_a``.

    Later fights win a key collision, which cannot normally happen — one ESPN competition is one
    bout — but is counted rather than silently resolved.
    """
    rows: dict[str, dict[str, Any]] = {}
    n_total = 0
    n_unmapped = 0
    n_undecided = 0
    n_collisions = 0

    for fight in fights:
        n_total += 1
        key = espn_form_fight_id(crosswalk, fight.fight_id)
        if key is None:
            n_unmapped += 1
            continue
        row = fight_result_row(fight, crosswalk, key)
        if row["outcome_class_a"] is None:
            n_undecided += 1
        if key in rows:
            n_collisions += 1
            if rows[key]["fight_date"] > row["fight_date"]:
                continue
        rows[key] = row

    ordered = sorted(rows.values(), key=lambda r: (r["fight_date"], r["espn_fight_id"]))

    return {
        "export_manifest": dict(manifest),
        "export_schema_version": export_schema_version,
        "as_of_date": as_of.isoformat(),
        "id_scheme": ID_SCHEME,
        "class_labels": list(CLASS_LABELS),
        "undecided_methods": list(UNDECIDED_METHODS),
        "counts": {
            "fights_seen": n_total,
            "exported": len(ordered),
            "unmapped_no_crosswalk": n_unmapped,
            "undecided_no_winner": n_undecided,
            "key_collisions": n_collisions,
        },
        "results": {r["espn_fight_id"]: r for r in ordered},
        "notes": (
            "winner_id / result_method per settled bout, keyed by "
            f"{ID_SCHEME}. outcome_class_a indexes class_labels from corner A's perspective "
            "(null for draw / no contest / DQ), matching the six-way probability vector "
            "produced from model_weights.json."
        ),
    }
