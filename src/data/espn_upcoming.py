"""
ESPN upcoming events — announced-but-unplayed UFC cards, listings only, **not** merged
into training CSVs (same site-only posture as ``ufcstats_upcoming.py``, ADR-23).

Reuses the exact ``fetch_fightcenter()`` call ``refresh_espn_fights_incremental``
already makes for completed events. That payload's ``cards[*].competitions[]``
already contains non-final (announced) bouts before ``_competition_is_final()``
filters them out for the training path — see
``docs/ufc-com-upcoming-scrape-plan.md`` §0 for the trace through the ingest code.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.espn_client import ESPNClient
from src.data.espn_crosswalk import CrosswalkStore, build_name_index
from src.data.espn_ingest import _PendingEvent, _competition_is_final, _iter_competitions, _resolve_season_events
from src.data.espn_normalize import _athlete_id_from_competitor, normalize_fighter_name, weight_class_from_note
from src.data.loader import _coerce_weight_class_from_cell
from src.data.schema import WeightClass

DEFAULT_ESPN_UPCOMING_CARDS_JSON = "espn_upcoming_cards.json"
FIGHTCENTER_SOURCE = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter"


def _collect_future_events(
    espn: ESPNClient,
    years: List[int],
    *,
    today: date,
) -> List[_PendingEvent]:
    """Upcoming-only twin of ``espn_ingest._collect_incremental_events``.

    Shares the season-index scan (``_resolve_season_events``) with the training-safe
    completed-events path but applies the opposite date filter here, never touching that
    other function (ADR-05: no today-or-future events in ``ufcstats_fights.csv``).
    """
    resolved, _skipped_undated = _resolve_season_events(espn, years)
    return [ev for ev in resolved if ev.event_date >= today and ev.event_id]


def _resolve_known_fighter_id(
    espn_athlete_id: str,
    display_name: str,
    *,
    crosswalk: CrosswalkStore,
    name_index: Dict[str, str],
) -> Optional[str]:
    """Read-only crosswalk/name lookup — never provisions a new fighter ID.

    Unlike the completed-fight path, an announced bout can be cancelled or replaced
    before fight night, and there is no result yet to justify minting a permanent
    UFCStats-style hex ID (ADR-23: upcoming ingestion must not create training-facing
    state). Fighters with no existing crosswalk/profile match ship name-only
    (``fighter_id=None``) until they actually fight and go through the normal
    completed-fight ingest + audit path.
    """
    existing = crosswalk.athlete_to_fighter.get(espn_athlete_id)
    if existing:
        return existing
    return name_index.get(normalize_fighter_name(display_name))


def _parse_future_bouts_from_fightcenter(
    fightcenter: Dict[str, Any],
    event_id: str,
    *,
    crosswalk: CrosswalkStore,
    name_index: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Extract announced (non-final) bouts from one event's fightcenter payload.

    ``bout_order`` follows ``_iter_competitions``'s natural (card-dict-then-list)
    order; whether that lines up with main-card-first display order is unverified
    against live data and should be confirmed locally before relying on it for UI
    ordering.
    """
    bouts: List[Dict[str, Any]] = []
    for i, comp in enumerate(_iter_competitions(fightcenter)):
        if _competition_is_final(comp):
            continue
        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            continue
        id_a = _athlete_id_from_competitor(competitors[0])
        id_b = _athlete_id_from_competitor(competitors[1])
        name_a = ((competitors[0].get("athlete") or {}).get("displayName") or "").strip()
        name_b = ((competitors[1].get("athlete") or {}).get("displayName") or "").strip()
        if not id_a or not id_b or not name_a or not name_b:
            continue

        wc_note = weight_class_from_note(comp.get("note"), (comp.get("type") or {}).get("text"))
        wc_enum, wc_raw = _coerce_weight_class_from_cell(wc_note or "")
        wc_value = wc_enum.value if wc_enum is not None else WeightClass.UNKNOWN.value

        comp_id = str(comp.get("id") or i)
        bouts.append(
            {
                "bout_order": i,
                "fight_id": f"espn_{event_id}_{comp_id}",
                "fight_url": None,
                "fighter_a_id": _resolve_known_fighter_id(
                    id_a, name_a, crosswalk=crosswalk, name_index=name_index
                ),
                "fighter_b_id": _resolve_known_fighter_id(
                    id_b, name_b, crosswalk=crosswalk, name_index=name_index
                ),
                "fighter_a_name": name_a,
                "fighter_b_name": name_b,
                "weight_class": wc_value,
                "weight_class_raw": wc_raw or wc_note,
            }
        )
    return bouts


def scrape_espn_upcoming_cards(
    data_dir: Path,
    *,
    client: Optional[ESPNClient] = None,
    max_events: Optional[int] = None,
    force_season_years: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable document of ESPN-sourced upcoming events and bouts.

    Same output shape as ``ufcstats_upcoming.scrape_upcoming_cards`` (``schema_version``,
    ``source``, ``scraped_at``, ``events[].bouts[]``) so
    ``src/export/upcoming_events_doc.py`` needs no changes to consume either source.
    Does not write the training fights CSV.
    """
    data_dir = Path(data_dir)
    espn = client or ESPNClient(cache_dir=data_dir / "cache" / "espn")
    crosswalk = CrosswalkStore(data_dir)
    name_index = build_name_index(data_dir / "fighter_profiles.csv", crosswalk)

    today = date.today()
    if force_season_years is not None:
        years = force_season_years
    else:
        # ESPN "season" year can lag the calendar year for early-year cards (same
        # caveat as refresh_espn_fights_incremental); keep the prior year too.
        years = [y for y in espn.list_season_years() if y >= today.year - 1]

    pending = _collect_future_events(espn, years, today=today)
    if max_events is not None:
        pending = pending[:max_events]

    events_out: List[Dict[str, Any]] = []
    for ev in pending:
        try:
            fightcenter = espn.fetch_fightcenter(ev.event_id)
        except RuntimeError as e:
            print(f"[espn upcoming] skip {ev.event_id} ({ev.event_name}): {e}", flush=True)
            continue
        bouts = _parse_future_bouts_from_fightcenter(
            fightcenter, ev.event_id, crosswalk=crosswalk, name_index=name_index
        )
        events_out.append(
            {
                "event_url": None,
                "event_id": ev.event_id,
                "event_title": ev.event_name,
                "event_date": ev.event_date.isoformat(),
                "location": None,
                "bouts": bouts,
            }
        )

    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "source": FIGHTCENTER_SOURCE,
        "scraped_at": scraped_at,
        "events": events_out,
    }


def scrape_espn_upcoming_cards_to_path(
    path: Path,
    data_dir: Path,
    *,
    client: Optional[ESPNClient] = None,
    max_events: Optional[int] = None,
    force_season_years: Optional[List[int]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = scrape_espn_upcoming_cards(
        data_dir, client=client, max_events=max_events, force_season_years=force_season_years
    )
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Scrape ESPN upcoming events into JSON (not for training).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None, help="Output file (default: data-dir/espn_upcoming_cards.json)")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args(argv)

    out = args.out or (Path(args.data_dir) / DEFAULT_ESPN_UPCOMING_CARDS_JSON)
    scrape_espn_upcoming_cards_to_path(out, args.data_dir, max_events=args.max_events)
    print(f"Wrote {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
