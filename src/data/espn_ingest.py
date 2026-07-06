"""
Incremental ESPN → ``ufcstats_fights.csv`` ingest with crosswalk ID preservation.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.espn_client import ESPNClient, ESPN_REQUEST_DELAY_SEC
from src.data.espn_crosswalk import (
    BoutIdentity,
    CrosswalkStore,
    _load_profiles_by_id,
    build_fight_index_from_csv,
    build_name_index,
    espn_placeholder_athlete_id,
    provision_ufcstats_fighter_id,
    remap_espn_placeholder_fight_ids,
    remap_espn_placeholders_in_fight_rows,
    resolve_fight_id,
    resolve_fighter_id,
)
from src.data.espn_normalize import (
    _athlete_id_from_competitor,
    build_fight_csv_row,
    espn_method_to_csv,
    fight_time_sec_from_status,
    normalize_fighter_name,
    parse_competitor_side,
    parse_event_date,
    weight_class_from_note,
)
from src.data.tier1_csv import CSV_FIELDS, DEFAULT_UFCSTATS_FIGHTS_CSV, LEGACY_FIGHTS_CSV

ESPN_INGEST_STATE_JSON = "espn_ingest_state.json"


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


@dataclass(frozen=True)
class _PendingEvent:
    event_ref: str
    event_id: str
    event_date: date
    event_name: str


def _resolve_season_events(espn: ESPNClient, years: List[int]) -> Tuple[List[_PendingEvent], int]:
    """
    Resolve event id/date/name for every event ref across the given season years.

    Shared scan step for both the training-safe completed-events path
    (:func:`_collect_incremental_events`, below) and the upcoming-events path
    (``espn_upcoming._collect_future_events``, future events only). This function applies
    **no** date filtering itself beyond dropping undated events (unusable by either caller)
    so the two call sites can never accidentally diverge on how they fetch/parse event
    metadata — each applies its own before/after-``today`` filter on the returned list.

    Returns ``(resolved_sorted_by_date, skipped_undated_count)``.
    """
    resolved: List[_PendingEvent] = []
    skipped_undated = 0
    for year in years:
        for event_ref in espn.list_event_refs(year):
            event = espn.fetch_event(event_ref)
            event_id = str(event.get("id") or "")
            event_date = parse_event_date(event.get("date") or "")
            event_name = (event.get("name") or event.get("shortName") or event_id).strip()
            if event_date is None:
                skipped_undated += 1
                continue
            resolved.append(
                _PendingEvent(
                    event_ref=event_ref,
                    event_id=event_id,
                    event_date=event_date,
                    event_name=event_name,
                )
            )
    resolved.sort(key=lambda e: (e.event_date, e.event_id))
    return resolved, skipped_undated


def _collect_incremental_events(
    espn: ESPNClient,
    years: List[int],
    *,
    max_date: Optional[date],
    today: date,
) -> Tuple[List[_PendingEvent], int]:
    """
    Resolve ESPN event metadata and return only **completed** cards we will pull (ADR-05).

    Returns ``(pending_sorted_by_date, skipped_count)`` where *skipped* is how
    many indexed events were dropped (before watermark, undated, or not yet held).
    """
    resolved, skipped = _resolve_season_events(espn, years)
    pending: List[_PendingEvent] = []
    for ev in resolved:
        if ev.event_date >= today:
            skipped += 1
            continue
        if max_date is not None and ev.event_date < max_date:
            skipped += 1
            continue
        pending.append(ev)
    return pending, skipped


def _fights_csv_path(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    primary = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    if primary.is_file():
        return primary
    legacy = data_dir / LEGACY_FIGHTS_CSV
    return legacy if legacy.is_file() else primary


def _load_ingest_state(data_dir: Path) -> Dict[str, Any]:
    path = Path(data_dir) / ESPN_INGEST_STATE_JSON
    if not path.is_file():
        return {"scraped_competition_ids": [], "seasons_touched": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_ingest_state(data_dir: Path, state: Dict[str, Any]) -> None:
    path = Path(data_dir) / ESPN_INGEST_STATE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _load_fights_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(row.get("fight_id") or "").strip(): row for row in csv.DictReader(f) if row.get("fight_id")}


def _max_fight_date(rows: Dict[str, Dict[str, Any]]) -> Optional[date]:
    best: Optional[date] = None
    for row in rows.values():
        raw = (row.get("date") or "").strip()
        if not raw:
            continue
        try:
            d = date.fromisoformat(raw)
        except ValueError:
            continue
        if best is None or d > best:
            best = d
    return best


def _write_fights_csv(path: Path, rows: Dict[str, Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r.get("date") or "", r.get("fight_id") or ""))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(ordered)
    return len(ordered)


def _iter_competitions(fightcenter: Dict[str, Any]) -> List[Dict[str, Any]]:
    comps: List[Dict[str, Any]] = []
    for card in (fightcenter.get("cards") or {}).values():
        if not isinstance(card, dict):
            continue
        for comp in card.get("competitions") or []:
            if isinstance(comp, dict):
                comps.append(comp)
    return comps


def _competition_is_final(comp: Dict[str, Any]) -> bool:
    st = comp.get("status") or {}
    if isinstance(st, dict) and st.get("$ref"):
        return True
    if not isinstance(st, dict):
        return False
    t = st.get("type")
    if isinstance(t, dict):
        if t.get("completed"):
            return True
        name = (t.get("name") or "").upper()
        if name.startswith("STATUS_FINAL") or (t.get("state") or "").lower() == "post":
            return True
    return False


def refresh_espn_fights_incremental(
    data_dir: Path,
    *,
    client: Optional[ESPNClient] = None,
    min_season_year: Optional[int] = None,
    max_seasons: Optional[int] = None,
    max_events: Optional[int] = None,
    max_competitions: Optional[int] = None,
    force_season_years: Optional[List[int]] = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Fetch new/updated UFC bouts from ESPN and merge into ``ufcstats_fights.csv``.

    Returns ``(total_rows_written, rows_updated_or_added_this_run)``.

    Existing rows are preserved by ``fight_id``; stats refresh when ESPN data
    is available. Never returns 0 rows when a non-empty CSV already exists and the
    network fails mid-run (caller should treat exceptions separately).
    """
    data_dir = Path(data_dir)
    fights_path = _fights_csv_path(data_dir)
    profiles_path = data_dir / "fighter_profiles.csv"
    existing = _load_fights_rows(fights_path)
    existing_count = len(existing)
    max_date = _max_fight_date(existing)

    crosswalk = CrosswalkStore(data_dir)
    profiles_by_id = _load_profiles_by_id(profiles_path)
    fight_index = (
        build_fight_index_from_csv(fights_path, profiles_path, crosswalk)
        if fights_path.is_file()
        else {}
    )
    name_index = build_name_index(profiles_path, crosswalk)

    state = _load_ingest_state(data_dir)
    scraped: Set[str] = set(state.get("scraped_competition_ids") or [])
    last_run: Dict[str, Any] = {"new_fighters": [], "new_fights": []}

    espn = client or ESPNClient(cache_dir=data_dir / "cache" / "espn")
    years = force_season_years or espn.list_season_years()
    if min_season_year is not None:
        years = [y for y in years if y >= min_season_year]
    if max_date is not None and not force_season_years:
        # ESPN "season" year != calendar year of every card; keep prior season in case
        # early-2026 cards live under the previous season slug, but skip old events below.
        years = [y for y in years if y >= max_date.year - 1]
    if max_seasons is not None:
        years = years[-max_seasons:]

    today = date.today()
    pending, skipped_indexed = _collect_incremental_events(
        espn, years, max_date=max_date, today=today
    )
    if max_events is not None:
        pending = pending[:max_events]

    _log(
        verbose,
        f"[espn incremental] data_dir={data_dir} fights={existing_count:,} rows "
        f"delay={espn.request_delay_sec}s cache={espn.cache_dir}",
    )
    if max_date is not None:
        _log(
            verbose,
            f"[espn incremental] latest CSV fight date: {max_date} — "
            f"pulling {len(pending)} event(s) on/after that date",
        )
    else:
        _log(
            verbose,
            f"[espn incremental] no fight dates in CSV — pulling {len(pending)} completed event(s)",
        )
    if skipped_indexed:
        _log(
            verbose,
            f"[espn incremental] ({skipped_indexed} older/future/undated cards in ESPN index — skipped)",
        )

    updated = 0
    n_pull = len(pending)

    for i, ev in enumerate(pending, start=1):
        try:
            fightcenter = espn.fetch_fightcenter(ev.event_id)
        except RuntimeError as e:
            _log(verbose, f"  [{i}/{n_pull}] {ev.event_date} {ev.event_name} | skip: {e}")
            continue

        comps = _iter_competitions(fightcenter)
        event_updated = 0
        for comp in comps:
            if max_competitions is not None and updated >= max_competitions:
                break
            comp_id = str(comp.get("id") or "")
            if not comp_id:
                continue
            if not _competition_is_final(comp):
                continue

            try:
                    row, ingest_meta = _ingest_competition(
                        espn,
                        event_id=ev.event_id,
                        event_date=ev.event_date,
                        competition=comp,
                        crosswalk=crosswalk,
                        fight_index=fight_index,
                        name_index=name_index,
                        profiles_by_id=profiles_by_id,
                        taken_fight_ids=set(existing.keys()),
                    )
            except RuntimeError as e:
                _log(verbose, f"    skip bout {comp_id}: {e}")
                continue
            if row is None:
                continue

            _append_last_run_entries(last_run, ingest_meta)

            fid = row["fight_id"]
            if existing.get(fid) != row:
                existing[fid] = row
                updated += 1
                event_updated += 1
            scraped.add(comp_id)

        _log(
            verbose,
            f"  [{i}/{n_pull}] {ev.event_date} {ev.event_name} | "
            f"{len(comps)} bouts | {event_updated} row updates",
        )

        if max_competitions is not None and updated >= max_competitions:
            break

    repair_espn_veteran_placeholders(
        existing,
        crosswalk,
        profiles_by_id,
        espn=espn,
        verbose=verbose,
    )
    _prune_resolved_fighters_from_last_run(last_run, crosswalk)
    crosswalk.save()
    state["scraped_competition_ids"] = sorted(scraped)
    state["seasons_touched"] = sorted(set(state.get("seasons_touched") or []) | set(years))
    state["last_run"] = last_run
    _save_ingest_state(data_dir, state)

    if not existing and existing_count > 0:
        return existing_count, 0
    if not existing:
        print("::warning::ESPN ingest produced 0 fight rows.", flush=True)
        return 0, 0

    total = _write_fights_csv(fights_path, existing)
    _log(
        verbose,
        f"[espn incremental] HTTP summary: {espn.network_requests} live requests, "
        f"{espn.cache_hits} cache hits",
    )
    print(
        f"[espn] Wrote {fights_path.name}: {total:,} rows ({updated} updated/added this run).",
        flush=True,
    )
    return total, updated


def repair_espn_veteran_placeholders(
    fights_rows: Dict[str, Dict[str, Any]],
    crosswalk: CrosswalkStore,
    profiles_by_id: Dict[str, Dict[str, str]],
    *,
    espn: ESPNClient,
    verbose: bool = True,
) -> int:
    """
    Upgrade ``espn_*`` placeholder fighter ids to provisioned hex ids when ESPN
    eventlog shows more than one UFC bout (not a debut).
    """
    from src.data.espn_audit import audit_rookie_ufc_history

    placeholder_aids: Set[str] = set()
    for row in fights_rows.values():
        for col in ("fighter_a_id", "fighter_b_id", "winner_id"):
            aid = espn_placeholder_athlete_id((row.get(col) or "").strip())
            if aid:
                placeholder_aids.add(aid)

    provisioned = 0
    for aid in sorted(placeholder_aids):
        mapped = crosswalk.athlete_to_fighter.get(aid)
        if mapped and not str(mapped).startswith("espn_"):
            continue
        try:
            athlete = espn.fetch_athlete(aid)
        except RuntimeError as e:
            _log(verbose, f"[espn repair] skip athlete {aid}: {e}")
            continue
        name = (athlete.get("displayName") or athlete.get("fullName") or aid).strip()
        rookie = audit_rookie_ufc_history(espn, aid, athlete_payload=athlete)
        ufc_n = rookie.get("ufc_bouts_in_eventlog")
        if ufc_n is None or ufc_n <= 1:
            continue
        taken: Set[str] = set(crosswalk.athlete_to_fighter.values())
        taken |= set(profiles_by_id.keys())
        ufc_id = provision_ufcstats_fighter_id(aid, taken)
        crosswalk.record_fighter(
            ufcstats_fighter_id=ufc_id,
            espn_athlete_id=aid,
            fighter_name=name,
            match_method="espn_veteran",
        )
        provisioned += 1
        _log(
            verbose,
            f"[espn repair] {name} ({aid}) -> {ufc_id} "
            f"({ufc_n} UFC bouts in eventlog)",
        )

    remapped = remap_espn_placeholders_in_fight_rows(fights_rows, crosswalk)
    fights_reid = remap_espn_placeholder_fight_ids(fights_rows, crosswalk)
    if provisioned or remapped or fights_reid:
        _log(
            verbose,
            f"[espn repair] provisioned {provisioned} fighter(s), "
            f"remapped {remapped} fighter cell(s), {fights_reid} fight id(s)",
        )
    return provisioned


def _prune_resolved_fighters_from_last_run(
    last_run: Dict[str, Any],
    crosswalk: CrosswalkStore,
) -> None:
    """Drop ``new_fighters`` entries that now map to a non-placeholder hex id."""
    kept: List[Dict[str, Any]] = []
    for entry in last_run.get("new_fighters") or []:
        aid = (entry.get("espn_athlete_id") or "").strip()
        mapped = crosswalk.athlete_to_fighter.get(aid) if aid else ""
        if mapped and not str(mapped).startswith("espn_"):
            continue
        kept.append(entry)
    last_run["new_fighters"] = kept

    kept_fights: List[Dict[str, Any]] = []
    for entry in last_run.get("new_fights") or []:
        comp = (entry.get("espn_competition_id") or "").strip()
        fid = (entry.get("fight_id") or "").strip()
        mapped = crosswalk.competition_to_fight.get(comp) if comp else ""
        if mapped and not str(mapped).startswith("espn_"):
            continue
        fa = (entry.get("fighter_a_id") or "").strip()
        fb = (entry.get("fighter_b_id") or "").strip()
        if fid.startswith("espn_") and fa and fb and not fa.startswith("espn_") and not fb.startswith("espn_"):
            continue
        kept_fights.append(entry)
    last_run["new_fights"] = kept_fights


def _append_last_run_entries(last_run: Dict[str, Any], meta: Dict[str, Any]) -> None:
    if meta.get("fight_match") == "espn_new":
        last_run.setdefault("new_fights", []).append(meta.get("fight_entry") or {})
    for fe in meta.get("new_fighter_entries") or []:
        last_run.setdefault("new_fighters", []).append(fe)


def _ingest_competition(
    espn: ESPNClient,
    *,
    event_id: str,
    event_date: date,
    competition: Dict[str, Any],
    crosswalk: CrosswalkStore,
    fight_index: Dict[Tuple[str, Tuple[str, str]], str],
    name_index: Dict[str, str],
    profiles_by_id: Optional[Dict[str, Dict[str, str]]] = None,
    taken_fight_ids: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    comp_id = str(competition.get("id") or "")
    empty_meta: Dict[str, Any] = {}
    competitors_raw = competition.get("competitors") or []
    if len(competitors_raw) != 2:
        return None, empty_meta

    # Prefer embedded competitors; fall back to core API list.
    comp_payload = espn.fetch_competition(event_id, comp_id)
    status_ref = (comp_payload.get("status") or {}).get("$ref")
    status = espn.fetch_competition_status(status_ref) if status_ref else {}
    result_name = ((status.get("result") or {}).get("name")) if isinstance(status.get("result"), dict) else None
    method = espn_method_to_csv(result_name)
    if method is None:
        return None, empty_meta

    round_len = 300
    fmt = comp_payload.get("format") or {}
    reg = fmt.get("regulation") or {}
    if reg.get("clock"):
        try:
            round_len = int(float(reg["clock"]))
        except (TypeError, ValueError):
            pass
    fight_time = fight_time_sec_from_status(status, round_length_sec=round_len)

    wc = weight_class_from_note(competition.get("note"), (competition.get("type") or {}).get("text"))

    if not wc:
        return None, empty_meta

    sides: List[Tuple[str, str, bool, Dict[str, Optional[int]]]] = []
    comp_refs = espn.list_competitor_refs(event_id, comp_id)
    if len(comp_refs) != 2:
        return None, empty_meta
    for ref in comp_refs:
        comptr = espn.fetch_competitor(ref)
        ath = comptr.get("athlete") or {}
        if isinstance(ath, dict) and ath.get("$ref") and not ath.get("displayName"):
            ath = espn.get_json(ath["$ref"])
            comptr = {**comptr, "athlete": ath}
        st_ref = (comptr.get("statistics") or {}).get("$ref")
        if not st_ref:
            return None, empty_meta
        stats_payload = espn.fetch_competitor_statistics(st_ref)
        sides.append(parse_competitor_side(comptr, stats_payload))

    if len(sides) != 2:
        return None, empty_meta

    names = (sides[0][1], sides[1][1])
    bout = BoutIdentity(
        espn_competition_id=comp_id,
        espn_event_id=event_id,
        event_date=event_date,
        espn_athlete_ids=(sides[0][0], sides[1][0]),
        fighter_names=names,
    )

    fighter_ids: List[str] = []
    fighter_meta: List[Dict[str, Any]] = []
    winner_id = ""
    for espn_aid, display_name, winner, stats in sides:
        ufc_fid, fmatch = resolve_fighter_id(
            espn_aid,
            display_name,
            crosswalk=crosswalk,
            name_index=name_index,
            profiles_by_id=profiles_by_id,
            espn=espn,
        )
        fighter_ids.append(ufc_fid)
        if fmatch in ("name", "fuzzy_phys", "fuzzy_name", "espn_veteran"):
            name_index[normalize_fighter_name(display_name)] = ufc_fid
        fighter_meta.append(
            {
                "espn_athlete_id": espn_aid,
                "fighter_id": ufc_fid,
                "display_name": display_name,
                "match_method": fmatch,
            }
        )
        if winner:
            winner_id = ufc_fid

    if method in ("draw", "no contest"):
        winner_id = ""

    taken: Set[str] = set(taken_fight_ids or ())
    taken |= set(fight_index.values())
    fight_id, fight_match = resolve_fight_id(
        bout,
        crosswalk=crosswalk,
        fight_index=fight_index,
        fighter_a_id=fighter_ids[0],
        fighter_b_id=fighter_ids[1],
        taken_fight_ids=taken,
    )

    row = build_fight_csv_row(
        fight_id=fight_id,
        event_date=event_date,
        fighter_a_id=fighter_ids[0],
        fighter_b_id=fighter_ids[1],
        winner_id=winner_id,
        method=method,
        weight_class=wc,
        fight_time_sec=fight_time,
        side_a=sides[0][3],
        side_b=sides[1][3],
    )

    new_fighter_entries = [m for m in fighter_meta if m.get("match_method") == "espn_new"]
    for m in new_fighter_entries:
        opp = fighter_meta[1] if m is fighter_meta[0] else fighter_meta[0]
        m["bout_summary"] = (
            f" @ {event_date.isoformat()} vs {opp.get('display_name')} "
            f"(fight {fight_id})"
        )

    ingest_meta: Dict[str, Any] = {
        "fight_match": fight_match,
        "new_fighter_entries": new_fighter_entries,
    }
    if fight_match == "espn_new":
        ingest_meta["fight_entry"] = {
            "fight_id": fight_id,
            "espn_competition_id": comp_id,
            "espn_event_id": event_id,
            "event_date": event_date.isoformat(),
            "match_method": fight_match,
            "fighter_a_id": fighter_ids[0],
            "fighter_b_id": fighter_ids[1],
            "fighter_a_name": names[0],
            "fighter_b_name": names[1],
        }

    return row, ingest_meta


def build_crosswalk_from_espn(
    data_dir: Path,
    *,
    client: Optional[ESPNClient] = None,
    season_years: Optional[List[int]] = None,
    max_events: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    Walk ESPN seasons and populate crosswalk tables without rewriting fight stats.

    Useful before relying on incremental ingest for historical ID stability.
    """
    data_dir = Path(data_dir)
    fights_path = _fights_csv_path(data_dir)
    if not fights_path.is_file():
        raise FileNotFoundError(f"Need existing fights CSV at {fights_path} to build crosswalk")

    crosswalk = CrosswalkStore(data_dir)
    profiles_path = data_dir / "fighter_profiles.csv"
    profiles_by_id = _load_profiles_by_id(profiles_path)
    fight_index = build_fight_index_from_csv(fights_path, profiles_path, crosswalk)
    name_index = build_name_index(profiles_path, crosswalk)
    espn = client or ESPNClient(cache_dir=data_dir / "cache" / "espn")
    years = season_years or espn.list_season_years()
    matched = 0
    events_seen = 0
    with open(fights_path, newline="", encoding="utf-8") as f:
        fights_csv_rows = sum(1 for _ in csv.DictReader(f))

    _log(
        verbose,
        f"[espn crosswalk] data_dir={data_dir} fights_csv={fights_csv_rows:,} rows "
        f"existing_maps fights={len(crosswalk.competition_to_fight):,} "
        f"fighters={len(crosswalk.athlete_to_fighter):,} seasons={years} "
        f"delay={espn.request_delay_sec}s cache={espn.cache_dir}",
    )

    for year in years:
        event_refs = espn.list_event_refs(year)
        _log(verbose, f"[espn crosswalk] season {year}: {len(event_refs)} events")
        for ei, event_ref in enumerate(event_refs):
            if max_events is not None and events_seen >= max_events:
                break
            event = espn.fetch_event(event_ref)
            event_id = str(event.get("id") or "")
            event_date = parse_event_date(event.get("date") or "")
            if event_date is None:
                continue
            events_seen += 1
            event_name = (event.get("name") or event.get("shortName") or event_id).strip()
            fights_before = len(crosswalk.competition_to_fight)
            fighters_before = len(crosswalk.athlete_to_fighter)
            event_matched = 0
            try:
                fightcenter = espn.fetch_fightcenter(event_id)
            except RuntimeError as e:
                _log(verbose, f"  event {ei + 1}/{len(event_refs)} {event_date} {event_name} | skip: {e}")
                continue
            comps = _iter_competitions(fightcenter)
            for comp in comps:
                comp_id = str(comp.get("id") or "")
                if not comp_id or comp_id in crosswalk.competition_to_fight:
                    continue
                comp_refs = espn.list_competitor_refs(event_id, comp_id)
                if len(comp_refs) != 2:
                    continue
                names: List[str] = []
                aids: List[str] = []
                for ref in comp_refs:
                    comptr = espn.fetch_competitor(ref)
                    ath = comptr.get("athlete") or {}
                    if isinstance(ath, dict) and ath.get("$ref") and not ath.get("displayName"):
                        ath = espn.get_json(ath["$ref"])
                    aids.append(_athlete_id_from_competitor({"athlete": ath}))
                    names.append((ath.get("displayName") or ath.get("fullName") or "").strip())
                if len(aids) != 2 or not all(aids):
                    continue
                bout = BoutIdentity(
                    espn_competition_id=comp_id,
                    espn_event_id=event_id,
                    event_date=event_date,
                    espn_athlete_ids=(aids[0], aids[1]),
                    fighter_names=(names[0], names[1]),
                )
                fid, method = resolve_fight_id(bout, crosswalk=crosswalk, fight_index=fight_index)
                if method in ("name_date", "crosswalk"):
                    matched += 1
                    event_matched += 1
                for aid, nm in zip(aids, names):
                    resolve_fighter_id(
                        aid,
                        nm,
                        crosswalk=crosswalk,
                        name_index=name_index,
                        profiles_by_id=profiles_by_id,
                        espn=espn,
                    )

            new_fights = len(crosswalk.competition_to_fight) - fights_before
            new_fighters = len(crosswalk.athlete_to_fighter) - fighters_before
            _log(
                verbose,
                f"  event {ei + 1}/{len(event_refs)} {event_date} {event_name} | "
                f"{len(comps)} bouts | +{new_fights} fight maps | +{new_fighters} fighter maps | "
                f"{event_matched} name/date matches",
            )

        if max_events is not None and events_seen >= max_events:
            break

    crosswalk.save()
    _log(
        verbose,
        f"[espn crosswalk] HTTP summary: {espn.network_requests} live requests, "
        f"{espn.cache_hits} cache hits",
    )
    print(
        f"[espn crosswalk] Done: {matched} name/date fight matches, "
        f"{len(crosswalk.competition_to_fight):,} fight maps, "
        f"{len(crosswalk.athlete_to_fighter):,} fighter maps -> {data_dir}",
        flush=True,
    )
    return matched


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="ESPN UFC ingest / crosswalk (cached, rate-limited).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--sleep",
        type=float,
        default=None,
        help=f"Override delay (default {ESPN_REQUEST_DELAY_SEC}s)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-event progress (summary lines still print).",
    )
    p.add_argument(
        "--log-network",
        action="store_true",
        help="Print each uncached HTTP URL (very noisy on first run).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_inc = sub.add_parser("incremental", help="Merge new bouts into ufcstats_fights.csv")
    p_inc.add_argument("--max-seasons", type=int, default=None)
    p_inc.add_argument("--max-events", type=int, default=None)
    p_inc.add_argument("--max-competitions", type=int, default=None)

    p_xw = sub.add_parser("crosswalk", help="Build ID crosswalk from existing CSV + ESPN")
    p_xw.add_argument("--season", type=int, action="append", dest="seasons")
    p_xw.add_argument("--max-events", type=int, default=None)

    args = p.parse_args(argv)
    delay = ESPN_REQUEST_DELAY_SEC if args.sleep is None else args.sleep
    verbose = not args.quiet
    client = ESPNClient(
        cache_dir=Path(args.data_dir) / "cache" / "espn",
        request_delay_sec=delay,
        log_network=args.log_network,
    )
    _log(verbose, f"[espn] command={args.cmd} data_dir={args.data_dir}")

    if args.cmd == "incremental":
        refresh_espn_fights_incremental(
            args.data_dir,
            client=client,
            max_seasons=args.max_seasons,
            max_events=args.max_events,
            max_competitions=args.max_competitions,
            verbose=verbose,
        )
    else:
        build_crosswalk_from_espn(
            args.data_dir,
            client=client,
            season_years=args.seasons,
            max_events=args.max_events,
            verbose=verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
