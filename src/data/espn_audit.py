"""
Post-ingest audit for new ``espn_*`` fighter and fight IDs.

Method 1: structured report with fuzzy name suggestions.
Auto-reject: fuzzy name match + identical height and reach vs an existing profile.
Method 5: fight-level consistency checks on the fights CSV.
Rookie check: ESPN athlete eventlog UFC bout count vs career record.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.espn_client import ESPNClient
from src.data.espn_crosswalk import CrosswalkStore
from src.data.espn_normalize import normalize_fighter_name
from src.data.tier1_csv import DEFAULT_UFCSTATS_FIGHTS_CSV

ESPN_INGEST_AUDIT_JSON = "espn_ingest_audit.json"

# Auto-reject when name similarity >= this and height_cm + reach_cm match exactly.
FUZZY_PHYS_AUTO_REJECT = 0.85

# High-confidence duplicate suggestion (warn in report; phys match triggers reject).
FUZZY_WARN = 0.85
FUZZY_FAIL = 0.92


@dataclass
class AuditFinding:
    severity: str  # info | warn | reject
    code: str
    message: str
    entity_type: str  # fighter | fight
    entity_id: str
    details: Dict[str, Any] = field(default_factory=dict)


def name_similarity(a: str, b: str) -> float:
    na = normalize_fighter_name(a)
    nb = normalize_fighter_name(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _float_cm(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        return round(float(raw), 2)
    except (TypeError, ValueError):
        return None


def _phys_match(a_h: Optional[float], a_r: Optional[float], b_h: Optional[float], b_r: Optional[float]) -> bool:
    if a_h is None or a_r is None or b_h is None or b_r is None:
        return False
    return a_h == b_h and a_r == b_r


def _load_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = (row.get("fighter_id") or "").strip()
            if fid:
                rows[fid] = row
    return rows


def _best_fuzzy_profile_match(
    display_name: str,
    profiles: Dict[str, Dict[str, str]],
    *,
    exclude_id: str,
    espn_height_cm: Optional[float],
    espn_reach_cm: Optional[float],
) -> Tuple[Optional[str], float, bool]:
    best_id: Optional[str] = None
    best_score = 0.0
    phys_identical = False
    for fid, row in profiles.items():
        if fid == exclude_id or fid.startswith("espn_"):
            continue
        pname = (row.get("name") or "").strip()
        score = name_similarity(display_name, pname)
        if score <= best_score:
            continue
        ph = _float_cm(row.get("height_cm"))
        pr = _float_cm(row.get("reach_cm"))
        phys = _phys_match(espn_height_cm, espn_reach_cm, ph, pr)
        best_score = score
        best_id = fid
        phys_identical = phys
    return best_id, best_score, phys_identical


def parse_record_string(record: str) -> Tuple[int, int, int]:
    """Parse ESPN-style record (e.g. ``15-3-0``) to wins, losses, draws."""
    raw = (record or "").strip()
    if not raw:
        return 0, 0, 0
    parts = re.split(r"[-–]", raw)
    nums: List[int] = []
    for p in parts[:3]:
        p = p.strip()
        if not p:
            nums.append(0)
            continue
        m = re.search(r"\d+", p)
        nums.append(int(m.group(0)) if m else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def count_ufc_bouts_in_eventlog(payload: Dict[str, Any]) -> int:
    events = payload.get("events")
    if events is None:
        events = payload.get("eventlog")
    if isinstance(events, dict):
        events = events.get("items") or events.get("events") or []
    if not isinstance(events, list):
        return 0
    count = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        league = ev.get("league") or {}
        if not isinstance(league, dict):
            league = {}
        slug = str(league.get("slug") or league.get("abbreviation") or "").lower()
        name = str(league.get("name") or league.get("displayName") or "").lower()
        if slug == "ufc" or name == "ufc" or "ufc" in name:
            count += 1
    return count


def _record_from_athlete(athlete: Dict[str, Any]) -> str:
    for key in ("displayRecord", "recordSummary", "record"):
        val = athlete.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            for sub in ("displayValue", "summary", "text"):
                s = val.get(sub)
                if isinstance(s, str) and s.strip():
                    return s.strip()
    return ""


def audit_rookie_ufc_history(
    espn: ESPNClient,
    espn_athlete_id: str,
    *,
    athlete_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    athlete = athlete_payload or espn.fetch_athlete(espn_athlete_id)
    try:
        eventlog = espn.fetch_athlete_eventlog(espn_athlete_id)
    except RuntimeError as e:
        return {
            "ufc_bouts_in_eventlog": None,
            "record": _record_from_athlete(athlete),
            "wins": None,
            "losses": None,
            "draws": None,
            "rookie_ok": None,
            "error": str(e),
        }
    ufc_count = count_ufc_bouts_in_eventlog(eventlog)
    rec = _record_from_athlete(athlete)
    w, l, d = parse_record_string(rec)
    career_bouts = w + l + d
    rookie_ok = ufc_count == 1 and career_bouts >= 1
    return {
        "ufc_bouts_in_eventlog": ufc_count,
        "record": rec,
        "wins": w,
        "losses": l,
        "draws": d,
        "career_bouts": career_bouts,
        "rookie_ok": rookie_ok,
    }


def audit_fighter_method1(
    entry: Dict[str, Any],
    profiles: Dict[str, Dict[str, str]],
    espn: Optional[ESPNClient],
    *,
    fetch_rookie: bool,
) -> Tuple[Dict[str, Any], List[AuditFinding]]:
    fighter_id = entry["fighter_id"]
    display_name = entry.get("display_name") or ""
    espn_athlete_id = entry.get("espn_athlete_id") or ""
    match_method = entry.get("match_method") or "espn_new"

    espn_h = _float_cm(entry.get("height_cm"))
    espn_r = _float_cm(entry.get("reach_cm"))
    if espn_h is None and espn_r is None and espn and espn_athlete_id:
        try:
            ath = espn.fetch_athlete(espn_athlete_id)
            hi = ath.get("height")
            ri = ath.get("reach")
            espn_h = round(float(hi) * 2.54, 2) if hi else None
            espn_r = round(float(ri) * 2.54, 2) if ri else None
        except (RuntimeError, TypeError, ValueError):
            pass

    suggested_id, score, phys_identical = _best_fuzzy_profile_match(
        display_name,
        profiles,
        exclude_id=fighter_id,
        espn_height_cm=espn_h,
        espn_reach_cm=espn_r,
    )

    findings: List[AuditFinding] = []
    if suggested_id and score >= FUZZY_WARN:
        sev = "warn"
        if score >= FUZZY_FAIL:
            sev = "warn"
        if phys_identical and score >= FUZZY_PHYS_AUTO_REJECT:
            sev = "reject"
            findings.append(
                AuditFinding(
                    severity="reject",
                    code="duplicate_fuzzy_phys",
                    message=(
                        f"Auto-reject: '{display_name}' ({fighter_id}) likely duplicate of "
                        f"{suggested_id} (score={score:.3f}, identical height/reach)"
                    ),
                    entity_type="fighter",
                    entity_id=fighter_id,
                    details={
                        "suggested_fighter_id": suggested_id,
                        "fuzzy_score": round(score, 4),
                        "height_cm": espn_h,
                        "reach_cm": espn_r,
                    },
                )
            )
        else:
            findings.append(
                AuditFinding(
                    severity=sev,
                    code="fuzzy_name_match",
                    message=(
                        f"Fuzzy match '{display_name}' -> {suggested_id} "
                        f"(score={score:.3f})"
                    ),
                    entity_type="fighter",
                    entity_id=fighter_id,
                    details={"suggested_fighter_id": suggested_id, "fuzzy_score": round(score, 4)},
                )
            )

    rookie: Dict[str, Any] = {}
    if fetch_rookie and espn and espn_athlete_id:
        rookie = audit_rookie_ufc_history(espn, espn_athlete_id)
        ufc_n = rookie.get("ufc_bouts_in_eventlog")
        if ufc_n is not None and ufc_n > 1:
            findings.append(
                AuditFinding(
                    severity="reject",
                    code="not_rookie_ufc_history",
                    message=(
                        f"ESPN eventlog shows {ufc_n} UFC bouts for '{display_name}' "
                        f"but id {fighter_id} was created as new"
                    ),
                    entity_type="fighter",
                    entity_id=fighter_id,
                    details=rookie,
                )
            )
        elif ufc_n == 1 and rookie.get("rookie_ok"):
            findings.append(
                AuditFinding(
                    severity="info",
                    code="rookie_ok",
                    message=(
                        f"Rookie check OK: 1 UFC bout in eventlog, "
                        f"record {rookie.get('record')}"
                    ),
                    entity_type="fighter",
                    entity_id=fighter_id,
                    details=rookie,
                )
            )
        elif ufc_n == 0:
            findings.append(
                AuditFinding(
                    severity="warn",
                    code="rookie_no_ufc_eventlog",
                    message=f"No UFC bouts in ESPN eventlog for '{display_name}'",
                    entity_type="fighter",
                    entity_id=fighter_id,
                    details=rookie,
                )
            )

    report_row = {
        "fighter_id": fighter_id,
        "espn_athlete_id": espn_athlete_id,
        "display_name": display_name,
        "match_method": match_method,
        "suggested_fighter_id": suggested_id or "",
        "fuzzy_score": round(score, 4) if suggested_id else 0.0,
        "phys_identical": phys_identical,
        "height_cm": espn_h,
        "reach_cm": espn_r,
        "rookie": rookie,
        "findings": [f.code for f in findings],
    }
    return report_row, findings


def _fighter_name_map(profiles: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for fid, row in profiles.items():
        name = (row.get("name") or "").strip()
        if fid and name:
            out[fid] = name
    return out


def audit_fight_method5(
    fight_entry: Dict[str, Any],
    fights_rows: Dict[str, Dict[str, Any]],
    id_to_name: Dict[str, str],
    *,
    max_date: Optional[date],
) -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    fight_id = fight_entry.get("fight_id") or ""
    row = fights_rows.get(fight_id)
    if not row:
        return findings

    fa = (row.get("fighter_a_id") or "").strip()
    fb = (row.get("fighter_b_id") or "").strip()
    d_raw = (row.get("date") or "").strip()
    fight_is_espn = fight_id.startswith("espn_")
    fa_espn = fa.startswith("espn_")
    fb_espn = fb.startswith("espn_")
    both_hex = not fa_espn and not fb_espn

    if fight_is_espn and both_hex:
        findings.append(
            AuditFinding(
                severity="reject",
                code="fight_espn_id_both_fighters_matched",
                message=(
                    f"Fight {fight_id}: both fighters have hex/crosswalk ids "
                    f"but fight_id is espn_*"
                ),
                entity_type="fight",
                entity_id=fight_id,
            )
        )

    if max_date and d_raw:
        try:
            fd = date.fromisoformat(d_raw)
            if fd < max_date and fight_entry.get("match_method") == "espn_new":
                findings.append(
                    AuditFinding(
                        severity="warn",
                        code="fight_before_watermark",
                        message=(
                            f"New espn fight {fight_id} dated {d_raw} before "
                            f"CSV max_date {max_date.isoformat()}"
                        ),
                        entity_type="fight",
                        entity_id=fight_id,
                    )
                )
        except ValueError:
            pass

    espn_side = fa if fa_espn else (fb if fb_espn else "")
    hex_side = fb if fa_espn else (fa if fb_espn else "")
    if espn_side and hex_side:
        espn_name = id_to_name.get(espn_side, espn_side)
        norm_espn = normalize_fighter_name(espn_name)
        for other_row in fights_rows.values():
            if (other_row.get("fight_id") or "").strip() == fight_id:
                continue
            oa = (other_row.get("fighter_a_id") or "").strip()
            ob = (other_row.get("fighter_b_id") or "").strip()
            if hex_side not in (oa, ob):
                continue
            other_fid = ob if oa == hex_side else oa
            other_name = id_to_name.get(other_fid, other_fid)
            if normalize_fighter_name(other_name) == norm_espn and other_fid != espn_side:
                findings.append(
                    AuditFinding(
                        severity="reject",
                        code="probable_duplicate_fighter_opponent_history",
                        message=(
                            f"Fighter {hex_side} already fought '{other_name}' "
                            f"(normalized match to new {espn_side} / {espn_name})"
                        ),
                        entity_type="fight",
                        entity_id=fight_id,
                        details={
                            "hex_fighter_id": hex_side,
                            "espn_fighter_id": espn_side,
                            "prior_opponent_id": other_fid,
                        },
                    )
                )
                break

    return findings


def run_espn_ingest_audit(
    data_dir: Path,
    *,
    last_run: Optional[Dict[str, Any]] = None,
    client: Optional[ESPNClient] = None,
    fetch_rookie: bool = True,
    fail_on_reject: bool = True,
) -> Tuple[Dict[str, Any], int]:
    """
    Build ``espn_ingest_audit.json`` from ``last_run`` in ingest state.

    Returns ``(audit_dict, exit_code)`` where exit_code is 1 if any reject and
    ``fail_on_reject``.
    """
    data_dir = Path(data_dir)
    profiles_path = data_dir / "fighter_profiles.csv"
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    profiles = _load_profiles(profiles_path)

    if last_run is None:
        state_path = data_dir / "espn_ingest_state.json"
        if state_path.is_file():
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            last_run = state.get("last_run") or {}
        else:
            last_run = {}

    new_fighters_raw: List[Dict[str, Any]] = list(last_run.get("new_fighters") or [])
    new_fights: List[Dict[str, Any]] = list(last_run.get("new_fights") or [])
    seen_fighters: Set[str] = set()
    new_fighters: List[Dict[str, Any]] = []
    for entry in new_fighters_raw:
        fid = (entry.get("fighter_id") or "").strip()
        if fid and fid in seen_fighters:
            continue
        if fid:
            seen_fighters.add(fid)
        new_fighters.append(entry)

    fights_rows: Dict[str, Dict[str, Any]] = {}
    max_date: Optional[date] = None
    if fights_path.is_file():
        with open(fights_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = (row.get("fight_id") or "").strip()
                if fid:
                    fights_rows[fid] = row
                d_raw = (row.get("date") or "").strip()
                if d_raw:
                    try:
                        d = date.fromisoformat(d_raw)
                        if max_date is None or d > max_date:
                            max_date = d
                    except ValueError:
                        pass

    espn = client
    if fetch_rookie and espn is None and new_fighters:
        espn = ESPNClient(cache_dir=data_dir / "cache" / "espn")

    crosswalk = CrosswalkStore(data_dir)
    id_to_name = _fighter_name_map(profiles)

    all_findings: List[AuditFinding] = []
    fighter_reports: List[Dict[str, Any]] = []
    fight_reports: List[Dict[str, Any]] = []

    for entry in new_fighters:
        if entry.get("espn_athlete_id") in crosswalk.athlete_to_fighter:
            mapped = crosswalk.athlete_to_fighter[entry["espn_athlete_id"]]
            if not str(mapped).startswith("espn_"):
                entry = {**entry, "crosswalk_note": f"athlete maps to {mapped}"}
        rep, findings = audit_fighter_method1(
            entry, profiles, espn, fetch_rookie=fetch_rookie
        )
        fighter_reports.append(rep)
        all_findings.extend(findings)

    for entry in new_fights:
        rep = dict(entry)
        findings = audit_fight_method5(
            entry, fights_rows, id_to_name, max_date=max_date
        )
        rep["findings"] = [f.code for f in findings]
        fight_reports.append(rep)
        all_findings.extend(findings)

    rejects = [f for f in all_findings if f.severity == "reject"]
    warns = [f for f in all_findings if f.severity == "warn"]

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_fighters_count": len(fighter_reports),
        "new_fights_count": len(fight_reports),
        "reject_count": len(rejects),
        "warn_count": len(warns),
        "passed": len(rejects) == 0,
        "new_fighters": fighter_reports,
        "new_fights": fight_reports,
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
                "entity_type": f.entity_type,
                "entity_id": f.entity_id,
                "details": f.details,
            }
            for f in all_findings
        ],
    }

    out_path = data_dir / ESPN_INGEST_AUDIT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    exit_code = 1 if (fail_on_reject and rejects) else 0
    return audit, exit_code


def format_audit_log_lines(audit: Dict[str, Any]) -> List[str]:
    """Human-readable lines for CI logs and weekly report."""
    lines: List[str] = []
    lines.append(
        f"ESPN ingest audit: {audit.get('new_fighters_count', 0)} new fighters, "
        f"{audit.get('new_fights_count', 0)} new fights, "
        f"{audit.get('reject_count', 0)} rejects, {audit.get('warn_count', 0)} warnings"
    )
    for f in audit.get("new_fighters") or []:
        name = f.get("display_name") or f.get("fighter_id")
        bout = f.get("bout_summary") or ""
        sug = f.get("suggested_fighter_id") or ""
        score = f.get("fuzzy_score") or 0
        rookie = f.get("rookie") or {}
        r_ok = rookie.get("rookie_ok")
        r_txt = ""
        if rookie.get("ufc_bouts_in_eventlog") is not None:
            r_txt = (
                f" | UFC bouts={rookie.get('ufc_bouts_in_eventlog')} "
                f"record={rookie.get('record')} rookie_ok={r_ok}"
            )
        extra = f" -> suggest {sug} ({score:.2f})" if sug and score >= FUZZY_WARN else ""
        lines.append(f"  fighter {f.get('fighter_id')}: {name}{bout}{extra}{r_txt}")
    for fight in audit.get("new_fights") or []:
        lines.append(
            f"  fight {fight.get('fight_id')}: {fight.get('event_date')} "
            f"{fight.get('fighter_a_name')} vs {fight.get('fighter_b_name')} "
            f"({fight.get('match_method')})"
        )
    for finding in audit.get("findings") or []:
        if finding.get("severity") in ("reject", "warn"):
            lines.append(f"  [{finding.get('severity')}] {finding.get('message')}")
    lines.append(f"Result: {'PASS' if audit.get('passed') else 'FAIL'}")
    return lines
