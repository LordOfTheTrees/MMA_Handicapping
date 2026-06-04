"""
Map ESPN competition payloads into ``ufcstats_fights.csv`` column conventions.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.data.loader import WEIGHT_CLASS_MAP


def normalize_fighter_name(name: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s


def parse_event_date(raw: str) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def espn_method_to_csv(result_name: Optional[str]) -> Optional[str]:
    if not result_name:
        return None
    s = re.sub(r"\s+", " ", str(result_name).strip().lower())
    s = s.replace("---", " ").replace("-", " ")
    if re.match(r"^(ko|tko)\b", s) or s in ("kotko", "ko tko"):
        return "ko/tko"
    if "submission" in s or s == "sub":
        return "submission"
    if "unanimous" in s:
        return "unanimous decision"
    if "split" in s:
        return "split decision"
    if "majority" in s:
        return "majority decision"
    if "draw" in s:
        return "draw"
    if "no contest" in s or s == "nc":
        return "no contest"
    if "disqualif" in s or s == "dq":
        return "dq"
    if "decision" in s:
        return "unanimous decision"
    return None


def weight_class_from_note(note: Optional[str], type_text: Optional[str] = None) -> Optional[str]:
    raw = (note or type_text or "").strip()
    if not raw:
        return None
    head = raw.split(" - ")[0].strip()
    wc = head.lower()
    if wc in WEIGHT_CLASS_MAP:
        return wc
    if "catch weight" in wc or "catchweight" in wc:
        return "catch_weight"
    for key in sorted(WEIGHT_CLASS_MAP.keys(), key=len, reverse=True):
        if key in wc:
            return key
    if "women" in wc and "strawweight" in wc:
        return "women's strawweight"
    if "women" in wc and "bantamweight" in wc:
        return "women's bantamweight"
    if "women" in wc and "flyweight" in wc:
        return "women's flyweight"
    if "women" in wc and "featherweight" in wc:
        return "women's featherweight"
    return wc if wc else None


def _stats_by_name(statistics_payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    splits = statistics_payload.get("splits") or {}
    for cat in splits.get("categories") or []:
        for stat in cat.get("stats") or []:
            name = stat.get("name")
            if not name:
                continue
            val = stat.get("value")
            if val is None:
                continue
            out[name] = float(val)
    return out


def fight_time_sec_from_status(
    status: Dict[str, Any],
    *,
    round_length_sec: int = 300,
) -> Optional[int]:
    period = status.get("period")
    clock = status.get("clock")
    if period is None or clock is None:
        return None
    try:
        p = int(period)
        c = float(clock)
    except (TypeError, ValueError):
        return None
    if p < 1:
        return None
    # ESPN ``clock`` on finished bouts is elapsed time in the final round.
    return (p - 1) * round_length_sec + int(round(c))


def _athlete_id_from_competitor(competitor: Dict[str, Any]) -> str:
    athlete = competitor.get("athlete") or {}
    if athlete.get("id"):
        return str(athlete["id"]).strip()
    ref = athlete.get("$ref") or ""
    m = re.search(r"/athletes/(\d+)", ref)
    return m.group(1) if m else ""


def parse_competitor_side(
    competitor: Dict[str, Any],
    statistics_payload: Dict[str, Any],
) -> Tuple[str, str, bool, Dict[str, Optional[int]]]:
    athlete = competitor.get("athlete") or {}
    athlete_id = _athlete_id_from_competitor(competitor)
    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
    winner = bool(competitor.get("winner"))
    stats = _stats_by_name(statistics_payload)
    return (
        athlete_id,
        name,
        winner,
        {
            "sig_landed": _int_or_none(stats.get("sigStrikesLanded")),
            "sig_attempted": _int_or_none(stats.get("sigStrikesAttempted")),
            "td_landed": _int_or_none(stats.get("takedownsLanded")),
            "td_attempted": _int_or_none(stats.get("takedownsAttempted")),
            "ctrl_sec": _int_or_none(stats.get("timeInControl")),
            "sub_attempts": _int_or_none(stats.get("submissions")),
        },
    )


def _int_or_none(val: Optional[float]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def build_fight_csv_row(
    *,
    fight_id: str,
    event_date: date,
    fighter_a_id: str,
    fighter_b_id: str,
    winner_id: str,
    method: str,
    weight_class: str,
    fight_time_sec: Optional[int],
    side_a: Dict[str, Optional[int]],
    side_b: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    id_a, id_b = sorted([fighter_a_id, fighter_b_id])
    if id_a == fighter_a_id:
        sa, sb = side_a, side_b
    else:
        sa, sb = side_b, side_a

    def cell(v: Optional[int]) -> Any:
        return v if v is not None else ""

    return {
        "fight_id": fight_id,
        "fighter_a_id": id_a,
        "fighter_b_id": id_b,
        "winner_id": winner_id or "",
        "method": method,
        "weight_class": weight_class,
        "date": event_date.isoformat(),
        "fight_time_sec": fight_time_sec if fight_time_sec is not None else "",
        "a_sig_str_landed": cell(sa.get("sig_landed")),
        "a_sig_str_attempted": cell(sa.get("sig_attempted")),
        "a_sig_str_absorbed": cell(sb.get("sig_landed")),
        "a_td_landed": cell(sa.get("td_landed")),
        "a_td_attempted": cell(sa.get("td_attempted")),
        "a_ctrl_time_sec": cell(sa.get("ctrl_sec")),
        "a_sub_attempts": cell(sa.get("sub_attempts")),
        "b_sig_str_landed": cell(sb.get("sig_landed")),
        "b_sig_str_attempted": cell(sb.get("sig_attempted")),
        "b_sig_str_absorbed": cell(sa.get("sig_landed")),
        "b_td_landed": cell(sb.get("td_landed")),
        "b_td_attempted": cell(sb.get("td_attempted")),
        "b_ctrl_time_sec": cell(sb.get("ctrl_sec")),
        "b_sub_attempts": cell(sb.get("sub_attempts")),
    }
