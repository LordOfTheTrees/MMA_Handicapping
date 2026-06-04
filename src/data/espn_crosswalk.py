"""
UFCStats hex IDs ↔ ESPN numeric IDs.

Crosswalk tables live under ``data/`` as ESPN-sourced artifacts; training CSVs
keep UFCStats column names and IDs wherever a mapping exists.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.data.espn_normalize import normalize_fighter_name

CROSSWALK_FIGHTS_CSV = "espn_crosswalk_fights.csv"
CROSSWALK_FIGHTERS_CSV = "espn_crosswalk_fighters.csv"

CROSSWALK_FIGHT_FIELDS = [
    "ufcstats_fight_id",
    "espn_competition_id",
    "espn_event_id",
    "event_date",
    "match_method",
]

CROSSWALK_FIGHTER_FIELDS = [
    "ufcstats_fighter_id",
    "espn_athlete_id",
    "fighter_name",
    "match_method",
]


@dataclass(frozen=True)
class BoutIdentity:
    espn_competition_id: str
    espn_event_id: str
    event_date: date
    espn_athlete_ids: Tuple[str, str]
    fighter_names: Tuple[str, str]


class CrosswalkStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.fight_path = self.data_dir / CROSSWALK_FIGHTS_CSV
        self.fighter_path = self.data_dir / CROSSWALK_FIGHTERS_CSV
        self.competition_to_fight: Dict[str, str] = {}
        self.fight_to_competition: Dict[str, str] = {}
        self.athlete_to_fighter: Dict[str, str] = {}
        self.fighter_to_athlete: Dict[str, str] = {}
        self._fight_rows: Dict[str, Dict[str, str]] = {}
        self._fighter_rows: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.fight_path.is_file():
            with open(self.fight_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ufc = (row.get("ufcstats_fight_id") or "").strip()
                    espn = (row.get("espn_competition_id") or "").strip()
                    if ufc and espn:
                        self.competition_to_fight[espn] = ufc
                        self.fight_to_competition[ufc] = espn
                        self._fight_rows[espn] = {k: row.get(k, "") or "" for k in CROSSWALK_FIGHT_FIELDS}
        if self.fighter_path.is_file():
            with open(self.fighter_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ufc = (row.get("ufcstats_fighter_id") or "").strip()
                    espn = (row.get("espn_athlete_id") or "").strip()
                    if ufc and espn:
                        self.athlete_to_fighter[espn] = ufc
                        self.fighter_to_athlete[ufc] = espn
                        self._fighter_rows[espn] = {k: row.get(k, "") or "" for k in CROSSWALK_FIGHTER_FIELDS}

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.fight_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CROSSWALK_FIGHT_FIELDS)
            w.writeheader()
            for espn_cid in sorted(self._fight_rows.keys(), key=lambda c: self._fight_rows[c].get("ufcstats_fight_id", "")):
                w.writerow(self._fight_rows[espn_cid])
        with open(self.fighter_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CROSSWALK_FIGHTER_FIELDS)
            w.writeheader()
            for espn_aid in sorted(self._fighter_rows.keys(), key=lambda a: self._fighter_rows[a].get("ufcstats_fighter_id", "")):
                w.writerow(self._fighter_rows[espn_aid])

    def record_fight(
        self,
        *,
        ufcstats_fight_id: str,
        espn_competition_id: str,
        espn_event_id: str,
        event_date: date,
        match_method: str,
    ) -> None:
        self.competition_to_fight[espn_competition_id] = ufcstats_fight_id
        self.fight_to_competition[ufcstats_fight_id] = espn_competition_id
        self._fight_rows[espn_competition_id] = {
            "ufcstats_fight_id": ufcstats_fight_id,
            "espn_competition_id": espn_competition_id,
            "espn_event_id": espn_event_id,
            "event_date": event_date.isoformat(),
            "match_method": match_method,
        }

    def record_fighter(
        self,
        *,
        ufcstats_fighter_id: str,
        espn_athlete_id: str,
        fighter_name: str,
        match_method: str,
    ) -> None:
        self.athlete_to_fighter[espn_athlete_id] = ufcstats_fighter_id
        self.fighter_to_athlete[ufcstats_fighter_id] = espn_athlete_id
        self._fighter_rows[espn_athlete_id] = {
            "ufcstats_fighter_id": ufcstats_fighter_id,
            "espn_athlete_id": espn_athlete_id,
            "fighter_name": fighter_name,
            "match_method": match_method,
        }


def _fight_pair_key(
    event_date: date,
    name_a: str,
    name_b: str,
) -> Tuple[str, Tuple[str, str]]:
    n1 = normalize_fighter_name(name_a)
    n2 = normalize_fighter_name(name_b)
    return event_date.isoformat(), tuple(sorted((n1, n2)))


def build_fight_index_from_csv(
    fights_csv: Path,
    profiles_csv: Path,
) -> Dict[Tuple[str, Tuple[str, str]], str]:
    id_to_name: Dict[str, str] = {}
    if profiles_csv.is_file():
        with open(profiles_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = (row.get("fighter_id") or "").strip()
                name = (row.get("name") or "").strip()
                if fid and name:
                    id_to_name[fid] = name

    index: Dict[Tuple[str, Tuple[str, str]], str] = {}
    with open(fights_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fight_id = (row.get("fight_id") or "").strip()
            d_raw = (row.get("date") or "").strip()
            fa = (row.get("fighter_a_id") or "").strip()
            fb = (row.get("fighter_b_id") or "").strip()
            if not fight_id or not d_raw or not fa or not fb:
                continue
            try:
                ev = date.fromisoformat(d_raw)
            except ValueError:
                continue
            na = id_to_name.get(fa, fa)
            nb = id_to_name.get(fb, fb)
            index[_fight_pair_key(ev, na, nb)] = fight_id
    return index


def resolve_fight_id(
    bout: BoutIdentity,
    *,
    crosswalk: CrosswalkStore,
    fight_index: Dict[Tuple[str, Tuple[str, str]], str],
) -> Tuple[str, str]:
    """Return (ufcstats_or_legacy_fight_id, match_method)."""
    if bout.espn_competition_id in crosswalk.competition_to_fight:
        return crosswalk.competition_to_fight[bout.espn_competition_id], "crosswalk"

    key = _fight_pair_key(bout.event_date, bout.fighter_names[0], bout.fighter_names[1])
    if key in fight_index:
        ufc_id = fight_index[key]
        crosswalk.record_fight(
            ufcstats_fight_id=ufc_id,
            espn_competition_id=bout.espn_competition_id,
            espn_event_id=bout.espn_event_id,
            event_date=bout.event_date,
            match_method="name_date",
        )
        return ufc_id, "name_date"

    return f"espn_{bout.espn_competition_id}", "espn_new"


def resolve_fighter_id(
    espn_athlete_id: str,
    display_name: str,
    *,
    crosswalk: CrosswalkStore,
    name_index: Dict[str, str],
) -> Tuple[str, str]:
    if espn_athlete_id in crosswalk.athlete_to_fighter:
        return crosswalk.athlete_to_fighter[espn_athlete_id], "crosswalk"

    norm = normalize_fighter_name(display_name)
    if norm in name_index:
        ufc_id = name_index[norm]
        crosswalk.record_fighter(
            ufcstats_fighter_id=ufc_id,
            espn_athlete_id=espn_athlete_id,
            fighter_name=display_name,
            match_method="name",
        )
        return ufc_id, "name"

    return f"espn_{espn_athlete_id}", "espn_new"


def build_name_index_from_profiles(profiles_csv: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not profiles_csv.is_file():
        return index
    with open(profiles_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = (row.get("fighter_id") or "").strip()
            name = (row.get("name") or "").strip()
            if fid and name:
                index[normalize_fighter_name(name)] = fid
    return index
