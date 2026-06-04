"""
Refresh ``fighter_profiles.csv`` for fighters linked in the ESPN crosswalk.

Only fetches athletes not yet present in the profiles CSV (or ``--force`` via CLI).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.data.espn_client import ESPNClient, ESPN_REQUEST_DELAY_SEC
from src.data.espn_crosswalk import CrosswalkStore
from src.data.tier1_csv import PROFILE_CSV_FIELDS

STANCE_MAP = {
    "orthodox": "orthodox",
    "southpaw": "southpaw",
    "switch": "switch",
    "open stance": "switch",
}


def _existing_profile_ids(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(row.get("fighter_id") or "").strip() for row in csv.DictReader(f) if row.get("fighter_id")}


def _parse_athlete_profile(athlete: Dict[str, Any], fighter_id: str) -> Dict[str, Any]:
    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
    height_in = athlete.get("height")
    reach_in = athlete.get("reach")
    height_cm = round(float(height_in) * 2.54, 2) if height_in else ""
    reach_cm = round(float(reach_in) * 2.54, 2) if reach_in else ""
    stance_raw = athlete.get("stance")
    if isinstance(stance_raw, dict):
        stance_raw = stance_raw.get("text") or ""
    stance = STANCE_MAP.get(str(stance_raw or "").strip().lower(), "")
    dob = ""
    raw_dob = athlete.get("dateOfBirth") or ""
    if raw_dob:
        try:
            dob = datetime.fromisoformat(str(raw_dob).replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            dob = ""
    return {
        "fighter_id": fighter_id,
        "name": name or fighter_id,
        "reach_cm": reach_cm,
        "height_cm": height_cm,
        "date_of_birth": dob,
        "stance": stance,
        "wrestling_pedigree": "0",
        "boxing_pedigree": "0",
        "bjj_pedigree": "0",
    }


def refresh_espn_profiles_incremental(
    data_dir: Path,
    *,
    client: Optional[ESPNClient] = None,
    force: bool = False,
) -> int:
    data_dir = Path(data_dir)
    profiles_path = data_dir / "fighter_profiles.csv"
    crosswalk = CrosswalkStore(data_dir)
    if not crosswalk.fighter_path.is_file():
        print("[espn profiles] No crosswalk fighters file; skip.", flush=True)
        return 0

    have = set() if force else _existing_profile_ids(profiles_path)
    espn = client or ESPNClient(cache_dir=data_dir / "cache" / "espn")
    added = 0
    rows: Dict[str, Dict[str, Any]] = {}
    if profiles_path.is_file():
        with open(profiles_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = (row.get("fighter_id") or "").strip()
                if fid:
                    rows[fid] = row

    with open(crosswalk.fighter_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ufc_id = (row.get("ufcstats_fighter_id") or "").strip()
            espn_id = (row.get("espn_athlete_id") or "").strip()
            if not ufc_id or not espn_id or ufc_id in have:
                continue
            athlete = espn.fetch_athlete(espn_id)
            profile = _parse_athlete_profile(athlete, ufc_id)
            rows[ufc_id] = profile
            have.add(ufc_id)
            added += 1

    if added:
        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profiles_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PROFILE_CSV_FIELDS)
            w.writeheader()
            for fid in sorted(rows.keys()):
                w.writerow(rows[fid])
    print(f"[espn profiles] Added/updated {added} profiles -> {profiles_path}", flush=True)
    return added


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ESPN athlete profiles via crosswalk.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--sleep", type=float, default=None)
    args = p.parse_args(argv)
    delay = ESPN_REQUEST_DELAY_SEC if args.sleep is None else args.sleep
    client = ESPNClient(cache_dir=args.data_dir / "cache" / "espn", request_delay_sec=delay)
    refresh_espn_profiles_incremental(args.data_dir, client=client, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
