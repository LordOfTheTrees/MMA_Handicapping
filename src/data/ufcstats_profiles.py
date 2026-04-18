"""
Build ``fighter_profiles.csv`` from UFCStats ``/fighter-details/<id>`` pages.

IDs are read from the UFCStats fights CSV (``fighter_a_id`` / ``fighter_b_id``).

Usage::

    python -m src.data.ufcstats_profiles --data-dir ./data
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup
from curl_cffi import requests

from src.data.ufcstats_scraper import (
    BASE,
    DEFAULT_UFCSTATS_FIGHTS_CSV,
    REQUEST_DELAY_SEC,
    fetch_soup,
    _session,
)

PROFILE_CSV_FIELDS = [
    "fighter_id",
    "name",
    "reach_cm",
    "height_cm",
    "date_of_birth",
    "stance",
    "wrestling_pedigree",
    "boxing_pedigree",
    "bjj_pedigree",
]

LEGACY_FIGHTS_CSV = "tier1_ufcstats.csv"


def fighter_ids_from_fights_csv(tier1_path: Path) -> List[str]:
    ids: Set[str] = set()
    with open(tier1_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("fighter_a_id", "fighter_b_id"):
                v = (row.get(key) or "").strip()
                if v:
                    ids.add(v)
    return sorted(ids)


def _height_ft_in_to_cm(raw: str) -> Optional[float]:
    t = raw.strip()
    if not t or t == "--":
        return None
    m = re.search(r"(\d+)'\s*(\d+)(?:\"|''|\u2032\u2032)?", t)
    if not m:
        return None
    inches = int(m.group(1)) * 12 + int(m.group(2))
    return round(inches * 2.54, 2)


def _reach_inches_to_cm(raw: str) -> Optional[float]:
    t = raw.strip()
    if not t or t == "--":
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*\"", t)
    if not m:
        return None
    return round(float(m.group(1)) * 2.54, 2)


def _parse_dob(raw: str) -> str:
    t = raw.strip()
    if not t or t == "--":
        return ""
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _parse_info_box_list(soup: BeautifulSoup) -> Dict[str, str]:
    box = soup.select_one(".b-fight-details .b-list__info-box ul.b-list__box-list")
    if not box:
        return {}
    out: Dict[str, str] = {}
    for li in box.select("li.b-list__box-list-item"):
        title = li.select_one("i.b-list__box-item-title")
        if not title:
            continue
        key = title.get_text(strip=True).rstrip(":").strip().lower()
        rest = li.get_text().replace(title.get_text(), "", 1).strip()
        out[key] = rest
    return out


def parse_fighter_profile_html(soup: BeautifulSoup, fighter_id: str) -> Optional[Dict[str, Any]]:
    name_el = soup.select_one("h2.b-content__title span.b-content__title-highlight")
    if not name_el:
        return None
    name = re.sub(r"\s+", " ", name_el.get_text(strip=True))
    if not name:
        return None

    fields = _parse_info_box_list(soup)
    height_cm = _height_ft_in_to_cm(fields.get("height", ""))
    reach_cm = _reach_inches_to_cm(fields.get("reach", ""))
    stance = fields.get("stance", "").strip().lower()
    dob = _parse_dob(fields.get("dob", ""))

    return {
        "fighter_id": fighter_id,
        "name": name,
        "reach_cm": reach_cm if reach_cm is not None else "",
        "height_cm": height_cm if height_cm is not None else "",
        "date_of_birth": dob,
        "stance": stance,
        "wrestling_pedigree": "0",
        "boxing_pedigree": "0",
        "bjj_pedigree": "0",
    }


def scrape_fighter_profiles_to_csv(
    fights_csv: Path,
    out_path: Path,
    *,
    sleep_sec: Optional[float] = None,
    max_fighters: Optional[int] = None,
    progress_every: int = 25,
    session: Optional[requests.Session] = None,
) -> int:
    delay = REQUEST_DELAY_SEC if sleep_sec is None else sleep_sec
    fights_csv = Path(fights_csv)
    if not fights_csv.is_file():
        raise FileNotFoundError(f"Fights CSV not found: {fights_csv}")

    ids = fighter_ids_from_fights_csv(fights_csv)
    if max_fighters is not None:
        ids = ids[:max_fighters]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sess = session or _session()
    referer = f"{BASE}/statistics/fighters"
    n_total = len(ids)

    print(
        f"Fighter profiles: {n_total} unique IDs from {fights_csv.name} -> {out_path} "
        f"(sleep {delay}s between requests)",
        flush=True,
    )
    if n_total == 0:
        print("No fighter IDs found; exiting.", flush=True)
        return 0

    rows: List[Dict[str, Any]] = []
    n_skip = 0
    t0 = time.perf_counter()
    for i, fid in enumerate(ids):
        if i:
            time.sleep(delay)
        url = f"{BASE}/fighter-details/{fid}"
        try:
            soup = fetch_soup(sess, url, referer=referer)
        except requests.RequestsError:
            n_skip += 1
            if progress_every > 0:
                print(
                    f"  [{i + 1}/{n_total}] HTTP error | id {fid} (skip total {n_skip})",
                    flush=True,
                )
            continue
        parsed = parse_fighter_profile_html(soup, fid)
        if not parsed:
            n_skip += 1
            if progress_every > 0:
                print(
                    f"  [{i + 1}/{n_total}] parse failed | id {fid} (skip total {n_skip})",
                    flush=True,
                )
            continue
        rows.append(parsed)

        done = i + 1
        if progress_every > 0 and (
            done == 1 or done % progress_every == 0 or done == n_total
        ):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  [{done}/{n_total}] ok={len(rows)} skip={n_skip} "
                f"~{rate:.1f} profiles/s | {parsed['name']}",
                flush=True,
            )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} fighter profiles -> {out_path} (skipped: {n_skip})", flush=True)
    return len(rows)


def _resolve_fights_csv(data_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        return Path(explicit)
    primary = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    if primary.is_file():
        return primary
    legacy = data_dir / LEGACY_FIGHTS_CSV
    if legacy.is_file():
        return legacy
    return primary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scrape UFCStats fighter profiles for IDs in fights CSV")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--fights-csv", type=Path, default=None, help="Defaults to ufcstats_fights.csv or tier1_ufcstats.csv under --data-dir")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--sleep",
        type=float,
        default=None,
        help=f"Seconds between requests (default: {REQUEST_DELAY_SEC}, same as ufcstats scraper)",
    )
    p.add_argument("--max-fighters", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args(argv)

    if args.data_dir:
        dd = Path(args.data_dir)
        tier1 = _resolve_fights_csv(dd, args.fights_csv)
        out = Path(args.out) if args.out else dd / "fighter_profiles.csv"
    else:
        tier1 = args.fights_csv or Path("data") / DEFAULT_UFCSTATS_FIGHTS_CSV
        if not Path(tier1).is_file():
            leg = Path("data") / LEGACY_FIGHTS_CSV
            if leg.is_file():
                tier1 = leg
        out = args.out or Path("data/fighter_profiles.csv")

    scrape_fighter_profiles_to_csv(
        tier1,
        out,
        sleep_sec=args.sleep,
        max_fighters=args.max_fighters,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
