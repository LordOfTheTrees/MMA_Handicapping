"""
Compare the UFCStats fights CSV to the universe of fights listed on UFCStats event pages.

Uses optional cached event inventory to avoid repeat full event crawls.

Usage::

    python -m src.data.ufcstats_gap_report --fetch-inventory-only --write-inventory-csv ./data/ufcstats_event_inventory.csv
    python -m src.data.ufcstats_gap_report --data-dir ./data --inventory-csv ./data/ufcstats_event_inventory.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from curl_cffi import requests

from src.data.tier1_inventory_io import load_inventory_csv, save_inventory_csv
from src.data.ufcstats_scraper import (
    DEFAULT_UFCSTATS_FIGHTS_CSV,
    ExpectedFight,
    diagnose_fight_parse_failure,
    fetch_soup,
    iter_expected_fights_from_completed_events,
    _session,
)

LEGACY_FIGHTS_CSV = "tier1_ufcstats.csv"


def local_fights_csv_row_issues(path: Path) -> List[str]:
    path = Path(path)
    issues: List[str] = []
    fight_counts: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            fid = (row.get("fight_id") or "").strip()
            a = (row.get("fighter_a_id") or "").strip()
            b = (row.get("fighter_b_id") or "").strip()
            w = (row.get("winner_id") or "").strip()
            method = (row.get("method") or "").strip().lower()
            if fid:
                fight_counts[fid] += 1
            if a and b and a == b:
                issues.append(f"line {i} fight_id={fid}: fighter_a_id == fighter_b_id")
            if w and a and b and w not in (a, b):
                issues.append(f"line {i} fight_id={fid}: winner_id not in {{a,b}}")
            if not w and method in ("ko/tko", "submission"):
                issues.append(f"line {i} fight_id={fid}: empty winner_id but method={method!r}")
    for fid, n in fight_counts.items():
        if n > 1:
            issues.append(f"duplicate fight_id {fid!r} appears {n} times")
    return issues


def fight_ids_from_fights_csv(path: Path) -> Set[str]:
    out: Set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("fight_id") or "").strip()
            if fid:
                out.add(fid)
    return out


def build_inventory(
    *,
    sleep_sec: float,
    max_events: Optional[int],
    session: Optional[requests.Session],
    progress_every: int,
) -> tuple[Dict[str, ExpectedFight], int]:
    inventory: Dict[str, ExpectedFight] = {}
    dup = 0
    t0 = time.perf_counter()
    for i, ef in enumerate(
        iter_expected_fights_from_completed_events(
            max_events=max_events,
            session=session,
            request_delay_sec=sleep_sec,
        )
    ):
        if ef.fight_id in inventory:
            dup += 1
        inventory[ef.fight_id] = ef
        if progress_every > 0 and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  inventory: {i + 1} fight slots, {len(inventory)} unique ids "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )
    return inventory, dup


def run_gap_report(
    fights_csv: Path,
    out_missing: Path,
    *,
    sleep_sec: float = 0.1,
    max_events: Optional[int] = None,
    diagnose: bool = True,
    inventory_progress_every: int = 500,
    diagnose_progress_every: int = 25,
    session: Optional[requests.Session] = None,
    inventory_csv: Optional[Path] = None,
    write_inventory_csv: Optional[Path] = None,
) -> None:
    fights_csv = Path(fights_csv)
    if not fights_csv.is_file():
        raise FileNotFoundError(fights_csv)

    have_ids = fight_ids_from_fights_csv(fights_csv)
    print(f"Fights CSV: {len(have_ids)} fight_id rows in {fights_csv}", flush=True)

    sess = session or _session()
    inv_dup = 0
    if inventory_csv:
        inv_path = Path(inventory_csv)
        print(f"Loading event inventory from {inv_path} (no event crawl) ...", flush=True)
        inventory = load_inventory_csv(inv_path)
    else:
        print("Building expected fight list from event pages (no per-fight fetch yet) ...", flush=True)
        inventory, inv_dup = build_inventory(
            sleep_sec=sleep_sec,
            max_events=max_events,
            session=sess,
            progress_every=inventory_progress_every,
        )
        if write_inventory_csv:
            wpath = Path(write_inventory_csv)
            save_inventory_csv(wpath, inventory)
            print(f"Saved inventory ({len(inventory)} fights) -> {wpath}", flush=True)
    expected_ids = set(inventory.keys())
    missing = sorted(expected_ids - have_ids)
    extra = sorted(have_ids - expected_ids)

    print("", flush=True)
    print(f"  Unique fight ids on event pages: {len(expected_ids)}", flush=True)
    print(f"  Duplicate fight_id slots (re-listed): {inv_dup}", flush=True)
    print(f"  Missing from fights CSV: {len(missing)}", flush=True)
    print(f"  Extra in CSV only (not on current event crawl): {len(extra)}", flush=True)

    fieldnames = [
        "fight_id",
        "fight_url",
        "event_url",
        "event_date",
        "diagnose_reason",
    ]
    rows_out: List[dict] = []

    if diagnose and missing:
        print(f"Diagnosing {len(missing)} missing fight pages ...", flush=True)
        t0 = time.perf_counter()
        for i, fid in enumerate(missing):
            if i:
                time.sleep(sleep_sec)
            ef = inventory[fid]
            reason = "http_error"
            try:
                soup = fetch_soup(sess, ef.fight_url, referer=ef.event_url)
                reason = diagnose_fight_parse_failure(soup, fid, ef.event_date)
            except requests.RequestsError:
                pass
            rows_out.append(
                {
                    "fight_id": fid,
                    "fight_url": ef.fight_url,
                    "event_url": ef.event_url,
                    "event_date": ef.event_date.isoformat(),
                    "diagnose_reason": reason,
                }
            )
            if diagnose_progress_every > 0 and (i + 1) % diagnose_progress_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"  diagnosed {i + 1}/{len(missing)} ({elapsed:.0f}s)", flush=True)

    out_missing = Path(out_missing)
    out_missing.parent.mkdir(parents=True, exist_ok=True)
    with open(out_missing, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {out_missing}", flush=True)


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
    p = argparse.ArgumentParser(description="Diff UFCStats site vs local fights CSV")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--fights-csv",
        type=Path,
        default=None,
        help=f"Defaults to {DEFAULT_UFCSTATS_FIGHTS_CSV} or {LEGACY_FIGHTS_CSV} under --data-dir",
    )
    p.add_argument(
        "--out-missing",
        type=Path,
        default=None,
        help="CSV of fights on site but not in local file (default: <data-dir>/ufcstats_missing_fights.csv)",
    )
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--no-diagnose", action="store_true")
    p.add_argument("--inventory-progress-every", type=int, default=500)
    p.add_argument("--diagnose-progress-every", type=int, default=25)
    p.add_argument("--inventory-csv", type=Path, default=None)
    p.add_argument("--write-inventory-csv", type=Path, default=None)
    p.add_argument("--fetch-inventory-only", action="store_true")
    p.add_argument(
        "--check-csv-only",
        action="store_true",
        help="Print local row issues only; no network",
    )
    args = p.parse_args(argv)

    if args.check_csv_only:
        if args.data_dir:
            fc = _resolve_fights_csv(Path(args.data_dir), args.fights_csv)
        else:
            fc = args.fights_csv or Path("data") / DEFAULT_UFCSTATS_FIGHTS_CSV
            if not fc.is_file():
                leg = Path("data") / LEGACY_FIGHTS_CSV
                if leg.is_file():
                    fc = leg
        issues = local_fights_csv_row_issues(fc)
        if not issues:
            print(f"No local row issues found in {fc}", flush=True)
        else:
            print(f"{len(issues)} issue(s) in {fc}:", flush=True)
            for line in issues[:200]:
                print(f"  {line}", flush=True)
            if len(issues) > 200:
                print(f"  ... and {len(issues) - 200} more", flush=True)
        return 0

    if args.fetch_inventory_only:
        if not args.write_inventory_csv:
            p.error("--fetch-inventory-only requires --write-inventory-csv")
        print("Fetching event pages only (inventory cache) ...", flush=True)
        sess = _session()
        inventory, dup = build_inventory(
            sleep_sec=args.sleep,
            max_events=args.max_events,
            session=sess,
            progress_every=args.inventory_progress_every,
        )
        save_inventory_csv(Path(args.write_inventory_csv), inventory)
        print(
            f"Wrote {len(inventory)} unique fight ids (duplicate card slots: {dup}) "
            f"-> {args.write_inventory_csv}",
            flush=True,
        )
        return 0

    if args.data_dir:
        dd = Path(args.data_dir)
        fights = _resolve_fights_csv(dd, args.fights_csv)
        out = Path(args.out_missing) if args.out_missing else dd / "ufcstats_missing_fights.csv"
    else:
        fights = args.fights_csv or Path("data") / DEFAULT_UFCSTATS_FIGHTS_CSV
        if not Path(fights).is_file():
            leg = Path("data") / LEGACY_FIGHTS_CSV
            if leg.is_file():
                fights = leg
        out = args.out_missing or Path("data/ufcstats_missing_fights.csv")

    run_gap_report(
        fights,
        out,
        sleep_sec=args.sleep,
        max_events=args.max_events,
        diagnose=not args.no_diagnose,
        inventory_progress_every=args.inventory_progress_every,
        diagnose_progress_every=args.diagnose_progress_every,
        inventory_csv=args.inventory_csv,
        write_inventory_csv=args.write_inventory_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
