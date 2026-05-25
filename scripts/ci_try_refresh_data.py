#!/usr/bin/env python3
"""
UFCStats refresh for GitHub Actions and local runs.

Default: require a live scrape; fail with a clear log if UFCStats serves a bot wall.

``--allow-stale-data``: use fights CSV from ``ci_restore_data_bundle`` when scrape is blocked.
Weekly/monthly workflows pass this only when you enable **allow_stale_data** on manual dispatch
(default false). Scheduled cron never uses stale fallback.

Documented in **docs/BACKEND_PIPELINE_INTEGRATION.md** (§ GitHub Actions admin) and
**docs/session-changelog-and-catchup.md**.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ufcstats_profiles import scrape_fighter_profiles_to_csv  # noqa: E402
from src.data.ufcstats_scraper import (  # noqa: E402
    DEFAULT_UFCSTATS_FIGHTS_CSV,
    probe_completed_events_index,
    scrape_ufcstats_fights_to_csv,
)
from src.data.ufcstats_upcoming import (  # noqa: E402
    DEFAULT_UPCOMING_CARDS_JSON,
    scrape_upcoming_cards_to_path,
)


def _count_fight_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="CI UFCStats refresh (strict or allow-stale).")
    p.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="If live scrape is blocked, use restored bundle CSVs instead of failing.",
    )
    args = p.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    before = _count_fight_rows(fights_path)

    probe = probe_completed_events_index()
    print(f"[ci_refresh] Index probe: {probe.detail}", flush=True)

    if probe.blocked:
        print("::group::UFCStats scrape blocked", flush=True)
        print(probe.detail, flush=True)
        print(
            "Live scrape is required unless --allow-stale-data is set and a prior run bundle "
            "was restored under data/.",
            flush=True,
        )
        print("::endgroup::", flush=True)
        if not args.allow_stale_data or before < 1:
            print(f"::error::{probe.detail}", file=sys.stderr)
            return 1
        print(
            f"::warning::--allow-stale-data: using restored fights CSV ({before:,} rows); "
            "skipping profile/upcoming scrape.",
            flush=True,
        )
        return 0

    scraped = scrape_ufcstats_fights_to_csv(fights_path)
    after = _count_fight_rows(fights_path)
    if after < 1:
        print(
            "::error::Scrape index was reachable but ufcstats_fights.csv has 0 rows.",
            file=sys.stderr,
        )
        return 1

    if scraped > 0 and after > before:
        print(
            f"[ci_refresh] Scraped {scraped} fight rows ({before} -> {after}); "
            "refreshing profiles and upcoming.",
            flush=True,
        )
        scrape_fighter_profiles_to_csv(fights_path, data_dir / "fighter_profiles.csv")
        scrape_upcoming_cards_to_path(data_dir / DEFAULT_UPCOMING_CARDS_JSON)
    elif before < 1 and not args.allow_stale_data:
        print(
            "::error::Live scrape added no fight rows and stale fallback is not allowed.",
            file=sys.stderr,
        )
        return 1
    elif before >= 1 and after == before:
        print(
            f"[ci_refresh] No new fight rows ({after:,}); using existing CSVs.",
            flush=True,
        )

    print(f"[ci_refresh] Ready: {after:,} fights in {fights_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
