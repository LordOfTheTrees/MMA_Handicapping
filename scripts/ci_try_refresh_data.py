#!/usr/bin/env python3
"""
Data refresh for GitHub Actions and local runs.

CI path (weekly/monthly workflows):

1. ``ci_restore_data_bundle`` — load last week's CSVs as the baseline.
2. ESPN incremental — add or update fights since that baseline (not a full rescrape).
3. By default **fail** if ESPN is unreachable or if zero fights were updated/added.

Use ``--allow-stale-data`` only for manual/debug runs when you accept no new data.

Documented in **docs/BACKEND_PIPELINE_INTEGRATION.md** and **docs/data-sources-espn.md**.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.espn_audit import format_audit_log_lines, run_espn_ingest_audit  # noqa: E402
from src.data.espn_ingest import refresh_espn_fights_incremental  # noqa: E402
from src.data.espn_profiles import refresh_espn_profiles_incremental  # noqa: E402
from src.data.ufcstats_scraper import DEFAULT_UFCSTATS_FIGHTS_CSV  # noqa: E402
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
    p = argparse.ArgumentParser(description="CI data refresh (ESPN incremental; bundle seeded).")
    p.add_argument(
        "--allow-stale-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Succeed with restored bundle only (no ESPN failure / no row updates required).",
    )
    args = p.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    before = _count_fight_rows(fights_path)

    try:
        print("[ci_refresh] ESPN incremental ingest ...", flush=True)
        _total, n_updated = refresh_espn_fights_incremental(data_dir)
    except Exception as e:
        print("::group::ESPN ingest failed", flush=True)
        print(str(e), flush=True)
        print("::endgroup::", flush=True)
        if args.allow_stale_data and before >= 1:
            print(
                f"::warning::--allow-stale-data: using restored fights CSV ({before:,} rows).",
                flush=True,
            )
            return 0
        print(f"::error::ESPN ingest failed: {e}", file=sys.stderr)
        return 1

    after = _count_fight_rows(fights_path)
    if after < 1:
        if args.allow_stale_data and before >= 1:
            print(
                f"::warning::ESPN produced 0 rows; using restored bundle ({before:,} rows).",
                flush=True,
            )
            return 0
        print("::error::ufcstats_fights.csv has 0 rows after ESPN refresh.", file=sys.stderr)
        return 1

    if n_updated < 1:
        if args.allow_stale_data:
            print(
                f"::warning::--allow-stale-data: ESPN changed 0 fights "
                f"({before:,} rows unchanged).",
                flush=True,
            )
        else:
            print(
                "::error::ESPN ingest completed but updated/added 0 fights. "
                "Weekly/monthly run has no new data to fold into the model.",
                file=sys.stderr,
            )
            return 1

    print(
        f"[ci_refresh] Fight rows {before:,} -> {after:,} ({n_updated} updated/added); "
        "refreshing ESPN-linked profiles.",
        flush=True,
    )
    refresh_espn_profiles_incremental(data_dir)

    print("[ci_refresh] ESPN ID audit (new espn_* fighters/fights) ...", flush=True)
    audit, audit_code = run_espn_ingest_audit(data_dir, fetch_rookie=True, fail_on_reject=True)
    print("::group::ESPN ingest audit", flush=True)
    for line in format_audit_log_lines(audit):
        print(line, flush=True)
    print("::endgroup::", flush=True)
    if audit_code != 0:
        print(
            "::error::ESPN ingest audit failed (probable duplicate or non-rookie espn_* id). "
            "See data/espn_ingest_audit.json.",
            file=sys.stderr,
        )
        return 1

    try:
        scrape_upcoming_cards_to_path(data_dir / DEFAULT_UPCOMING_CARDS_JSON)
    except Exception as e:
        print(f"[ci_refresh] Upcoming cards skipped: {e}", flush=True)

    print(f"[ci_refresh] Ready: {after:,} fights in {fights_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
