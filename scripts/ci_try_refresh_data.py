#!/usr/bin/env python3
"""
Data refresh for GitHub Actions and local runs.

CI path (weekly/monthly workflows):

1. ``ci_restore_data_bundle`` — load last week's CSVs as the baseline.
2. :func:`refresh_data` — ESPN incremental, profiles, audit (not a full rescrape).
3. By default **fail** if ESPN is unreachable or if zero fights were updated/added.

Use ``--allow-stale-data`` only for manual/debug runs when you accept no new data.

Writes ``espn_upcoming_scraped`` / ``ufcstats_upcoming_scraped`` step outputs (``true``/
``false``) so a later workflow step can gate ``export_upcoming_events.py`` on whether either
source actually completed this run (ESPN preferred — reliable in CI; UFCStats as fallback),
instead of re-exporting a stale ``*_cards.json`` carried over from a prior run.

Documented in **docs/BACKEND_PIPELINE_INTEGRATION.md** and **docs/data-sources-espn.md**.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.refresh import DataRefreshError, refresh_data  # noqa: E402


def _write_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


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

    try:
        print("[ci_refresh] refresh_data (ESPN + audit) ...", flush=True)
        result = refresh_data(
            data_dir,
            run_audit=True,
            fail_on_audit_reject=True,
            require_fight_updates=not args.allow_stale_data,
            fetch_rookie_audit=True,
        )
    except DataRefreshError as e:
        print("::group::Data refresh failed", flush=True)
        print(str(e), flush=True)
        print("::endgroup::", flush=True)
        _write_output("espn_upcoming_scraped", "false")
        _write_output("ufcstats_upcoming_scraped", "false")
        if args.allow_stale_data:
            print(f"::warning::--allow-stale-data: continuing without refresh ({e}).", flush=True)
            return 0
        print(f"::error::{e}", file=sys.stderr)
        return 1

    _write_output("espn_upcoming_scraped", "true" if result.espn_upcoming_cards_scraped else "false")
    _write_output("ufcstats_upcoming_scraped", "true" if result.upcoming_cards_scraped else "false")
    print("[ci_refresh] Ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
