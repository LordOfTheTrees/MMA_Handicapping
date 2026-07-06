"""
Optional hook: repopulate ``data_dir`` with CSVs before a full-rebuild train run.

``main.py train --full-rebuild`` and ``scripts/weekly_update.py`` call :func:`refresh_data`.

Tier-1 fight stats use ESPN (cached, rate-limited) with UFCStats hex IDs preserved
via crosswalk tables. UFCStats HTML scrape is no longer the primary path.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .espn_audit import format_audit_log_lines, run_espn_ingest_audit
from .espn_ingest import refresh_espn_fights_incremental
from .espn_profiles import refresh_espn_profiles_incremental
from .espn_upcoming import DEFAULT_ESPN_UPCOMING_CARDS_JSON, scrape_espn_upcoming_cards_to_path
from .ufcstats_profiles import scrape_fighter_profiles_to_csv
from .ufcstats_scraper import DEFAULT_UFCSTATS_FIGHTS_CSV, probe_completed_events_index
from .ufcstats_upcoming import DEFAULT_UPCOMING_CARDS_JSON, scrape_upcoming_cards_to_path


class DataRefreshError(RuntimeError):
    """ESPN ingest, row-count guard, or post-ingest audit rejected the refresh."""


@dataclass(frozen=True)
class RefreshResult:
    fights_total: int
    fights_updated: int
    audit_passed: bool
    audit_reject_count: int = 0
    audit_warn_count: int = 0
    upcoming_cards_scraped: bool = False
    espn_upcoming_cards_scraped: bool = False


def _count_fight_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def refresh_data(
    data_dir: Path,
    *,
    run_audit: bool = True,
    fail_on_audit_reject: bool = True,
    require_fight_updates: bool = False,
    fetch_rookie_audit: bool = True,
    ufcstats_gap_fill: bool = True,
    espn_upcoming: bool = True,
    espn_max_events: Optional[int] = None,
    espn_max_competitions: Optional[int] = None,
    espn_verbose: bool = True,
) -> RefreshResult:
    """Refresh fights (ESPN), profiles, audit, ESPN upcoming cards, then UFCStats gap-fill.

    The fights CSV uses only **completed** events (see ADR-05). ``espn_upcoming_cards.json``
    (ESPN's ``fightcenter`` payload, already-reliable in CI — see
    ``docs/ufc-com-upcoming-scrape-plan.md`` §0) is attempted independently of UFCStats and
    written to its own file so a bad UFCStats scrape can never clobber it or vice versa.
    ``upcoming_cards.json`` still uses UFCStats when reachable; otherwise the prior file is
    left in place. Either way, :attr:`RefreshResult.upcoming_cards_scraped` /
    :attr:`RefreshResult.espn_upcoming_cards_scraped` are ``False`` when that source didn't
    produce fresh data this run, so callers (``weekly_update.py``, CI) know not to re-export a
    stale ``upcoming_events.json`` from it.

    Raises :class:`DataRefreshError` when ingest fails, ``require_fight_updates`` is set
    but zero fights changed, or audit rejects and ``fail_on_audit_reject`` is true.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    before_rows = _count_fight_rows(fights_path)

    print("[refresh] ESPN incremental fights (cached) ...", flush=True)
    try:
        total, n_updated = refresh_espn_fights_incremental(
            data_dir,
            max_events=espn_max_events,
            max_competitions=espn_max_competitions,
            verbose=espn_verbose,
        )
    except Exception as e:
        raise DataRefreshError(f"ESPN incremental ingest failed: {e}") from e

    after_rows = _count_fight_rows(fights_path)
    if after_rows < 1 and before_rows < 1:
        raise DataRefreshError("ufcstats_fights.csv has 0 rows after ESPN refresh.")

    if require_fight_updates and n_updated < 1:
        raise DataRefreshError(
            f"ESPN ingest updated/added 0 fights ({before_rows:,} rows unchanged)."
        )

    print(
        f"[refresh] Fight rows {before_rows:,} -> {after_rows:,} "
        f"({n_updated} updated/added this run); ESPN profiles ...",
        flush=True,
    )
    refresh_espn_profiles_incremental(data_dir)

    audit_passed = True
    reject_count = 0
    warn_count = 0
    if run_audit:
        audit, audit_code = run_espn_ingest_audit(
            data_dir,
            fetch_rookie=fetch_rookie_audit,
            fail_on_reject=False,
            print_terminal=True,
        )
        reject_count = int(audit.get("reject_count") or 0)
        warn_count = int(audit.get("warn_count") or 0)
        audit_passed = bool(audit.get("passed"))
        if fail_on_audit_reject and not audit_passed:
            raise DataRefreshError(
                "ESPN ingest audit failed (probable duplicate or non-rookie espn_* id). "
                f"See {data_dir / 'espn_ingest_audit.json'}."
            )

    espn_upcoming_cards_scraped = False
    if espn_upcoming:
        print("[refresh] ESPN upcoming cards ...", flush=True)
        try:
            scrape_espn_upcoming_cards_to_path(data_dir / DEFAULT_ESPN_UPCOMING_CARDS_JSON, data_dir)
            espn_upcoming_cards_scraped = True
        except Exception as e:
            print(f"[refresh] ESPN upcoming cards scrape failed: {e}", flush=True)

    upcoming_cards_scraped = False
    if ufcstats_gap_fill:
        probe = probe_completed_events_index()
        if probe.blocked:
            print(
                f"[refresh] UFCStats blocked ({probe.detail}); "
                "skipping UFCStats profile/upcoming scrape.",
                flush=True,
            )
            return RefreshResult(
                fights_total=total,
                fights_updated=n_updated,
                audit_passed=audit_passed,
                audit_reject_count=reject_count,
                audit_warn_count=warn_count,
                upcoming_cards_scraped=False,
                espn_upcoming_cards_scraped=espn_upcoming_cards_scraped,
            )

        print("[refresh] UFCStats fighter profiles (gap-fill) ...", flush=True)
        if fights_path.is_file():
            scrape_fighter_profiles_to_csv(fights_path, data_dir / "fighter_profiles.csv")

        print("[refresh] UFCStats upcoming cards ...", flush=True)
        try:
            scrape_upcoming_cards_to_path(data_dir / DEFAULT_UPCOMING_CARDS_JSON)
            upcoming_cards_scraped = True
        except Exception as e:
            print(f"[refresh] Upcoming cards scrape skipped: {e}", flush=True)

    return RefreshResult(
        fights_total=total,
        fights_updated=n_updated,
        audit_passed=audit_passed,
        audit_reject_count=reject_count,
        audit_warn_count=warn_count,
        upcoming_cards_scraped=upcoming_cards_scraped,
        espn_upcoming_cards_scraped=espn_upcoming_cards_scraped,
    )
