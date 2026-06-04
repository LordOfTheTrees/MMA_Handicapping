"""
Optional hook: repopulate ``data_dir`` with CSVs before a full-rebuild train run.

``main.py train --full-rebuild`` calls :func:`refresh_data`.

Tier-1 fight stats use ESPN (cached, rate-limited) with UFCStats hex IDs preserved
via crosswalk tables. UFCStats HTML scrape is no longer the primary path.
"""
from pathlib import Path

from .espn_ingest import refresh_espn_fights_incremental
from .espn_profiles import refresh_espn_profiles_incremental
from .ufcstats_profiles import scrape_fighter_profiles_to_csv
from .ufcstats_scraper import DEFAULT_UFCSTATS_FIGHTS_CSV, probe_completed_events_index
from .ufcstats_upcoming import DEFAULT_UPCOMING_CARDS_JSON, scrape_upcoming_cards_to_path


def refresh_data(data_dir: Path) -> None:
    """Refresh fights (ESPN), profiles (ESPN crosswalk + UFCStats fallback), upcoming cards.

    The fights CSV uses only **completed** events (see ADR-05). ``upcoming_cards.json``
    still uses UFCStats when reachable; otherwise the prior file is left in place.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV

    print("[refresh] ESPN incremental fights (cached) ...", flush=True)
    refresh_espn_fights_incremental(data_dir)

    print("[refresh] ESPN profiles for crosswalked fighters ...", flush=True)
    refresh_espn_profiles_incremental(data_dir)

    probe = probe_completed_events_index()
    if probe.blocked:
        print(
            f"[refresh] UFCStats blocked ({probe.detail}); "
            "skipping UFCStats profile/upcoming scrape.",
            flush=True,
        )
        return

    print("[refresh] UFCStats fighter profiles (gap-fill) ...", flush=True)
    if fights_path.is_file():
        scrape_fighter_profiles_to_csv(fights_path, data_dir / "fighter_profiles.csv")

    print("[refresh] UFCStats upcoming cards ...", flush=True)
    try:
        scrape_upcoming_cards_to_path(data_dir / DEFAULT_UPCOMING_CARDS_JSON)
    except Exception as e:
        print(f"[refresh] Upcoming cards scrape skipped: {e}", flush=True)
