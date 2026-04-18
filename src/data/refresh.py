"""
Optional hook: repopulate ``data_dir`` with CSVs before a full-rebuild train run.

``main.py train --full-rebuild`` calls :func:`refresh_data`.
"""

from pathlib import Path

from .ufcstats_profiles import scrape_fighter_profiles_to_csv
from .ufcstats_scraper import DEFAULT_UFCSTATS_FIGHTS_CSV, scrape_ufcstats_fights_to_csv


def refresh_data(data_dir: Path) -> None:
    """Regenerate UFCStats fights CSV and fighter profiles under *data_dir*."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fights_path = data_dir / DEFAULT_UFCSTATS_FIGHTS_CSV
    scrape_ufcstats_fights_to_csv(fights_path)
    scrape_fighter_profiles_to_csv(fights_path, data_dir / "fighter_profiles.csv")
