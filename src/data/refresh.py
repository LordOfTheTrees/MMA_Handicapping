"""
Optional hook: repopulate ``data_dir`` with CSVs before a full-rebuild train run.

``main.py train --full-rebuild`` calls :func:`refresh_data`. Replace the body with
your scraper or orchestration (write tier*.csv and fighter_profiles.csv).
"""

from pathlib import Path


def refresh_data(data_dir: Path) -> None:
    """Download or regenerate CSVs under *data_dir*, then return."""
    raise NotImplementedError(
        "Implement refresh_data() in src/data/refresh.py to fetch CSVs into data_dir, "
        "or run `python main.py train` without --full-rebuild to use existing files."
    )
