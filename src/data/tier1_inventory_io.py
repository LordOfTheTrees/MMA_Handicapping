"""Save/load event-inventory CSV for :mod:`ufcstats_gap_report` (avoid repeat event crawls)."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict

from src.data.ufcstats_scraper import ExpectedFight

INVENTORY_FIELDS = ["fight_id", "fight_url", "event_url", "event_date"]


def save_inventory_csv(path: Path, inventory: Dict[str, ExpectedFight]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(inventory.values(), key=lambda ef: (ef.event_date, ef.fight_id))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        w.writeheader()
        for ef in rows:
            w.writerow(
                {
                    "fight_id": ef.fight_id,
                    "fight_url": ef.fight_url,
                    "event_url": ef.event_url,
                    "event_date": ef.event_date.isoformat(),
                }
            )


def load_inventory_csv(path: Path) -> Dict[str, ExpectedFight]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    inventory: Dict[str, ExpectedFight] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("fight_id") or "").strip()
            if not fid:
                continue
            d_raw = (row.get("event_date") or "").strip()
            try:
                ev_date = date.fromisoformat(d_raw)
            except ValueError:
                continue
            inventory[fid] = ExpectedFight(
                fight_id=fid,
                fight_url=(row.get("fight_url") or "").strip(),
                event_url=(row.get("event_url") or "").strip(),
                event_date=ev_date,
            )
    return inventory
