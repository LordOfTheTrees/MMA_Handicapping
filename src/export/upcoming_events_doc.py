"""Pure transform: ``upcoming_cards.json`` ingest doc -> deploy ``upcoming_events`` shape."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from src.export.git_meta import git_sha_training_repo

EXPORT_SCHEMA_VERSION = "mma-handicapping-upcoming-v1"


def build_upcoming_events_doc(cards: Dict[str, Any]) -> Dict[str, Any]:
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": exported_at,
        "git_sha_training_repo": git_sha_training_repo(),
        "source_upstream_scraped_at": cards.get("scraped_at"),
        "source_upstream_schema": cards.get("schema_version"),
    }

    events_out: list[dict[str, Any]] = []
    for ev in cards.get("events", []):
        bouts_clean: list[dict[str, Any]] = []
        for b in ev.get("bouts", []):
            bouts_clean.append(
                {
                    "bout_order": b["bout_order"],
                    "fight_id": b["fight_id"],
                    "fight_url": b.get("fight_url"),
                    "fighter_a_id": b["fighter_a_id"],
                    "fighter_b_id": b["fighter_b_id"],
                    "fighter_a_name": b.get("fighter_a_name"),
                    "fighter_b_name": b.get("fighter_b_name"),
                    "weight_class": b.get("weight_class"),
                    "weight_class_raw": b.get("weight_class_raw"),
                }
            )
        events_out.append(
            {
                "event_id": ev.get("event_id"),
                "event_title": ev.get("event_title"),
                "event_date": ev.get("event_date"),
                "event_url": ev.get("event_url"),
                "location": ev.get("location"),
                "bouts": bouts_clean,
            }
        )

    return {
        "export_manifest": manifest,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "events": events_out,
    }
