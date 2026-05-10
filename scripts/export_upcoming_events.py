#!/usr/bin/env python3
"""
Turn ``data/upcoming_cards.json`` (from ``src.data.ufcstats_upcoming``) into deploy JSON.

::

    python scripts/export_upcoming_events.py --cards data/upcoming_cards.json --out JSON_exports/upcoming_events.json
    python scripts/export_upcoming_events.py ... --copy-to-mma-ai
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT_SCHEMA_VERSION = "mma-handicapping-upcoming-v1"


def _git_sha() -> Optional[str]:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if cp.returncode == 0 and cp.stdout:
            return cp.stdout.strip()
    except OSError:
        pass
    return None


def build_upcoming_events_doc(cards: Dict[str, Any]) -> Dict[str, Any]:
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": exported_at,
        "git_sha_training_repo": _git_sha(),
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


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Export upcoming_cards.json to upcoming_events.json for deploy.")
    p.add_argument("--cards", type=Path, required=True, help="Path to upcoming_cards.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--copy-to-mma-ai",
        action="store_true",
        help="After write, copy output file to sibling mma.ai/artifacts",
    )
    p.add_argument(
        "--mma-ai-artifacts-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override deploy dir (default: <repo>/../mma.ai/artifacts)",
    )
    args = p.parse_args(argv)

    cards = json.loads(Path(args.cards).read_text(encoding="utf-8"))
    doc = build_upcoming_events_doc(cards)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {outp.resolve()}", flush=True)

    if args.copy_to_mma_ai:
        _scripts = Path(__file__).resolve().parent
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        import copy_exports_to_mma_ai as _cex

        dest = (
            Path(args.mma_ai_artifacts_dir).resolve()
            if args.mma_ai_artifacts_dir
            else _cex.default_mma_ai_artifacts_dir()
        )
        dst = _cex.copy_json_file(outp, dest)
        print(f"Copied -> {dst}", flush=True)


if __name__ == "__main__":
    main()
