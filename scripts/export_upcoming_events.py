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
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export.upcoming_events_doc import build_upcoming_events_doc  # noqa: E402


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
