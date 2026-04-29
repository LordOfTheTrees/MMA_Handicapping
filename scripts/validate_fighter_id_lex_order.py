#!/usr/bin/env python3
"""
Verify fighter_a_id < fighter_b_id (lexicographic) on every row of fight CSVs.

Matches the UFCStats scrape convention: corners stored as sorted(string ids), not page order.

Usage::

    python scripts/validate_fighter_id_lex_order.py
    python scripts/validate_fighter_id_lex_order.py --data-dir ./data
    python scripts/validate_fighter_id_lex_order.py --csv path/to/fights.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same discovery order as MMAPredictor.load_data (tier1 uses first match only for loading,
# but we validate every present file so merged/manual exports are checked too.)
_DATA_DIR_DEFAULT_CSVS = (
    "ufcstats_fights.csv",
    "tier1_ufcstats.csv",
    "tier2_bellator.csv",
    "tier2_one.csv",
    "tier2_pfl.csv",
    "tier2_rizin.csv",
    "tier3_sherdog.csv",
)


def _check_csv(path: Path) -> tuple[int, list[str]]:
    """Return (rows_checked, violation_messages)."""
    violations: list[str] = []
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return 0, [f"{path}: empty or unreadable"]
        fa_key = "fighter_a_id"
        fb_key = "fighter_b_id"
        if fa_key not in reader.fieldnames or fb_key not in reader.fieldnames:
            return 0, [
                f"{path}: missing {fa_key!r} / {fb_key!r} columns "
                f"(have {reader.fieldnames!r})"
            ]
        for i, row in enumerate(reader, start=2):  # header line 1
            n += 1
            a = (row.get(fa_key) or "").strip()
            b = (row.get(fb_key) or "").strip()
            fid = (row.get("fight_id") or "").strip() or f"row {i}"
            if a == b:
                violations.append(f"{path}:{i} fight_id={fid!r} fighter_a_id == fighter_b_id {a!r}")
            elif not (a < b):
                violations.append(
                    f"{path}:{i} fight_id={fid!r} expected fighter_a_id < fighter_b_id; "
                    f"got {a!r} vs {b!r}"
                )
    return n, violations


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate lex order fighter_a_id < fighter_b_id.")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Directory containing CSVs")
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single fights CSV (if set, --data-dir discovery is skipped)",
    )
    args = ap.parse_args()

    paths: list[Path]
    if args.csv is not None:
        paths = [args.csv]
    else:
        data_dir = args.data_dir
        paths = [data_dir / name for name in _DATA_DIR_DEFAULT_CSVS]
        paths = [p for p in paths if p.exists()]

    if not paths:
        print(
            f"No fight CSVs found under {args.data_dir} "
            f"(expected names like ufcstats_fights.csv). Nothing to validate.",
            file=sys.stderr,
        )
        return 0

    total_rows = 0
    all_violations: list[str] = []
    for p in paths:
        n, viol = _check_csv(p)
        total_rows += n
        all_violations.extend(viol)

    print(f"Checked {total_rows:,} rows across {len(paths)} file(s).")
    if all_violations:
        print(f"FAIL: {len(all_violations)} violation(s):", file=sys.stderr)
        for line in all_violations[:50]:
            print(line, file=sys.stderr)
        if len(all_violations) > 50:
            print(f"... and {len(all_violations) - 50} more", file=sys.stderr)
        return 1
    print("OK: every row has fighter_a_id < fighter_b_id (lexicographic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
