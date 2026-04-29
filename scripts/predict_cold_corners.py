#!/usr/bin/env python3
"""Cold-corner prediction: synthetic v synthetic or synthetic v known fighter id."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import resolve_date, resolve_weight_class  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402
from src.synthetic_matchups import SyntheticCorner, predict_cold_corner_matchup  # noqa: E402


def _corner_arg(s: str) -> SyntheticCorner | str:
    """Parse 'id:<hex>' for known UFC id, else build SyntheticCorner."""
    if s.startswith("id:"):
        return s[3:].strip()
    parts = [p.strip() for p in s.split("|")]
    name = parts[0] if parts else "Unknown"

    def _opt_float(i: int) -> Optional[float]:
        if len(parts) <= i or not parts[i]:
            return None
        return float(parts[i])

    def _ped(i: int, default: float) -> float:
        v = _opt_float(i)
        return default if v is None else v

    reach = _opt_float(1)
    height = _opt_float(2)
    box = _ped(3, 0.34)
    wrestle = _ped(4, 0.33)
    bjj = _ped(5, 0.33)
    return SyntheticCorner(
        display_name=name,
        reach_cm=reach,
        height_cm=height,
        boxing_pedigree=box,
        wrestling_pedigree=wrestle,
        bjj_pedigree=bjj,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Corner syntax: "
            "`id:<fighter_id>` for a known UFCStats id from your data, "
            "or `Name|reach|height|box|wrestle|bjj` (pipe fields; pedigree 0..1 defaults 0.34/0.33/0.33)."
        )
    )
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--corner-a", type=str, required=True)
    ap.add_argument("--corner-b", type=str, required=True)
    ap.add_argument("--weight-class", type=str, required=True)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    pred = MMAPredictor.load(args.model_path.resolve())
    wc = resolve_weight_class(args.weight_class)
    fd = resolve_date(args.date) if args.date else date.today()

    ca = _corner_arg(args.corner_a)
    cb = _corner_arg(args.corner_b)

    predict_cold_corner_matchup(pred, ca, cb, wc, fd, verbose=True)


if __name__ == "__main__":
    main()
