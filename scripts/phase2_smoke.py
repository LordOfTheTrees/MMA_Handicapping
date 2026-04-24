#!/usr/bin/env python3
"""
Phase 2 smoke checks: finite training matrix, label balance, symmetry (swap A/B), ELO snapshot.

Run from repo root::

    python scripts/phase2_smoke.py
    python scripts/phase2_smoke.py --model-path model.pkl --fighter-a ID --fighter-b ID --wc lightweight
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import WEIGHT_CLASS_MAP  # noqa: E402
from src.data.schema import WeightClass  # noqa: E402
from src.model.regression import CLASS_LABELS, N_CLASSES  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402


def _resolve_wc(raw: str) -> WeightClass:
    from src.cli.common import resolve_weight_class  # noqa: E402

    return resolve_weight_class(raw)


def _wc_from_csv_cell(cell: str) -> WeightClass:
    key = (cell or "lightweight").strip().lower().replace("-", "_").replace(" ", "_")
    if key in WEIGHT_CLASS_MAP:
        return WEIGHT_CLASS_MAP[key]
    if key.replace("_", " ") in WEIGHT_CLASS_MAP:
        return WEIGHT_CLASS_MAP[key.replace("_", " ")]
    return _resolve_wc(key)


def _sample_pair_from_fights_csv(path: Path) -> tuple[str, str, WeightClass, date]:
    """Use the first row whose weight_class maps cleanly (skip exotic raw titles)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("weight_class") or "").strip()
            key = raw.lower().replace("-", "_").replace(" ", "_")
            if key in WEIGHT_CLASS_MAP or raw.lower().replace("-", " ") in WEIGHT_CLASS_MAP:
                a = row["fighter_a_id"].strip()
                b = row["fighter_b_id"].strip()
                wc = _wc_from_csv_cell(raw)
                fd = date.fromisoformat(row["date"])
                return a, b, wc, fd
    raise ValueError(f"No row with mapped weight class in {path}")


def _symmetry_pairs() -> list[tuple[int, int]]:
    """For probs from A vs B vs swapped B vs A: r1[i] should equal r2[j]."""
    return [(0, 4), (1, 5), (2, 3)]


def run_smoke(
    model_path: Path,
    fights_csv: Path | None,
    fighter_a: str | None,
    fighter_b: str | None,
    wc_s: str | None,
    fight_date: date | None,
    elo_wc: str,
    symmetry_tol: float,
    top_k: int,
) -> int:
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        print("Train first:  python main.py train --data-dir ./data", file=sys.stderr)
        return 2

    print(f"Loading {model_path} ...", flush=True)
    p = MMAPredictor.load(model_path)

    if p.regression is None or p._X_train is None:
        print("Loaded model has no regression / training matrix.", file=sys.stderr)
        return 2

    X = p._X_train
    y = p._y_train
    print(f"Training matrix: X shape {X.shape}, y shape {y.shape}", flush=True)

    if not np.isfinite(X).all():
        bad = np.sum(~np.isfinite(X))
        print(f"FAIL: {bad} non-finite values in _X_train", file=sys.stderr)
        return 1
    print("OK: all finite _X_train", flush=True)

    counts = Counter(y.tolist())
    print("Class counts:", ", ".join(f"{CLASS_LABELS[k]}={counts.get(k, 0)}" for k in range(N_CLASSES)), flush=True)
    if min(counts.get(k, 0) for k in range(N_CLASSES)) < 1:
        print("WARN: at least one outcome class has zero training rows.", file=sys.stderr)

    if fights_csv and fights_csv.is_file() and (fighter_a is None or fighter_b is None):
        fighter_a, fighter_b, wc, fight_date = _sample_pair_from_fights_csv(fights_csv)
        print(f"Sample matchup from {fights_csv.name}: {fighter_a} vs {fighter_b} | {wc.value} | {fight_date}", flush=True)
    else:
        if not fighter_a or not fighter_b or not wc_s:
            print("Provide --fighter-a/--fighter-b/--wc (and optional --date) or a readable --fights-csv", file=sys.stderr)
            return 2
        wc = _resolve_wc(wc_s)
        if fight_date is None:
            fight_date = date.today()

    p1 = p.predict_proba_point_only(fighter_a, fighter_b, wc, fight_date)
    p2 = p.predict_proba_point_only(fighter_b, fighter_a, wc, fight_date)
    s1, s2 = float(p1.sum()), float(p2.sum())
    if abs(s1 - 1.0) > 1e-5 or abs(s2 - 1.0) > 1e-5:
        print(f"FAIL: prob sums {s1}, {s2} (expected 1.0)", file=sys.stderr)
        return 1

    max_err = 0.0
    for i, j in _symmetry_pairs():
        max_err = max(max_err, abs(float(p1[i]) - float(p2[j])))
        max_err = max(max_err, abs(float(p1[j]) - float(p2[i])))
    print(f"Symmetry max |delta| (swap A/B): {max_err:.2e} (tol {symmetry_tol})", flush=True)
    if max_err > symmetry_tol:
        print("FAIL: symmetry check", file=sys.stderr)
        return 1
    print("OK: symmetry (point probs)", flush=True)

    wc_elo = _resolve_wc(elo_wc)
    as_of = date.today()
    ids: set[str] = set()
    for f in p.fights:
        if f.weight_class == wc_elo:
            ids.add(f.fighter_a_id)
            ids.add(f.fighter_b_id)
    ranked = [(fid, p.elo_model.get_elo(fid, wc_elo, as_of)) for fid in ids]
    ranked.sort(key=lambda t: t[1], reverse=True)
    print(f"\nELO snapshot ({wc_elo.value}, as_of={as_of}, n={len(ranked)} fighters with fights in class):", flush=True)
    for fid, elo in ranked[:top_k]:
        name = p.profiles.get(fid)
        nm = name.name if name else fid[:12]
        print(f"  top  {elo:8.1f}  {nm}", flush=True)
    for fid, elo in ranked[-top_k:]:
        name = p.profiles.get(fid)
        nm = name.name if name else fid[:12]
        print(f"  tail {elo:8.1f}  {nm}", flush=True)

    print("\nPhase 2 smoke checks passed.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 pipeline smoke checks")
    ap.add_argument("--model-path", type=Path, default=ROOT / "model.pkl")
    ap.add_argument("--fights-csv", type=Path, default=ROOT / "data" / "ufcstats_fights.csv")
    ap.add_argument("--fighter-a", default=None)
    ap.add_argument("--fighter-b", default=None)
    ap.add_argument("--wc", default=None, help="Weight class alias, e.g. lightweight")
    ap.add_argument("--date", default=None, help="Fight date YYYY-MM-DD")
    ap.add_argument("--elo-wc", default="lightweight", help="Weight class for ELO top/tail listing")
    ap.add_argument(
        "--symmetry-tol",
        type=float,
        default=0.12,
        help="Max |p1[i]-p2[j]| for mirrored classes (interactions break exact mirror; default ~12%%)",
    )
    ap.add_argument(
        "--strict-symmetry",
        action="store_true",
        help="Use tolerance 1e-5 (often fails with current matchup interaction features)",
    )
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    fdate = date.fromisoformat(args.date) if args.date else None
    tol = 1e-5 if args.strict_symmetry else args.symmetry_tol
    return run_smoke(
        args.model_path,
        args.fights_csv,
        args.fighter_a,
        args.fighter_b,
        args.wc,
        fdate,
        args.elo_wc,
        tol,
        args.top_k,
    )


if __name__ == "__main__":
    raise SystemExit(main())
