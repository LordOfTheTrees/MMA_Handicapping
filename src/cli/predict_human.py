"""
Interactive ``predict-human``: resolve fighter IDs from profile names (exact +
fuzzy fallback), optionally pick among past fights between the pair, run
``predict`` / ``explain``.
"""
from __future__ import annotations

import difflib
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.schema import FightRecord, FighterProfile, WeightClass
from ..data.fighter_names import fighter_ids_for_exact_name

from argparse import ArgumentParser, Namespace


def _norm(s: str) -> str:
    return " ".join(str(s).strip().split()).casefold()


def _profile_display_name(fid: str, profiles: Dict[str, FighterProfile]) -> str:
    p = profiles.get(fid)
    if p and p.name and str(p.name).strip():
        return f"{p.name} ({fid})"
    return fid


def _fuzzy_candidates(
    query: str,
    profiles: Dict[str, FighterProfile],
    *,
    min_ratio: float = 0.55,
    limit: int = 25,
) -> List[Tuple[float, str, str]]:
    qn = _norm(query)
    if not qn:
        return []

    scored: List[Tuple[float, str, str]] = []
    for fid, p in profiles.items():
        pn = _norm(p.name) if p.name else ""
        if not pn:
            continue
        r = float(difflib.SequenceMatcher(None, qn, pn).ratio())
        bonus = 0.0
        if pn.startswith(qn) or qn in pn:
            bonus = 0.12
        r = min(1.0, r + bonus)
        if r >= min_ratio:
            scored.append((r, fid, str(p.name).strip()))

    scored.sort(key=lambda t: (-t[0], t[2]))
    return scored[:limit]


def _read_choice(prompt: str, n: int) -> int:
    while True:
        msg = f"{prompt} [1-{n}, q=quit] " if prompt else f"Select [1-{n}, q=quit] "
        raw = input(msg).strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("Cancelled.", flush=True)
            sys.exit(0)
        try:
            k = int(raw)
            if 1 <= k <= n:
                return k
        except ValueError:
            pass
        print("  Invalid choice — try again.", flush=True)


def _enumerate_pick(lines: Sequence[str]) -> int:
    for i, line in enumerate(lines, start=1):
        print(f"    {i:2}.  {line}", flush=True)
    return _read_choice("", len(lines)) - 1


def _interactive_pick_fighter(
    label: str,
    needle: Optional[str],
    profiles: Dict[str, FighterProfile],
) -> str:
    if needle is None:
        needle = input(f"{label} (fighter name): ").strip()
        if not needle:
            print("Empty name.", file=sys.stderr)
            sys.exit(1)

    ids = fighter_ids_for_exact_name(needle, profiles)
    if len(ids) == 1:
        fid = ids[0]
        print(f"  {label}: {_profile_display_name(fid, profiles)}", flush=True)
        return fid

    if len(ids) > 1:
        print(
            f"\n  Multiple profiles match {needle!r} exactly ({len(ids)}). Pick one:",
            flush=True,
        )
        rows = [_profile_display_name(x, profiles) for x in ids]
        return ids[_enumerate_pick(rows)]

    fuzz = _fuzzy_candidates(needle, profiles)
    print(f"\n  No exact name match for {needle!r}.", flush=True)

    if fuzz:
        print("  Did you mean (by similarity)?", flush=True)
        rows: List[Tuple[str, str]] = []
        for i, (rr, fid, name) in enumerate(fuzz, start=1):
            print(f"    {i:2}.  score={rr:.3f}  {name} ({fid})", flush=True)
            rows.append((f"{name} ({fid})", fid))
        rows.append(("None of these — enter a different name", "__retry__"))

        idx = _read_choice("\nChoose a fighter:", len(rows))
        chosen = rows[idx - 1][1]
        if chosen == "__retry__":
            redo = input("  Type name again (or empty to abort): ").strip()
            if not redo:
                sys.exit(1)
            return _interactive_pick_fighter(label, redo, profiles)
        fid = chosen
        print(f"  {label}: {_profile_display_name(fid, profiles)}", flush=True)
        return fid

    print(
        "  No similar names above the similarity cutoff. "
        "Check spelling or use fighter_id with:  python main.py predict …",
        file=sys.stderr,
    )
    sys.exit(1)


def _fights_between(
    fights: List[FightRecord],
    fid_a: str,
    fid_b: str,
) -> List[FightRecord]:
    out: List[FightRecord] = []
    for f in fights:
        if {f.fighter_a_id, f.fighter_b_id} == {fid_a, fid_b}:
            out.append(f)
    out.sort(key=lambda ff: ff.fight_date, reverse=True)
    return out


def _fight_one_line(
    f: FightRecord,
    profiles: Dict[str, FighterProfile],
) -> str:
    wc = f.weight_class.value
    prom = (getattr(f, "promotion", None) or "")[:28]
    d = f.fight_date.isoformat()
    tl = getattr(f.tier, "name", str(f.tier))
    na = _profile_display_name(f.fighter_a_id, profiles)
    nb = _profile_display_name(f.fighter_b_id, profiles)
    return f"{d}  {wc:<18}  {prom}  tier={tl}  |  {f.fight_id}\n        {na}  vs  {nb}"


def register_predict_human_arguments(p: ArgumentParser) -> None:
    p.add_argument(
        "name_corner_a",
        nargs="?",
        default=None,
        metavar="NAME_A",
        help="First fighter display name (corner A in model output). Omit to be prompted.",
    )
    p.add_argument(
        "name_corner_b",
        nargs="?",
        default=None,
        metavar="NAME_B",
        help="Second fighter (corner B). Omit to be prompted.",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Print log-odds decomposition instead of full predict table.",
    )
    p.add_argument(
        "--model-path",
        dest="predict_human_model_path",
        default=None,
        metavar="PATH",
        help="Model pickle (uses top-level --model-path if omitted).",
    )
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="With no matching fight in DB: required hypothetical fight date. "
        "Ignored when a single fight row is auto-selected unless --force-context.",
    )
    p.add_argument(
        "--weight-class",
        default=None,
        dest="weight_class_raw",
        metavar="CLASS",
        help="With no matching fight in DB: weight class for hypothetical matchup.",
    )
    p.add_argument(
        "--force-context",
        action="store_true",
        help="Use --date and --weight-class even when database fights exist (hypothetical).",
    )


def cmd_predict_human(args: Namespace) -> None:
    from ..pipeline import MMAPredictor

    model_path = Path(
        getattr(args, "predict_human_model_path", None)
        or getattr(args, "model_path", None)
        or "model.pkl"
    )
    if not model_path.exists():
        print(f"No trained model found at {model_path}.", flush=True)
        sys.exit(1)

    predictor: MMAPredictor = MMAPredictor.load(model_path)
    profiles = predictor.profiles
    fights = predictor.fights

    if not profiles:
        print(
            "This model has no fighter_profiles (empty training data). "
            "Add data/fighter_profiles.csv and re-train.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nModel: {model_path.resolve()}\n")
    print("Corner A = first name (fighter_a in predict); corner B = second name.\n")

    fid_a = _interactive_pick_fighter("Corner A", args.name_corner_a, profiles)
    fid_b = _interactive_pick_fighter("Corner B", args.name_corner_b, profiles)
    if fid_a == fid_b:
        print("Corners must be two different fighters.", file=sys.stderr)
        sys.exit(1)

    pair = _fights_between(fights, fid_a, fid_b)
    wc: WeightClass
    fdate: date

    force = getattr(args, "force_context", False)
    has_cli_ctx = bool(args.date and args.weight_class_raw)

    if pair and not (force and has_cli_ctx):
        if len(pair) == 1:
            row = pair[0]
            wc, fdate = row.weight_class, row.fight_date
            print(f"\n  Using sole database fight: {row.fight_id}  ({fdate})  {wc.value}\n")
        else:
            print(
                f"\n  Found {len(pair)} fights between these fighters in loaded data. Pick one:\n",
                flush=True,
            )
            lines = [_fight_one_line(f, profiles) for f in pair]
            idx = _enumerate_pick(lines)
            row = pair[idx]
            wc, fdate = row.weight_class, row.fight_date
            print(f"\n  Selected fight_id={row.fight_id}\n")
    else:
        if not pair:
            print("\n  No fight between this pair in loaded data - hypothetical matchup.\n")
        else:
            print("\n  --force-context: ignoring database fights - hypothetical.\n")

        if args.weight_class_raw and args.date:
            wc = resolve_weight_class(args.weight_class_raw)
            fdate = resolve_date(args.date)
        else:
            print("  Enter weight class and date for the hypothetical fight.")
            raw_wc = args.weight_class_raw or input("  Weight class: ").strip()
            if not raw_wc:
                print("Weight class required.", file=sys.stderr)
                sys.exit(1)
            wc = resolve_weight_class(raw_wc)
            raw_d = args.date or input("  Date YYYY-MM-DD: ").strip()
            if not raw_d:
                print("Date required.", file=sys.stderr)
                sys.exit(1)
            fdate = resolve_date(raw_d)

    print(f"\n{'='*60}")
    print(
        f"Predict: {_profile_display_name(fid_a, profiles)}  vs  "
        f"{_profile_display_name(fid_b, profiles)}\n          {wc.value}  |  {fdate}\n{'='*60}\n",
        flush=True,
    )

    if args.explain:
        predictor.explain(fid_a, fid_b, wc, fdate)
    else:
        result = predictor.predict(fid_a, fid_b, wc, fdate, verbose=True)
        print(f"\nDerived:")
        print(f"  Total win %    {result.total_win:.2f}")
        print(f"  Finish win %   {result.finish_win:.2f}")
        print(f"  Finish lose %  {result.finish_lose:.2f}")
        print(f"  Decision %     {result.go_to_decision:.2f}\n")
