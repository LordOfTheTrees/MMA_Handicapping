"""
Post-hoc walk-forward +EV book: OOS model P vs posted lines.

Odds never enter training, ``Config`` search, or ``score_tier1_fight_slice``.
Uses production ``Config()`` (frozen 2022 Phase-3 winner) and refits **weights W**
only — no ``--selection-search`` / random walks.

Stake rules (fixed in advance): among listed contracts on a fight, stake the
single +EV market with the largest edge ``e = P*d - 1``. Three parallel books:
full Kelly, half Kelly, and 1 unit flat. Two-way (A vs B) and six-class method
props are **separate** books (no parlays).

Local only — not a GitHub Action; do not write ``JSON_exports/`` (mma.ai sync
globs every ``*.json`` there).

From repo root::

    python -m src.eval.market_book --data-dir ./data --out-dir ./data/market_eval \\
        --start-year 2013 --end-year 2025

Sidecars (gitignored, not redistributed)::

    data/Public datasets/Kaggle/jurek betting odds/UFC_betting_odds.csv
    data/Public datasets/Kaggle/mdabbert ultimate/ufc-master.csv

Outputs ``market_book.json``, ``market_book_yoy.png``, and ``market_book_slices.png``
under ``--out-dir``. Slices (one-way only, not crossed): card slot, gender, weight
class. Card slots use mdabbert billed order with a fixed 5-fight main card.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..data.espn_normalize import normalize_fighter_name
from ..data.loader import load_fighter_profiles, load_ufcstats_fights
from ..data.schema import FightRecord, FighterProfile, WeightClass
from ..model.regression import encode_outcome
from .fight_scoring import filter_tier1_fights_in_calendar_year
from .tuning_harness import fit_predictor_for_train_before, train_before_for_eval_year

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_EXPORTS_DIR = (REPO_ROOT / "JSON_exports").resolve()

_WIN = frozenset({0, 1, 2})
_KELLY_F_MAX = 1.0 - 1e-12
_START_UNITS_1U = 100.0
_DEFAULT_JUREK = Path("Public datasets") / "Kaggle" / "jurek betting odds" / "UFC_betting_odds.csv"
_DEFAULT_MDABBERT = Path("Public datasets") / "Kaggle" / "mdabbert ultimate" / "ufc-master.csv"

# Pre-registered card layout (not searched). mdabbert lists fights billed/main-first.
# Index 0 = billed headliner; first MAIN_CARD_SIZE rows = main card; index MAIN_CARD_SIZE
# = featured prelim (prelim main event); later rows = generic prelims (ESPN + early).
# Dates with DOUBLEHEADER_MIN+ fights skip card slots (two events mashed by date).
MAIN_CARD_SIZE = 5
DOUBLEHEADER_MIN = 16
CARD_SLOTS: Tuple[str, ...] = (
    "title",
    "main_event",
    "main_card",
    "prelim_main_event",
    "generic_prelims",
)
GENDERS: Tuple[str, ...] = ("men", "women", "other")
_WOMEN_WC = frozenset(
    {
        WeightClass.W_STRAWWEIGHT,
        WeightClass.W_FLYWEIGHT,
        WeightClass.W_BANTAMWEIGHT,
        WeightClass.W_FEATHERWEIGHT,
    }
)
_MEN_WC = frozenset(
    {
        WeightClass.STRAWWEIGHT,
        WeightClass.FLYWEIGHT,
        WeightClass.BANTAMWEIGHT,
        WeightClass.FEATHERWEIGHT,
        WeightClass.LIGHTWEIGHT,
        WeightClass.WELTERWEIGHT,
        WeightClass.MIDDLEWEIGHT,
        WeightClass.LIGHT_HEAVYWEIGHT,
        WeightClass.HEAVYWEIGHT,
    }
)


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def american_to_decimal(american: float) -> Optional[float]:
    """American odds → decimal. ``None`` if non-finite or zero."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or a == 0.0:
        return None
    if a > 0.0:
        d = a / 100.0 + 1.0
    else:
        d = 100.0 / abs(a) + 1.0
    return d if d > 1.0 else None


def parse_decimal(raw: Any) -> Optional[float]:
    """Parse a posted decimal (already decimal, not American)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        d = float(s)
    except ValueError:
        return None
    if not math.isfinite(d) or d <= 1.0:
        return None
    return d


def parse_american(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return american_to_decimal(float(s))
    except ValueError:
        return None


def median_float(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def edge(p: float, decimal_odds: float) -> float:
    return float(p) * float(decimal_odds) - 1.0


def kelly_fraction(e: float, decimal_odds: float) -> float:
    """Full Kelly ``e / (d-1)`` clipped to ``(0, 1)``."""
    b = float(decimal_odds) - 1.0
    if b <= 0.0 or e <= 0.0:
        return 0.0
    f = e / b
    if f <= 0.0:
        return 0.0
    return float(min(f, _KELLY_F_MAX))


def implied_log_growth(p: float, f: float, decimal_odds: float) -> float:
    """Ex-ante log-wealth growth if model ``p`` is true and we stake ``f``."""
    if f <= 0.0:
        return 0.0
    b = decimal_odds - 1.0
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    f = min(max(float(f), 0.0), _KELLY_F_MAX)
    return p * math.log(1.0 + f * b) + (1.0 - p) * math.log(1.0 - f)


def realized_multiplier(hit: bool, f: float, decimal_odds: float) -> float:
    if f <= 0.0:
        return 1.0
    if hit:
        return 1.0 + f * (decimal_odds - 1.0)
    return 1.0 - f


def realized_flat_pnl(hit: bool, decimal_odds: float) -> float:
    return (decimal_odds - 1.0) if hit else -1.0


def hex_id_from_url(url: str) -> Optional[str]:
    u = (url or "").strip().rstrip("/")
    if not u:
        return None
    tail = u.rsplit("/", 1)[-1].strip().lower()
    return tail or None


def swap_method_decimals_to_a(
    d_f1: Tuple[Optional[float], Optional[float], Optional[float]],
    d_f2: Tuple[Optional[float], Optional[float], Optional[float]],
    f1_is_a: bool,
) -> Tuple[Optional[float], ...]:
    """
    Map (f1_ko, f1_sub, f1_dec) / (f2_*) onto classes 0–5 from fighter A's view.
    Classes: 0 A-KO, 1 A-sub, 2 A-dec, 3 B-dec, 4 B-KO, 5 B-sub.
    """
    if f1_is_a:
        a_ko, a_sub, a_dec = d_f1
        b_ko, b_sub, b_dec = d_f2
    else:
        a_ko, a_sub, a_dec = d_f2
        b_ko, b_sub, b_dec = d_f1
    return (a_ko, a_sub, a_dec, b_dec, b_ko, b_sub)


def assert_out_dir_allowed(out_dir: Path) -> Path:
    """Refuse paths that would land under ``JSON_exports/`` (mma.ai sync glob)."""
    resolved = Path(out_dir).expanduser().resolve()
    json_exp = JSON_EXPORTS_DIR
    if resolved == json_exp or json_exp in resolved.parents or resolved.name == "JSON_exports":
        raise SystemExit(
            f"refusing --out-dir {resolved}: JSON_exports is copied to mma.ai. "
            "Use ./data/market_eval (gitignored)."
        )
    return resolved


# ---------------------------------------------------------------------------
# Posted lines join
# ---------------------------------------------------------------------------


@dataclass
class PostedLines:
    source: str
    d_a: Optional[float] = None
    d_b: Optional[float] = None
    method: Tuple[Optional[float], ...] = (None, None, None, None, None, None)

    def has_two_way(self) -> bool:
        return self.d_a is not None and self.d_b is not None

    def has_method(self) -> bool:
        return all(x is not None for x in self.method)


def _parse_iso_date(raw: str) -> Optional[date]:
    s = (raw or "").strip()
    if not s:
        return None
    s = s[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def load_jurek_by_fight_id(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Group jurek rows by UFCStats ``fight_id`` (from ``fight_url``).
    Median decimal per column (``adding_date`` is a bulk-load stamp, not a line time).
    """
    buckets: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {
            "f1_id": [],
            "f2_id": [],
            "odds_1": [],
            "odds_2": [],
            "f1_ko": [],
            "f1_sub": [],
            "f1_dec": [],
            "f2_ko": [],
            "f2_sub": [],
            "f2_dec": [],
        }
    )
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            fid = hex_id_from_url(row.get("fight_url") or "")
            if not fid:
                continue
            ev = _parse_iso_date(row.get("event_date") or "")
            if ev is not None and ev.year >= 2027:
                continue
            b = buckets[fid]
            a1 = hex_id_from_url(row.get("fighter_1_url") or "")
            a2 = hex_id_from_url(row.get("fighter_2_url") or "")
            if a1:
                b["f1_id"].append(a1)
            if a2:
                b["f2_id"].append(a2)
            for key, col in (
                ("odds_1", "odds_1"),
                ("odds_2", "odds_2"),
                ("f1_ko", "f1_ko_odds"),
                ("f1_sub", "f1_sub_odds"),
                ("f1_dec", "f1_dec_odds"),
                ("f2_ko", "f2_ko_odds"),
                ("f2_sub", "f2_sub_odds"),
                ("f2_dec", "f2_dec_odds"),
            ):
                d = parse_decimal(row.get(col))
                if d is not None:
                    b[key].append(d)
    out: Dict[str, Dict[str, Any]] = {}
    for fid, b in buckets.items():
        f1 = b["f1_id"][0] if b["f1_id"] else None
        f2 = b["f2_id"][0] if b["f2_id"] else None
        out[fid] = {
            "f1_id": f1,
            "f2_id": f2,
            "odds_1": median_float(b["odds_1"]),
            "odds_2": median_float(b["odds_2"]),
            "f1_ko": median_float(b["f1_ko"]),
            "f1_sub": median_float(b["f1_sub"]),
            "f1_dec": median_float(b["f1_dec"]),
            "f2_ko": median_float(b["f2_ko"]),
            "f2_sub": median_float(b["f2_sub"]),
            "f2_dec": median_float(b["f2_dec"]),
        }
    return out


def jurek_row_to_posted(row: Mapping[str, Any], fight: FightRecord) -> Optional[PostedLines]:
    f1 = row.get("f1_id")
    f2 = row.get("f2_id")
    a, b = fight.fighter_a_id, fight.fighter_b_id
    if f1 == a and f2 == b:
        f1_is_a = True
    elif f1 == b and f2 == a:
        f1_is_a = False
    else:
        return None
    d1, d2 = row.get("odds_1"), row.get("odds_2")
    if f1_is_a:
        d_a, d_b = d1, d2
    else:
        d_a, d_b = d2, d1
    method = swap_method_decimals_to_a(
        (row.get("f1_ko"), row.get("f1_sub"), row.get("f1_dec")),
        (row.get("f2_ko"), row.get("f2_sub"), row.get("f2_dec")),
        f1_is_a,
    )
    return PostedLines(source="jurek", d_a=d_a, d_b=d_b, method=method)


def load_mdabbert_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            dt = _parse_iso_date(row.get("date") or "")
            nr = normalize_fighter_name(row.get("R_fighter") or "")
            nb = normalize_fighter_name(row.get("B_fighter") or "")
            if dt is None or not nr or not nb or nr == nb:
                continue
            rows.append(
                {
                    "date": dt,
                    "nr": nr,
                    "nb": nb,
                    "d_r": parse_american(row.get("R_odds")),
                    "d_b": parse_american(row.get("B_odds")),
                    "r_ko": parse_american(row.get("r_ko_odds")),
                    "r_sub": parse_american(row.get("r_sub_odds")),
                    "r_dec": parse_american(row.get("r_dec_odds")),
                    "b_ko": parse_american(row.get("b_ko_odds")),
                    "b_sub": parse_american(row.get("b_sub_odds")),
                    "b_dec": parse_american(row.get("b_dec_odds")),
                    "title_bout": _truthy(row.get("title_bout")),
                }
            )
    return rows


def mdabbert_to_posted(row: Mapping[str, Any], fight: FightRecord, na: str, nb: str) -> Optional[PostedLines]:
    nr, nblue = row["nr"], row["nb"]
    if nr == na and nblue == nb:
        r_is_a = True
    elif nr == nb and nblue == na:
        r_is_a = False
    else:
        return None
    d_r, d_blu = row.get("d_r"), row.get("d_b")
    if r_is_a:
        d_a, d_b = d_r, d_blu
    else:
        d_a, d_b = d_blu, d_r
    method = swap_method_decimals_to_a(
        (row.get("r_ko"), row.get("r_sub"), row.get("r_dec")),
        (row.get("b_ko"), row.get("b_sub"), row.get("b_dec")),
        r_is_a,
    )
    return PostedLines(source="mdabbert", d_a=d_a, d_b=d_b, method=method)


def join_posted_lines(
    fights: Sequence[FightRecord],
    profiles: Mapping[str, FighterProfile],
    jurek_path: Path,
    mdabbert_path: Path,
) -> Tuple[Dict[str, PostedLines], Dict[str, int]]:
    """
    One source per fight: jurek ``fight_id`` first, mdabbert name+date fill.
    Do not blend columns across sources.
    """
    jurek = load_jurek_by_fight_id(jurek_path) if jurek_path.is_file() else {}
    md_rows = load_mdabbert_rows(mdabbert_path) if mdabbert_path.is_file() else []

    name_of: Dict[str, str] = {}
    for fid, prof in profiles.items():
        name_of[fid] = normalize_fighter_name(prof.name)

    by_name_date: Dict[Tuple[date, frozenset], List[FightRecord]] = defaultdict(list)
    for fight in fights:
        na = name_of.get(fight.fighter_a_id, "")
        nb = name_of.get(fight.fighter_b_id, "")
        if not na or not nb or na == nb:
            continue
        by_name_date[(fight.fight_date, frozenset({na, nb}))].append(fight)

    md_by_key: Dict[Tuple[date, frozenset], List[Dict[str, Any]]] = defaultdict(list)
    for row in md_rows:
        md_by_key[(row["date"], frozenset({row["nr"], row["nb"]}))].append(row)

    posted: Dict[str, PostedLines] = {}
    n_jurek = 0
    n_md = 0
    n_two_way = 0
    n_method = 0

    for fight in fights:
        pl: Optional[PostedLines] = None
        raw = jurek.get(fight.fight_id)
        if raw is not None:
            pl = jurek_row_to_posted(raw, fight)
            if pl is not None:
                n_jurek += 1
        if pl is None:
            na = name_of.get(fight.fighter_a_id, "")
            nb = name_of.get(fight.fighter_b_id, "")
            key = (fight.fight_date, frozenset({na, nb}))
            candidates = md_by_key.get(key, [])
            fights_here = by_name_date.get(key, [])
            if len(candidates) == 1 and len(fights_here) == 1 and fights_here[0].fight_id == fight.fight_id:
                pl = mdabbert_to_posted(candidates[0], fight, na, nb)
                if pl is not None:
                    n_md += 1
        if pl is None:
            continue
        posted[fight.fight_id] = pl
        if pl.has_two_way():
            n_two_way += 1
        if pl.has_method():
            n_method += 1

    stats = {
        "n_jurek": n_jurek,
        "n_mdabbert_fill": n_md,
        "n_joined": len(posted),
        "n_two_way": n_two_way,
        "n_method": n_method,
    }
    return posted, stats


def _truthy(raw: Any) -> bool:
    s = str(raw or "").strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def gender_of_weight_class(wc: WeightClass) -> str:
    if wc in _WOMEN_WC:
        return "women"
    if wc in _MEN_WC:
        return "men"
    return "other"


def card_slots_from_billed_index(
    billed_index: Optional[int],
    n_on_card: int,
    *,
    is_title: bool,
) -> Tuple[str, ...]:
    """
    Overlapping labels. ``title`` is independent of billed index.
    Returns () card-position tags when index is missing or the date looks like a doubleheader.
    """
    slots: List[str] = []
    if is_title:
        slots.append("title")
    if billed_index is None or n_on_card >= DOUBLEHEADER_MIN or billed_index < 0:
        return tuple(slots)
    if billed_index == 0:
        slots.append("main_event")
    if billed_index < MAIN_CARD_SIZE:
        slots.append("main_card")
    elif billed_index == MAIN_CARD_SIZE:
        slots.append("prelim_main_event")
    else:
        slots.append("generic_prelims")
    return tuple(slots)


@dataclass(frozen=True)
class FightSliceTags:
    gender: str
    weight_class: str
    card: Tuple[str, ...]


def assign_slice_tags(
    fights: Sequence[FightRecord],
    profiles: Mapping[str, FighterProfile],
    md_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, FightSliceTags]:
    """Gender/weight from UFCStats records; card slots from mdabbert billed order + title flag."""
    name_of: Dict[str, str] = {
        fid: normalize_fighter_name(prof.name) for fid, prof in profiles.items()
    }
    by_name_date: Dict[Tuple[date, frozenset], List[FightRecord]] = defaultdict(list)
    for fight in fights:
        na = name_of.get(fight.fighter_a_id, "")
        nb = name_of.get(fight.fighter_b_id, "")
        if not na or not nb or na == nb:
            continue
        by_name_date[(fight.fight_date, frozenset({na, nb}))].append(fight)

    billed: Dict[str, Tuple[int, int, bool]] = {}  # fight_id -> (index, n, title)
    by_date_rows: Dict[date, List[Mapping[str, Any]]] = defaultdict(list)
    for row in md_rows:
        by_date_rows[row["date"]].append(row)
    for dt, rows in by_date_rows.items():
        n = len(rows)
        for i, row in enumerate(rows):
            key = (dt, frozenset({row["nr"], row["nb"]}))
            matched = by_name_date.get(key, [])
            if len(matched) != 1:
                continue
            billed[matched[0].fight_id] = (i, n, bool(row.get("title_bout")))

    out: Dict[str, FightSliceTags] = {}
    for fight in fights:
        raw = (fight.weight_class_raw or "").lower()
        title_raw = "title" in raw
        idx_n_t = billed.get(fight.fight_id)
        if idx_n_t is not None:
            idx, n, title_md = idx_n_t
            is_title = title_md or title_raw
        else:
            idx, n, is_title = None, 0, title_raw
        out[fight.fight_id] = FightSliceTags(
            gender=gender_of_weight_class(fight.weight_class),
            weight_class=fight.weight_class.value,
            card=card_slots_from_billed_index(idx, n, is_title=is_title),
        )
    return out


# ---------------------------------------------------------------------------
# Per-fight book + yearly rollup
# ---------------------------------------------------------------------------


@dataclass
class StakePick:
    contract: str
    p: float
    decimal_odds: float
    e: float
    hit: bool


def pick_max_edge(
    probs_and_odds: Sequence[Tuple[str, float, float]],
    hit_of: Mapping[str, bool],
) -> Optional[StakePick]:
    """Among listed contracts, the +EV market with largest ``e``. None if all e<=0."""
    best: Optional[StakePick] = None
    for name, p, d in probs_and_odds:
        if d is None or d <= 1.0 or p <= 0.0:
            continue
        e = edge(p, d)
        if e <= 0.0:
            continue
        cand = StakePick(contract=name, p=float(p), decimal_odds=float(d), e=float(e), hit=bool(hit_of[name]))
        if best is None or cand.e > best.e:
            best = cand
    return best


def fill_two_way_hits(y: int) -> Dict[str, bool]:
    a_won = y in _WIN
    return {"A": a_won, "B": not a_won}


def method_candidates(p6: np.ndarray, lines: PostedLines) -> List[Tuple[str, float, float]]:
    if not lines.has_method():
        return []
    out: List[Tuple[str, float, float]] = []
    labels = ("A_ko", "A_sub", "A_dec", "B_dec", "B_ko", "B_sub")
    for i, lab in enumerate(labels):
        d = lines.method[i]
        if d is None:
            continue
        out.append((lab, float(p6[i]), float(d)))
    return out


def fill_method_hits(y: int) -> Dict[str, bool]:
    labels = ("A_ko", "A_sub", "A_dec", "B_dec", "B_ko", "B_sub")
    return {lab: (i == y) for i, lab in enumerate(labels)}


def _kelly_path(picks: Sequence[StakePick], fraction_scale: float) -> Dict[str, Any]:
    """fraction_scale 1.0 = full Kelly, 0.5 = half. Wealth starts at 1."""
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    ruin_zero = False
    ruin_50 = False
    log_proj = 0.0
    log_real = 0.0
    gs: List[float] = []
    log_rets: List[float] = []
    fs: List[float] = []
    for pk in picks:
        f_star = kelly_fraction(pk.e, pk.decimal_odds)
        f = min(f_star * fraction_scale, _KELLY_F_MAX)
        fs.append(f)
        g = implied_log_growth(pk.p, f, pk.decimal_odds)
        gs.append(g)
        log_proj += g
        m = realized_multiplier(pk.hit, f, pk.decimal_odds)
        m = max(m, 1e-15)
        log_r = math.log(m)
        log_real += log_r
        log_rets.append(log_r)
        wealth *= m
        if wealth > peak:
            peak = wealth
        dd = 1.0 - wealth / peak if peak > 0 else 1.0
        if dd > max_dd:
            max_dd = dd
        if wealth <= 1e-12:
            ruin_zero = True
        if dd >= 0.5:
            ruin_50 = True
    mu = float(np.mean(log_rets)) if log_rets else 0.0
    var = float(np.var(log_rets)) if len(log_rets) > 1 else 0.0
    if not log_rets:
        approx = float("nan")
    elif var <= 0.0:
        approx = 0.0 if mu > 0.0 else 1.0
    elif mu <= 0.0:
        approx = 1.0
    else:
        approx = float(min(1.0, math.exp(-2.0 * mu * math.log(2.0) / var)))
    return {
        "n_bets": len(picks),
        "mean_f": float(np.mean(fs)) if fs else float("nan"),
        "projected_log_growth": log_proj,
        "realized_log_growth": log_real,
        "projected_wealth": math.exp(log_proj) if picks else 1.0,
        "realized_wealth": wealth if picks else 1.0,
        "max_drawdown": max_dd,
        "ruin_zero": ruin_zero,
        "ruin_50pct_drawdown": ruin_50,
        "approx_ruin_50pct_brownian": approx,
    }


def _flat_1u_path(picks: Sequence[StakePick]) -> Dict[str, Any]:
    bank = _START_UNITS_1U
    peak = bank
    max_dd_units = 0.0
    projected = 0.0
    realized = 0.0
    bust = False
    down_50 = False
    for pk in picks:
        projected += pk.e
        pnl = realized_flat_pnl(pk.hit, pk.decimal_odds)
        realized += pnl
        bank += pnl
        if bank > peak:
            peak = bank
        dd = peak - bank
        if dd > max_dd_units:
            max_dd_units = dd
        if bank <= 0.0:
            bust = True
        if bank <= _START_UNITS_1U - 50.0:
            down_50 = True
    n = len(picks)
    roi_real = realized / _START_UNITS_1U
    roi_proj = projected / _START_UNITS_1U
    return {
        "n_bets": n,
        "start_units": _START_UNITS_1U,
        "projected_profit_units": projected,
        "realized_profit_units": realized,
        "terminal_units": bank,
        "projected_roi": roi_proj,
        "realized_roi": roi_real,
        "max_drawdown_units": max_dd_units,
        "bust": bust,
        "down_50_units": down_50,
    }


def rollup_picks(picks: Sequence[StakePick], n_priced: int) -> Dict[str, Any]:
    n_plus = len(picks)
    mean_e = float(np.mean([pk.e for pk in picks])) if picks else float("nan")
    hits = sum(1 for pk in picks if pk.hit)
    hit_rate = (hits / n_plus) if n_plus else float("nan")
    mean_implied = float(np.mean([1.0 / pk.decimal_odds for pk in picks])) if picks else float("nan")
    return {
        "n_priced": n_priced,
        "n_plus_ev": n_plus,
        "coverage": (n_plus / n_priced) if n_priced else float("nan"),
        "mean_edge": mean_e,
        "hit_rate": hit_rate,
        "mean_posted_implied": mean_implied,
        "full_kelly": _kelly_path(picks, 1.0),
        "half_kelly": _kelly_path(picks, 0.5),
        "flat_1u": _flat_1u_path(picks),
    }


class _Bucket:
    __slots__ = ("picks", "n_priced")

    def __init__(self) -> None:
        self.picks: List[StakePick] = []
        self.n_priced = 0

    def add(self, pick: Optional[StakePick], priced: bool) -> None:
        if priced:
            self.n_priced += 1
        if pick is not None:
            self.picks.append(pick)


def _extend_bucket(dst: _Bucket, src: _Bucket) -> None:
    dst.picks.extend(src.picks)
    dst.n_priced += src.n_priced


class BookAccum:
    """One-way slice buckets (not crossed). Overall + card + gender + weight class."""

    def __init__(self) -> None:
        self.two_way = _Bucket()
        self.method = _Bucket()
        self.tw_card: Dict[str, _Bucket] = {s: _Bucket() for s in CARD_SLOTS}
        self.mh_card: Dict[str, _Bucket] = {s: _Bucket() for s in CARD_SLOTS}
        self.tw_gender: Dict[str, _Bucket] = {g: _Bucket() for g in GENDERS}
        self.mh_gender: Dict[str, _Bucket] = {g: _Bucket() for g in GENDERS}
        self.tw_wc: Dict[str, _Bucket] = defaultdict(_Bucket)
        self.mh_wc: Dict[str, _Bucket] = defaultdict(_Bucket)

    def add_fight(
        self,
        tags: FightSliceTags,
        tw_pick: Optional[StakePick],
        tw_priced: bool,
        meth_pick: Optional[StakePick],
        meth_priced: bool,
    ) -> None:
        self.two_way.add(tw_pick, tw_priced)
        self.method.add(meth_pick, meth_priced)
        self.tw_gender[tags.gender].add(tw_pick, tw_priced)
        self.mh_gender[tags.gender].add(meth_pick, meth_priced)
        self.tw_wc[tags.weight_class].add(tw_pick, tw_priced)
        self.mh_wc[tags.weight_class].add(meth_pick, meth_priced)
        for slot in tags.card:
            self.tw_card[slot].add(tw_pick, tw_priced)
            self.mh_card[slot].add(meth_pick, meth_priced)

    def merge(self, other: "BookAccum") -> None:
        _extend_bucket(self.two_way, other.two_way)
        _extend_bucket(self.method, other.method)
        for s in CARD_SLOTS:
            _extend_bucket(self.tw_card[s], other.tw_card[s])
            _extend_bucket(self.mh_card[s], other.mh_card[s])
        for g in GENDERS:
            _extend_bucket(self.tw_gender[g], other.tw_gender[g])
            _extend_bucket(self.mh_gender[g], other.mh_gender[g])
        for k, b in other.tw_wc.items():
            _extend_bucket(self.tw_wc[k], b)
        for k, b in other.mh_wc.items():
            _extend_bucket(self.mh_wc[k], b)

    def as_report(self) -> Dict[str, Any]:
        return {
            "two_way": rollup_picks(self.two_way.picks, self.two_way.n_priced),
            "method": rollup_picks(self.method.picks, self.method.n_priced),
            "by_card": {
                s: {
                    "two_way": rollup_picks(self.tw_card[s].picks, self.tw_card[s].n_priced),
                    "method": rollup_picks(self.mh_card[s].picks, self.mh_card[s].n_priced),
                }
                for s in CARD_SLOTS
            },
            "by_gender": {
                g: {
                    "two_way": rollup_picks(self.tw_gender[g].picks, self.tw_gender[g].n_priced),
                    "method": rollup_picks(self.mh_gender[g].picks, self.mh_gender[g].n_priced),
                }
                for g in GENDERS
            },
            "by_weight_class": {
                wc: {
                    "two_way": rollup_picks(self.tw_wc[wc].picks, self.tw_wc[wc].n_priced),
                    "method": rollup_picks(self.mh_wc[wc].picks, self.mh_wc[wc].n_priced),
                }
                for wc in sorted(set(self.tw_wc) | set(self.mh_wc))
            },
        }


_DEFAULT_TAGS = FightSliceTags(gender="other", weight_class="unknown", card=())


def book_year(
    fights: Sequence[FightRecord],
    posted: Mapping[str, PostedLines],
    predict_p6,
    tags: Optional[Mapping[str, FightSliceTags]] = None,
) -> BookAccum:
    acc = BookAccum()
    tag_map = tags or {}
    for fight in fights:
        y = encode_outcome(fight, fight.fighter_a_id)
        if y is None:
            continue
        lines = posted.get(fight.fight_id)
        if lines is None:
            continue
        p6 = np.asarray(
            predict_p6(fight.fighter_a_id, fight.fighter_b_id, fight.weight_class, fight.fight_date),
            dtype=float,
        )
        tw_pick: Optional[StakePick] = None
        tw_priced = False
        meth_pick: Optional[StakePick] = None
        meth_priced = False
        if lines.has_two_way() and lines.d_a is not None and lines.d_b is not None:
            tw_priced = True
            cands = [
                ("A", float(p6[0] + p6[1] + p6[2]), float(lines.d_a)),
                ("B", float(p6[3] + p6[4] + p6[5]), float(lines.d_b)),
            ]
            tw_pick = pick_max_edge(cands, fill_two_way_hits(y))
        if lines.has_method():
            meth_priced = True
            meth_pick = pick_max_edge(method_candidates(p6, lines), fill_method_hits(y))
        acc.add_fight(
            tag_map.get(fight.fight_id, _DEFAULT_TAGS),
            tw_pick,
            tw_priced,
            meth_pick,
            meth_priced,
        )
    return acc


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def default_sidecar(data_dir: Path, rel: Path) -> Path:
    return (Path(data_dir) / rel).resolve()


def run_market_book(
    data_dir: Path,
    out_dir: Path,
    *,
    start_year: int = 2013,
    end_year: int = 2025,
    jurek_path: Optional[Path] = None,
    mdabbert_path: Optional[Path] = None,
    elo_cache: Optional[Path] = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    out_dir = assert_out_dir_allowed(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jurek_path = Path(jurek_path).resolve() if jurek_path else default_sidecar(data_dir, _DEFAULT_JUREK)
    mdabbert_path = Path(mdabbert_path).resolve() if mdabbert_path else default_sidecar(data_dir, _DEFAULT_MDABBERT)
    if not jurek_path.is_file() and not mdabbert_path.is_file():
        raise FileNotFoundError(
            f"Need at least one sidecar CSV. Missing:\n  {jurek_path}\n  {mdabbert_path}"
        )
    elo_cache = Path(elo_cache).resolve() if elo_cache else (out_dir / "elo_walkforward_cache.pkl")

    fights_csv = data_dir / "ufcstats_fights.csv"
    profiles_csv = data_dir / "fighter_profiles.csv"
    fights = load_ufcstats_fights(fights_csv)
    profiles = load_fighter_profiles(profiles_csv)
    posted, join_stats = join_posted_lines(fights, profiles, jurek_path, mdabbert_path)
    md_rows = load_mdabbert_rows(mdabbert_path) if mdabbert_path.is_file() else []
    tags = assign_slice_tags(fights, profiles, md_rows)
    print(
        f"[market_book] join  jurek={join_stats['n_jurek']}  "
        f"mdabbert_fill={join_stats['n_mdabbert_fill']}  "
        f"two_way={join_stats['n_two_way']}  method={join_stats['n_method']}",
        flush=True,
    )

    cfg = Config()
    years_out: Dict[str, Any] = {}
    pooled = BookAccum()
    for y in range(int(start_year), int(end_year) + 1):
        print(f"[market_book] fit year {y} (train_before={train_before_for_eval_year(y)}) ...", flush=True)
        pred = fit_predictor_for_train_before(
            cfg,
            data_dir,
            train_before_for_eval_year(y),
            skip_bootstrap=True,
            elo_cache_path=elo_cache,
        )
        year_fights = filter_tier1_fights_in_calendar_year(pred.fights, pred.config.master_start_year, y)
        year_acc = book_year(
            year_fights,
            posted,
            pred.predict_proba_point_only,
            tags,
        )
        pooled.merge(year_acc)
        years_out[str(y)] = year_acc.as_report()
        tw = years_out[str(y)]["two_way"]
        mh = years_out[str(y)]["method"]
        print(
            f"  two_way n_priced={tw['n_priced']} n_plus_ev={tw['n_plus_ev']}  "
            f"method n_priced={mh['n_priced']} n_plus_ev={mh['n_plus_ev']}",
            flush=True,
        )

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "config": "Config() frozen 2022 Phase-3 winner; skip_bootstrap=True",
        "stake_rules": ["full_kelly", "half_kelly", "flat_1u"],
        "slice_rules": {
            "card": (
                "mdabbert billed order (main-first); "
                f"main_card=first {MAIN_CARD_SIZE}; "
                f"prelim_main_event=index {MAIN_CARD_SIZE}; "
                "generic_prelims=later; title=mdabbert title_bout or weight_class_raw; "
                "overlapping labels; not crossed with gender/weight"
            ),
            "gender": "men/women from WeightClass; catch/unknown=other",
            "weight_class": "FightRecord.weight_class; one-way only",
        },
        "join": join_stats,
        "sidecars": {"jurek": str(jurek_path), "mdabbert": str(mdabbert_path)},
        "years": years_out,
        "slices_pooled": pooled.as_report(),
    }
    json_path = out_dir / "market_book.json"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[market_book] wrote {json_path}", flush=True)
    png_path = out_dir / "market_book_yoy.png"
    from .tuning_plots import plot_market_book_slices, plot_market_book_yoy

    plot_market_book_yoy(report, png_path)
    print(f"[market_book] wrote {png_path}", flush=True)
    slices_png = out_dir / "market_book_slices.png"
    plot_market_book_slices(report, slices_png)
    print(f"[market_book] wrote {slices_png}", flush=True)
    return report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Walk-forward +EV book on OOS Config() P vs posted two-way and method odds. "
            "Local only; odds never train. Stakes: full Kelly, half Kelly, 1 unit on every +EV pick."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV data directory")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "market_eval",
        help="JSON+PNG output (must not be JSON_exports/)",
    )
    p.add_argument("--start-year", type=int, default=2013)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--jurek", type=Path, default=None, help="Override jurek UFC_betting_odds.csv")
    p.add_argument("--mdabbert", type=Path, default=None, help="Override mdabbert ufc-master.csv")
    p.add_argument(
        "--elo-cache",
        type=Path,
        default=None,
        help="PIT ELO cache (default: <out-dir>/elo_walkforward_cache.pkl)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    run_market_book(
        args.data_dir,
        args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        jurek_path=args.jurek,
        mdabbert_path=args.mdabbert,
        elo_cache=args.elo_cache,
    )


if __name__ == "__main__":
    main()
