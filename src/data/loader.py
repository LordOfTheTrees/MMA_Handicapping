"""
Data loading and ingestion functions.

Each loader reads a specific CSV format and returns canonical schema objects.
The expected column names are documented per function. Rows with unrecognized
weight class or method strings are skipped with a warning rather than crashing.

Directory layout expected by MMAPredictor.load_data():
    data/
        ufcstats_fights.csv (or legacy tier1_ufcstats.csv)
        tier2_bellator.csv
        tier2_one.csv
        tier2_pfl.csv
        tier2_rizin.csv
        tier3_sherdog.csv
        fighter_profiles.csv
"""
import csv
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schema import (
    DataTier, FightRecord, FightStats, FighterProfile,
    ResultMethod, Stance, WeightClass,
)


# ---------------------------------------------------------------------------
# String normalisation maps
# ---------------------------------------------------------------------------

WEIGHT_CLASS_MAP: Dict[str, WeightClass] = {
    "strawweight": WeightClass.STRAWWEIGHT,
    "flyweight": WeightClass.FLYWEIGHT,
    "bantamweight": WeightClass.BANTAMWEIGHT,
    "featherweight": WeightClass.FEATHERWEIGHT,
    "lightweight": WeightClass.LIGHTWEIGHT,
    "welterweight": WeightClass.WELTERWEIGHT,
    "middleweight": WeightClass.MIDDLEWEIGHT,
    "light heavyweight": WeightClass.LIGHT_HEAVYWEIGHT,
    "light_heavyweight": WeightClass.LIGHT_HEAVYWEIGHT,
    "heavyweight": WeightClass.HEAVYWEIGHT,
    "women's strawweight": WeightClass.W_STRAWWEIGHT,
    "women's flyweight": WeightClass.W_FLYWEIGHT,
    "women's bantamweight": WeightClass.W_BANTAMWEIGHT,
    "women's featherweight": WeightClass.W_FEATHERWEIGHT,
    "w strawweight": WeightClass.W_STRAWWEIGHT,
    "w flyweight": WeightClass.W_FLYWEIGHT,
    "w bantamweight": WeightClass.W_BANTAMWEIGHT,
    "w featherweight": WeightClass.W_FEATHERWEIGHT,
    "catch_weight": WeightClass.CATCH_WEIGHT,
}

METHOD_MAP: Dict[str, ResultMethod] = {
    "ko": ResultMethod.KO_TKO,
    "tko": ResultMethod.KO_TKO,
    "ko/tko": ResultMethod.KO_TKO,
    "ko tko": ResultMethod.KO_TKO,
    "submission": ResultMethod.SUBMISSION,
    "sub": ResultMethod.SUBMISSION,
    "unanimous decision": ResultMethod.UNANIMOUS_DECISION,
    "unanimous": ResultMethod.UNANIMOUS_DECISION,
    "split decision": ResultMethod.SPLIT_DECISION,
    "split": ResultMethod.SPLIT_DECISION,
    "majority decision": ResultMethod.MAJORITY_DECISION,
    "majority": ResultMethod.MAJORITY_DECISION,
    "draw": ResultMethod.DRAW,
    "no contest": ResultMethod.NO_CONTEST,
    "nc": ResultMethod.NO_CONTEST,
    "dq": ResultMethod.DQ,
    "disqualification": ResultMethod.DQ,
}

STANCE_MAP: Dict[str, Stance] = {
    "orthodox": Stance.ORTHODOX,
    "southpaw": Stance.SOUTHPAW,
    "switch": Stance.SWITCH,
}


def _coerce_weight_class_from_cell(raw: str) -> tuple[Optional[WeightClass], Optional[str]]:
    """
    Map CSV ``weight_class`` to :class:`WeightClass`.
    Non-canonical labels (tournament-only wording, etc.) become ``UNKNOWN`` with the
    original cell preserved. The scraper writes ``catch_weight`` for catch-weight bouts.
    """
    cell = (raw or "").strip()
    if not cell:
        return None, None
    mapped = WEIGHT_CLASS_MAP.get(cell.lower())
    if mapped is not None:
        return mapped, None
    return WeightClass.UNKNOWN, cell


def _parse_method(raw: str) -> Optional[ResultMethod]:
    """
    Map CSV ``method`` to :class:`ResultMethod`.

    UFCStats long labels (e.g. ``TKO - Doctor's Stoppage``) normalize to ``KO_TKO``:
    treated as a finish for ``winner_id`` — damage inflicted by the winner led to stoppage.
    """
    if raw is None or not str(raw).strip():
        return None
    s = re.sub(r"\s+", " ", str(raw).strip().lower())
    if re.match(r"^(tko|ko)\b", s):
        return ResultMethod.KO_TKO
    return METHOD_MAP.get(s)


def _parse_date(raw: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _int_or_none(val) -> Optional[int]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _float_or_none(val) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tier 1: UFCStats
# ---------------------------------------------------------------------------

def load_ufcstats_fights(data_path: Path) -> List[FightRecord]:
    """
    Load Tier 1 UFC fight records with full per-fight stats.

    Required columns:
        fight_id, fighter_a_id, fighter_b_id, winner_id, method,
        weight_class, date

    Non-standard ``weight_class`` strings (e.g. catch weight, tournament titles)
    load as :attr:`WeightClass.UNKNOWN` with :attr:`FightRecord.weight_class_raw`
    set to the CSV cell.

    Optional stat columns (populated when present):
        a_sig_str_landed, a_sig_str_attempted, a_sig_str_absorbed,
        a_td_landed, a_td_attempted, a_ctrl_time_sec, a_sub_attempts,
        b_sig_str_landed, b_sig_str_attempted, b_sig_str_absorbed,
        b_td_landed, b_td_attempted, b_ctrl_time_sec, b_sub_attempts,
        fight_time_sec
    """
    records: List[FightRecord] = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = _parse_method(row.get("method", ""))
            weight_class, weight_class_raw = _coerce_weight_class_from_cell(
                row.get("weight_class", "")
            )
            fight_date = _parse_date(row.get("date", ""))

            if method is None or weight_class is None or fight_date is None:
                warnings.warn(
                    f"Skipping UFC fight {row.get('fight_id')}: "
                    f"unrecognised method={row.get('method')!r} "
                    f"or weight_class={row.get('weight_class')!r} "
                    f"or date={row.get('date')!r}"
                )
                continue

            fight_time = _int_or_none(row.get("fight_time_sec"))

            stats_a = FightStats(
                significant_strikes_landed=_int_or_none(row.get("a_sig_str_landed")),
                significant_strikes_attempted=_int_or_none(row.get("a_sig_str_attempted")),
                significant_strikes_absorbed=_int_or_none(row.get("a_sig_str_absorbed")),
                takedowns_landed=_int_or_none(row.get("a_td_landed")),
                takedowns_attempted=_int_or_none(row.get("a_td_attempted")),
                control_time_seconds=_int_or_none(row.get("a_ctrl_time_sec")),
                submission_attempts=_int_or_none(row.get("a_sub_attempts")),
                total_fight_time_seconds=fight_time,
            )
            stats_b = FightStats(
                significant_strikes_landed=_int_or_none(row.get("b_sig_str_landed")),
                significant_strikes_attempted=_int_or_none(row.get("b_sig_str_attempted")),
                significant_strikes_absorbed=_int_or_none(row.get("b_sig_str_absorbed")),
                takedowns_landed=_int_or_none(row.get("b_td_landed")),
                takedowns_attempted=_int_or_none(row.get("b_td_attempted")),
                control_time_seconds=_int_or_none(row.get("b_ctrl_time_sec")),
                submission_attempts=_int_or_none(row.get("b_sub_attempts")),
                total_fight_time_seconds=fight_time,
            )

            records.append(FightRecord(
                fight_id=row["fight_id"],
                fighter_a_id=row["fighter_a_id"],
                fighter_b_id=row["fighter_b_id"],
                winner_id=row.get("winner_id") or None,
                result_method=method,
                weight_class=weight_class,
                fight_date=fight_date,
                promotion="UFC",
                tier=DataTier.TIER_1,
                fighter_a_stats=stats_a,
                fighter_b_stats=stats_b,
                weight_class_raw=weight_class_raw,
            ))
    return records


# ---------------------------------------------------------------------------
# Tier 2: Major promotions
# ---------------------------------------------------------------------------

def load_major_promotion_fights(data_path: Path, promotion: str) -> List[FightRecord]:
    """
    Load Tier 2 fight records from a major promotion.

    Required columns:
        fight_id, fighter_a_id, fighter_b_id, winner_id, method,
        weight_class, date
    """
    records: List[FightRecord] = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = _parse_method(row.get("method", ""))
            weight_class, weight_class_raw = _coerce_weight_class_from_cell(
                row.get("weight_class", "")
            )
            fight_date = _parse_date(row.get("date", ""))

            if method is None or weight_class is None or fight_date is None:
                warnings.warn(
                    f"Skipping {promotion} fight {row.get('fight_id')}: "
                    f"unrecognised method, weight_class, or date"
                )
                continue

            records.append(FightRecord(
                fight_id=row["fight_id"],
                fighter_a_id=row["fighter_a_id"],
                fighter_b_id=row["fighter_b_id"],
                winner_id=row.get("winner_id") or None,
                result_method=method,
                weight_class=weight_class,
                fight_date=fight_date,
                promotion=promotion,
                tier=DataTier.TIER_2,
                weight_class_raw=weight_class_raw,
            ))
    return records


# ---------------------------------------------------------------------------
# Tier 3: Sherdog
# ---------------------------------------------------------------------------

def load_sherdog_fights(data_path: Path) -> List[FightRecord]:
    """
    Load Tier 3 fight records from Sherdog Fight Finder.

    Outcomes only — no stats. Used for ELO construction only.

    Required columns:
        fight_id, fighter_a_id, fighter_b_id, winner_id, method,
        weight_class, date, promotion
    """
    records: List[FightRecord] = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = _parse_method(row.get("method", ""))
            weight_class, weight_class_raw = _coerce_weight_class_from_cell(
                row.get("weight_class", "")
            )
            fight_date = _parse_date(row.get("date", ""))

            if method is None or weight_class is None or fight_date is None:
                warnings.warn(
                    f"Skipping Sherdog fight {row.get('fight_id')}: "
                    f"unrecognised method, weight_class, or date"
                )
                continue

            records.append(FightRecord(
                fight_id=row["fight_id"],
                fighter_a_id=row["fighter_a_id"],
                fighter_b_id=row["fighter_b_id"],
                winner_id=row.get("winner_id") or None,
                result_method=method,
                weight_class=weight_class,
                fight_date=fight_date,
                promotion=row.get("promotion", "Unknown"),
                tier=DataTier.TIER_3,
                weight_class_raw=weight_class_raw,
            ))
    return records


# ---------------------------------------------------------------------------
# Fighter profiles
# ---------------------------------------------------------------------------

def load_fighter_profiles(data_path: Path) -> Dict[str, FighterProfile]:
    """
    Load fighter physical attributes and pedigree signals.

    Required columns:
        fighter_id, name

    Optional columns:
        reach_cm, height_cm, date_of_birth (YYYY-MM-DD), stance,
        wrestling_pedigree, boxing_pedigree, bjj_pedigree
    """
    profiles: Dict[str, FighterProfile] = {}
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dob = _parse_date(row.get("date_of_birth", "")) if row.get("date_of_birth") else None
            stance = STANCE_MAP.get(row.get("stance", "").strip().lower(), Stance.UNKNOWN)

            profiles[row["fighter_id"]] = FighterProfile(
                fighter_id=row["fighter_id"],
                name=row["name"],
                reach_cm=_float_or_none(row.get("reach_cm")),
                height_cm=_float_or_none(row.get("height_cm")),
                date_of_birth=dob,
                stance=stance,
                wrestling_pedigree=float(row.get("wrestling_pedigree") or 0),
                boxing_pedigree=float(row.get("boxing_pedigree") or 0),
                bjj_pedigree=float(row.get("bjj_pedigree") or 0),
            )
    return profiles


# ---------------------------------------------------------------------------
# Sorting and filtering utilities
# ---------------------------------------------------------------------------

def sort_fights_chronologically(fights: List[FightRecord]) -> List[FightRecord]:
    """Return a new list sorted by fight_date ascending."""
    return sorted(fights, key=lambda f: f.fight_date)


def filter_tier1_post_era(
    fights: List[FightRecord], master_start_year: int
) -> List[FightRecord]:
    """Return only Tier 1 fights on or after ``Config.master_start_year`` (inclusive)."""
    return [
        f for f in fights
        if f.tier == DataTier.TIER_1 and f.fight_date.year >= master_start_year
    ]
