"""
Shared CLI helpers: date parsing, weight-class resolution.
"""
from __future__ import annotations

import sys
from datetime import date, datetime

from ..data.schema import WeightClass

_WC_ALIASES = {
    "strawweight":       WeightClass.STRAWWEIGHT,
    "straw":             WeightClass.STRAWWEIGHT,
    "flyweight":         WeightClass.FLYWEIGHT,
    "fly":               WeightClass.FLYWEIGHT,
    "bantamweight":      WeightClass.BANTAMWEIGHT,
    "bantam":            WeightClass.BANTAMWEIGHT,
    "featherweight":     WeightClass.FEATHERWEIGHT,
    "feather":           WeightClass.FEATHERWEIGHT,
    "lightweight":       WeightClass.LIGHTWEIGHT,
    "light":             WeightClass.LIGHTWEIGHT,
    "welterweight":      WeightClass.WELTERWEIGHT,
    "welter":            WeightClass.WELTERWEIGHT,
    "middleweight":      WeightClass.MIDDLEWEIGHT,
    "middle":            WeightClass.MIDDLEWEIGHT,
    "light_heavyweight": WeightClass.LIGHT_HEAVYWEIGHT,
    "lhw":               WeightClass.LIGHT_HEAVYWEIGHT,
    "heavyweight":       WeightClass.HEAVYWEIGHT,
    "heavy":             WeightClass.HEAVYWEIGHT,
    "hw":                WeightClass.HEAVYWEIGHT,
    "w_strawweight":     WeightClass.W_STRAWWEIGHT,
    "w_straw":           WeightClass.W_STRAWWEIGHT,
    "w_flyweight":       WeightClass.W_FLYWEIGHT,
    "w_fly":             WeightClass.W_FLYWEIGHT,
    "w_bantamweight":    WeightClass.W_BANTAMWEIGHT,
    "w_bantam":          WeightClass.W_BANTAMWEIGHT,
    "w_featherweight":   WeightClass.W_FEATHERWEIGHT,
    "w_feather":         WeightClass.W_FEATHERWEIGHT,
}


def _normalize_wc_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def try_resolve_weight_class(raw: str) -> WeightClass | None:
    """Return a weight class if *raw* matches a known alias; otherwise ``None``."""
    if not raw.strip():
        return None
    return _WC_ALIASES.get(_normalize_wc_key(raw))


def resolve_weight_class(raw: str) -> WeightClass:
    wc = try_resolve_weight_class(raw)
    if wc is None:
        valid = sorted(set(_WC_ALIASES.keys()))
        print(f"Unknown weight class: {raw!r}")
        print(f"Valid options: {', '.join(valid)}")
        sys.exit(1)
    return wc


def resolve_date(raw=None) -> date:
    if raw is None:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {raw!r}  (expected YYYY-MM-DD)")
        sys.exit(1)
