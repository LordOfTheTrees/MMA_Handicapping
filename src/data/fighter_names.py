"""
Resolve display names from ``fighter_profiles.csv`` to ``fighter_id`` keys.

Matching is **exact** on the profile ``name`` field (case-insensitive, stripped).
For substring or fuzzy search, build your own filter over ``profiles.values()``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .schema import FighterProfile


def fighter_ids_for_exact_name(name: str, profiles: Dict[str, FighterProfile]) -> List[str]:
    """
    Return every ``fighter_id`` whose profile ``name`` equals *name* after strip + casefold.

    Empty *name* returns an empty list. Multiple UFCStats rows can share a spelling
    in edge cases — callers should handle len != 1.
    """
    if not name or not str(name).strip():
        return []
    target = str(name).strip().casefold()
    return [
        fid
        for fid, p in profiles.items()
        if p.name and p.name.strip().casefold() == target
    ]


def resolve_fighter_id(name: str, profiles: Dict[str, FighterProfile]) -> Optional[str]:
    """
    If exactly one profile matches :func:`fighter_ids_for_exact_name`, return its id.

    Returns ``None`` if there are zero matches or more than one (ambiguous).
    """
    ids = fighter_ids_for_exact_name(name, profiles)
    if len(ids) == 1:
        return ids[0]
    return None


def require_fighter_id(name: str, profiles: Dict[str, FighterProfile]) -> str:
    """
    Resolve *name* to a unique ``fighter_id`` or raise ``ValueError`` with a short reason.
    """
    ids = fighter_ids_for_exact_name(name, profiles)
    if not ids:
        raise ValueError(f"No profile name matches {name!r} (exact, case-insensitive).")
    if len(ids) > 1:
        raise ValueError(
            f"Ambiguous name {name!r}: {len(ids)} profiles match — use fighter_id directly."
        )
    return ids[0]
