"""
Synthetic / unknown fighters for ``predict``: register cold-start corners and run matchups.

- **Known** fighters: pass their existing ``fighter_id`` string (already in trained data).
- Unknown fighters: pass a :class:`SyntheticCorner` — we mint ``unk_<hex>`` ids, attach a
  :class:`FighterProfile`, and seed ELO in this division via :meth:`ELOModel.initialize_from_pedigree`.

Style axes resolve to cold-start blending (empty fight list in corpus for that id x division).
If both corners lack prior bouts **in this weight class** before ``fight_date``, CIs route to
``cauchy_wc_debut`` (same as weight-class debut in production).

Corners are registered on the predictor in-memory until process restarts — do not collide ids.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Optional, Union

from .data.schema import FighterProfile, PredictionResult, WeightClass
from .pipeline import MMAPredictor


@dataclass(frozen=True)
class SyntheticCorner:
    """Cold-start fighter with no UFC history in our data (prior-based profile only)."""

    display_name: str = "Unknown"
    reach_cm: Optional[float] = None
    height_cm: Optional[float] = None
    boxing_pedigree: float = 0.34
    wrestling_pedigree: float = 0.33
    bjj_pedigree: float = 0.33

    def validate(self) -> None:
        ped = (
            float(self.boxing_pedigree),
            float(self.wrestling_pedigree),
            float(self.bjj_pedigree),
        )
        if any(not (0.0 <= x <= 1.0) for x in ped):
            raise ValueError("Pedigree sliders must lie in [0, 1].")


def _profile_display_name(predictor: MMAPredictor, fighter_id: str) -> str:
    prof = predictor.profiles.get(fighter_id)
    if prof is not None and getattr(prof, "name", None):
        return str(prof.name).strip()
    return fighter_id[:16] + ("..." if len(fighter_id) > 16 else "")


def _corner_banner_block(
    predictor: MMAPredictor,
    corner: Union[str, SyntheticCorner],
    resolved_id: str,
    slot: str,
) -> str:
    """Multi-line ASCII block describing one corner (A perspective = fighter A)."""
    name = _profile_display_name(predictor, resolved_id)
    if isinstance(corner, SyntheticCorner):
        kind = (
            "Synthetic (minted id; no bouts in loaded data - style/ELO from profile priors)"
        )
    else:
        kind = "Known (fighter id from your corpus - loaded history + profile)"

    if slot == "A":
        lead = "  Fighter A - All rows labeled 'Win ...' below are victories for fighter A."
    else:
        lead = (
            "  Fighter B - Rows labeled 'Lose ...' below are victories for fighter B "
            "(wins by your corner-B pick from fighter A's perspective)."
        )

    return f"{lead}\n  Display name / profile label: {name}\n    {kind}\n    fighter_id={resolved_id}"


def _print_cold_matchup_banner(
    predictor: MMAPredictor,
    corner_a: Union[str, SyntheticCorner],
    corner_b: Union[str, SyntheticCorner],
    id_a: str,
    id_b: str,
    wc: WeightClass,
    fight_date: date,
) -> None:
    print("", flush=True)
    print("=" * 74, flush=True)
    print(
        "Cold-corner prediction  (multiclass softmax uses fighter-A perspective)",
        flush=True,
    )
    print("-" * 74, flush=True)
    print(_corner_banner_block(predictor, corner_a, id_a, "A"), flush=True)
    print("", flush=True)
    print(_corner_banner_block(predictor, corner_b, id_b, "B"), flush=True)
    print("", flush=True)
    print(f"  Division: {wc.value}  |  Scheduled date: {fight_date}", flush=True)
    print("=" * 74, flush=True)
    print("", flush=True)


def _print_derived_like_predict(result: PredictionResult) -> None:
    """Match ``main.py cmd_predict`` derived block — values are fractions 0–1."""
    print("", flush=True)
    print(
        "Derived - values are fractions 0-1 (same as python main.py predict; multiply by 100 for %):"
    )
    print(f"  Total win %    {result.total_win:.2f}")
    print(f"  Finish win %   {result.finish_win:.2f}")
    print(f"  Finish lose %  {result.finish_lose:.2f}")
    print(f"  Decision %     {result.go_to_decision:.2f}")
    print("", flush=True)


def mint_synthetic_id() -> str:
    """Return a unique fighter id prefixed ``unk_``."""
    return f"unk_{uuid.uuid4().hex[:14]}"


def register_synthetic_fighter(
    predictor: MMAPredictor,
    corner: SyntheticCorner,
    wc: WeightClass,
    *,
    fighter_id: Optional[str] = None,
    display_suffix: Optional[str] = None,
) -> str:
    """
    Attach a synthetic fighter to ``predictor``: profile + pedigree ELO cold start **in wc**.

    Returns the ``fighter_id`` to pass to :meth:`~MMAPredictor.predict`.

    Raises if ``fighter_id`` is supplied and already present in profiles.
    """
    corner.validate()
    fid = fighter_id or mint_synthetic_id()
    if fid in predictor.profiles:
        raise ValueError(f"fighter_id {fid!r} already registered on this predictor.")

    dn = corner.display_name.strip()
    if display_suffix:
        dn = f"{dn} ({display_suffix})"

    profile = FighterProfile(
        fighter_id=fid,
        name=dn,
        reach_cm=corner.reach_cm,
        height_cm=corner.height_cm,
        wrestling_pedigree=corner.wrestling_pedigree,
        boxing_pedigree=corner.boxing_pedigree,
        bjj_pedigree=corner.bjj_pedigree,
    )
    predictor.profiles[fid] = profile
    predictor.elo_model.initialize_from_pedigree(fid, wc, profile)
    return fid


def _resolve_corner_id(
    predictor: MMAPredictor,
    corner: Union[str, SyntheticCorner],
    wc: WeightClass,
    *,
    display_suffix: Optional[str] = None,
) -> str:
    if isinstance(corner, str):
        return corner
    if not isinstance(corner, SyntheticCorner):
        raise TypeError(
            "corner must be a fighter id (str) or SyntheticCorner; "
            f"got {type(corner).__name__}"
        )
    return register_synthetic_fighter(
        predictor, corner, wc, display_suffix=display_suffix
    )


def predict_cold_corner_matchup(
    predictor: MMAPredictor,
    corner_a: Union[str, SyntheticCorner],
    corner_b: Union[str, SyntheticCorner],
    wc: WeightClass,
    fight_date: date,
    *,
    verbose: bool = True,
    **predict_kwargs,
) -> PredictionResult:
    """
    Predict where each corner may be a real ``fighter_id`` or a :class:`SyntheticCorner`.

    Two synthetic corners → unknown × unknown.
    Synthetic + ``str`` → unknown × known.

    Probabilities for classes 0–2 remain **corner A wins** (fighter A perspective).
    """
    if isinstance(corner_a, SyntheticCorner):
        corner_a.validate()
    if isinstance(corner_b, SyntheticCorner):
        corner_b.validate()

    dual_synth = isinstance(corner_a, SyntheticCorner) and isinstance(corner_b, SyntheticCorner)

    id_a = _resolve_corner_id(
        predictor,
        corner_a,
        wc,
        display_suffix="A" if dual_synth and isinstance(corner_a, SyntheticCorner) else None,
    )
    id_b = _resolve_corner_id(
        predictor,
        corner_b,
        wc,
        display_suffix="B" if dual_synth and isinstance(corner_b, SyntheticCorner) else None,
    )

    if verbose:
        _print_cold_matchup_banner(
            predictor, corner_a, corner_b, id_a, id_b, wc, fight_date
        )

    result = predictor.predict(
        id_a, id_b, wc, fight_date, verbose=verbose, **predict_kwargs
    )

    if verbose:
        _print_derived_like_predict(result)

    return result
