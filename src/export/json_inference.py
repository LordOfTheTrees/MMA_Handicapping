"""Point probabilities from exported JSON artifacts (deploy parity with ``predict_proba_point_only``).

Contract: ``elo_states.json`` / ``style_axes.json`` are snapshots at their shared ``as_of_date``.
:class:`~src.pipeline.MMAPredictor` resolves ELO/style at an arbitrary calendar date.

Snapshot inference is only comparable to pickle **when** ``fight_date`` equals that ``as_of_date``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from src.data.schema import (
    DataTier,
    ELOState,
    FighterProfile,
    Stance,
    StyleAxes,
    WeightClass,
)
from src.matchup.interactions import FEATURE_NAMES, build_matchup_features, features_to_array
from src.model.regression import MultinomialLogisticModel, N_CLASSES

EXPECTED_EXPORT_SCHEMA = "mma-handicapping-export-v1"

_ARTIFACT_KEYS = ("model_weights", "elo_states", "style_axes", "fighter_profiles")

JsonSource = Union[Path, Mapping[str, Any]]

__all__ = [
    "EXPECTED_EXPORT_SCHEMA",
    "predict_proba_snapshot",
    "load_four_artifact_documents",
]


def load_four_artifact_documents(source: JsonSource) -> tuple[dict[str, Any], ...]:
    """
    Load the four inference JSON docs.

    ``source`` is either a directory containing the standard filenames, or a mapping with keys
    ``model_weights``, ``elo_states``, ``style_axes``, ``fighter_profiles``.
    Each value is the parsed dict (as ``json.loads``), not an on-disk path.
    """
    if isinstance(source, Path):
        dir_path = source
        return tuple(
            json.loads((dir_path / f"{key}.json").read_text(encoding="utf-8"))
            for key in ("model_weights", "elo_states", "style_axes", "fighter_profiles")
        )

    missing = [k for k in _ARTIFACT_KEYS if k not in source]
    if missing:
        raise KeyError(f"artifact mapping missing keys: {missing}")

    mw = source["model_weights"]
    elo = source["elo_states"]
    sx = source["style_axes"]
    prof = source["fighter_profiles"]

    return (
        mw if isinstance(mw, dict) else dict(mw),
        elo if isinstance(elo, dict) else dict(elo),
        sx if isinstance(sx, dict) else dict(sx),
        prof if isinstance(prof, dict) else dict(prof),
    )


def _assert_schema_versions(mw: dict, elo: dict, sx: dict, prof: dict) -> None:
    for tag, doc in (
        ("model_weights", mw),
        ("elo_states", elo),
        ("style_axes", sx),
        ("fighter_profiles", prof),
    ):
        v = doc.get("export_schema_version")
        if v != EXPECTED_EXPORT_SCHEMA:
            raise ValueError(f"{tag}: expected export_schema_version {EXPECTED_EXPORT_SCHEMA!r}, got {v!r}")


def _coerce_weight_class(wc: WeightClass | str) -> WeightClass:
    if isinstance(wc, WeightClass):
        return wc
    try:
        return WeightClass(wc)
    except ValueError as exc:
        raise ValueError(f"invalid weight_class: {wc!r}") from exc


def _coerce_primary_tier(v: Any) -> DataTier:
    return DataTier(int(v))


def _parse_stance(raw: Any) -> Stance:
    if raw is None or raw == "":
        return Stance.UNKNOWN
    if isinstance(raw, Stance):
        return raw
    try:
        return Stance(str(raw))
    except ValueError:
        return Stance.UNKNOWN


def _profile_from_export_dict(fid: str, d: Mapping[str, Any]) -> FighterProfile:
    dob = d.get("date_of_birth")
    dob_dt: date | None
    if dob is None or dob == "":
        dob_dt = None
    else:
        dob_dt = date.fromisoformat(str(dob))

    return FighterProfile(
        fighter_id=str(d["fighter_id"]),
        name=str(d.get("name", fid)),
        reach_cm=float(d["reach_cm"]) if d.get("reach_cm") is not None else None,
        height_cm=float(d["height_cm"]) if d.get("height_cm") is not None else None,
        date_of_birth=dob_dt,
        stance=_parse_stance(d.get("stance")),
        wrestling_pedigree=float(d.get("wrestling_pedigree", 0.0)),
        boxing_pedigree=float(d.get("boxing_pedigree", 0.0)),
        bjj_pedigree=float(d.get("bjj_pedigree", 0.0)),
    )


def _empty_profile(fighter_id: str) -> FighterProfile:
    return FighterProfile(fighter_id=fighter_id, name=fighter_id)


def _elo_state(fid: str, wc: WeightClass, cell: Mapping[str, Any]) -> ELOState:
    lfd = cell.get("last_fight_date")
    return ELOState(
        fighter_id=fid,
        weight_class=wc,
        elo=float(cell["elo"]),
        uncertainty=float(cell["uncertainty"]),
        last_fight_date=date.fromisoformat(str(lfd)) if lfd not in (None, "") else None,
        n_fights=int(cell["n_fights"]),
        primary_tier=_coerce_primary_tier(cell["primary_tier"]),
    )


def _style_axes(fid: str, wc: WeightClass, cell: Mapping[str, Any]) -> StyleAxes:
    return StyleAxes(
        fighter_id=fid,
        weight_class=wc,
        striker_score=float(cell["striker_score"]),
        grappler_score=float(cell["grappler_score"]),
        finish_threat=float(cell["finish_threat"]),
        finish_vulnerability=float(cell["finish_vulnerability"]),
        striker_uncertainty=float(cell["striker_uncertainty"]),
        grappler_uncertainty=float(cell["grappler_uncertainty"]),
        n_quality_fights=float(cell["n_quality_fights"]),
    )


def predict_proba_snapshot(
    artifact_root: JsonSource,
    fighter_a_id: str,
    fighter_b_id: str,
    wc: WeightClass | str,
    fight_date: date,
) -> np.ndarray:
    """
    Softmax probabilities (6,) for fighter ``fighter_a_id`` vs ``fighter_b_id`` using **only**
    artifact JSON loaded from ``artifact_root``.

    Raises:
        ValueError: if ``fight_date`` differs from bundled ``as_of_date`` — snapshot vectors
          do not replicate temporal ``get_state`` / ``get_style_axes`` logic.
        KeyError: missing fighter ids or missing weight-class slice under ``states`` / ``axes``.
    """
    mw, elo_doc, sx_doc, prof_doc = load_four_artifact_documents(artifact_root)
    _assert_schema_versions(mw, elo_doc, sx_doc, prof_doc)

    as_elo_str = elo_doc.get("as_of_date")
    as_style_str = sx_doc.get("as_of_date")
    if not as_elo_str or not as_style_str:
        raise ValueError("elo_states / style_axes missing top-level as_of_date")
    as_elo = date.fromisoformat(str(as_elo_str))
    as_style = date.fromisoformat(str(as_style_str))
    if as_elo != as_style:
        raise ValueError(f"elo as_of_date {as_elo} != style_axes as_of_date {as_style}")
    if fight_date != as_elo:
        raise ValueError(
            f"fight_date {fight_date} must equal artifact as_of_date {as_elo} "
            "(JSON exports are snapshots; pickles evaluate at arbitrary fight_date)."
        )

    feats = mw.get("feature_names")
    if list(feats) != FEATURE_NAMES:
        raise ValueError("model_weights feature_names mismatch vs FEATURE_NAMES")

    W_arr = np.asarray(mw["W"], dtype=np.float64)
    if W_arr.shape != (N_CLASSES, len(FEATURE_NAMES)):
        raise ValueError(f"model_weights W unexpected shape {W_arr.shape}")

    wc_enum = _coerce_weight_class(wc)
    wkey = wc_enum.value

    states_by_fighter = elo_doc["states"]
    axes_by_fighter = sx_doc["axes"]
    raw_profiles = prof_doc.get("profiles", {})

    def _elo_for(fid: str) -> ELOState:
        fc = states_by_fighter[fid][wkey]
        return _elo_state(fid, wc_enum, fc)

    def _axes_for(fid: str) -> StyleAxes:
        ac = axes_by_fighter[fid][wkey]
        return _style_axes(fid, wc_enum, ac)

    def _prof(fid: str) -> FighterProfile:
        pr = raw_profiles.get(fid)
        if pr is None:
            return _empty_profile(fid)
        return _profile_from_export_dict(fid, pr)

    reg = MultinomialLogisticModel(n_features=len(FEATURE_NAMES))
    rr = mw.get("regression", {})
    reg.delta = float(rr.get("huber_delta", 1.35))
    reg.l2_lambda = float(rr.get("l2_lambda", 1e-4))
    reg.W = W_arr
    reg.is_fitted = True

    features = build_matchup_features(
        _elo_for(fighter_a_id),
        _elo_for(fighter_b_id),
        _axes_for(fighter_a_id),
        _axes_for(fighter_b_id),
        _prof(fighter_a_id),
        _prof(fighter_b_id),
        fight_date,
    )
    x = features_to_array(features)
    return np.asarray(reg.predict_proba(x), dtype=np.float64)
