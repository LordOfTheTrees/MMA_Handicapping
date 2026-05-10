#!/usr/bin/env python3
"""
Export trained :class:`~src.pipeline.MMAPredictor` state to portable JSON for the deploy repo.

Run from repo root::

    python scripts/export_artifacts.py --model-path data/model.pkl --out-dir JSON_exports
    python scripts/export_artifacts.py ... --copy-to-mma-ai

Emits ``model_weights.json``, ``elo_states.json``, ``style_axes.json``, ``fighter_profiles.json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export.git_meta import git_sha_training_repo  # noqa: E402
from src.matchup.interactions import FEATURE_NAMES  # noqa: E402
from src.model.regression import CLASS_LABELS, N_CLASSES  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402


EXPORT_SCHEMA_VERSION = "mma-handicapping-export-v1"


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _config_snapshot(predictor: MMAPredictor) -> dict[str, Any]:
    cfg = predictor.config
    return {
        "master_start_year": cfg.master_start_year,
        "holdout_start_date": cfg.holdout_start_date,
        "elo": dataclasses.asdict(cfg.elo),
        "features": dataclasses.asdict(cfg.features),
        "model": dataclasses.asdict(cfg.model),
    }


def _export_model_weights(predictor: MMAPredictor, manifest: dict[str, Any]) -> dict[str, Any]:
    reg = predictor.regression
    if reg is None or reg.W is None:
        raise RuntimeError("Predictor must have trained regression weights (train_regression completed).")

    W = np.asarray(reg.W, dtype=float)
    if W.shape != (N_CLASSES, len(FEATURE_NAMES)):
        raise ValueError(f"Unexpected W shape {W.shape}; expected ({N_CLASSES}, {len(FEATURE_NAMES)})")

    bootstrap = getattr(predictor, "_bootstrap_W", None)
    boot_list: Optional[list]
    if bootstrap is None:
        boot_list = None
    else:
        boot_arr = np.asarray(bootstrap, dtype=float)
        if boot_arr.ndim != 3 or boot_arr.shape[1:] != W.shape:
            raise ValueError(f"Unexpected bootstrap_W shape {boot_arr.shape}")
        boot_list = boot_arr.tolist()

    return {
        "export_manifest": manifest,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(CLASS_LABELS),
        "n_classes": N_CLASSES,
        "W": W.tolist(),
        "bootstrap_W": boot_list,
        "regression": {
            "huber_delta": float(reg.delta),
            "l2_lambda": float(reg.l2_lambda),
            "n_features": int(reg.n_features),
            "is_fitted": bool(reg.is_fitted),
        },
        "training_config": _json_sanitize(_config_snapshot(predictor)),
    }


def _as_of_date(predictor: MMAPredictor, override: Optional[date]) -> date:
    if override is not None:
        return override
    if predictor.fights:
        return predictor.fights[-1].fight_date
    return date.today()


def _export_elo_states(predictor: MMAPredictor, as_of: date, manifest: dict[str, Any]) -> dict[str, Any]:
    em = predictor.elo_model
    if em is None:
        raise RuntimeError("Predictor missing elo_model.")

    states = getattr(em, "_states", {})
    out: dict[str, dict[str, Any]] = {}
    for fid, wc in states:
        st = em.get_state(fid, wc, as_of)
        out.setdefault(fid, {})
        out[fid][st.weight_class.value] = {
            "elo": float(st.elo),
            "uncertainty": float(st.uncertainty),
            "last_fight_date": st.last_fight_date.isoformat() if st.last_fight_date else None,
            "n_fights": int(st.n_fights),
            "primary_tier": st.primary_tier.value,
        }
    return {
        "export_manifest": manifest,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "as_of_date": as_of.isoformat(),
        "states": out,
    }


def _export_style_axes(predictor: MMAPredictor, as_of: date, manifest: dict[str, Any]) -> dict[str, Any]:
    em = predictor.elo_model
    if em is None:
        raise RuntimeError("Predictor missing elo_model.")

    states = getattr(em, "_states", {})
    out: dict[str, dict[str, Any]] = {}
    for fid, wc in states:
        ax = predictor.get_style_axes(fid, wc, as_of)
        out.setdefault(fid, {})
        out[fid][wc.value] = {
            "striker_score": float(ax.striker_score),
            "grappler_score": float(ax.grappler_score),
            "finish_threat": float(ax.finish_threat),
            "finish_vulnerability": float(ax.finish_vulnerability),
            "striker_uncertainty": float(ax.striker_uncertainty),
            "grappler_uncertainty": float(ax.grappler_uncertainty),
            "n_quality_fights": float(ax.n_quality_fights),
        }
    return {
        "export_manifest": manifest,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "as_of_date": as_of.isoformat(),
        "axes": out,
    }


def _export_fighter_profiles(predictor: MMAPredictor, manifest: dict[str, Any]) -> dict[str, Any]:
    profs = {}
    for fid, p in predictor.profiles.items():
        profs[fid] = _json_sanitize(dataclasses.asdict(p))
    return {
        "export_manifest": manifest,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "profiles": profs,
    }


def export_all(
    predictor: MMAPredictor,
    out_dir: Path,
    *,
    as_of: Optional[date] = None,
) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    as_of_d = _as_of_date(predictor, as_of)
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": exported_at,
        "git_sha_training_repo": git_sha_training_repo(cwd=ROOT),
        "as_of_date": as_of_d.isoformat(),
        "notes": "Produced by MMA_Handicapping scripts/export_artifacts.py",
    }

    writers = [
        ("model_weights.json", _export_model_weights(predictor, manifest)),
        ("elo_states.json", _export_elo_states(predictor, as_of_d, manifest)),
        ("style_axes.json", _export_style_axes(predictor, as_of_d, manifest)),
        ("fighter_profiles.json", _export_fighter_profiles(predictor, manifest)),
    ]
    written: list[Path] = []
    for name, doc in writers:
        path = out_dir / name
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written[0], written[1], written[2], written[3]


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Export MMAPredictor to JSON artifacts for mma.ai / OctagonELO.")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="YYYY-MM-DD for ELO/style export (default: last fight date in model, else today)",
    )
    p.add_argument(
        "--copy-to-mma-ai",
        action="store_true",
        help="After export, copy all *.json under --out-dir to sibling mma.ai/artifacts",
    )
    p.add_argument(
        "--mma-ai-artifacts-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override deploy dir (default: <repo>/../mma.ai/artifacts)",
    )
    args = p.parse_args(argv)

    as_of: Optional[date] = None
    if args.as_of_date:
        as_of = date.fromisoformat(args.as_of_date)

    predictor = MMAPredictor.load(Path(args.model_path))
    out_dir = Path(args.out_dir)
    export_all(predictor, out_dir, as_of=as_of)
    print(f"Wrote 4 JSON files under {out_dir.resolve()}", flush=True)

    if args.copy_to_mma_ai:
        _scripts = Path(__file__).resolve().parent
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        import copy_exports_to_mma_ai as _cex

        dest = (
            Path(args.mma_ai_artifacts_dir).resolve()
            if args.mma_ai_artifacts_dir
            else _cex.default_mma_ai_artifacts_dir()
        )
        copied = _cex.copy_json_from_dir(out_dir, dest)
        print(f"Copied {len(copied)} JSON file(s) -> {dest}", flush=True)


if __name__ == "__main__":
    main()
