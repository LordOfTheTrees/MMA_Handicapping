#!/usr/bin/env python3
"""
Generate the cross-repo parity fixture consumed by ``mma.ai``'s test suite.

``mma.ai`` re-implements the 12-feature matchup vector in ``api/matchup.py`` because the
deployed service does not import this package. That duplication is only safe if something
proves the two implementations agree, and a test that needs both repos checked out cannot
run in ``mma.ai``'s CI.

So this script freezes the answer. It runs **this** repo's implementation — the one the
model was trained with — over a matrix of edge cases and writes inputs plus expected
outputs to JSON. ``mma.ai`` then asserts its own port reproduces them, with no dependency
on this repo at test time.

The cases are synthetic and hand-chosen rather than drawn from the live artifacts, so the
fixture never goes stale when the weekly export refreshes. What is pinned is *code*
semantics — imputation of missing physicals, stance normalisation, interaction terms, the
softmax — which is where drift actually happens.

Usage:
    python scripts/export_parity_vectors.py --out ../mma.ai/tests/fixtures/parity_vectors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import WeightClass  # noqa: E402
from src.export.json_inference import (  # noqa: E402
    _elo_state,
    _profile_from_export_dict,
    _style_axes,
)
from src.matchup.interactions import (  # noqa: E402
    FEATURE_NAMES,
    build_matchup_features,
    features_to_array,
)
from src.model.regression import (  # noqa: E402
    CLASS_LABELS,
    MultinomialLogisticModel,
    N_CLASSES,
)

FIXTURE_VERSION = 1

#: Frozen weights so expected probabilities do not move when the model retrains. Values are
#: arbitrary but fixed; the point is that both repos apply the same softmax to the same x.
FROZEN_W_SEED = 20260907

WC = WeightClass.LIGHTWEIGHT
FIGHT_DATE = date(2026, 5, 16)


def _elo(fid: str, elo: float, *, n_fights: int = 12, last: str | None = "2025-11-08") -> Dict[str, Any]:
    return {
        "fighter_id": fid,
        "elo": elo,
        "uncertainty": 118.5,
        "last_fight_date": last,
        "n_fights": n_fights,
        "primary_tier": 1,
    }


def _axes(fid: str, striker: float, grappler: float, threat: float, vuln: float) -> Dict[str, Any]:
    return {
        "fighter_id": fid,
        "striker_score": striker,
        "grappler_score": grappler,
        "finish_threat": threat,
        "finish_vulnerability": vuln,
        "striker_uncertainty": 0.09,
        "grappler_uncertainty": 0.11,
        "n_quality_fights": 9.0,
    }


def _profile(fid: str, **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "fighter_id": fid,
        "name": fid.upper(),
        "reach_cm": 183.0,
        "height_cm": 180.5,
        "date_of_birth": "1993-07-19",
        "stance": "orthodox",
    }
    base.update(overrides)
    return base


def build_cases() -> List[Dict[str, Any]]:
    """Every branch in feature construction that the two implementations could disagree on."""
    elo_a, elo_b = _elo("a", 1682.4), _elo("b", 1497.1)
    axes_a = _axes("a", 0.71, 0.38, 0.62, 0.24)
    axes_b = _axes("b", 0.29, 0.77, 0.41, 0.58)

    variants: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = [
        ("baseline_complete_profiles", _profile("a"), _profile("b")),
        # Physical imputation: missing values must become 0.0, not propagate as null.
        ("missing_reach_corner_a", _profile("a", reach_cm=None), _profile("b")),
        ("missing_reach_both", _profile("a", reach_cm=None), _profile("b", reach_cm=None)),
        ("missing_height_corner_b", _profile("a"), _profile("b", height_cm=None)),
        ("missing_dob_corner_a", _profile("a", date_of_birth=None), _profile("b")),
        ("missing_all_physicals", _profile("a", reach_cm=None, height_cm=None, date_of_birth=None), _profile("b")),
        # Stance mismatch is 1 only for orthodox-vs-southpaw, in either order.
        ("stance_orthodox_vs_southpaw", _profile("a", stance="orthodox"), _profile("b", stance="southpaw")),
        ("stance_southpaw_vs_orthodox", _profile("a", stance="southpaw"), _profile("b", stance="orthodox")),
        ("stance_orthodox_vs_orthodox", _profile("a", stance="orthodox"), _profile("b", stance="orthodox")),
        ("stance_southpaw_vs_southpaw", _profile("a", stance="southpaw"), _profile("b", stance="southpaw")),
        ("stance_switch_vs_southpaw", _profile("a", stance="switch"), _profile("b", stance="southpaw")),
        ("stance_unknown_vs_orthodox", _profile("a", stance="unknown"), _profile("b", stance="orthodox")),
        ("stance_empty_vs_southpaw", _profile("a", stance=""), _profile("b", stance="southpaw")),
        ("stance_null_vs_southpaw", _profile("a", stance=None), _profile("b", stance="southpaw")),
        # Normalisation: a raw-cased or padded cell must not read as a different stance on
        # one side than the other. These two cases caught a genuine train/serve split.
        ("stance_mixed_case_vs_orthodox", _profile("a", stance="Southpaw"), _profile("b", stance="orthodox")),
        ("stance_padded_vs_orthodox", _profile("a", stance="  southpaw  "), _profile("b", stance="orthodox")),
        ("stance_unrecognised_vs_orthodox", _profile("a", stance="sideways"), _profile("b", stance="orthodox")),
        # Sign conventions: swapping corners must negate every signed difference.
        ("swapped_corners", _profile("b"), _profile("a")),
        # Numeric edge: integer-typed physicals must coerce the same way as floats.
        ("integer_physicals", _profile("a", reach_cm=183, height_cm=180), _profile("b", reach_cm=175, height_cm=177)),
        # Age sign: A older than B gives a positive age_diff_days.
        ("corner_a_older", _profile("a", date_of_birth="1988-01-02"), _profile("b", date_of_birth="1998-01-02")),
    ]

    cases: List[Dict[str, Any]] = []
    for name, prof_a, prof_b in variants:
        swapped = name == "swapped_corners"
        ea, eb = (elo_b, elo_a) if swapped else (elo_a, elo_b)
        aa, ab = (axes_b, axes_a) if swapped else (axes_a, axes_b)

        features = build_matchup_features(
            _elo_state("a", WC, ea),
            _elo_state("b", WC, eb),
            _style_axes("a", WC, aa),
            _style_axes("b", WC, ab),
            _profile_from_export_dict("a", prof_a),
            _profile_from_export_dict("b", prof_b),
            FIGHT_DATE,
        )
        x = features_to_array(features)
        cases.append(
            {
                "name": name,
                "weight_class": WC.value,
                "fight_date": FIGHT_DATE.isoformat(),
                "elo_a": ea,
                "elo_b": eb,
                "axes_a": aa,
                "axes_b": ab,
                "profile_a": prof_a,
                "profile_b": prof_b,
                "expected_features": dict(zip(FEATURE_NAMES, (float(v) for v in x))),
                "expected_x": [float(v) for v in x],
            }
        )
    return cases


def frozen_w() -> np.ndarray:
    rng = np.random.default_rng(FROZEN_W_SEED)
    return np.round(rng.normal(0.0, 0.01, size=(N_CLASSES, len(FEATURE_NAMES))), 8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Destination parity_vectors.json")
    args = ap.parse_args()

    cases = build_cases()
    W = frozen_w()

    model = MultinomialLogisticModel(n_features=len(FEATURE_NAMES))
    model.W = W
    model.is_fitted = True
    for case in cases:
        probs = model.predict_proba(np.asarray(case["expected_x"], dtype=np.float64))
        case["expected_probs"] = [float(p) for p in probs]

    doc = {
        "fixture_version": FIXTURE_VERSION,
        "generated_by": "MMA_Handicapping/scripts/export_parity_vectors.py",
        "note": (
            "Golden vectors from the training implementation. Regenerate only when feature "
            "semantics change on purpose; a diff here means training and serving disagree."
        ),
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(CLASS_LABELS),
        "frozen_W": [[float(v) for v in row] for row in W],
        "cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} parity cases -> {args.out}")


if __name__ == "__main__":
    main()
