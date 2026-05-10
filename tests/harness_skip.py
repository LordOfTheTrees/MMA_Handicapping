"""Resolve optional ``model.pkl`` for integration / parity tests."""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def harness_model_path() -> Path | None:
    raw = os.environ.get("MMA_HARNESS_MODEL", "").strip()
    if raw and Path(raw).expanduser().resolve().is_file():
        return Path(raw).expanduser().resolve()
    fixed = (_REPO / "tests" / "fixtures" / "parity" / "model.pkl").resolve()
    if fixed.is_file():
        return fixed
    return None


HAS_HARNESS_MODEL = harness_model_path() is not None
