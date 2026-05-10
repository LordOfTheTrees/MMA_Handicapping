"""Resolve optional ``model.pkl`` for integration / parity tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_ENV_VAR = "MMA_HARNESS_MODEL"
_DEFAULT_DATA_PKL = _REPO / "data" / "model.pkl"
_FIXTURE_PKL = _REPO / "tests" / "fixtures" / "parity" / "model.pkl"


def _display_path(p: Path) -> str:
    """Prefer repo-relative path when under repo (shorter logs)."""
    try:
        return str(p.resolve().relative_to(_REPO))
    except ValueError:
        return str(p.resolve())


def harness_env_raw() -> str:
    """Raw env value (may be unset or wrong path)."""
    return os.environ.get(_ENV_VAR, "").strip()


def harness_model_path() -> Path | None:
    """First match wins: env override, then ``data/model.pkl``, then fixture path."""
    raw = harness_env_raw()
    if raw:
        cand = Path(raw).expanduser().resolve()
        if cand.is_file():
            return cand
    data_pkl = _DEFAULT_DATA_PKL.resolve()
    if data_pkl.is_file():
        return data_pkl
    fixed = _FIXTURE_PKL.resolve()
    if fixed.is_file():
        return fixed
    return None


HAS_HARNESS_MODEL = harness_model_path() is not None


def print_harness_integration_preamble(*, module: str, description: str) -> None:
    """
    Human-oriented banner for parity / export integration tests.

    Always prints to stderr so unittest stdout stays ordered; verbosity is intentional.
    """
    lines = [
        "",
        "=" * 80,
        f" MMA_HARNESS_INTEGRATION  module={module}",
        f" {description}",
        "-" * 80,
        " These tests need a trained MMAPredictor pickle (same kind as:",
        '   python main.py --model-path ./data/model.pkl predict ...)',
        "",
        f" Resolution order:",
        f"   1. Environment variable {_ENV_VAR} (optional override)",
        f"   2. Default repo path   {_display_path(_DEFAULT_DATA_PKL)}",
        f"   3. Test fixture        {_display_path(_FIXTURE_PKL)}",
        "",
    ]
    raw = harness_env_raw()
    if raw:
        expanded = Path(raw).expanduser().resolve()
        exists = expanded.is_file()
        lines.append(f" {_ENV_VAR} is set to: {raw!r}")
        lines.append(f"   expanded..: {_display_path(expanded)}")
        lines.append(f"   is_file:    {exists}")
        if not exists:
            lines.append("   (missing -> fall through to default data path, then fixture)")
    else:
        lines.append(f" {_ENV_VAR} is not set (using default path if present).")

    lines.append(f" Default: {_display_path(_DEFAULT_DATA_PKL)}")
    lines.append(f"   exists: {_DEFAULT_DATA_PKL.resolve().is_file()}")
    lines.append(f" Fixture: {_display_path(_FIXTURE_PKL)}")
    lines.append(f"   exists: {_FIXTURE_PKL.resolve().is_file()}")
    lines.append("")

    resolved = harness_model_path()
    if resolved is not None:
        lines.append(f" RESULT: INTEGRATION TESTS IN THIS FILE WILL RUN")
        lines.append(f" Using pickle: {_display_path(resolved)}")
        lines.append("")
        lines.append(" Contract: export uses as_of_date = last fight date in the model (or today if no fights).")
        lines.append(" Parity: predict_proba_point_only (pickle) vs predict_proba_snapshot (JSON) must match exactly.")
    else:
        lines.append(" RESULT: INTEGRATION TESTS IN THIS FILE ARE SKIPPED (no pickle found).")
        lines.append("")
        lines.append(" Put a trained model at the default path, or set the env override, or use the fixture path:")
        lines.append(f"   -> {_display_path(_DEFAULT_DATA_PKL)}  (default after train)")
        lines.append(f"   -> {_ENV_VAR}=<path>  ( optional )")
        lines.append(f"   -> {_display_path(_FIXTURE_PKL)}  ( optional )")
        lines.append("")
        lines.append(" Then: python -m unittest tests.test_export_artifacts_smoke tests.test_artifact_parity -v")

    lines.append("=" * 80)
    lines.append("")
    print("\n".join(lines), flush=True, file=sys.stderr)
