"""Resolve optional ``model.pkl`` for integration / parity tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_ENV_VAR = "MMA_HARNESS_MODEL"
_DEFAULT_DATA_PKL = _REPO / "data" / "model.pkl"


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
    """First match wins: env override when file exists, else ``data/model.pkl``."""
    raw = harness_env_raw()
    if raw:
        cand = Path(raw).expanduser().resolve()
        if cand.is_file():
            return cand
    data_pkl = _DEFAULT_DATA_PKL.resolve()
    if data_pkl.is_file():
        return data_pkl
    return None


HAS_HARNESS_MODEL = harness_model_path() is not None


def _harness_miss_detail_parts() -> list[str]:
    """Facts for unittest skipReason / banners when no pickle is resolved."""
    parts: list[str] = []
    raw = harness_env_raw()
    if raw:
        expanded = Path(raw).expanduser().resolve()
        parts.append(f"env={raw!r} ->{_display_path(expanded)} exists={expanded.is_file()}")
    else:
        parts.append("env unset")

    dd = _DEFAULT_DATA_PKL.resolve()
    parts.append(f"default data/model.pkl {_display_path(dd)} exists={dd.is_file()}")
    return parts


# Shown verbatim on unittest's 'skipped …' line — do not rely on stderr ordering alone.
HARNESS_SKIP_REASON = (
    "No model.pkl resolved (checks: "
    + " ; ".join(_harness_miss_detail_parts())
    + "). Train to data/model.pkl or set MMA_HARNESS_MODEL=<path>; stderr has MMA_HARNESS_INTEGRATION banner."
    if not HAS_HARNESS_MODEL
    else "OK: model pickle resolved (never shown as skip reason)"
)


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
            lines.append("   (missing -> fall through to default data/model.pkl)")
    else:
        lines.append(f" {_ENV_VAR} is not set (using default path if present).")

    lines.append(f" Default: {_display_path(_DEFAULT_DATA_PKL)}")
    lines.append(f"   exists: {_DEFAULT_DATA_PKL.resolve().is_file()}")
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
        lines.append(" Train a model to the default path or set the env override:")
        lines.append(f"   -> {_display_path(_DEFAULT_DATA_PKL)}  (usual after train)")
        lines.append(f"   -> {_ENV_VAR}=<path>  (override)")
        lines.append("")
        lines.append(" Then: python scripts/run_harness.py integration")

    lines.append("=" * 80)
    lines.append("")
    print("\n".join(lines), flush=True, file=sys.stderr)
