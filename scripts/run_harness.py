#!/usr/bin/env python3
"""
Human-friendly test harness (wraps unittest). Run from repo root::

    python scripts/run_harness.py
    python scripts/run_harness.py quick
    python scripts/run_harness.py site
    python scripts/run_harness.py integration
    python scripts/run_harness.py integration --model path/to/model.pkl

Suites:
  all           Default. Every test under tests/ (unittest discover).
  quick         Offline unit tests only (no model.pkl needed).
  site          JSON_exports/ contracts vs docs/website_elements.md pages (events, rankings, profiles, bout JSON, about, reference_distributions).
  integration   Export smoke + pickle vs JSON parity (needs model; see tests/harness_skip.py).

Model resolution matches tests: **`MMA_HARNESS_MODEL`** if set and the file exists, else **`data/model.pkl`**.
Use --model to set MMA_HARNESS_MODEL for this run only (optional).

Site suite: optional env MMA_SITE_EXPORT_DIR overrides JSON_exports path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run MMA_Handicapping unittest suites without memorizing discovery flags.",
    )
    p.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=("all", "quick", "site", "integration"),
        help="Test suite name (default: all). "
        '"quick" = no pickle; "site" = JSON_exports vs website pages; '
        '"integration" = export + parity vs model.pkl;',
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help="Set MMA_HARNESS_MODEL for this process only (optional; integration suite).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Omit unittest -v (less output).",
    )
    args = p.parse_args(argv)

    repo = _repo_root()
    exe = sys.executable
    verbose = not args.quiet

    # For "discover", -v must NOT come before "discover" (Python 3.11+ unittest treats
    # trailing -s/-p as errors: they only apply to discover's sub-parser).
    if args.suite == "all":
        cmd = [exe, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]
        if verbose:
            cmd.append("-v")
    else:
        cmd = [exe, "-m", "unittest"]
        if verbose:
            cmd.append("-v")
        if args.suite == "quick":
            cmd += [
                "tests.test_json_snapshot_inference",
                "tests.test_upcoming_events_export",
                "tests.test_upcoming_bouts_parse",
            ]
        elif args.suite == "site":
            cmd.append("tests.test_site_export_pages")
        else:
            cmd += [
                "tests.test_export_artifacts_smoke",
                "tests.test_artifact_parity",
            ]

    env = os.environ.copy()
    if args.model:
        env["MMA_HARNESS_MODEL"] = str(Path(args.model).expanduser().resolve())

    banner = (
        f"[run_harness] cwd={repo} suite={args.suite!r} cmd={' '.join(cmd)!r}"
        + (f" MMA_HARNESS_MODEL={env['MMA_HARNESS_MODEL']!r}" if args.model else "")
    )
    print(banner, flush=True, file=sys.stderr)

    proc = subprocess.run(cmd, cwd=str(repo), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
