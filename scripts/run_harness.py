#!/usr/bin/env python3
"""
Human-friendly test harness (wraps unittest). Run from repo root::

    python scripts/run_harness.py
    python scripts/run_harness.py quick
    python scripts/run_harness.py integration
    python scripts/run_harness.py integration --model path/to/model.pkl

Suites:
  all           Default. Every test under tests/ (unittest discover).
  quick         Offline unit tests only (no model.pkl needed).
  integration   Export smoke + pickle vs JSON parity (needs model; see tests/harness_skip.py).

Model resolution matches tests: MMA_HARNESS_MODEL, else data/model.pkl, else tests/fixtures/parity/model.pkl.
Use --model to set MMA_HARNESS_MODEL for this run only (optional).
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
        choices=("all", "quick", "integration"),
        help="Test suite name (default: all). "
        '"quick" = no pickle; "integration" = export + parity vs model.pkl;',
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
    cmd: list[str] = [exe, "-m", "unittest"]
    if not args.quiet:
        cmd.append("-v")

    if args.suite == "all":
        cmd += ["discover", "-s", "tests", "-p", "test*.py"]
    elif args.suite == "quick":
        cmd += [
            "tests.test_json_snapshot_inference",
            "tests.test_upcoming_events_export",
            "tests.test_upcoming_bouts_parse",
        ]
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
