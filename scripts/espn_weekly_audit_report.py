#!/usr/bin/env python3
"""
Print human-readable ESPN ingest audit lines and upload-friendly summary.

Reads ``data/espn_ingest_audit.json`` (from the latest ingest) or rebuilds it
from ``espn_ingest_state.json`` ``last_run`` when ``--run-audit`` is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.espn_audit import (  # noqa: E402
    ESPN_INGEST_AUDIT_JSON,
    format_audit_log_lines,
    run_espn_ingest_audit,
)


def main() -> int:
    p = argparse.ArgumentParser(description="ESPN weekly audit report (log + cached JSON).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--run-audit",
        action="store_true",
        help="Re-run audit from last_run in espn_ingest_state.json (needs network for rookie check).",
    )
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 even when audit has rejects (report-only workflow).",
    )
    p.add_argument(
        "--skip-rookie-fetch",
        action="store_true",
        help="Skip ESPN eventlog calls (offline / cached report only).",
    )
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    audit_path = data_dir / ESPN_INGEST_AUDIT_JSON

    if args.run_audit:
        audit, code = run_espn_ingest_audit(
            data_dir,
            fetch_rookie=not args.skip_rookie_fetch,
            fail_on_reject=not args.no_fail,
        )
    elif audit_path.is_file():
        with open(audit_path, encoding="utf-8") as f:
            audit = json.load(f)
        code = 0 if audit.get("passed", True) else 1
    else:
        print(
            f"[espn_audit] No {ESPN_INGEST_AUDIT_JSON}; run ingest first or pass --run-audit.",
            flush=True,
        )
        return 0

    print("::group::ESPN ingest audit report", flush=True)
    for line in format_audit_log_lines(audit):
        print(line, flush=True)
    print("::endgroup::", flush=True)
    print(f"[espn_audit] Wrote {audit_path}", flush=True)

    if args.no_fail:
        return 0
    return code


if __name__ == "__main__":
    raise SystemExit(main())
