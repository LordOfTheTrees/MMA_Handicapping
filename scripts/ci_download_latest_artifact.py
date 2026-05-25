#!/usr/bin/env python3
"""
Download the newest non-expired GitHub Actions artifact ZIP and extract it.

Used by ``sync-json-to-mma-ai``. Uses ``curl`` for the ZIP URL so Azure redirects work.

Selection order:

1. If ``artifact_name`` is ``run-bundle``, ``AUTO``, or legacy ``mma-json-exports`` (case-insensitive):
   skip exact match; pick the newest non-expired artifact whose name starts with any
   ``--fallback-prefix`` (default: ``weekly-refresh-``, ``monthly-retrain-``).
2. If env ``TRIGGERING_WORKFLOW_RUN_ID`` is set (``workflow_run`` sync), prefer bundles from that run.
3. Otherwise: prefer newest artifact whose **name equals** ``artifact_name``; if none, prefix fallback.

CLI::

    python scripts/ci_download_latest_artifact.py <artifact_name|run-bundle|AUTO> <extract_to_dir> [--fallback-prefix PREFIX ...]

Environment:

- ``GITHUB_REPOSITORY``, ``GITHUB_TOKEN`` — required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple


class _RedirectBlobSafe(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        joined = urllib.parse.urljoin(req.full_url, newurl.strip())
        return urllib.request.Request(
            joined,
            method="GET",
            headers={},
            origin_req_host=req.origin_req_host,
            unverifiable=True,
        )


def _download_artifact_zip(repo: str, aid: str, token: str, dest: Path) -> None:
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{aid}/zip"
    if shutil.which("curl"):
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-H",
                f"Authorization: Bearer {token}",
                "-H",
                "Accept: application/vnd.github+json",
                "-o",
                str(dest),
                url,
            ],
            check=True,
            timeout=600,
        )
        return

    zip_req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = urllib.request.build_opener(_RedirectBlobSafe())
    with opener.open(zip_req, timeout=600) as resp:
        dest.write_bytes(resp.read())


def _fetch_artifact_pages(repo: str, token: str, *, max_pages: int = 15) -> List[dict[str, Any]]:
    """List artifacts (newest first per page); concatenate pages."""
    hdr = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    all_rows: List[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        api = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100&page={page}"
        req = urllib.request.Request(api, headers=hdr)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        batch = data.get("artifacts") or []
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 100:
            break
    return all_rows


def _pick_artifact(
    rows: List[dict[str, Any]],
    exact_name: Optional[str],
    fallback_prefixes: List[str],
    *,
    workflow_run_id: Optional[int] = None,
) -> Tuple[Optional[int], Optional[str]]:
    alive = [a for a in rows if not a.get("expired")]
    alive.sort(key=lambda a: a["created_at"], reverse=True)

    if workflow_run_id is not None:
        for a in alive:
            wr = a.get("workflow_run") or {}
            if wr.get("id") != workflow_run_id:
                continue
            name = str(a.get("name") or "")
            if exact_name and name == exact_name:
                return int(a["id"]), name
            if any(name.startswith(p) for p in fallback_prefixes):
                return int(a["id"]), name

    if exact_name:
        for a in alive:
            if a.get("name") == exact_name:
                return int(a["id"]), str(a["name"])

    for a in alive:
        name = str(a.get("name") or "")
        if any(name.startswith(p) for p in fallback_prefixes):
            return int(a["id"]), name

    return None, None


def main() -> int:
    p = argparse.ArgumentParser(description="Download latest Actions artifact for JSON sync (ZIP extract).")
    p.add_argument(
        "artifact_name",
        help="Exact artifact name, or run-bundle / AUTO = newest weekly-refresh-* / monthly-retrain-*",
    )
    p.add_argument(
        "extract_to",
        type=Path,
        nargs="?",
        default=Path("_artifact_extract"),
        help="Directory to extract ZIP into",
    )
    p.add_argument(
        "--fallback-prefix",
        action="append",
        dest="fallback_prefixes",
        default=None,
        metavar="PREFIX",
        help="When exact name missing or artifact_name is run-bundle/AUTO: use newest artifact whose "
        "name starts with PREFIX (repeatable). Defaults: weekly-refresh- and monthly-retrain-",
    )
    args = p.parse_args()

    prefixes = args.fallback_prefixes
    if prefixes is None:
        prefixes = ["weekly-refresh-", "monthly-retrain-"]

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 1

    try:
        rows = _fetch_artifact_pages(repo, token)
    except urllib.error.HTTPError as e:
        print(f"::error::Listing artifacts failed: {e}", file=sys.stderr)
        return 1

    raw = args.artifact_name.strip()
    # Legacy workflows used ``mma-json-exports``; bundles are ``weekly-refresh-*`` / ``monthly-retrain-*``.
    if raw.casefold() in ("run-bundle", "auto", "mma-json-exports"):
        exact = None
    else:
        exact = raw

    trigger_run_id = os.environ.get("TRIGGERING_WORKFLOW_RUN_ID", "").strip()
    aid, picked_name = _pick_artifact(
        rows,
        exact,
        prefixes,
        workflow_run_id=int(trigger_run_id) if trigger_run_id.isdigit() else None,
    )
    if aid is None or picked_name is None:
        if exact is None:
            print(
                f"::error::No non-expired artifact matching prefixes {prefixes!r}. "
                "Run weekly/monthly CI first.",
                file=sys.stderr,
            )
        else:
            print(
                f"::error::No usable artifact: exact {exact!r} or prefixes {prefixes!r} "
                "(non-expired). Run weekly/monthly CI first.",
                file=sys.stderr,
            )
        return 1

    extract_to = Path(args.extract_to).resolve()
    extract_to.mkdir(parents=True, exist_ok=True)

    tmp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"artifact-{aid}.zip"
    try:
        _download_artifact_zip(repo, str(aid), token, tmp)
    except (urllib.error.HTTPError, OSError, subprocess.CalledProcessError) as e:
        print(f"::error::Download failed for artifact id {aid}: {e}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(tmp, "r") as zf:
        zf.extractall(extract_to)
    tmp.unlink(missing_ok=True)

    print(f"Extracted artifact id {aid} ({picked_name}) -> {extract_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
