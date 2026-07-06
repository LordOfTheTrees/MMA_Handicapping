#!/usr/bin/env python3
"""
Restore ``data/`` CSV/JSON from the newest non-expired weekly/monthly **run bundle**
artifact (``weekly-refresh-*`` or ``monthly-retrain-*``).

Used before a best-effort scrape in CI so a Cloudflare block does not leave an empty
``ufcstats_fights.csv``.

Environment: ``GITHUB_REPOSITORY``, ``GITHUB_TOKEN``, optional ``GITHUB_OUTPUT``.

Outputs (``GITHUB_OUTPUT``):

- ``data_restored`` — ``true`` / ``false``
- ``fight_rows`` — data rows in restored fights CSV (0 if none)
"""
from __future__ import annotations

import csv
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

BUNDLE_PREFIXES = ("weekly-refresh-", "monthly-retrain-")
FILES = (
    "data/ufcstats_fights.csv",
    "data/fighter_profiles.csv",
    "data/espn_crosswalk_fights.csv",
    "data/espn_crosswalk_fighters.csv",
    "data/espn_ingest_state.json",
    "data/espn_ingest_audit.json",
    "data/upcoming_cards.json",
    "data/espn_upcoming_cards.json",
)


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


def _fetch_artifacts(repo: str, token: str) -> List[dict[str, Any]]:
    hdr = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    rows: List[dict[str, Any]] = []
    for page in range(1, 16):
        api = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100&page={page}"
        req = urllib.request.Request(api, headers=hdr)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        batch = data.get("artifacts") or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
    return rows


def _pick_bundle(rows: List[dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    alive = [a for a in rows if not a.get("expired")]
    alive.sort(key=lambda a: a["created_at"], reverse=True)
    for a in alive:
        name = str(a.get("name") or "")
        if any(name.startswith(p) for p in BUNDLE_PREFIXES):
            return int(a["id"]), name
    return None, None


def _count_fight_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _write_output(data_restored: str, fight_rows: int) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        Path(out).write_text(
            f"data_restored={data_restored}\nfight_rows={fight_rows}\n",
            encoding="utf-8",
        )


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 1

    try:
        rows = _fetch_artifacts(repo, token)
    except urllib.error.HTTPError as e:
        print(f"::error::GitHub API listing artifacts failed: {e}", file=sys.stderr)
        return 1

    aid, name = _pick_bundle(rows)
    Path("data").mkdir(parents=True, exist_ok=True)
    fights_path = Path("data/ufcstats_fights.csv")

    if aid is None:
        _write_output("false", _count_fight_rows(fights_path))
        print("No weekly/monthly run bundle artifact; cold start for data CSVs.")
        return 0

    tmp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"data-bundle-{aid}.zip"
    try:
        _download_artifact_zip(repo, str(aid), token, tmp)
    except (urllib.error.HTTPError, OSError, subprocess.CalledProcessError) as e:
        print(f"::error::Failed to download artifact {aid} ({name}): {e}", file=sys.stderr)
        return 1

    restored_any = False
    with zipfile.ZipFile(tmp, "r") as zf:
        names = set(zf.namelist())
        for rel in FILES:
            if rel not in names:
                alt = rel.split("/", 1)[-1]
                candidates = [n for n in names if n.endswith(alt)]
                if not candidates:
                    continue
                rel = candidates[0]
            dest = Path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(rel))
            restored_any = True
            print(f"Restored {dest} from artifact {name} (id {aid})")

    tmp.unlink(missing_ok=True)
    fight_rows = _count_fight_rows(fights_path)
    if restored_any and fight_rows == 0:
        print(
            "::warning::Bundle restored but ufcstats_fights.csv has 0 data rows.",
            flush=True,
        )

    _write_output("true" if restored_any else "false", fight_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
