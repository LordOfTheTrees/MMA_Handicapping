#!/usr/bin/env python3
"""
Download the newest non-expired GitHub Actions artifact ZIP by name and extract it.

Used by ``sync-json-to-mma-ai`` workflow. Uses ``curl`` for the ZIP URL so Azure redirects work.

Environment:

- ``GITHUB_REPOSITORY`` — ``owner/repo`` (Actions sets this)
- ``GITHUB_TOKEN`` — token with ``actions: read``

CLI::

    python scripts/ci_download_latest_artifact.py <artifact_name> <extract_to_dir>

Example::

    python scripts/ci_download_latest_artifact.py mma-json-exports ./_bundle
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


def main() -> int:
    p = argparse.ArgumentParser(description="Download latest Actions artifact by name (ZIP extract).")
    p.add_argument("artifact_name", help="Exact artifact name, e.g. mma-json-exports")
    p.add_argument(
        "extract_to",
        type=Path,
        nargs="?",
        default=Path("_artifact_extract"),
        help="Directory to extract ZIP contents into (created if missing)",
    )
    args = p.parse_args()

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 1

    name = args.artifact_name
    api = (
        "https://api.github.com/repos/"
        + repo
        + "/actions/artifacts?name="
        + urllib.parse.quote(name)
        + "&per_page=50"
    )
    req = urllib.request.Request(
        api,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"::error::Listing artifacts failed: {e}", file=sys.stderr)
        return 1

    arts = [a for a in data.get("artifacts", []) if not a.get("expired")]
    arts.sort(key=lambda a: a["created_at"], reverse=True)
    if not arts:
        print(f"::error::No non-expired artifact named {name!r}", file=sys.stderr)
        return 1

    aid = arts[0]["id"]
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

    print(f"Extracted artifact id {aid} ({name}) -> {extract_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
