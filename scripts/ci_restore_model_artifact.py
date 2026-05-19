#!/usr/bin/env python3
"""
Restore ``data/model.pkl`` from the newest non-expired GitHub Actions artifact named
``mma-model-state`` (same repository). Used by scheduled CI to forward model state
without external blob storage.

Environment (GitHub Actions provides these automatically):

- ``GITHUB_REPOSITORY`` — ``owner/repo``
- ``GITHUB_TOKEN`` — workflow token (needs ``actions: read``)
- ``GITHUB_OUTPUT`` — step output file for ``seed=true|false``

Exit codes: ``0`` — restored or cold-start (no artifact); ``1`` — download/unpack error.

Implementation note: GitHub returns a 302 to Azure Blob Storage for artifact ZIPs.
``urllib`` forwarding GitHub headers (or ``Authorization``) to Azure often yields 401/400.
We download via ``curl`` when available (drops ``Authorization`` on cross-host redirects),
else a redirect handler that sends **no** custom headers to the redirect target.
"""
from __future__ import annotations

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


ARTIFACT_NAME = "mma-model-state"


class _RedirectBlobSafe(urllib.request.HTTPRedirectHandler):
    """Follow redirects with a plain GET; no GitHub/Azure-conflicting headers."""

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


def _write_output(seed: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        Path(path).write_text(f"seed={seed}\n", encoding="utf-8")


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 1

    api = (
        "https://api.github.com/repos/"
        + repo
        + "/actions/artifacts?name="
        + urllib.parse.quote(ARTIFACT_NAME)
        + "&per_page=30"
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
        print(f"::error::GitHub API listing artifacts failed: {e}", file=sys.stderr)
        return 1

    arts = [a for a in data.get("artifacts", []) if not a.get("expired")]
    arts.sort(key=lambda a: a["created_at"], reverse=True)

    Path("data").mkdir(parents=True, exist_ok=True)
    if not arts:
        _write_output("false")
        print("No mma-model-state artifact; cold start.")
        return 0

    aid = arts[0]["id"]
    tmp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "mma-model-state.zip"
    try:
        _download_artifact_zip(repo, str(aid), token, tmp)
    except (urllib.error.HTTPError, OSError, subprocess.CalledProcessError) as e:
        print(f"::error::Failed to download artifact {aid}: {e}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(tmp, "r") as zf:
        zf.extractall(".")
    tmp.unlink(missing_ok=True)

    pkl = Path("data/model.pkl")
    if not pkl.is_file():
        loose = Path("model.pkl")
        if loose.is_file():
            loose.rename(pkl)
    if not pkl.is_file():
        print("::error::Artifact zip did not contain data/model.pkl", file=sys.stderr)
        return 1

    _write_output("true")
    print(f"Restored data/model.pkl from artifact id {aid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
