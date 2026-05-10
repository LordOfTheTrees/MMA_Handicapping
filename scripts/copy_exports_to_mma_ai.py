#!/usr/bin/env python3
"""
Copy exported ``*.json`` from this repo (e.g. ``JSON_exports/``) into the sibling ``mma.ai/artifacts/``.

Run from repo root::

    python scripts/copy_exports_to_mma_ai.py
    python scripts/copy_exports_to_mma_ai.py --src JSON_exports --dest ../mma.ai/artifacts
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_mma_ai_artifacts_dir() -> Path:
    """``<repo>/../mma.ai/artifacts`` (resolved)."""
    return (REPO_ROOT.parent / "mma.ai" / "artifacts").resolve()


def copy_json_from_dir(src_dir: Path, dest_dir: Path) -> list[Path]:
    """Copy every ``*.json`` under *src_dir* into *dest_dir* (creates *dest_dir*)."""
    src_dir = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source is not a directory: {src_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for p in sorted(src_dir.glob("*.json")):
        dst = dest_dir / p.name
        shutil.copy2(p, dst)
        out.append(dst)
    return out


def copy_json_file(src_file: Path, dest_dir: Path) -> Path:
    """Copy a single JSON file into *dest_dir* keeping its basename."""
    src_file = Path(src_file).resolve()
    dest_dir = Path(dest_dir).resolve()
    if not src_file.is_file():
        raise FileNotFoundError(f"Not a file: {src_file}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src_file.name
    shutil.copy2(src_file, dst)
    return dst


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Copy JSON_exports (or folder) to mma.ai/artifacts.")
    p.add_argument(
        "--src",
        type=Path,
        default=REPO_ROOT / "JSON_exports",
        help=f"Directory of JSON snapshots (default: {REPO_ROOT / 'JSON_exports'})",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Deploy artifacts dir (default: {default_mma_ai_artifacts_dir()})",
    )
    args = p.parse_args(argv)

    dest = args.dest.resolve() if args.dest else default_mma_ai_artifacts_dir()
    copied = copy_json_from_dir(args.src, dest)
    if not copied:
        print(f"No *.json files in {args.src.resolve()}", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"Copied {len(copied)} file(s) to {dest}", flush=True)
    for c in copied:
        print(f"  {c.name}", flush=True)


if __name__ == "__main__":
    main()
