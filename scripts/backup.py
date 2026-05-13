#!/usr/bin/env python3
"""Create a restorable Fresh FA backup without media files from 01.Video."""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP_DIR = ROOT / "backups"
DB_PATH = ROOT / "fresh_fa.db"

INCLUDE_DIRS = (
    "02.Transcript",
    "03.Evaluation",
    "04.Meta",
    "config",
)
INCLUDE_FILES = (
    ".env",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _copy_sqlite_db(target: Path) -> None:
    if not DB_PATH.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30.0)
    try:
        dest = sqlite3.connect(str(target), timeout=30.0)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()


def _copy_dir(name: str, stage_root: Path) -> None:
    src = ROOT / name
    if not src.exists():
        return
    shutil.copytree(
        src,
        stage_root / name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )


def _copy_file(name: str, stage_root: Path) -> None:
    src = ROOT / name
    if src.is_file():
        shutil.copy2(src, stage_root / name)


def _write_manifest(stage_root: Path, archive_name: str) -> None:
    manifest = stage_root / "BACKUP_MANIFEST.txt"
    lines = [
        f"archive={archive_name}",
        f"created_at={datetime.now(timezone.utc).isoformat()}",
        f"project_root={ROOT}",
        "includes=fresh_fa.db,.env,02.Transcript,03.Evaluation,04.Meta,config",
        "excludes=01.Video",
        "",
        "Restore outline:",
        "1. Stop the Fresh FA server.",
        "2. Extract this archive into the project root.",
        "3. Start the Fresh FA server.",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune_old_backups(backup_dir: Path, keep_days: int) -> list[Path]:
    if keep_days <= 0 or not backup_dir.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    deleted: list[Path] = []
    for path in backup_dir.glob("fresh-fa-*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        path.unlink()
        deleted.append(path)
    return deleted


def create_backup(backup_dir: Path, keep_days: int) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.chmod(0o700)
    archive_name = f"fresh-fa-{_timestamp()}.tar.gz"
    archive_path = backup_dir / archive_name

    with tempfile.TemporaryDirectory(prefix="fresh-fa-backup-") as tmp:
        stage_root = Path(tmp) / "Fresh-FA"
        stage_root.mkdir()
        _copy_sqlite_db(stage_root / "fresh_fa.db")
        for name in INCLUDE_FILES:
            _copy_file(name, stage_root)
        for name in INCLUDE_DIRS:
            _copy_dir(name, stage_root)
        _write_manifest(stage_root, archive_name)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(stage_root, arcname="Fresh-FA")
    archive_path.chmod(0o600)

    _prune_old_backups(backup_dir, keep_days)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup Fresh FA without 01.Video media files.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(os.environ.get("FRESH_FA_BACKUP_DIR", DEFAULT_BACKUP_DIR)),
        help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})",
    )
    parser.add_argument(
        "--keep-days",
        type=int,
        default=int(os.environ.get("FRESH_FA_BACKUP_KEEP_DAYS", "30")),
        help="Delete fresh-fa-*.tar.gz backups older than this many days; 0 disables pruning.",
    )
    args = parser.parse_args()
    archive = create_backup(args.backup_dir.expanduser().resolve(), max(0, args.keep_days))
    print(archive)


if __name__ == "__main__":
    main()
