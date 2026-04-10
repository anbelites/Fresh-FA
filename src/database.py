"""SQLite database for video metadata, evaluations index, and job state."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "fresh_fa.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    stem TEXT PRIMARY KEY,
    filename TEXT,
    manager_id TEXT,
    manager_name TEXT,
    location_id TEXT,
    location_name TEXT,
    interaction_date TEXT,
    tags TEXT DEFAULT '[]',
    uploaded_at TEXT,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS managers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    stem TEXT,
    kind TEXT,
    status TEXT DEFAULT 'queued',
    stage TEXT DEFAULT 'queued',
    video_file TEXT,
    error TEXT,
    transcript TEXT,
    evaluation TEXT,
    tone_file TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_stem ON jobs(stem);
CREATE INDEX IF NOT EXISTS idx_videos_manager ON videos(manager_id);
CREATE INDEX IF NOT EXISTS idx_videos_location ON videos(location_id);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


class DB:
    """Thin synchronous wrapper around SQLite for use from server threads."""

    def __init__(self) -> None:
        self._conn = get_connection()

    def close(self) -> None:
        self._conn.close()

    # --- managers ---
    def list_managers(self) -> list[dict[str, str]]:
        rows = self._conn.execute("SELECT id, name FROM managers ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def add_manager(self, mid: str, name: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO managers (id, name) VALUES (?, ?)", (mid, name)
        )
        self._conn.commit()

    def delete_manager(self, mid: str) -> bool:
        cur = self._conn.execute("DELETE FROM managers WHERE id = ?", (mid,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- locations ---
    def list_locations(self) -> list[dict[str, str]]:
        rows = self._conn.execute("SELECT id, name FROM locations ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def add_location(self, lid: str, name: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO locations (id, name) VALUES (?, ?)", (lid, name)
        )
        self._conn.commit()

    def delete_location(self, lid: str) -> bool:
        cur = self._conn.execute("DELETE FROM locations WHERE id = ?", (lid,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- video meta ---
    def get_video_meta(self, stem: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM videos WHERE stem = ?", (stem,)).fetchone()
        if not row:
            return {}
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        return d

    def upsert_video_meta(self, stem: str, **kw: Any) -> dict[str, Any]:
        existing = self.get_video_meta(stem)
        tags = kw.get("tags", existing.get("tags", []))
        if isinstance(tags, list):
            tags_json = json.dumps(tags, ensure_ascii=False)
        else:
            tags_json = "[]"

        self._conn.execute(
            """INSERT INTO videos (stem, filename, manager_id, manager_name, location_id,
               location_name, interaction_date, tags, uploaded_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(stem) DO UPDATE SET
               manager_id=excluded.manager_id, manager_name=excluded.manager_name,
               location_id=excluded.location_id, location_name=excluded.location_name,
               interaction_date=excluded.interaction_date, tags=excluded.tags,
               status=COALESCE(excluded.status, videos.status)""",
            (
                stem,
                kw.get("filename", existing.get("filename")),
                kw.get("manager_id", existing.get("manager_id")),
                kw.get("manager_name", existing.get("manager_name")),
                kw.get("location_id", existing.get("location_id")),
                kw.get("location_name", existing.get("location_name")),
                kw.get("interaction_date", existing.get("interaction_date")),
                tags_json,
                kw.get("uploaded_at", existing.get("uploaded_at", _utc_now())),
                kw.get("status", existing.get("status", "pending")),
            ),
        )
        self._conn.commit()
        return self.get_video_meta(stem)

    def delete_video(self, stem: str) -> bool:
        cur = self._conn.execute("DELETE FROM videos WHERE stem = ?", (stem,))
        self._conn.execute("DELETE FROM jobs WHERE stem = ?", (stem,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- jobs ---
    def upsert_job(self, job_id: str, **kw: Any) -> None:
        existing = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if existing:
            sets = []
            vals = []
            for k, v in kw.items():
                if v is not None:
                    sets.append(f"{k} = ?")
                    vals.append(v)
            if sets:
                sets.append("updated_at = ?")
                vals.append(_utc_now())
                vals.append(job_id)
                self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
        else:
            kw.setdefault("created_at", _utc_now())
            kw.setdefault("updated_at", _utc_now())
            cols = ["id"] + list(kw.keys())
            placeholders = ", ".join(["?"] * len(cols))
            self._conn.execute(
                f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({placeholders})",
                [job_id] + list(kw.values()),
            )
        self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def latest_job_by_stem(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM jobs WHERE id IN (
                SELECT id FROM jobs GROUP BY stem
                HAVING updated_at = MAX(updated_at)
            ) ORDER BY updated_at DESC"""
        ).fetchall()
        return {r["stem"]: dict(r) for r in rows if r["stem"]}

    def delete_jobs_for_stem(self, stem: str) -> None:
        self._conn.execute("DELETE FROM jobs WHERE stem = ?", (stem,))
        self._conn.commit()

    # --- migration: import from YAML config files ---
    def import_managers_from_yaml(self, items: list[dict[str, str]]) -> None:
        for item in items:
            self.add_manager(item["id"], item["name"])

    def import_locations_from_yaml(self, items: list[dict[str, str]]) -> None:
        for item in items:
            self.add_location(item["id"], item["name"])
