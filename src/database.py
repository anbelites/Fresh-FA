"""SQLite database for video metadata, evaluations index, and job state."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIG_DIR, DEFAULT_CHECKLIST_SLUG, PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "fresh_fa.db"

ACTIVE_CHECKLIST_KEY = "active_checklist_slug"

# Не импортировать в чеклисты (справочники и проч.)
_CONFIG_YAML_EXCLUDE = frozenset(
    {
        "managers.yaml",
        "managers.yml",
        "locations.yaml",
        "locations.yml",
    }
)

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

CREATE TABLE IF NOT EXISTS checklists (
    slug TEXT PRIMARY KEY,
    version TEXT NOT NULL DEFAULT '1',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checklist_criteria (
    checklist_slug TEXT NOT NULL REFERENCES checklists(slug) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL,
    criterion_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (checklist_slug, criterion_id)
);

CREATE INDEX IF NOT EXISTS idx_checklist_criteria_slug ON checklist_criteria(checklist_slug);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_videos_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(videos)").fetchall()
    cols = {str(r[1]) for r in rows}
    if "display_title" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN display_title TEXT")
        conn.commit()


def _migrate_jobs_stream_log(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(jobs)").fetchall()
    cols = {str(r[1]) for r in rows}
    if "stream_log" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN stream_log TEXT")
        conn.commit()


def _migrate_checklist_slugs_strip_extensions(conn: sqlite3.Connection) -> None:
    """Переименовать slug вида foo.yaml → foo (однократно, идемпотентно)."""
    rows = conn.execute(
        """
        SELECT slug FROM checklists
        WHERE lower(slug) LIKE '%.yaml' OR lower(slug) LIKE '%.yml'
        """
    ).fetchall()
    for row in rows:
        old = str(row["slug"])
        new = Path(old).stem
        if not new or new == old:
            continue
        clash = conn.execute("SELECT 1 FROM checklists WHERE slug = ?", (new,)).fetchone()
        if clash:
            continue
        meta = conn.execute(
            "SELECT version, updated_at FROM checklists WHERE slug = ?",
            (old,),
        ).fetchone()
        if not meta:
            continue
        conn.execute(
            """INSERT INTO checklists (slug, version, updated_at) VALUES (?, ?, ?)""",
            (new, meta["version"], meta["updated_at"]),
        )
        conn.execute(
            "UPDATE checklist_criteria SET checklist_slug = ? WHERE checklist_slug = ?",
            (new, old),
        )
        conn.execute(
            "UPDATE app_settings SET value = ? WHERE key = ? AND value = ?",
            (new, ACTIVE_CHECKLIST_KEY, old),
        )
        conn.execute("DELETE FROM checklists WHERE slug = ?", (old,))
    conn.commit()


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA)
    _migrate_videos_columns(conn)
    _migrate_jobs_stream_log(conn)
    _migrate_checklist_slugs_strip_extensions(conn)
    conn.commit()
    conn.close()


class DB:
    """Thin synchronous wrapper around SQLite for use from server threads."""

    def __init__(self) -> None:
        self._conn = get_connection()
        _migrate_videos_columns(self._conn)
        _migrate_jobs_stream_log(self._conn)
        _migrate_checklist_slugs_strip_extensions(self._conn)

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
               location_name, interaction_date, tags, uploaded_at, status, display_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(stem) DO UPDATE SET
               manager_id=excluded.manager_id, manager_name=excluded.manager_name,
               location_id=excluded.location_id, location_name=excluded.location_name,
               interaction_date=excluded.interaction_date, tags=excluded.tags,
               display_title=excluded.display_title,
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
                kw.get("display_title", existing.get("display_title")),
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

    # --- checklists ---
    def checklist_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM checklists").fetchone()
        return int(row[0]) if row else 0

    def list_checklist_files(self) -> list[dict[str, str]]:
        rows = self._conn.execute("SELECT slug FROM checklists ORDER BY slug COLLATE NOCASE").fetchall()
        return [{"name": r["slug"]} for r in rows]

    def checklist_exists(self, slug: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM checklists WHERE slug = ? LIMIT 1", (slug,)
        ).fetchone()
        return row is not None

    def get_setting(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_active_checklist_slug(self) -> str:
        raw = self.get_setting(ACTIVE_CHECKLIST_KEY)
        if raw and self.checklist_exists(raw.strip()):
            return raw.strip()
        rows = self._conn.execute("SELECT slug FROM checklists ORDER BY slug COLLATE NOCASE LIMIT 1").fetchall()
        if rows:
            slug = str(rows[0]["slug"])
            self.set_setting(ACTIVE_CHECKLIST_KEY, slug)
            return slug
        return DEFAULT_CHECKLIST_SLUG

    def set_active_checklist_slug(self, slug: str) -> None:
        if not self.checklist_exists(slug):
            raise ValueError(f"Чеклист не найден: {slug}")
        self.set_setting(ACTIVE_CHECKLIST_KEY, slug)

    def get_checklist_content(self, slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT slug, version FROM checklists WHERE slug = ?", (slug,)
        ).fetchone()
        if not row:
            return None
        crit_rows = self._conn.execute(
            "SELECT criterion_id, name, description FROM checklist_criteria WHERE checklist_slug = ? ORDER BY sort_order, criterion_id",
            (slug,),
        ).fetchall()
        criteria = [
            {
                "id": r["criterion_id"],
                "name": r["name"],
                "description": r["description"] or "",
            }
            for r in crit_rows
        ]
        n = self.checklist_count()
        return {
            "filename": row["slug"],
            "version": row["version"],
            "criteria": criteria,
            "can_delete": n > 1,
        }

    def replace_checklist(
        self,
        slug: str,
        version: str,
        criteria: list[dict[str, Any]],
    ) -> None:
        now = _utc_now()
        self._conn.execute("DELETE FROM checklist_criteria WHERE checklist_slug = ?", (slug,))
        self._conn.execute(
            """INSERT INTO checklists (slug, version, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at""",
            (slug, version, now),
        )
        for i, c in enumerate(criteria):
            cid = str(c.get("id", "")).strip()
            if not cid:
                continue
            self._conn.execute(
                """INSERT INTO checklist_criteria (checklist_slug, sort_order, criterion_id, name, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    slug,
                    i,
                    cid,
                    str(c.get("name", cid)).strip(),
                    str(c.get("description", "") or "").strip(),
                ),
            )
        self._conn.commit()

    def insert_checklist(
        self,
        slug: str,
        version: str,
        criteria: list[dict[str, Any]],
    ) -> None:
        if self.checklist_exists(slug):
            raise ValueError(f"Чеклист уже есть: {slug}")
        now = _utc_now()
        self._conn.execute(
            "INSERT INTO checklists (slug, version, updated_at) VALUES (?, ?, ?)",
            (slug, version, now),
        )
        for i, c in enumerate(criteria):
            cid = str(c.get("id", "")).strip()
            if not cid:
                continue
            self._conn.execute(
                """INSERT INTO checklist_criteria (checklist_slug, sort_order, criterion_id, name, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    slug,
                    i,
                    cid,
                    str(c.get("name", cid)).strip(),
                    str(c.get("description", "") or "").strip(),
                ),
            )
        self._conn.commit()

    def delete_checklist(self, slug: str) -> bool:
        cur = self._conn.execute("DELETE FROM checklists WHERE slug = ?", (slug,))
        self._conn.commit()
        if cur.rowcount and self.get_setting(ACTIVE_CHECKLIST_KEY) == slug:
            self._conn.execute("DELETE FROM app_settings WHERE key = ?", (ACTIVE_CHECKLIST_KEY,))
            self._conn.commit()
        return cur.rowcount > 0

    # --- migration: import from YAML config files ---
    def import_managers_from_yaml(self, items: list[dict[str, str]]) -> None:
        for item in items:
            self.add_manager(item["id"], item["name"])

    def import_locations_from_yaml(self, items: list[dict[str, str]]) -> None:
        for item in items:
            self.add_location(item["id"], item["name"])


def migrate_checklists_from_config(
    db: DB,
    *,
    config_dir: Path | None = None,
    active_marker: Path | None = None,
    delete_source_files: bool = True,
) -> None:
    """Импорт YAML из config/ в БД при пустой таблице checklists; затем удаление исходных файлов."""
    if db.checklist_count() > 0:
        return

    cd = config_dir or CONFIG_DIR
    marker = active_marker or (cd / ".active_criteria")
    cd.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for pat in ("*.yaml", "*.yml"):
        for p in sorted(cd.glob(pat)):
            if not p.is_file():
                continue
            if p.name in _CONFIG_YAML_EXCLUDE:
                continue
            paths.append(p)

    seen: set[str] = set()
    unique_paths: list[Path] = []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            unique_paths.append(p)

    imported_source_files: list[Path] = []
    for p in unique_paths:
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeError):
            continue
        if not isinstance(raw, dict):
            continue
        version = str(raw.get("version", "1"))
        crits: list[dict[str, Any]] = []
        for row in raw.get("criteria") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id", "")).strip()
            if not cid:
                continue
            crits.append(
                {
                    "id": cid,
                    "name": str(row.get("name", cid)).strip(),
                    "description": str(row.get("description", "")).strip(),
                }
            )
        if not crits:
            crits = [
                {
                    "id": "criterion_1",
                    "name": "Новый критерий",
                    "description": "Опишите, что проверяет ИИ по этому пункту.",
                }
            ]
        slug = p.stem
        try:
            db.insert_checklist(slug, version, crits)
            imported_source_files.append(p)
        except (ValueError, sqlite3.Error):
            continue

    if db.checklist_count() == 0:
        db.insert_checklist(
            DEFAULT_CHECKLIST_SLUG,
            "1",
            [
                {
                    "id": "criterion_1",
                    "name": "Новый критерий",
                    "description": "Опишите, что проверяет ИИ по этому пункту.",
                }
            ],
        )

    active: str | None = None
    if marker.is_file():
        try:
            raw_a = marker.read_text(encoding="utf-8").strip()
            if raw_a:
                candidates: list[str] = []
                if raw_a.lower().endswith((".yaml", ".yml")):
                    candidates.append(Path(raw_a).stem)
                candidates.append(raw_a)
                for cand in candidates:
                    if cand and db.checklist_exists(cand):
                        active = cand
                        break
        except OSError:
            pass
    if not active and db.checklist_exists(DEFAULT_CHECKLIST_SLUG):
        active = DEFAULT_CHECKLIST_SLUG
    if not active:
        files = db.list_checklist_files()
        if files:
            active = files[0]["name"]
    if active and db.checklist_exists(active):
        db.set_active_checklist_slug(active)
    else:
        fl = db.list_checklist_files()
        if fl:
            db.set_active_checklist_slug(fl[0]["name"])

    if delete_source_files and imported_source_files:
        for fp in imported_source_files:
            if fp.is_file():
                try:
                    fp.unlink()
                except OSError:
                    pass
        if marker.is_file():
            try:
                marker.unlink()
            except OSError:
                pass
