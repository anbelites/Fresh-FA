"""SQLite database for metadata, auth, checklists, and job state."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIG_DIR, DEFAULT_CHECKLIST_SLUG, PROJECT_ROOT
from src.seed_data import build_seed_criteria

DB_PATH = PROJECT_ROOT / "fresh_fa.db"

ACTIVE_CHECKLIST_KEY = "active_checklist_slug"
USER_ROLE_USER = "user"
USER_ROLE_ADMIN = "admin"
AUTH_SOURCE_LOCAL = "local"
AUTH_SOURCE_AD = "ad"
_UNSET = object()

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
    display_title TEXT,
    manager_id TEXT,
    manager_name TEXT,
    location_id TEXT,
    location_name TEXT,
    interaction_date TEXT,
    training_type_slug TEXT,
    expected_speaker_count INTEGER NOT NULL DEFAULT 2,
    tags TEXT DEFAULT '[]',
    uploaded_at TEXT,
    status TEXT DEFAULT 'pending',
    uploaded_by TEXT,
    uploaded_by_name TEXT,
    uploader_location_id TEXT,
    uploader_location_name TEXT,
    uploader_location_crm_id TEXT,
    training_type_name_snapshot TEXT,
    checklist_slug_snapshot TEXT,
    checklist_display_name_snapshot TEXT,
    checklist_version_snapshot TEXT,
    checklist_department_snapshot TEXT,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    file_sha256 TEXT,
    dedupe_key TEXT,
    delete_requested_at TEXT,
    delete_requested_by TEXT,
    deleted_at TEXT,
    deleted_by TEXT
);

CREATE TABLE IF NOT EXISTS managers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    crm_name TEXT,
    crm_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    stem TEXT,
    kind TEXT,
    status TEXT DEFAULT 'queued',
    stage TEXT DEFAULT 'queued',
    args_json TEXT,
    video_file TEXT,
    error TEXT,
    transcript TEXT,
    evaluation TEXT,
    tone_file TEXT,
    stream_log TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_stem ON jobs(stem);
CREATE INDEX IF NOT EXISTS idx_videos_manager ON videos(manager_id);
CREATE INDEX IF NOT EXISTS idx_videos_location ON videos(location_id);

CREATE TABLE IF NOT EXISTS checklists (
    slug TEXT PRIMARY KEY,
    display_name TEXT,
    department TEXT,
    version TEXT NOT NULL DEFAULT '1',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checklist_criteria (
    checklist_slug TEXT NOT NULL REFERENCES checklists(slug) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL,
    criterion_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    weight INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (checklist_slug, criterion_id)
);

CREATE INDEX IF NOT EXISTS idx_checklist_criteria_slug ON checklist_criteria(checklist_slug);

CREATE TABLE IF NOT EXISTS training_types (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    checklist_slug TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_types_checklist ON training_types(checklist_slug);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL DEFAULT '',
    display_name TEXT,
    full_name TEXT,
    auth_source TEXT NOT NULL DEFAULT 'local',
    location_id TEXT,
    department TEXT,
    profile_completed_at TEXT,
    onboarding_version INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT 'user',
    daily_upload_limit INTEGER,
    max_queued_jobs INTEGER,
    max_running_jobs INTEGER,
    quotas_updated_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_eval_states (
    stem TEXT NOT NULL,
    criteria_slug TEXT NOT NULL,
    draft_saved_at TEXT,
    published_at TEXT,
    published_by TEXT,
    compared_at TEXT,
    compared_by TEXT,
    PRIMARY KEY (stem, criteria_slug)
);

CREATE TABLE IF NOT EXISTS evaluation_comparisons (
    stem TEXT NOT NULL,
    criteria_slug TEXT NOT NULL,
    ai_overall REAL,
    human_overall REAL,
    diff_percent REAL,
    status_color TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (stem, criteria_slug)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    cols = _table_columns(conn, table)
    if column in cols:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    return True


def _normalize_weight(value: Any) -> int:
    try:
        weight = int(value)
    except (TypeError, ValueError):
        weight = 1
    return max(1, min(weight, 100000))


def _normalize_user_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return USER_ROLE_ADMIN if role == USER_ROLE_ADMIN else USER_ROLE_USER


def _normalize_auth_source(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return AUTH_SOURCE_AD if raw == AUTH_SOURCE_AD else AUTH_SOURCE_LOCAL


def _normalize_optional_limit(value: Any) -> int | None:
    if value is _UNSET or value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        out = int(raw)
    except (TypeError, ValueError):
        return None
    return max(0, min(out, 1_000_000))


def _normalize_onboarding_version(value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = 0
    return max(0, min(out, 1_000_000))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_videos_columns(conn: sqlite3.Connection) -> None:
    changed = False
    changed |= _ensure_column(conn, "videos", "display_title", "TEXT")
    changed |= _ensure_column(conn, "videos", "training_type_slug", "TEXT")
    changed |= _ensure_column(conn, "videos", "expected_speaker_count", "INTEGER NOT NULL DEFAULT 2")
    changed |= _ensure_column(conn, "videos", "uploaded_by", "TEXT")
    changed |= _ensure_column(conn, "videos", "uploaded_by_name", "TEXT")
    changed |= _ensure_column(conn, "videos", "uploader_location_id", "TEXT")
    changed |= _ensure_column(conn, "videos", "uploader_location_name", "TEXT")
    changed |= _ensure_column(conn, "videos", "uploader_location_crm_id", "TEXT")
    changed |= _ensure_column(conn, "videos", "training_type_name_snapshot", "TEXT")
    changed |= _ensure_column(conn, "videos", "checklist_slug_snapshot", "TEXT")
    changed |= _ensure_column(conn, "videos", "checklist_display_name_snapshot", "TEXT")
    changed |= _ensure_column(conn, "videos", "checklist_version_snapshot", "TEXT")
    changed |= _ensure_column(conn, "videos", "checklist_department_snapshot", "TEXT")
    changed |= _ensure_column(conn, "videos", "file_size_bytes", "INTEGER NOT NULL DEFAULT 0")
    changed |= _ensure_column(conn, "videos", "file_sha256", "TEXT")
    changed |= _ensure_column(conn, "videos", "dedupe_key", "TEXT")
    changed |= _ensure_column(conn, "videos", "delete_requested_at", "TEXT")
    changed |= _ensure_column(conn, "videos", "delete_requested_by", "TEXT")
    changed |= _ensure_column(conn, "videos", "deleted_at", "TEXT")
    changed |= _ensure_column(conn, "videos", "deleted_by", "TEXT")
    if changed:
        conn.commit()


def _migrate_jobs_stream_log(conn: sqlite3.Connection) -> None:
    changed = False
    changed |= _ensure_column(conn, "jobs", "stream_log", "TEXT")
    changed |= _ensure_column(conn, "jobs", "args_json", "TEXT")
    if changed:
        conn.commit()


def _migrate_checklist_criteria_weight(conn: sqlite3.Connection) -> None:
    if _ensure_column(conn, "checklist_criteria", "weight", "INTEGER NOT NULL DEFAULT 1"):
        conn.commit()


def _migrate_locations_columns(conn: sqlite3.Connection) -> None:
    changed = False
    changed |= _ensure_column(conn, "locations", "crm_name", "TEXT")
    changed |= _ensure_column(conn, "locations", "crm_id", "TEXT")
    changed |= _ensure_column(conn, "locations", "is_active", "INTEGER NOT NULL DEFAULT 1")
    if changed:
        conn.commit()


def _migrate_users_columns(conn: sqlite3.Connection) -> None:
    changed = False
    changed |= _ensure_column(conn, "users", "full_name", "TEXT")
    changed |= _ensure_column(conn, "users", "auth_source", "TEXT NOT NULL DEFAULT 'local'")
    changed |= _ensure_column(conn, "users", "location_id", "TEXT")
    changed |= _ensure_column(conn, "users", "department", "TEXT")
    changed |= _ensure_column(conn, "users", "profile_completed_at", "TEXT")
    changed |= _ensure_column(conn, "users", "onboarding_version", "INTEGER NOT NULL DEFAULT 0")
    changed |= _ensure_column(conn, "users", "daily_upload_limit", "INTEGER")
    changed |= _ensure_column(conn, "users", "max_queued_jobs", "INTEGER")
    changed |= _ensure_column(conn, "users", "max_running_jobs", "INTEGER")
    changed |= _ensure_column(conn, "users", "quotas_updated_at", "TEXT")
    if changed:
        conn.commit()


def _migrate_training_type_columns(conn: sqlite3.Connection) -> None:
    if _ensure_column(conn, "training_types", "department", "TEXT"):
        conn.commit()


def _migrate_checklist_columns(conn: sqlite3.Connection) -> None:
    changed = False
    changed |= _ensure_column(conn, "checklists", "display_name", "TEXT")
    changed |= _ensure_column(conn, "checklists", "department", "TEXT")
    if changed:
        conn.commit()


def _migrate_checklist_display_names(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE checklists
        SET display_name = COALESCE(NULLIF(trim(display_name), ''), slug)
        WHERE display_name IS NULL OR trim(display_name) = ''
        """
    )
    conn.commit()


def _migrate_user_display_names(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE users
        SET full_name = COALESCE(NULLIF(trim(full_name), ''), NULLIF(trim(display_name), ''))
        WHERE full_name IS NULL OR trim(full_name) = ''
        """
    )
    conn.execute(
        """
        UPDATE users
        SET profile_completed_at = COALESCE(profile_completed_at, updated_at)
        WHERE NULLIF(trim(full_name), '') IS NOT NULL
        """
    )
    conn.commit()


def _ensure_seed_manager_sync(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO managers (id, name)
        SELECT username, COALESCE(NULLIF(trim(full_name), ''), NULLIF(trim(display_name), ''), username)
        FROM users
        WHERE NULLIF(trim(username), '') IS NOT NULL
        """
    )
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
            "UPDATE training_types SET checklist_slug = ? WHERE checklist_slug = ?",
            (new, old),
        )
        conn.execute(
            "UPDATE app_settings SET value = ? WHERE key = ? AND value = ?",
            (new, ACTIVE_CHECKLIST_KEY, old),
        )
        conn.execute("DELETE FROM checklists WHERE slug = ?", (old,))
    conn.commit()


def _ensure_post_migration_indexes(conn: sqlite3.Connection) -> None:
    video_cols = _table_columns(conn, "videos")
    if "training_type_slug" in video_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_training_type ON videos(training_type_slug)")
    if "uploaded_by" in video_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_uploaded_by ON videos(uploaded_by)")
    if "uploaded_by" in video_cols and "uploaded_at" in video_cols:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_videos_uploaded_by_uploaded_at ON videos(uploaded_by, uploaded_at)"
        )
    if "deleted_at" in video_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_deleted_at ON videos(deleted_at)")
    if "file_sha256" in video_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_file_sha256 ON videos(file_sha256)")
    job_cols = _table_columns(conn, "jobs")
    if "status" in job_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    if "created_at" in job_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
    if "kind" in job_cols and "status" in job_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status)")
    location_cols = _table_columns(conn, "locations")
    if "crm_id" in location_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_crm_id ON locations(crm_id)")
    user_cols = _table_columns(conn, "users")
    if "location_id" in user_cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_location_id ON users(location_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_comparisons_status ON evaluation_comparisons(status_color)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_comparisons_updated ON evaluation_comparisons(updated_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)")


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_videos_columns(conn)
    _migrate_jobs_stream_log(conn)
    _migrate_checklist_criteria_weight(conn)
    _migrate_locations_columns(conn)
    _migrate_users_columns(conn)
    _migrate_training_type_columns(conn)
    _migrate_checklist_columns(conn)
    _migrate_checklist_slugs_strip_extensions(conn)
    _migrate_checklist_display_names(conn)
    _migrate_user_display_names(conn)
    _ensure_seed_manager_sync(conn)
    _ensure_post_migration_indexes(conn)
    conn.commit()


def init_db() -> None:
    conn = get_connection()
    _run_schema_migrations(conn)
    conn.close()


class DB:
    """Thin synchronous wrapper around SQLite for use from server threads."""

    def __init__(self) -> None:
        self._conn = get_connection()
        _run_schema_migrations(self._conn)

    def close(self) -> None:
        self._conn.close()

    # --- managers ---
    def list_managers(self) -> list[dict[str, str]]:
        rows = self._conn.execute("SELECT id, name FROM managers ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def add_manager(self, mid: str, name: str) -> None:
        self._conn.execute(
            """
            INSERT INTO managers (id, name)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name
            """,
            (mid, name),
        )
        self._conn.commit()

    def delete_manager(self, mid: str) -> bool:
        cur = self._conn.execute("DELETE FROM managers WHERE id = ?", (mid,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- locations ---
    def list_locations(self) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """
            SELECT id, name, crm_name, crm_id, is_active
            FROM locations
            ORDER BY name COLLATE NOCASE, id COLLATE NOCASE
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_location(self, lid: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, name, crm_name, crm_id, is_active FROM locations WHERE id = ?",
            (lid,),
        ).fetchone()
        return dict(row) if row else None

    def add_location(
        self,
        lid: str,
        name: str,
        *,
        crm_name: str | None = None,
        crm_id: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        existing = self._conn.execute(
            """
            SELECT id
            FROM locations
            WHERE id = ?
               OR (? IS NOT NULL AND crm_id = ?)
               OR lower(name) = lower(?)
            LIMIT 1
            """,
            (lid, crm_id, crm_id, name),
        ).fetchone()
        target_id = str(existing["id"]) if existing else lid
        self._conn.execute(
            """
            INSERT INTO locations (id, name, crm_name, crm_id, is_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                crm_name = COALESCE(excluded.crm_name, locations.crm_name),
                crm_id = COALESCE(excluded.crm_id, locations.crm_id),
                is_active = excluded.is_active
            """,
            (
                target_id,
                name,
                crm_name or name,
                crm_id,
                1 if is_active else 0,
            ),
        )
        self._conn.commit()
        item = self.get_location(target_id)
        if item is None:
            raise RuntimeError(f"Не удалось сохранить локацию: {target_id}")
        return item

    def delete_location(self, lid: str) -> bool:
        cur = self._conn.execute("DELETE FROM locations WHERE id = ?", (lid,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- training types ---
    def list_training_types(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                tt.slug,
                tt.name,
                tt.department,
                tt.checklist_slug,
                c.display_name AS checklist_display_name,
                c.department AS checklist_department,
                c.version AS checklist_version,
                tt.sort_order,
                tt.created_at,
                tt.updated_at
            FROM training_types tt
            LEFT JOIN checklists c ON c.slug = tt.checklist_slug
            ORDER BY tt.sort_order, tt.name COLLATE NOCASE, tt.slug COLLATE NOCASE
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def training_type_exists(self, slug: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM training_types WHERE slug = ? LIMIT 1",
            (slug,),
        ).fetchone()
        return row is not None

    def get_training_type(self, slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT
                tt.slug,
                tt.name,
                tt.department,
                tt.checklist_slug,
                c.display_name AS checklist_display_name,
                c.department AS checklist_department,
                c.version AS checklist_version,
                tt.sort_order,
                tt.created_at,
                tt.updated_at
            FROM training_types tt
            LEFT JOIN checklists c ON c.slug = tt.checklist_slug
            WHERE tt.slug = ?
            """,
            (slug,),
        ).fetchone()
        return dict(row) if row else None

    def _next_training_type_sort_order(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(sort_order), -1) FROM training_types").fetchone()
        return int(row[0]) + 1 if row else 0

    def upsert_training_type(
        self,
        slug: str,
        *,
        name: str,
        department: str | None = None,
        checklist_slug: str | None = None,
        sort_order: int | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        existing = self.get_training_type(slug)
        order = sort_order if sort_order is not None else (
            existing.get("sort_order") if existing else self._next_training_type_sort_order()
        )
        self._conn.execute(
            """
            INSERT INTO training_types (slug, name, department, checklist_slug, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                name = excluded.name,
                department = excluded.department,
                checklist_slug = excluded.checklist_slug,
                sort_order = excluded.sort_order,
                updated_at = excluded.updated_at
            """,
            (
                slug,
                name,
                department or existing.get("department"),
                checklist_slug,
                int(order),
                existing.get("created_at", now) if existing else now,
                now,
            ),
        )
        self._conn.commit()
        item = self.get_training_type(slug)
        if item is None:
            raise RuntimeError(f"Не удалось сохранить тип тренировки: {slug}")
        return item

    def delete_training_type(self, slug: str) -> bool:
        self._conn.execute(
            "UPDATE videos SET training_type_slug = NULL WHERE training_type_slug = ?",
            (slug,),
        )
        cur = self._conn.execute("DELETE FROM training_types WHERE slug = ?", (slug,))
        self._conn.commit()
        return cur.rowcount > 0

    def count_training_types_for_checklist(self, checklist_slug: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM training_types WHERE checklist_slug = ?",
            (checklist_slug,),
        ).fetchone()
        return int(row[0]) if row else 0

    # --- local users ---
    def list_users(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                username,
                display_name,
                full_name,
                auth_source,
                location_id,
                department,
                profile_completed_at,
                onboarding_version,
                role,
                daily_upload_limit,
                max_queued_jobs,
                max_running_jobs,
                quotas_updated_at,
                is_active,
                created_at,
                updated_at
            FROM users
            ORDER BY username COLLATE NOCASE
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_user(self, username: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT
                username,
                password_hash,
                display_name,
                full_name,
                auth_source,
                location_id,
                department,
                profile_completed_at,
                onboarding_version,
                role,
                daily_upload_limit,
                max_queued_jobs,
                max_running_jobs,
                quotas_updated_at,
                is_active,
                created_at,
                updated_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_user(
        self,
        username: str,
        *,
        password_hash: str | None = None,
        display_name: str | None = None,
        full_name: str | None = None,
        auth_source: str = AUTH_SOURCE_LOCAL,
        location_id: str | None = None,
        department: str | None = None,
        profile_completed_at: str | None = None,
        onboarding_version: int | object = _UNSET,
        role: str = USER_ROLE_USER,
        daily_upload_limit: int | None | object = _UNSET,
        max_queued_jobs: int | None | object = _UNSET,
        max_running_jobs: int | None | object = _UNSET,
        is_active: bool = True,
    ) -> dict[str, Any]:
        now = _utc_now()
        existing = self.get_user(username) or {}
        created_at = existing.get("created_at", now) if existing else now
        final_full_name = (full_name or existing.get("full_name") or "").strip() or None
        final_display_name = (
            (display_name or "").strip()
            or final_full_name
            or str(existing.get("display_name") or "").strip()
            or None
        )
        final_location_id = location_id if location_id is not None else existing.get("location_id")
        final_department = department if department is not None else existing.get("department")
        final_profile_completed_at = profile_completed_at
        if final_profile_completed_at is None:
            final_profile_completed_at = existing.get("profile_completed_at")
        if final_full_name and final_location_id and final_department and not final_profile_completed_at:
            final_profile_completed_at = now
        final_onboarding_version = (
            existing.get("onboarding_version", 0)
            if onboarding_version is _UNSET
            else _normalize_onboarding_version(onboarding_version)
        )
        final_daily_upload_limit = (
            existing.get("daily_upload_limit")
            if daily_upload_limit is _UNSET
            else _normalize_optional_limit(daily_upload_limit)
        )
        final_max_queued_jobs = (
            existing.get("max_queued_jobs")
            if max_queued_jobs is _UNSET
            else _normalize_optional_limit(max_queued_jobs)
        )
        final_max_running_jobs = (
            existing.get("max_running_jobs")
            if max_running_jobs is _UNSET
            else _normalize_optional_limit(max_running_jobs)
        )
        final_quotas_updated_at = existing.get("quotas_updated_at")
        if (
            daily_upload_limit is not _UNSET
            or max_queued_jobs is not _UNSET
            or max_running_jobs is not _UNSET
        ):
            if (
                final_daily_upload_limit != existing.get("daily_upload_limit")
                or final_max_queued_jobs != existing.get("max_queued_jobs")
                or final_max_running_jobs != existing.get("max_running_jobs")
            ):
                final_quotas_updated_at = now
        final_password_hash = (
            password_hash
            if password_hash is not None
            else str(existing.get("password_hash") or "")
        )
        self._conn.execute(
            """
            INSERT INTO users (
                username,
                password_hash,
                display_name,
                full_name,
                auth_source,
                location_id,
                department,
                profile_completed_at,
                onboarding_version,
                role,
                daily_upload_limit,
                max_queued_jobs,
                max_running_jobs,
                quotas_updated_at,
                is_active,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash = excluded.password_hash,
                display_name = excluded.display_name,
                full_name = excluded.full_name,
                auth_source = excluded.auth_source,
                location_id = excluded.location_id,
                department = excluded.department,
                profile_completed_at = excluded.profile_completed_at,
                onboarding_version = excluded.onboarding_version,
                role = excluded.role,
                daily_upload_limit = excluded.daily_upload_limit,
                max_queued_jobs = excluded.max_queued_jobs,
                max_running_jobs = excluded.max_running_jobs,
                quotas_updated_at = excluded.quotas_updated_at,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (
                username,
                final_password_hash,
                final_display_name,
                final_full_name,
                _normalize_auth_source(auth_source),
                final_location_id,
                final_department,
                final_profile_completed_at,
                final_onboarding_version,
                _normalize_user_role(role),
                final_daily_upload_limit,
                final_max_queued_jobs,
                final_max_running_jobs,
                final_quotas_updated_at,
                1 if is_active else 0,
                created_at,
                now,
            ),
        )
        self._conn.commit()
        final_name = final_full_name or final_display_name or username
        self.add_manager(username, final_name)
        item = self.get_user(username)
        if item is None:
            raise RuntimeError(f"Не удалось сохранить пользователя: {username}")
        return item

    def complete_user_profile(
        self,
        username: str,
        *,
        full_name: str,
        location_id: str,
        department: str,
    ) -> dict[str, Any]:
        existing = self.get_user(username)
        if not existing:
            raise ValueError(f"Пользователь не найден: {username}")
        item = self.upsert_user(
            username,
            password_hash=None,
            display_name=full_name,
            full_name=full_name,
            auth_source=str(existing.get("auth_source") or AUTH_SOURCE_LOCAL),
            location_id=location_id,
            department=department,
            profile_completed_at=_utc_now(),
            role=str(existing.get("role") or USER_ROLE_USER),
            daily_upload_limit=existing.get("daily_upload_limit"),
            max_queued_jobs=existing.get("max_queued_jobs"),
            max_running_jobs=existing.get("max_running_jobs"),
            is_active=bool(existing.get("is_active")),
        )
        return item

    def set_user_onboarding_version(self, username: str, version: int) -> dict[str, Any]:
        normalized = str(username or "").strip()
        if not normalized:
            raise ValueError("Не указан пользователь")
        existing = self.get_user(normalized)
        if not existing:
            raise ValueError(f"Пользователь не найден: {normalized}")
        self._conn.execute(
            """
            UPDATE users
            SET onboarding_version = ?, updated_at = ?
            WHERE username = ?
            """,
            (_normalize_onboarding_version(version), _utc_now(), normalized),
        )
        self._conn.commit()
        item = self.get_user(normalized)
        if item is None:
            raise RuntimeError(f"Не удалось обновить onboarding_version: {normalized}")
        return item

    def count_user_uploaded_videos_between(
        self,
        username: str,
        *,
        start_at: str,
        end_at: str,
    ) -> int:
        normalized = str(username or "").strip()
        if not normalized:
            return 0
        row = self._conn.execute(
            """
            SELECT COUNT(*)
            FROM videos
            WHERE lower(COALESCE(uploaded_by, '')) = lower(?)
              AND COALESCE(uploaded_at, '') >= ?
              AND COALESCE(uploaded_at, '') < ?
            """,
            (normalized, start_at, end_at),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_user_jobs(
        self,
        username: str,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> int:
        normalized = str(username or "").strip()
        if not normalized:
            return 0
        sql = [
            """
            SELECT COUNT(*)
            FROM jobs j
            LEFT JOIN videos v ON v.stem = j.stem
            WHERE lower(COALESCE(v.uploaded_by, '')) = lower(?)
            """
        ]
        params: list[Any] = [normalized]
        if statuses:
            marks = ", ".join(["?"] * len(statuses))
            sql.append(f"AND j.status IN ({marks})")
            params.extend(list(statuses))
        row = self._conn.execute("\n".join(sql), params).fetchone()
        return int(row[0]) if row else 0

    def list_user_quota_usage(self, *, start_at: str, end_at: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        uploaded_rows = self._conn.execute(
            """
            SELECT lower(COALESCE(uploaded_by, '')) AS username_key, COUNT(*) AS uploaded_count
            FROM videos
            WHERE COALESCE(TRIM(uploaded_by), '') <> ''
              AND COALESCE(uploaded_at, '') >= ?
              AND COALESCE(uploaded_at, '') < ?
            GROUP BY lower(COALESCE(uploaded_by, ''))
            """,
            (start_at, end_at),
        ).fetchall()
        for row in uploaded_rows:
            key = str(row["username_key"] or "").strip()
            if not key:
                continue
            item = out.setdefault(
                key,
                {"daily_uploaded_count": 0, "queued_count": 0, "running_count": 0},
            )
            item["daily_uploaded_count"] = int(row["uploaded_count"] or 0)
        job_rows = self._conn.execute(
            """
            SELECT
                lower(COALESCE(v.uploaded_by, '')) AS username_key,
                SUM(CASE WHEN j.status = 'queued' THEN 1 ELSE 0 END) AS queued_count,
                SUM(CASE WHEN j.status = 'running' THEN 1 ELSE 0 END) AS running_count
            FROM jobs j
            LEFT JOIN videos v ON v.stem = j.stem
            WHERE COALESCE(TRIM(v.uploaded_by), '') <> ''
              AND j.status IN ('queued', 'running')
            GROUP BY lower(COALESCE(v.uploaded_by, ''))
            """
        ).fetchall()
        for row in job_rows:
            key = str(row["username_key"] or "").strip()
            if not key:
                continue
            item = out.setdefault(
                key,
                {"daily_uploaded_count": 0, "queued_count": 0, "running_count": 0},
            )
            item["queued_count"] = int(row["queued_count"] or 0)
            item["running_count"] = int(row["running_count"] or 0)
        return out

    def count_running_jobs_by_user(self) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT lower(COALESCE(v.uploaded_by, '')) AS username_key, COUNT(*) AS running_count
            FROM jobs j
            LEFT JOIN videos v ON v.stem = j.stem
            WHERE j.status = 'running'
              AND COALESCE(TRIM(v.uploaded_by), '') <> ''
            GROUP BY lower(COALESCE(v.uploaded_by, ''))
            """
        ).fetchall()
        return {
            str(row["username_key"] or "").strip(): int(row["running_count"] or 0)
            for row in rows
            if str(row["username_key"] or "").strip()
        }

    def set_user_password_hash(self, username: str, password_hash: str) -> bool:
        cur = self._conn.execute(
            """
            UPDATE users
            SET password_hash = ?, updated_at = ?
            WHERE username = ?
            """,
            (password_hash, _utc_now(), username),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def set_user_active(self, username: str, is_active: bool) -> bool:
        cur = self._conn.execute(
            """
            UPDATE users
            SET is_active = ?, updated_at = ?
            WHERE username = ?
            """,
            (1 if is_active else 0, _utc_now(), username),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_user(self, username: str) -> bool:
        cur = self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- audit log ---
    def add_audit_event(
        self,
        *,
        actor: str | None,
        action: str,
        target_type: str,
        target_id: str | None = None,
        status: str = "ok",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = _utc_now()
        payload = json.dumps(details or {}, ensure_ascii=False)
        cur = self._conn.execute(
            """
            INSERT INTO audit_log (actor, action, target_type, target_id, status, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (actor, action, target_type, target_id, status, payload, created_at),
        )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id, actor, action, target_type, target_id, status, details_json, created_at
            FROM audit_log
            WHERE id = ?
            """,
            (int(cur.lastrowid),),
        ).fetchone()
        if not row:
            raise RuntimeError("Не удалось сохранить audit log")
        item = dict(row)
        try:
            item["details"] = json.loads(item.get("details_json") or "{}")
        except json.JSONDecodeError:
            item["details"] = {}
        return item

    def list_audit_events(
        self,
        *,
        limit: int = 200,
        actor: str | None = None,
        action: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        sql = [
            """
            SELECT id, actor, action, target_type, target_id, status, details_json, created_at
            FROM audit_log
            WHERE 1 = 1
            """
        ]
        params: list[Any] = []
        if actor:
            sql.append("AND actor = ?")
            params.append(actor)
        if action:
            sql.append("AND action = ?")
            params.append(action)
        sql.append("ORDER BY created_at DESC, id DESC LIMIT ?")
        params.append(safe_limit)
        rows = self._conn.execute("\n".join(sql), params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item.get("details_json") or "{}")
            except json.JSONDecodeError:
                item["details"] = {}
            out.append(item)
        return out

    # --- video meta ---
    def get_video_meta(self, stem: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM videos WHERE stem = ?", (stem,)).fetchone()
        if not row:
            return {}
        data = dict(row)
        data["tags"] = json.loads(data.get("tags") or "[]")
        return data

    def upsert_video_meta(self, stem: str, **kw: Any) -> dict[str, Any]:
        existing = self.get_video_meta(stem)
        tags = kw.get("tags", existing.get("tags", []))
        tags_json = json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else "[]"
        uploaded_at = existing.get("uploaded_at") or kw.get("uploaded_at") or _utc_now()
        training_type_slug = kw.get("training_type_slug", existing.get("training_type_slug"))
        if training_type_slug == "":
            training_type_slug = None
        raw_speaker_count = kw.get("expected_speaker_count", existing.get("expected_speaker_count", 2))
        try:
            expected_speaker_count = int(raw_speaker_count)
        except (TypeError, ValueError):
            expected_speaker_count = 2
        expected_speaker_count = max(1, min(expected_speaker_count, 8))
        training_type_name_snapshot = kw.get(
            "training_type_name_snapshot", existing.get("training_type_name_snapshot")
        )
        checklist_slug_snapshot = kw.get(
            "checklist_slug_snapshot", existing.get("checklist_slug_snapshot")
        )
        checklist_display_name_snapshot = kw.get(
            "checklist_display_name_snapshot", existing.get("checklist_display_name_snapshot")
        )
        checklist_version_snapshot = kw.get(
            "checklist_version_snapshot", existing.get("checklist_version_snapshot")
        )
        checklist_department_snapshot = kw.get(
            "checklist_department_snapshot", existing.get("checklist_department_snapshot")
        )
        file_size_bytes = kw.get("file_size_bytes", existing.get("file_size_bytes", 0))
        file_sha256 = kw.get("file_sha256", existing.get("file_sha256"))
        dedupe_key = kw.get("dedupe_key", existing.get("dedupe_key"))
        delete_requested_at = kw.get("delete_requested_at", existing.get("delete_requested_at"))
        delete_requested_by = kw.get("delete_requested_by", existing.get("delete_requested_by"))
        deleted_at = kw.get("deleted_at", existing.get("deleted_at"))
        deleted_by = kw.get("deleted_by", existing.get("deleted_by"))

        self._conn.execute(
            """
            INSERT INTO videos (
                stem,
                filename,
                display_title,
                manager_id,
                manager_name,
                location_id,
                location_name,
                interaction_date,
                training_type_slug,
                expected_speaker_count,
                tags,
                uploaded_at,
                status,
                uploaded_by,
                uploaded_by_name,
                uploader_location_id,
                uploader_location_name,
                uploader_location_crm_id,
                training_type_name_snapshot,
                checklist_slug_snapshot,
                checklist_display_name_snapshot,
                checklist_version_snapshot,
                checklist_department_snapshot,
                file_size_bytes,
                file_sha256,
                dedupe_key,
                delete_requested_at,
                delete_requested_by,
                deleted_at,
                deleted_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stem) DO UPDATE SET
                filename = COALESCE(excluded.filename, videos.filename),
                display_title = excluded.display_title,
                manager_id = excluded.manager_id,
                manager_name = excluded.manager_name,
                location_id = excluded.location_id,
                location_name = excluded.location_name,
                interaction_date = excluded.interaction_date,
                training_type_slug = excluded.training_type_slug,
                expected_speaker_count = COALESCE(excluded.expected_speaker_count, videos.expected_speaker_count),
                tags = excluded.tags,
                status = COALESCE(excluded.status, videos.status),
                uploaded_by = COALESCE(excluded.uploaded_by, videos.uploaded_by),
                uploaded_by_name = COALESCE(excluded.uploaded_by_name, videos.uploaded_by_name),
                uploader_location_id = COALESCE(excluded.uploader_location_id, videos.uploader_location_id),
                uploader_location_name = COALESCE(excluded.uploader_location_name, videos.uploader_location_name),
                uploader_location_crm_id = COALESCE(excluded.uploader_location_crm_id, videos.uploader_location_crm_id),
                training_type_name_snapshot = COALESCE(excluded.training_type_name_snapshot, videos.training_type_name_snapshot),
                checklist_slug_snapshot = COALESCE(excluded.checklist_slug_snapshot, videos.checklist_slug_snapshot),
                checklist_display_name_snapshot = COALESCE(excluded.checklist_display_name_snapshot, videos.checklist_display_name_snapshot),
                checklist_version_snapshot = COALESCE(excluded.checklist_version_snapshot, videos.checklist_version_snapshot),
                checklist_department_snapshot = COALESCE(excluded.checklist_department_snapshot, videos.checklist_department_snapshot),
                file_size_bytes = COALESCE(excluded.file_size_bytes, videos.file_size_bytes),
                file_sha256 = COALESCE(excluded.file_sha256, videos.file_sha256),
                dedupe_key = COALESCE(excluded.dedupe_key, videos.dedupe_key),
                delete_requested_at = excluded.delete_requested_at,
                delete_requested_by = excluded.delete_requested_by,
                deleted_at = excluded.deleted_at,
                deleted_by = excluded.deleted_by
            """,
            (
                stem,
                kw.get("filename", existing.get("filename")),
                kw.get("display_title", existing.get("display_title")),
                kw.get("manager_id", existing.get("manager_id")),
                kw.get("manager_name", existing.get("manager_name")),
                kw.get("location_id", existing.get("location_id")),
                kw.get("location_name", existing.get("location_name")),
                kw.get("interaction_date", existing.get("interaction_date")),
                training_type_slug,
                expected_speaker_count,
                tags_json,
                uploaded_at,
                kw.get("status", existing.get("status", "pending")),
                kw.get("uploaded_by", existing.get("uploaded_by")),
                kw.get("uploaded_by_name", existing.get("uploaded_by_name")),
                kw.get("uploader_location_id", existing.get("uploader_location_id")),
                kw.get("uploader_location_name", existing.get("uploader_location_name")),
                kw.get("uploader_location_crm_id", existing.get("uploader_location_crm_id")),
                training_type_name_snapshot,
                checklist_slug_snapshot,
                checklist_display_name_snapshot,
                checklist_version_snapshot,
                checklist_department_snapshot,
                file_size_bytes,
                file_sha256,
                dedupe_key,
                delete_requested_at,
                delete_requested_by,
                deleted_at,
                deleted_by,
            ),
        )
        self._conn.commit()
        return self.get_video_meta(stem)

    def request_video_deletion(self, stem: str, username: str | None) -> dict[str, Any]:
        meta = self.get_video_meta(stem)
        if not meta:
            raise ValueError("Видео не найдено")
        now = _utc_now()
        return self.upsert_video_meta(
            stem,
            delete_requested_at=now,
            delete_requested_by=username,
            status=meta.get("status", "pending"),
        )

    def restore_video_from_deletion_request(self, stem: str) -> dict[str, Any]:
        meta = self.get_video_meta(stem)
        if not meta:
            raise ValueError("Видео не найдено")
        return self.upsert_video_meta(
            stem,
            delete_requested_at=None,
            delete_requested_by=None,
            deleted_at=None,
            deleted_by=None,
            status=meta.get("status", "pending"),
        )

    def find_video_by_sha256(self, sha256: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        if not sha256:
            return None
        sql = "SELECT * FROM videos WHERE file_sha256 = ?"
        params: list[Any] = [sha256]
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY uploaded_at DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        if not row:
            return None
        data = dict(row)
        data["tags"] = json.loads(data.get("tags") or "[]")
        return data

    def get_latest_human_eval_state_for_stem(self, stem: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT stem, criteria_slug, draft_saved_at, published_at, published_by, compared_at, compared_by
            FROM human_eval_states
            WHERE stem = ?
            ORDER BY COALESCE(published_at, draft_saved_at, compared_at) DESC, criteria_slug DESC
            LIMIT 1
            """,
            (stem,),
        ).fetchone()
        return dict(row) if row else None

    def delete_video(self, stem: str) -> bool:
        cur = self._conn.execute("DELETE FROM videos WHERE stem = ?", (stem,))
        self._conn.execute("DELETE FROM jobs WHERE stem = ?", (stem,))
        self._conn.execute("DELETE FROM human_eval_states WHERE stem = ?", (stem,))
        self._conn.execute("DELETE FROM evaluation_comparisons WHERE stem = ?", (stem,))
        self._conn.commit()
        return cur.rowcount > 0

    def resolve_checklist_slug_for_video(self, stem: str) -> str:
        meta = self.get_video_meta(stem)
        snapshot_slug = str(meta.get("checklist_slug_snapshot") or "").strip()
        if snapshot_slug and self.checklist_exists(snapshot_slug):
            return snapshot_slug
        return self.get_current_checklist_slug_for_video(stem)

    def get_current_checklist_slug_for_video(self, stem: str) -> str:
        meta = self.get_video_meta(stem)
        training_slug = str(meta.get("training_type_slug") or "").strip()
        if training_slug:
            training_type = self.get_training_type(training_slug)
            checklist_slug = str((training_type or {}).get("checklist_slug") or "").strip()
            if checklist_slug and self.checklist_exists(checklist_slug):
                return checklist_slug
        return self.get_active_checklist_slug()

    # --- human eval state ---
    def get_human_eval_state(self, stem: str, criteria_slug: str) -> dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT stem, criteria_slug, draft_saved_at, published_at, published_by, compared_at, compared_by
            FROM human_eval_states
            WHERE stem = ? AND criteria_slug = ?
            """,
            (stem, criteria_slug),
        ).fetchone()
        return dict(row) if row else {}

    def upsert_human_eval_state(
        self,
        stem: str,
        criteria_slug: str,
        *,
        draft_saved_at: str | None = None,
        published_at: str | None = None,
        published_by: str | None = None,
        compared_at: str | None = None,
        compared_by: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_human_eval_state(stem, criteria_slug)
        self._conn.execute(
            """
            INSERT INTO human_eval_states (
                stem,
                criteria_slug,
                draft_saved_at,
                published_at,
                published_by,
                compared_at,
                compared_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stem, criteria_slug) DO UPDATE SET
                draft_saved_at = excluded.draft_saved_at,
                published_at = COALESCE(excluded.published_at, human_eval_states.published_at),
                published_by = COALESCE(excluded.published_by, human_eval_states.published_by),
                compared_at = COALESCE(excluded.compared_at, human_eval_states.compared_at),
                compared_by = COALESCE(excluded.compared_by, human_eval_states.compared_by)
            """,
            (
                stem,
                criteria_slug,
                draft_saved_at if draft_saved_at is not None else existing.get("draft_saved_at"),
                published_at,
                published_by,
                compared_at,
                compared_by,
            ),
        )
        self._conn.commit()
        return self.get_human_eval_state(stem, criteria_slug)

    def mark_human_eval_draft_saved(self, stem: str, criteria_slug: str) -> dict[str, Any]:
        return self.upsert_human_eval_state(
            stem,
            criteria_slug,
            draft_saved_at=_utc_now(),
        )

    def publish_human_eval(self, stem: str, criteria_slug: str, username: str | None) -> dict[str, Any]:
        state = self.get_human_eval_state(stem, criteria_slug)
        if state.get("published_at"):
            raise ValueError("Чеклист уже опубликован")
        return self.upsert_human_eval_state(
            stem,
            criteria_slug,
            draft_saved_at=state.get("draft_saved_at") or _utc_now(),
            published_at=_utc_now(),
            published_by=username,
        )

    def mark_human_eval_compared(self, stem: str, criteria_slug: str, username: str | None) -> dict[str, Any]:
        return self.upsert_human_eval_state(
            stem,
            criteria_slug,
            compared_at=_utc_now(),
            compared_by=username,
        )

    # --- persisted AI vs human comparison ---
    def save_evaluation_comparison(
        self,
        stem: str,
        criteria_slug: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = _utc_now()
        existing = self.get_evaluation_comparison(stem, criteria_slug)
        self._conn.execute(
            """
            INSERT INTO evaluation_comparisons (
                stem,
                criteria_slug,
                ai_overall,
                human_overall,
                diff_percent,
                status_color,
                payload_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stem, criteria_slug) DO UPDATE SET
                ai_overall = excluded.ai_overall,
                human_overall = excluded.human_overall,
                diff_percent = excluded.diff_percent,
                status_color = excluded.status_color,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                stem,
                criteria_slug,
                payload.get("ai_overall"),
                payload.get("human_overall"),
                payload.get("overall_diff"),
                payload.get("status_color"),
                json.dumps(payload, ensure_ascii=False),
                existing.get("created_at", now) if existing else now,
                now,
            ),
        )
        self._conn.commit()
        item = self.get_evaluation_comparison(stem, criteria_slug)
        if item is None:
            raise RuntimeError("Не удалось сохранить сравнение оценок")
        return item

    def get_evaluation_comparison(self, stem: str, criteria_slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT stem, criteria_slug, ai_overall, human_overall, diff_percent, status_color, payload_json, created_at, updated_at
            FROM evaluation_comparisons
            WHERE stem = ? AND criteria_slug = ?
            """,
            (stem, criteria_slug),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["payload"] = json.loads(data.get("payload_json") or "{}")
        except json.JSONDecodeError:
            data["payload"] = {}
        return data

    def get_latest_evaluation_comparison_for_stem(self, stem: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT stem, criteria_slug, ai_overall, human_overall, diff_percent, status_color, payload_json, created_at, updated_at
            FROM evaluation_comparisons
            WHERE stem = ?
            ORDER BY updated_at DESC, criteria_slug DESC
            LIMIT 1
            """,
            (stem,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["payload"] = json.loads(data.get("payload_json") or "{}")
        except json.JSONDecodeError:
            data["payload"] = {}
        return data

    # --- jobs ---
    def upsert_job(self, job_id: str, **kw: Any) -> None:
        existing = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if existing:
            sets = []
            vals = []
            for key, value in kw.items():
                if value is not None:
                    sets.append(f"{key} = ?")
                    vals.append(value)
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

    def list_jobs(
        self,
        *,
        limit: int = 200,
        status: str | None = None,
        kind: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        sql = [
            """
            SELECT
                j.*,
                v.display_title,
                v.manager_id,
                v.manager_name,
                v.location_id,
                v.location_name,
                v.training_type_slug,
                v.uploaded_by,
                v.uploaded_by_name,
                v.deleted_at,
                v.delete_requested_at,
                v.status AS video_status
            FROM jobs j
            LEFT JOIN videos v ON v.stem = j.stem
            WHERE 1 = 1
            """
        ]
        params: list[Any] = []
        if status:
            sql.append("AND j.status = ?")
            params.append(status)
        if kind:
            sql.append("AND j.kind = ?")
            params.append(kind)
        if query:
            like = f"%{str(query).strip().lower()}%"
            sql.append(
                """
                AND (
                    lower(COALESCE(j.id, '')) LIKE ?
                    OR lower(COALESCE(j.stem, '')) LIKE ?
                    OR lower(COALESCE(j.video_file, '')) LIKE ?
                    OR lower(COALESCE(v.display_title, '')) LIKE ?
                    OR lower(COALESCE(v.manager_name, '')) LIKE ?
                    OR lower(COALESCE(v.location_name, '')) LIKE ?
                    OR lower(COALESCE(v.uploaded_by, '')) LIKE ?
                    OR lower(COALESCE(v.uploaded_by_name, '')) LIKE ?
                )
                """
            )
            params.extend([like, like, like, like, like, like, like, like])
        sql.append("ORDER BY COALESCE(j.updated_at, j.created_at) DESC, j.created_at DESC, j.id DESC LIMIT ?")
        params.append(safe_limit)
        rows = self._conn.execute("\n".join(sql), params).fetchall()
        return [dict(r) for r in rows]

    def count_jobs(self, *, statuses: tuple[str, ...] | None = None) -> int:
        if statuses:
            marks = ", ".join(["?"] * len(statuses))
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})",
                list(statuses),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0]) if row else 0

    def latest_job_by_stem(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM (
                SELECT
                    jobs.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY stem
                        ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC, id DESC
                    ) AS row_num
                FROM jobs
                WHERE COALESCE(TRIM(stem), '') <> ''
            ) latest_jobs
            WHERE row_num = 1
            ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC, id DESC
            """
        ).fetchall()
        return {r["stem"]: dict(r) for r in rows if r["stem"]}

    def list_dispatchable_jobs(self, *, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 5000))
        rows = self._conn.execute(
            """
            SELECT
                j.*,
                v.display_title,
                v.uploaded_by,
                v.uploaded_by_name,
                v.filename AS source_filename
            FROM jobs j
            LEFT JOIN videos v ON v.stem = j.stem
            WHERE j.status = 'queued'
            ORDER BY COALESCE(j.created_at, j.updated_at) ASC, j.id ASC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def requeue_jobs(self, *, from_statuses: tuple[str, ...], stage: str = "queued") -> int:
        if not from_statuses:
            return 0
        marks = ", ".join(["?"] * len(from_statuses))
        cur = self._conn.execute(
            f"""
            UPDATE jobs
            SET status = 'queued', stage = ?, updated_at = ?
            WHERE status IN ({marks})
            """,
            [stage, _utc_now(), *list(from_statuses)],
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def delete_jobs_for_stem(self, stem: str) -> None:
        self._conn.execute("DELETE FROM jobs WHERE stem = ?", (stem,))
        self._conn.commit()

    # --- checklists ---
    def checklist_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM checklists").fetchone()
        return int(row[0]) if row else 0

    def list_checklist_files(self) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """
            SELECT slug, COALESCE(NULLIF(trim(display_name), ''), slug) AS display_name, department, version
            FROM checklists
            ORDER BY department COLLATE NOCASE, display_name COLLATE NOCASE, slug COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "name": r["slug"],
                "display_name": r["display_name"],
                "department": r["department"],
                "version": r["version"],
            }
            for r in rows
        ]

    def checklist_exists(self, slug: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM checklists WHERE slug = ? LIMIT 1",
            (slug,),
        ).fetchone()
        return row is not None

    def get_setting(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def list_settings(self, prefix: str | None = None) -> list[dict[str, str]]:
        if prefix:
            rows = self._conn.execute(
                """
                SELECT key, value
                FROM app_settings
                WHERE key LIKE ?
                ORDER BY key COLLATE NOCASE
                """,
                (f"{prefix}%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT key, value
                FROM app_settings
                ORDER BY key COLLATE NOCASE
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def set_setting(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO app_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._conn.commit()

    def delete_setting(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def get_active_checklist_slug(self) -> str:
        raw = self.get_setting(ACTIVE_CHECKLIST_KEY)
        if raw and self.checklist_exists(raw.strip()):
            return raw.strip()
        rows = self._conn.execute(
            "SELECT slug FROM checklists ORDER BY slug COLLATE NOCASE LIMIT 1"
        ).fetchall()
        if rows:
            slug = str(rows[0]["slug"])
            self.set_setting(ACTIVE_CHECKLIST_KEY, slug)
            return slug
        return DEFAULT_CHECKLIST_SLUG

    def set_active_checklist_slug(self, slug: str) -> None:
        if not self.checklist_exists(slug):
            raise ValueError(f"Чеклист не найден: {slug}")
        self.set_setting(ACTIVE_CHECKLIST_KEY, slug)

    def get_checklist_meta(self, slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT slug, COALESCE(NULLIF(trim(display_name), ''), slug) AS display_name, department, version, updated_at
            FROM checklists
            WHERE slug = ?
            """,
            (slug,),
        ).fetchone()
        return dict(row) if row else None

    def get_checklist_content(self, slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT slug, COALESCE(NULLIF(trim(display_name), ''), slug) AS display_name, department, version
            FROM checklists
            WHERE slug = ?
            """,
            (slug,),
        ).fetchone()
        if not row:
            return None
        crit_rows = self._conn.execute(
            """
            SELECT criterion_id, name, description, weight
            FROM checklist_criteria
            WHERE checklist_slug = ?
            ORDER BY sort_order, criterion_id
            """,
            (slug,),
        ).fetchall()
        criteria = [
            {
                "id": r["criterion_id"],
                "name": r["name"],
                "description": r["description"] or "",
                "weight": _normalize_weight(r["weight"]),
            }
            for r in crit_rows
        ]
        return {
            "filename": row["slug"],
            "display_name": row["display_name"],
            "department": row["department"],
            "version": row["version"],
            "criteria": criteria,
            "can_delete": self.checklist_count() > 1,
            "in_use_by_training_types": self.count_training_types_for_checklist(str(row["slug"])),
        }

    def replace_checklist(
        self,
        slug: str,
        version: str,
        criteria: list[dict[str, Any]],
        *,
        display_name: str | None = None,
        department: str | None = None,
    ) -> None:
        now = _utc_now()
        self._conn.execute("DELETE FROM checklist_criteria WHERE checklist_slug = ?", (slug,))
        existing = self.get_checklist_meta(slug) or {}
        self._conn.execute(
            """
            INSERT INTO checklists (slug, display_name, department, version, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                display_name = excluded.display_name,
                department = excluded.department,
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (
                slug,
                (display_name or existing.get("display_name") or slug),
                department if department is not None else existing.get("department"),
                version,
                now,
            ),
        )
        for i, item in enumerate(criteria):
            cid = str(item.get("id", "")).strip()
            if not cid:
                continue
            self._conn.execute(
                """
                INSERT INTO checklist_criteria (
                    checklist_slug,
                    sort_order,
                    criterion_id,
                    name,
                    description,
                    weight
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    i,
                    cid,
                    str(item.get("name", cid)).strip(),
                    str(item.get("description", "") or "").strip(),
                    _normalize_weight(item.get("weight", 1)),
                ),
            )
        self._conn.commit()

    def insert_checklist(
        self,
        slug: str,
        version: str,
        criteria: list[dict[str, Any]],
        *,
        display_name: str | None = None,
        department: str | None = None,
    ) -> None:
        if self.checklist_exists(slug):
            raise ValueError(f"Чеклист уже есть: {slug}")
        now = _utc_now()
        self._conn.execute(
            "INSERT INTO checklists (slug, display_name, department, version, updated_at) VALUES (?, ?, ?, ?, ?)",
            (slug, display_name or slug, department, version, now),
        )
        for i, item in enumerate(criteria):
            cid = str(item.get("id", "")).strip()
            if not cid:
                continue
            self._conn.execute(
                """
                INSERT INTO checklist_criteria (
                    checklist_slug,
                    sort_order,
                    criterion_id,
                    name,
                    description,
                    weight
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    i,
                    cid,
                    str(item.get("name", cid)).strip(),
                    str(item.get("description", "") or "").strip(),
                    _normalize_weight(item.get("weight", 1)),
                ),
            )
        self._conn.commit()

    def ensure_seed_checklist(
        self,
        slug: str,
        *,
        display_name: str,
        department: str,
        criteria_texts: list[str],
        version: str = "1",
    ) -> None:
        if self.checklist_exists(slug):
            meta = self.get_checklist_meta(slug) or {}
            self._conn.execute(
                """
                UPDATE checklists
                SET display_name = COALESCE(NULLIF(trim(display_name), ''), ?),
                    department = COALESCE(department, ?)
                WHERE slug = ?
                """,
                (display_name, department, slug),
            )
            self._conn.commit()
            return
        criteria = build_seed_criteria(criteria_texts)
        self.insert_checklist(
            slug,
            version,
            criteria,
            display_name=display_name,
            department=department,
        )

    def delete_checklist(self, slug: str) -> bool:
        if self.count_training_types_for_checklist(slug) > 0:
            raise ValueError("Чеклист привязан к типу тренировки")
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
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(cd.glob(pattern)):
            if not path.is_file():
                continue
            if path.name in _CONFIG_YAML_EXCLUDE:
                continue
            paths.append(path)

    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        unique_paths.append(path)

    imported_source_files: list[Path] = []
    for path in unique_paths:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        version = str(raw.get("version", "1"))
        criteria: list[dict[str, Any]] = []
        for row in raw.get("criteria") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id", "")).strip()
            if not cid:
                continue
            criteria.append(
                {
                    "id": cid,
                    "name": str(row.get("name", cid)).strip(),
                    "description": str(row.get("description", "")).strip(),
                    "weight": _normalize_weight(row.get("weight", 1)),
                }
            )
        if not criteria:
            criteria = [
                {
                    "id": "criterion_1",
                    "name": "Новый критерий",
                    "description": "Опишите, что проверяет ИИ по этому пункту.",
                    "weight": 1,
                }
            ]
        slug = path.stem
        try:
            db.insert_checklist(slug, version, criteria, display_name=slug, department=None)
            imported_source_files.append(path)
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
                    "weight": 1,
                }
            ],
            display_name=DEFAULT_CHECKLIST_SLUG,
            department=None,
        )

    active: str | None = None
    if marker.is_file():
        try:
            raw_active = marker.read_text(encoding="utf-8").strip()
            if raw_active:
                candidates: list[str] = []
                if raw_active.lower().endswith((".yaml", ".yml")):
                    candidates.append(Path(raw_active).stem)
                candidates.append(raw_active)
                for candidate in candidates:
                    if candidate and db.checklist_exists(candidate):
                        active = candidate
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
        files = db.list_checklist_files()
        if files:
            db.set_active_checklist_slug(files[0]["name"])

    if delete_source_files and imported_source_files:
        for fp in imported_source_files:
            if not fp.is_file():
                continue
            try:
                fp.unlink()
            except OSError:
                pass
        if marker.is_file():
            try:
                marker.unlink()
            except OSError:
                pass
