"""
Веб-интерфейс: загрузка видео/аудио, списки транскриптов/оценок, прогресс пайплайна.
Запуск из корня проекта: uvicorn web.server:app --reload --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from src.cuda_runtime_path import ensure_nvidia_pip_libs

ensure_nvidia_pip_libs()

from pydantic import BaseModel

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from src.ad_auth import (
    ad_auth_enabled,
    auth_enabled,
    auth_mode,
    fetch_ad_display_name,
    local_auth_enabled,
    require_session_secret,
    validate_ad_config_at_startup,
    verify_user_password,
)
from src.eval_schema import compute_eval_totals, normalize_eval_criteria, normalize_loaded_evaluation
from src.local_auth import (
    hash_local_password,
    normalize_local_username,
    validate_local_password_strength,
    verify_local_user_password,
)
from src.paths import (
    CONFIG_DIR,
    CRITERIA_FILE,
    DEFAULT_CHECKLIST_SLUG,
    EVALUATION_DIR,
    LOCATIONS_FILE,
    MANAGERS_FILE,
    META_DIR,
    TRANSCRIPT_DIR,
    VIDEO_DIR,
    human_evaluation_json_path,
    meta_json_path,
)
from src.artifacts import delete_derived_artifacts_for_stem
from src.pipeline import (
    evaluate_only_from_transcript,
    find_video_for_stem,
    list_transcripts,
    list_videos,
    process_one_video,
    process_one_video_resume,
)
from src.database import (
    AUTH_SOURCE_AD,
    AUTH_SOURCE_LOCAL,
    DB,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    _normalize_auth_source,
    init_db,
    migrate_checklists_from_config,
)
from src.errors import PipelineCancelled
from src.seed_data import CHECKLIST_SEEDS, LOCATION_SEEDS, make_location_internal_id

DEPARTMENT_LABELS = {
    "ОО": "Отдел оценки",
    "ОП": "Отдел продаж",
}

_MEDIA_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wma": "audio/x-ms-wma",
    ".amr": "audio/amr",
}
_MEDIA_UPLOAD_EXTENSIONS = frozenset(_MEDIA_MIME)

STATIC_DIR = Path(__file__).resolve().parent / "static"

OPS_MAX_WORKERS_KEY = "ops.max_workers"
OPS_ADMIN_REFRESH_SECONDS_KEY = "ops.admin_auto_refresh_seconds"
AUTH_LOCAL_REGISTRATION_ENABLED_KEY = "auth.local_registration_enabled"
OPS_MAX_QUEUE_DEPTH_KEY = "ops.max_queue_depth"
QUOTA_DEFAULT_DAILY_UPLOAD_LIMIT_KEY = "quota.default_daily_upload_limit"
QUOTA_DEFAULT_MAX_QUEUED_JOBS_KEY = "quota.default_max_queued_jobs"
QUOTA_DEFAULT_MAX_RUNNING_JOBS_KEY = "quota.default_max_running_jobs"
_MAX_WORKERS_MIN = 1
_MAX_WORKERS_MAX = 16
_MAX_QUEUE_DEPTH_DEFAULT = 100
_MAX_QUEUE_DEPTH_MAX = 10_000
_DEFAULT_DAILY_UPLOAD_LIMIT = 20
_DEFAULT_DAILY_UPLOAD_LIMIT_MAX = 1_000
_DEFAULT_MAX_QUEUED_JOBS = 5
_DEFAULT_MAX_QUEUED_JOBS_MAX = 1_000
_DEFAULT_MAX_RUNNING_JOBS = 1
_DEFAULT_MAX_RUNNING_JOBS_MAX = 16
_ONBOARDING_TOUR_VERSION = 1

init_db()

app = FastAPI(title="Fresh FA", version="2")

_SESSION_MAX_AGE = int(os.environ.get("SESSION_MAX_AGE_SEC", str(60 * 60 * 24 * 7)))
_SESSION_HTTPS_ONLY = os.environ.get("SESSION_COOKIE_SECURE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


class _AuthEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not auth_enabled():
            return await call_next(request)
        path = request.scope.get("path") or ""
        if path.startswith("/static/"):
            return await call_next(request)
        public_api = frozenset(
            {
                "/api/health",
                "/api/auth/login",
                "/api/auth/logout",
                "/api/auth/me",
                "/api/auth/register",
                "/api/auth/locations",
                "/api/auth/status",
            }
        )
        if path in public_api:
            return await call_next(request)
        if path.startswith("/api/"):
            if not request.session.get("user"):
                return JSONResponse(
                    {"detail": "Требуется вход в систему"},
                    status_code=401,
                )
            return await call_next(request)
        if path == "/login.html":
            return await call_next(request)
        if path in ("/", "/dashboard", "/admin"):
            if not request.session.get("user"):
                return RedirectResponse(url="/login.html", status_code=302)
            return await call_next(request)
        return await call_next(request)


app.add_middleware(_AuthEnforcementMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=require_session_secret(),
    session_cookie="fa_session",
    max_age=_SESSION_MAX_AGE,
    same_site="lax",
    https_only=_SESSION_HTTPS_ONLY,
)


import concurrent.futures
import threading

_ACTIVE_CRITERIA_MARKER = CONFIG_DIR / ".active_criteria"

_db = DB()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        out = default
    return max(minimum, min(out, maximum))


def _registration_allowed() -> bool:
    if not local_auth_enabled():
        return False
    raw = _db.get_setting(AUTH_LOCAL_REGISTRATION_ENABLED_KEY)
    if raw is None:
        return True
    return _coerce_bool(raw, True)


def _configured_max_workers() -> int:
    raw = _db.get_setting(OPS_MAX_WORKERS_KEY)
    if raw is None or not str(raw).strip():
        raw = os.environ.get("FA_MAX_WORKERS", "2")
    return _coerce_int(raw, 2, _MAX_WORKERS_MIN, _MAX_WORKERS_MAX)


def _configured_admin_refresh_seconds() -> int:
    raw = _db.get_setting(OPS_ADMIN_REFRESH_SECONDS_KEY)
    if raw is None or not str(raw).strip():
        return 10
    return _coerce_int(raw, 10, 3, 120)


def _configured_max_queue_depth() -> int:
    raw = _db.get_setting(OPS_MAX_QUEUE_DEPTH_KEY)
    if raw is None or not str(raw).strip():
        return _MAX_QUEUE_DEPTH_DEFAULT
    return _coerce_int(raw, _MAX_QUEUE_DEPTH_DEFAULT, 1, _MAX_QUEUE_DEPTH_MAX)


def _configured_default_daily_upload_limit() -> int:
    raw = _db.get_setting(QUOTA_DEFAULT_DAILY_UPLOAD_LIMIT_KEY)
    if raw is None or not str(raw).strip():
        return _DEFAULT_DAILY_UPLOAD_LIMIT
    return _coerce_int(raw, _DEFAULT_DAILY_UPLOAD_LIMIT, 0, _DEFAULT_DAILY_UPLOAD_LIMIT_MAX)


def _configured_default_max_queued_jobs() -> int:
    raw = _db.get_setting(QUOTA_DEFAULT_MAX_QUEUED_JOBS_KEY)
    if raw is None or not str(raw).strip():
        return _DEFAULT_MAX_QUEUED_JOBS
    return _coerce_int(raw, _DEFAULT_MAX_QUEUED_JOBS, 0, _DEFAULT_MAX_QUEUED_JOBS_MAX)


def _configured_default_max_running_jobs() -> int:
    raw = _db.get_setting(QUOTA_DEFAULT_MAX_RUNNING_JOBS_KEY)
    if raw is None or not str(raw).strip():
        return _DEFAULT_MAX_RUNNING_JOBS
    return _coerce_int(raw, _DEFAULT_MAX_RUNNING_JOBS, 0, _DEFAULT_MAX_RUNNING_JOBS_MAX)


def _runtime_settings_payload() -> dict[str, Any]:
    configured_workers = _configured_max_workers()
    return {
        "max_workers": {
            "current": _MAX_WORKERS,
            "configured": configured_workers,
            "applied": _MAX_WORKERS == configured_workers,
            "minimum": _MAX_WORKERS_MIN,
            "maximum": _MAX_WORKERS_MAX,
        },
        "admin_auto_refresh_seconds": _configured_admin_refresh_seconds(),
        "local_registration_enabled": _registration_allowed(),
        "max_queue_depth": _configured_max_queue_depth(),
        "default_daily_upload_limit": _configured_default_daily_upload_limit(),
        "default_max_queued_jobs": _configured_default_max_queued_jobs(),
        "default_max_running_jobs": _configured_default_max_running_jobs(),
    }


def _audit_log(
    request: Request | None,
    *,
    action: str,
    target_type: str,
    target_id: str | None = None,
    status: str = "ok",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actor = None
    if request is not None:
        actor = str(request.session.get("user") or "").strip() or None
    return _db.add_audit_event(
        actor=actor,
        action=action,
        target_type=target_type,
        target_id=target_id,
        status=status,
        details=details,
    )


def _apply_max_workers_setting(new_value: int) -> dict[str, Any]:
    global _executor, _MAX_WORKERS
    safe_value = _coerce_int(new_value, _MAX_WORKERS, _MAX_WORKERS_MIN, _MAX_WORKERS_MAX)
    active_jobs = _db.count_jobs(statuses=("queued", "running"))
    if safe_value == _MAX_WORKERS:
        return {
            "requested": safe_value,
            "current": _MAX_WORKERS,
            "configured": _configured_max_workers(),
            "active_jobs": active_jobs,
            "applied": True,
            "changed": False,
        }
    if active_jobs > 0:
        return {
            "requested": safe_value,
            "current": _MAX_WORKERS,
            "configured": _configured_max_workers(),
            "active_jobs": active_jobs,
            "applied": False,
            "changed": True,
            "reason": "Есть активные или ожидающие задачи; новое значение вступит в силу после завершения очереди или перезапуска сервера.",
        }
    old_executor = _executor
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=safe_value,
        thread_name_prefix="fa-worker",
    )
    _MAX_WORKERS = safe_value
    old_executor.shutdown(wait=False, cancel_futures=False)
    _dispatch_jobs()
    return {
        "requested": safe_value,
        "current": _MAX_WORKERS,
        "configured": _configured_max_workers(),
        "active_jobs": active_jobs,
        "applied": True,
        "changed": True,
    }


def _seed_builtin_locations() -> None:
    for item in LOCATION_SEEDS:
        name = str(item.get("crm_name") or "").strip()
        crm_id = str(item.get("crm_id") or "").strip()
        if not name or not crm_id:
            continue
        _db.add_location(
            make_location_internal_id(name),
            name,
            crm_name=name,
            crm_id=crm_id,
            is_active=True,
        )


def _seed_builtin_checklists() -> None:
    for item in CHECKLIST_SEEDS:
        _db.ensure_seed_checklist(
            str(item["slug"]),
            display_name=str(item["display_name"]),
            department=str(item["department"]),
            criteria_texts=list(item["criteria"]),
            version="1",
        )


def _seed_builtin_training_types() -> None:
    for index, item in enumerate(CHECKLIST_SEEDS):
        checklist_slug = str(item["slug"])
        _db.upsert_training_type(
            checklist_slug,
            name=str(item["display_name"]),
            department=str(item["department"]),
            checklist_slug=checklist_slug,
            sort_order=index,
        )


@app.on_event("startup")
def _migrate_yaml_to_db() -> None:
    """Импорт справочников и чеклистов из YAML при пустых таблицах."""
    validate_ad_config_at_startup()
    if not _db.list_managers():
        _db.import_managers_from_yaml(_load_yaml_list(MANAGERS_FILE, "managers"))
    if not _db.list_locations():
        _db.import_locations_from_yaml(_load_yaml_list(LOCATIONS_FILE, "locations"))
    migrate_checklists_from_config(
        _db,
        config_dir=CONFIG_DIR,
        active_marker=_ACTIVE_CRITERIA_MARKER,
        delete_source_files=True,
    )
    _seed_builtin_locations()
    _seed_builtin_checklists()
    _seed_builtin_training_types()
    _recover_jobs_after_restart()


_MAX_WORKERS = _configured_max_workers()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="fa-worker")

_job_cancel_lock = threading.Lock()
_job_cancel_events: dict[str, threading.Event] = {}
_running_jobs_lock = threading.Lock()
_running_job_ids: set[str] = set()
_dispatch_lock = threading.Lock()
_upload_intake_lock = threading.Lock()
_comparison_tasks_lock = threading.Lock()
_comparison_running_keys: set[tuple[str, str]] = set()
_comparison_errors: dict[tuple[str, str], str] = {}


def _safe_criteria_filename(name: str) -> str:
    s = (name or "").strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        raise HTTPException(400, "Некорректное имя чеклиста")
    lower = s.lower()
    if lower.endswith(".yaml"):
        s = s[: -len(".yaml")]
    elif lower.endswith(".yml"):
        s = s[: -len(".yml")]
    if not s or not re.match(r"^[a-zA-Z0-9_\-\u0400-\u04FF]+$", s):
        raise HTTPException(
            400,
            "Допустимы латиница, цифры, дефис, подчёркивание и кириллица, без расширения",
        )
    return s


def _safe_training_type_slug(name: str) -> str:
    s = (name or "").strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        raise HTTPException(400, "Некорректный тип тренировки")
    if not re.match(r"^[a-zA-Z0-9_\-\u0400-\u04FF]+$", s):
        raise HTTPException(
            400,
            "Допустимы латиница, цифры, дефис, подчёркивание и кириллица",
        )
    return s


def _normalize_department_code(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text in DEPARTMENT_LABELS:
        return text
    return None


def _department_label(code: str | None) -> str | None:
    return DEPARTMENT_LABELS.get(str(code or "").strip().upper() or "", None)


def _require_department_code(raw: Any, *, error_message: str = "Выберите корректный отдел") -> str:
    code = _normalize_department_code(raw)
    if not code:
        raise HTTPException(400, error_message)
    return code


def _admin_users() -> set[str]:
    raw = os.environ.get("ADMIN_USERS", "")
    out: set[str] = set()
    for part in raw.split(","):
        name = part.strip().lower()
        if name:
            out.add(name)
    return out


def _effective_user_role(username: str | None, role_hint: str | None = None) -> str:
    uname = (username or "").strip().lower()
    if uname and uname in _admin_users():
        return USER_ROLE_ADMIN
    if str(role_hint or "").strip().lower() == USER_ROLE_ADMIN:
        return USER_ROLE_ADMIN
    return "user"


def _session_is_admin(request: Request) -> bool:
    if not auth_enabled():
        return True
    return str(request.session.get("role") or "").strip().lower() == USER_ROLE_ADMIN


def _user_profile_complete(user_row: dict[str, Any] | None) -> bool:
    if not user_row:
        return False
    full_name = str(user_row.get("full_name") or user_row.get("display_name") or "").strip()
    location_id = str(user_row.get("location_id") or "").strip()
    department = _normalize_department_code((user_row or {}).get("department"))
    return bool(full_name and location_id and department)


def _user_onboarding_version(user_row: dict[str, Any] | None) -> int:
    try:
        return max(0, int((user_row or {}).get("onboarding_version") or 0))
    except (TypeError, ValueError):
        return 0


def _user_location_payload(user_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not user_row:
        return None
    location_id = str(user_row.get("location_id") or "").strip()
    if not location_id:
        return None
    loc = _db.get_location(location_id)
    if not loc:
        return {"id": location_id}
    return {
        "id": loc.get("id"),
        "name": loc.get("name"),
        "crm_name": loc.get("crm_name") or loc.get("name"),
        "crm_id": loc.get("crm_id"),
        "is_active": bool(loc.get("is_active", 1)),
    }


def _session_user_row(request: Request) -> dict[str, Any] | None:
    user = request.session.get("user")
    if not user:
        return None
    return _db.get_user(str(user))


def _session_user_display_name(user_row: dict[str, Any] | None, fallback: str | None = None) -> str | None:
    if user_row:
        full_name = str(user_row.get("full_name") or "").strip()
        if full_name:
            return full_name
        display_name = str(user_row.get("display_name") or "").strip()
        if display_name:
            return display_name
    fb = str(fallback or "").strip()
    return fb or None


def _session_auth_payload(request: Request) -> dict[str, Any]:
    if not auth_enabled():
        return {
            "auth_enabled": False,
            "auth_type": auth_mode(),
            "user": None,
            "display_name": None,
            "department": None,
            "department_label": None,
            "role": USER_ROLE_ADMIN,
            "is_admin": True,
            "onboarding_seen_version": 0,
            "onboarding_current_version": _ONBOARDING_TOUR_VERSION,
            "onboarding_completed": False,
        }
    user = request.session.get("user")
    user_row = _session_user_row(request)
    role = _effective_user_role(user, (user_row or {}).get("role") or request.session.get("role"))
    location = _user_location_payload(user_row)
    display_name = _session_user_display_name(user_row, request.session.get("display_name"))
    department = _normalize_department_code((user_row or {}).get("department"))
    onboarding_seen_version = _user_onboarding_version(user_row)
    return {
        "auth_enabled": auth_enabled(),
        "auth_type": auth_mode(),
        "user": user,
        "display_name": display_name,
        "full_name": str((user_row or {}).get("full_name") or display_name or "").strip() or None,
        "location": location,
        "location_id": (location or {}).get("id"),
        "department": department,
        "department_label": _department_label(department),
        "profile_complete": bool(user and _user_profile_complete(user_row)),
        "role": role if user else None,
        "is_admin": bool(user and role == USER_ROLE_ADMIN),
        "onboarding_seen_version": onboarding_seen_version,
        "onboarding_current_version": _ONBOARDING_TOUR_VERSION,
        "onboarding_completed": bool(user and onboarding_seen_version >= _ONBOARDING_TOUR_VERSION),
    }


def _require_admin(request: Request) -> None:
    if not _session_is_admin(request):
        raise HTTPException(403, "Доступно только admin")


def _require_authenticated_user(request: Request) -> dict[str, Any]:
    user_row = _session_user_row(request)
    if not user_row:
        raise HTTPException(401, "Требуется вход в систему")
    return user_row


def _require_completed_profile(request: Request) -> dict[str, Any]:
    user_row = _require_authenticated_user(request)
    if not _user_profile_complete(user_row):
        raise HTTPException(409, "Сначала заполните профиль: ФИО, локацию и отдел")
    return user_row


def _request_access_context(request: Request) -> dict[str, Any]:
    if not auth_enabled():
        return {
            "user": None,
            "location_id": None,
            "department": None,
            "is_admin": True,
        }
    user_row = _require_authenticated_user(request)
    username = str(user_row.get("username") or request.session.get("user") or "").strip() or None
    return {
        "user": username,
        "location_id": str(user_row.get("location_id") or "").strip() or None,
        "department": _normalize_department_code(user_row.get("department")),
        "is_admin": _session_is_admin(request),
    }


def _utc_day_window(now_dt: datetime | None = None) -> tuple[str, str]:
    base = now_dt.astimezone(timezone.utc) if now_dt else datetime.now(timezone.utc)
    day_start = datetime(base.year, base.month, base.day, tzinfo=timezone.utc)
    next_day = day_start + timedelta(days=1)
    return day_start.isoformat(), next_day.isoformat()


def _coerce_optional_limit(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0, out)


def _effective_user_queue_limits(user_row: dict[str, Any] | None) -> dict[str, int]:
    return {
        "daily_upload_limit": _coerce_optional_limit((user_row or {}).get("daily_upload_limit"))
        if (user_row or {}).get("daily_upload_limit") is not None
        else _configured_default_daily_upload_limit(),
        "max_queued_jobs": _coerce_optional_limit((user_row or {}).get("max_queued_jobs"))
        if (user_row or {}).get("max_queued_jobs") is not None
        else _configured_default_max_queued_jobs(),
        "max_running_jobs": _coerce_optional_limit((user_row or {}).get("max_running_jobs"))
        if (user_row or {}).get("max_running_jobs") is not None
        else _configured_default_max_running_jobs(),
    }


def _user_queue_usage(username: str | None) -> dict[str, int]:
    normalized = str(username or "").strip()
    if not normalized:
        return {"daily_uploaded_count": 0, "queued_count": 0, "running_count": 0}
    start_at, end_at = _utc_day_window()
    return {
        "daily_uploaded_count": _db.count_user_uploaded_videos_between(
            normalized, start_at=start_at, end_at=end_at
        ),
        "queued_count": _db.count_user_jobs(normalized, statuses=("queued",)),
        "running_count": _db.count_user_jobs(normalized, statuses=("running",)),
    }


def _upload_quota_payload(user_row: dict[str, Any] | None) -> dict[str, Any]:
    global_queue_full = _db.count_jobs(statuses=("queued",)) >= _configured_max_queue_depth()
    if not auth_enabled():
        return {
            "auth_enabled": False,
            "daily_limit": None,
            "daily_uploaded_count": 0,
            "daily_remaining": None,
            "queued_limit": None,
            "queued_count": 0,
            "running_limit": None,
            "running_count": 0,
            "blocked": global_queue_full,
            "blocked_reasons": ["Глобальная очередь обработки заполнена"] if global_queue_full else [],
            "reset_at": _utc_day_window()[1],
        }
    username = str((user_row or {}).get("username") or "").strip() or None
    if not username:
        raise HTTPException(401, "Требуется вход в систему")
    limits = _effective_user_queue_limits(user_row)
    usage = _user_queue_usage(username)
    daily_limit = int(limits["daily_upload_limit"])
    queued_limit = int(limits["max_queued_jobs"])
    running_limit = int(limits["max_running_jobs"])
    remaining = max(0, daily_limit - usage["daily_uploaded_count"])
    blocked_reasons: list[str] = []
    if remaining <= 0:
        blocked_reasons.append("Дневная квота загрузок исчерпана")
    if usage["queued_count"] >= queued_limit:
        blocked_reasons.append("Превышен лимит ожидающих задач пользователя")
    if global_queue_full:
        blocked_reasons.append("Глобальная очередь обработки заполнена")
    return {
        "auth_enabled": True,
        "user": username,
        "daily_limit": daily_limit,
        "daily_uploaded_count": usage["daily_uploaded_count"],
        "daily_remaining": remaining,
        "queued_limit": queued_limit,
        "queued_count": usage["queued_count"],
        "running_limit": running_limit,
        "running_count": usage["running_count"],
        "reset_at": _utc_day_window()[1],
        "blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
    }


def _assert_upload_intake_allowed(user_row: dict[str, Any] | None) -> dict[str, Any]:
    quota = _upload_quota_payload(user_row)
    if quota.get("daily_remaining") is not None and int(quota.get("daily_remaining") or 0) <= 0:
        raise HTTPException(429, "Дневная квота загрузок исчерпана. Попробуйте завтра.")
    if quota.get("queued_limit") is not None and int(quota.get("queued_count") or 0) >= int(
        quota.get("queued_limit") or 0
    ):
        raise HTTPException(429, "У пользователя слишком много задач в очереди. Дождитесь завершения обработки.")
    if quota.get("blocked"):
        reasons = quota.get("blocked_reasons") or []
        raise HTTPException(429, str(reasons[0]) if reasons else "Загрузка временно недоступна")
    return quota


def _next_video_destination(filename: str) -> Path:
    stem = _safe_stem(filename)
    ext = Path(filename).suffix.lower()
    dest = VIDEO_DIR / f"{stem}{ext}"
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = VIDEO_DIR / f"{stem}_{suffix}{ext}"
    return dest


async def _write_upload_temp_file(file: UploadFile, temp_path: Path) -> tuple[str, int]:
    sha256 = hashlib.sha256()
    total_size = 0
    with temp_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            sha256.update(chunk)
            total_size += len(chunk)
    return sha256.hexdigest(), total_size


def _video_owner_username(meta: dict[str, Any]) -> str | None:
    return str(meta.get("uploaded_by") or meta.get("manager_id") or "").strip() or None


def _video_location_id(meta: dict[str, Any]) -> str | None:
    return (
        str(meta.get("location_id") or meta.get("uploader_location_id") or "").strip() or None
    )


def _video_department_code(meta: dict[str, Any]) -> str | None:
    snapshot = _normalize_department_code(meta.get("checklist_department_snapshot"))
    if snapshot:
        return snapshot
    training_type = _get_training_type_or_none(meta.get("training_type_slug"))
    return _normalize_department_code((training_type or {}).get("department"))


def _video_permissions(
    meta: dict[str, Any],
    context: dict[str, Any],
    *,
    include_foreign_scope: bool,
) -> dict[str, Any]:
    owner = _video_owner_username(meta)
    is_admin = bool(context.get("is_admin"))
    current_user = str(context.get("user") or "").strip() or None
    is_owner = bool(current_user and owner and current_user == owner)
    same_scope = False
    if not is_admin and not is_owner and include_foreign_scope:
        same_scope = bool(
            context.get("location_id")
            and _video_location_id(meta)
            and context.get("department")
            and _video_department_code(meta)
            and context["location_id"] == _video_location_id(meta)
            and context["department"] == _video_department_code(meta)
        )
    can_view = is_admin or is_owner or same_scope
    can_mutate = is_admin or is_owner
    return {
        "owner_username": owner,
        "is_owner": is_owner,
        "is_admin": is_admin,
        "scope_match": same_scope,
        "can_view": can_view,
        "read_only": bool(can_view and not can_mutate),
        "can_delete": can_mutate,
        "can_restore": can_mutate,
        "can_re_evaluate": is_admin,
        "can_save_human_eval": can_mutate,
        "can_publish_human_eval": can_mutate,
        "can_compare": can_mutate,
        "can_restart_comparison": is_admin,
        "can_control_jobs": is_admin,
        "can_restart_failed_pipeline": can_mutate,
        "can_view_ai_log": is_admin,
        "can_edit_meta": is_admin,
    }


def _require_video_permissions(
    request: Request,
    stem_key: str,
    *,
    include_foreign_scope: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = _db.get_video_meta(stem_key)
    if not meta:
        raise HTTPException(404, "Запись не найдена")
    permissions = _video_permissions(
        meta,
        _request_access_context(request),
        include_foreign_scope=include_foreign_scope,
    )
    if not permissions.get("can_view"):
        raise HTTPException(403, "Запись недоступна")
    return meta, permissions


def _require_video_write_access(request: Request, stem_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    meta, permissions = _require_video_permissions(request, stem_key, include_foreign_scope=True)
    if not permissions.get("can_delete"):
        raise HTTPException(403, "Запись доступна только для чтения")
    return meta, permissions


def _require_video_admin_write_access(
    request: Request,
    stem_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta, permissions = _require_video_write_access(request, stem_key)
    if not permissions.get("is_admin"):
        raise HTTPException(403, "Доступно только admin")
    return meta, permissions


def _set_auth_session(request: Request, user_row: dict[str, Any]) -> dict[str, Any]:
    username = str(user_row.get("username") or "").strip()
    role = _effective_user_role(username, user_row.get("role"))
    display_name = _session_user_display_name(user_row, username)
    department = _normalize_department_code(user_row.get("department"))
    onboarding_seen_version = _user_onboarding_version(user_row)
    request.session["user"] = username
    request.session["display_name"] = display_name
    request.session["role"] = role
    request.session["auth_type"] = str(user_row.get("auth_source") or auth_mode())
    request.session["profile_complete"] = _user_profile_complete(user_row)
    return {
        "ok": True,
        "user": username,
        "display_name": display_name,
        "full_name": str(user_row.get("full_name") or display_name or "").strip() or None,
        "role": role,
        "auth_type": str(user_row.get("auth_source") or auth_mode()),
        "is_admin": role == USER_ROLE_ADMIN,
        "profile_complete": _user_profile_complete(user_row),
        "location": _user_location_payload(user_row),
        "department": department,
        "department_label": _department_label(department),
        "onboarding_seen_version": onboarding_seen_version,
        "onboarding_current_version": _ONBOARDING_TOUR_VERSION,
        "onboarding_completed": onboarding_seen_version >= _ONBOARDING_TOUR_VERSION,
    }


def _resolve_criteria_query_param(raw: str | None, stem: str | None = None) -> str:
    """Slug чеклиста из query; при пустом/некорректном — связанный с тренировкой или активный."""
    fallback = _db.resolve_checklist_slug_for_video(stem) if stem else _db.get_active_checklist_slug()
    if not raw or not str(raw).strip():
        return fallback
    try:
        name = _safe_criteria_filename(str(raw).strip())
    except HTTPException:
        return fallback
    if _db.checklist_exists(name):
        return name
    return fallback


def _get_training_type_or_none(slug: str | None) -> dict[str, Any] | None:
    name = str(slug or "").strip()
    if not name:
        return None
    return _db.get_training_type(name)


def _current_checklist_meta_for_training_type(training_type_slug: str | None) -> dict[str, Any] | None:
    training_type = _get_training_type_or_none(training_type_slug)
    checklist_slug = str((training_type or {}).get("checklist_slug") or "").strip()
    if not checklist_slug:
        return None
    meta = _db.get_checklist_meta(checklist_slug)
    if not meta:
        return None
    out = dict(meta)
    out["training_type"] = training_type
    return out


def _checklist_stale_payload(video_meta: dict[str, Any]) -> dict[str, Any] | None:
    snapshot_slug = str(video_meta.get("checklist_slug_snapshot") or "").strip()
    snapshot_version = str(video_meta.get("checklist_version_snapshot") or "").strip()
    current_meta = _current_checklist_meta_for_training_type(video_meta.get("training_type_slug"))
    if not snapshot_slug or not snapshot_version or not current_meta:
        return None
    current_slug = str(current_meta.get("slug") or "").strip()
    current_version = str(current_meta.get("version") or "").strip()
    stale = snapshot_slug != current_slug or snapshot_version != current_version
    return {
        "is_stale": stale,
        "snapshot_slug": snapshot_slug,
        "snapshot_version": snapshot_version,
        "current_slug": current_slug,
        "current_version": current_version,
        "current_display_name": current_meta.get("display_name"),
        "current_department": current_meta.get("department"),
    }


def _resolve_checklist_content(stem: str, criteria_name: str) -> dict[str, Any]:
    data = _db.get_checklist_content(criteria_name)
    if data:
        return data
    fallback_slug = _db.resolve_checklist_slug_for_video(stem)
    fallback = _db.get_checklist_content(fallback_slug)
    if fallback:
        return fallback
    raise HTTPException(404, "Чеклист не найден")


def _resolve_bound_checklist_slug_or_raise(stem: str, raw: str | None = None) -> str:
    meta = _db.get_video_meta(stem)
    training_type = _get_training_type_or_none(meta.get("training_type_slug"))
    if training_type:
        checklist_slug = str(training_type.get("checklist_slug") or "").strip()
        if not checklist_slug:
            raise ValueError("У выбранного типа тренировки не привязан чеклист")
    return _resolve_criteria_query_param(raw, stem)


def _normalized_ai_eval(stem: str, criteria_name: str, criteria_content: dict[str, Any]) -> dict[str, Any] | None:
    raw = _load_evaluation_for_stem(stem, criteria_name)
    return normalize_loaded_evaluation(raw, list(criteria_content.get("criteria") or []))


def _normalized_human_eval(stem: str, criteria_name: str, criteria_content: dict[str, Any]) -> dict[str, Any] | None:
    raw = _load_human_eval_for_stem(stem, criteria_name)
    return normalize_loaded_evaluation(raw, list(criteria_content.get("criteria") or []))


def _load_evaluation_for_stem(stem: str, criteria_name: str) -> dict | None:
    """Оценка для пары (stem, чеклист); legacy: stem__criteria.yaml.eval.json и stem.json для дефолта."""
    path = EVALUATION_DIR / f"{stem}__{criteria_name}.eval.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
    if criteria_name == DEFAULT_CHECKLIST_SLUG:
        alt = EVALUATION_DIR / f"{stem}__{CRITERIA_FILE.name}.eval.json"
        if alt.is_file():
            try:
                return json.loads(alt.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                return None
    if criteria_name in (DEFAULT_CHECKLIST_SLUG, CRITERIA_FILE.name):
        legacy = EVALUATION_DIR / f"{stem}.json"
        if legacy.is_file():
            try:
                return json.loads(legacy.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                return None
    return None


def _load_human_eval_for_stem(stem: str, criteria_name: str) -> dict | None:
    path = human_evaluation_json_path(stem, criteria_name)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    if criteria_name == DEFAULT_CHECKLIST_SLUG:
        legacy = EVALUATION_DIR / f"{stem}__{CRITERIA_FILE.name}.human.json"
        if legacy.is_file():
            try:
                return json.loads(legacy.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _save_human_eval_for_stem(stem: str, criteria_name: str, data: dict) -> Path:
    path = human_evaluation_json_path(stem, criteria_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _stem_has_any_evaluation(stem: str) -> bool:
    if (EVALUATION_DIR / f"{stem}.json").is_file():
        return True
    return any(EVALUATION_DIR.glob(f"{stem}__*.eval.json"))


_COMPARE_EVAL_PAYLOAD_VERSION = 2


def _build_compare_eval_payload(
    ai_eval: dict[str, Any],
    human_eval: dict[str, Any],
) -> dict[str, Any]:
    ai_criteria = {c["id"]: c for c in (ai_eval.get("criteria") or [])}
    human_criteria = {c["id"]: c for c in (human_eval.get("criteria") or [])}
    comparison_rows: list[dict[str, Any]] = []
    all_ids = list(
        dict.fromkeys(
            [c["id"] for c in ai_eval.get("criteria", [])]
            + [c["id"] for c in human_eval.get("criteria", [])]
        )
    )
    for cid in all_ids:
        ai_c = ai_criteria.get(cid, {})
        hu_c = human_criteria.get(cid, {})
        weight = ai_c.get("weight") or hu_c.get("weight") or 1
        ai_passed = ai_c.get("passed")
        hu_passed = hu_c.get("passed")
        same = ai_passed == hu_passed if ai_passed is not None and hu_passed is not None else None
        comparison_rows.append(
            {
                "id": cid,
                "name": ai_c.get("name") or hu_c.get("name") or cid,
                "weight": weight,
                "ai_passed": ai_passed,
                "ai_comment": ai_c.get("comment", ""),
                "human_passed": hu_passed,
                "human_comment": hu_c.get("comment", ""),
                "same": same,
                "ai_awarded_weight": ai_c.get("awarded_weight", 0),
                "human_awarded_weight": hu_c.get("awarded_weight", 0),
            }
        )
    compared_count = sum(
        1
        for row in comparison_rows
        if row.get("ai_passed") is not None or row.get("human_passed") is not None
    )
    mismatch_count = sum(
        1
        for row in comparison_rows
        if (row.get("ai_passed") is not None or row.get("human_passed") is not None)
        and row.get("ai_passed") != row.get("human_passed")
    )
    ai_percent = ai_eval.get("overall_average")
    human_percent = human_eval.get("overall_average")
    overall_diff = (
        round(mismatch_count * 100.0 / compared_count, 1)
        if compared_count > 0
        else None
    )
    return {
        "compare_version": _COMPARE_EVAL_PAYLOAD_VERSION,
        "ai_overall": ai_eval.get("earned_score"),
        "human_overall": human_eval.get("earned_score"),
        "ai_percent": ai_percent,
        "human_percent": human_percent,
        "max_score": ai_eval.get("max_score") or human_eval.get("max_score"),
        "overall_diff": overall_diff,
        "mismatch_count": mismatch_count,
        "compared_count": compared_count,
        "status_color": _comparison_status_color(overall_diff),
        "ai_signature": _comparison_eval_signature(ai_eval),
        "human_signature": _comparison_eval_signature(human_eval),
        "rows": comparison_rows,
    }


def _comparison_eval_signature(evaluation: dict[str, Any]) -> dict[str, Any]:
    criteria = sorted(
        [
            {
                "id": str(row.get("id") or ""),
                "passed": row.get("passed"),
                "weight": row.get("weight"),
                "comment": str(row.get("comment") or ""),
                "awarded_weight": row.get("awarded_weight"),
            }
            for row in (evaluation.get("criteria") or [])
        ],
        key=lambda row: row["id"],
    )
    return {
        "evaluated_at": str(evaluation.get("evaluated_at") or ""),
        "earned_score": evaluation.get("earned_score"),
        "max_score": evaluation.get("max_score"),
        "overall_average": evaluation.get("overall_average"),
        "criteria": criteria,
    }


def _comparison_payload_is_current(
    payload: dict[str, Any] | None,
    ai_eval: dict[str, Any],
    human_eval: dict[str, Any],
) -> bool:
    if not payload:
        return False
    return (
        payload.get("compare_version") == _COMPARE_EVAL_PAYLOAD_VERSION
        and payload.get("ai_signature") == _comparison_eval_signature(ai_eval)
        and payload.get("human_signature") == _comparison_eval_signature(human_eval)
    )


def _format_comparison_diff_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "—"
    overall_diff = payload.get("overall_diff")
    mismatch_count = payload.get("mismatch_count")
    compared_count = payload.get("compared_count")
    if overall_diff is None:
        return "—"
    if isinstance(mismatch_count, (int, float)) and isinstance(compared_count, (int, float)) and compared_count:
        return f"{float(overall_diff):.1f}% ({int(mismatch_count)} из {int(compared_count)} критериев)"
    return f"{float(overall_diff):.1f}%"


def _is_deepseek_v4_model_name(name: str | None) -> bool:
    return str(name or "").strip().lower().startswith("deepseek-v4-")


def _ensure_published_eval_comparison(
    stem: str,
    criteria_name: str,
    *,
    criteria_content: dict[str, Any] | None = None,
    actor_username: str | None = None,
    attach_llm: bool = True,
    force: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    state = _db.get_human_eval_state(stem, criteria_name)
    if not state.get("published_at"):
        return None, state
    if criteria_content is None:
        criteria_content = _resolve_checklist_content(stem, criteria_name)
    ai_eval = _normalized_ai_eval(stem, criteria_name, criteria_content)
    human_eval = _normalized_human_eval(stem, criteria_name, criteria_content)
    if not ai_eval or not human_eval:
        return None, state
    existing = _db.get_evaluation_comparison(stem, criteria_name)
    existing_payload = (existing or {}).get("payload")
    existing_is_current = (not force) and _comparison_payload_is_current(existing_payload, ai_eval, human_eval)
    needs_llm_enrichment = bool(
        attach_llm
        and existing_is_current
        and existing_payload
        and not existing_payload.get("llm_analysis")
    )
    if existing_is_current and not needs_llm_enrichment:
        if not state.get("compared_at"):
            state = _db.mark_human_eval_compared(stem, criteria_name, actor_username)
        return existing_payload, state
    payload = (
        dict(existing_payload)
        if existing_is_current and existing_payload
        else _build_compare_eval_payload(ai_eval, human_eval)
    )
    if attach_llm:
        payload = _attach_compare_llm_analysis(payload)
    _db.save_evaluation_comparison(stem, criteria_name, payload)
    state = _db.mark_human_eval_compared(stem, criteria_name, actor_username)
    return payload, state


def _attach_compare_llm_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload or payload.get("llm_analysis"):
        return payload
    comparison_rows = list(payload.get("rows") or [])
    llm_analysis: str | None = None
    try:
        from openai import OpenAI as _OAI
        import os as _os

        base = _os.environ.get("OPENAI_BASE_URL")
        key = _os.environ.get("OPENAI_API_KEY", "")
        if key:
            kw_cl: dict[str, Any] = {"api_key": key}
            if base:
                kw_cl["base_url"] = base
            client = _OAI(**kw_cl)
            model = _os.environ.get("OPENAI_EVAL_MODEL", "deepseek-v4-pro")
            rows_text = "\n".join(
                f"- {r['name']} (id={r['id']}, вес={r['weight']}): ИИ={r['ai_passed']}, Человек={r['human_passed']}, совпадение={r['same']}"
                for r in comparison_rows
            )
            prompt = (
                "Сравни два чеклиста оценки звонка/визита: ИИ и человек. "
                "Кратко, по-русски, без таблиц. "
                "Ответ адресуй пользователю интерфейса, а не разработчику системы. "
                "Дай: 1) общий вывод, 2) 2-4 ключевых расхождения, "
                "3) 2-3 практические рекомендации по этой конкретной записи для проверяющего или сотрудника. "
                "Не предлагай менять промпт, чеклист, модель, систему, интерфейс, процесс или настройки. "
                "Не давай системных рекомендаций и не ссылайся на внутренние инструкции. "
                f"\nИтог ИИ: {payload.get('ai_overall')} / {payload.get('max_score')} ({payload.get('ai_percent')}%)"
                f"\nИтог Человек: {payload.get('human_overall')} / {payload.get('max_score')} ({payload.get('human_percent')}%)"
                f"\nДоля несовпавших ответов: {_format_comparison_diff_text(payload)}"
                f"\nПострочно:\n{rows_text}"
            )
            create_kw: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты эксперт по аудиту качества продаж и клиентского сервиса. "
                            "Пиши только пользовательские выводы по конкретной записи. "
                            "Не предлагай менять систему, промпты, чеклисты, модель или настройки."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            if _is_deepseek_v4_model_name(model):
                effort = str(_os.environ.get("OPENAI_EVAL_REASONING_EFFORT", "high") or "high").strip().lower()
                if effort not in {"high", "max"}:
                    effort = "high"
                create_kw["reasoning_effort"] = effort
                create_kw["extra_body"] = {"thinking": {"type": "enabled"}}
            elif "reasoner" not in str(model or "").lower():
                create_kw["temperature"] = 0.2
            resp = client.chat.completions.create(
                **create_kw,
            )
            llm_analysis = (resp.choices[0].message.content or "").strip()
    except Exception:
        llm_analysis = None

    if not llm_analysis:
        return payload
    enriched = dict(payload)
    enriched["llm_analysis"] = llm_analysis
    return enriched


def _comparison_task_key(stem: str, criteria_name: str) -> tuple[str, str]:
    return (str(stem or "").strip(), str(criteria_name or "").strip())


def _comparison_runtime_payload(stem: str, criteria_name: str) -> dict[str, Any]:
    key = _comparison_task_key(stem, criteria_name)
    with _comparison_tasks_lock:
        return {
            "pending": key in _comparison_running_keys,
            "error": _comparison_errors.get(key) or None,
        }


def _trigger_eval_comparison_background(
    stem: str,
    criteria_name: str,
    *,
    criteria_content: dict[str, Any] | None = None,
    actor_username: str | None = None,
    attach_llm: bool = True,
    force: bool = False,
) -> bool:
    key = _comparison_task_key(stem, criteria_name)
    if not key[0] or not key[1]:
        return False
    with _comparison_tasks_lock:
        if key in _comparison_running_keys:
            return False
        _comparison_running_keys.add(key)
        _comparison_errors.pop(key, None)

    def _runner() -> None:
        try:
            _ensure_published_eval_comparison(
                stem,
                criteria_name,
                criteria_content=criteria_content,
                actor_username=actor_username,
                attach_llm=attach_llm,
                force=force,
            )
        except Exception as exc:
            with _comparison_tasks_lock:
                _comparison_errors[key] = str(exc) or exc.__class__.__name__
        finally:
            with _comparison_tasks_lock:
                _comparison_running_keys.discard(key)

    threading.Thread(target=_runner, name=f"compare-{key[0]}", daemon=True).start()
    return True


def _run_eval_only_job(job_id: str, stem_key: str, criteria_name: str | None = None) -> None:
    tr = TRANSCRIPT_DIR / f"{stem_key}.json"
    try:
        with _job_cancel_lock:
            ev = _job_cancel_events.get(job_id)

        def cancel_check() -> None:
            if ev is not None and ev.is_set():
                raise PipelineCancelled()
            jx = _db.get_job(job_id)
            if jx and jx.get("status") == "cancelled":
                raise PipelineCancelled()

        j0 = _db.get_job(job_id)
        if j0 and j0.get("status") == "cancelled":
            return

        _job_set(job_id, status="running", stage="eval_prep", updated_at=_utc_now())

        if not tr.is_file():
            _job_set(
                job_id,
                status="error",
                stage="error",
                error="Нет файла транскрипта — сначала дождитесь распознавания.",
                updated_at=_utc_now(),
            )
            return

        slug = criteria_name if criteria_name else _resolve_bound_checklist_slug_or_raise(stem_key)
        if not _db.checklist_exists(slug):
            raise FileNotFoundError(f"Чеклист не найден в БД: {slug}")
        cpath = Path(slug)
        cancel_check()
        _job_set(job_id, stage="evaluating", stream_log="", updated_at=_utc_now())
        stream_cb = _job_stream_callback_throttled(job_id)
        evaluate_only_from_transcript(
            tr,
            criteria_path=cpath,
            cancel_check=cancel_check,
            db=_db,
            stream_callback=stream_cb,
        )
        _trigger_eval_comparison_background(stem_key, slug, actor_username=None, attach_llm=True)
        j_done = _db.get_job(job_id)
        if j_done and j_done.get("status") == "cancelled":
            return
        _job_set(
            job_id,
            status="done",
            stage="done",
            stem=stem_key,
            stream_log="",
            updated_at=_utc_now(),
        )
    except PipelineCancelled:
        _job_set(
            job_id,
            status="cancelled",
            stage="cancelled",
            updated_at=_utc_now(),
        )
    except Exception as e:
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=str(e),
            stream_log="",
            updated_at=_utc_now(),
        )
    finally:
        with _job_cancel_lock:
            _job_cancel_events.pop(job_id, None)


def _job_set(job_id: str, **updates: object) -> None:
    _db.upsert_job(job_id, **updates)


def _job_stream_callback_throttled(job_id: str):
    """Обновляет jobs.stream_log для live-текста оценки (без агрессивного throttle — иначе мелкие чанки не доходят до UI)."""
    state: dict[str, str | None] = {"last": None}

    def cb(text: str) -> None:
        s = text if isinstance(text, str) else ""
        if s == state["last"]:
            return
        state["last"] = s
        _job_set(job_id, stream_log=s, updated_at=_utc_now())

    return cb


def _stem_from_path_param(stem: str) -> str:
    """Stem из URL: без путей и «..»."""
    s = (stem or "").strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        raise HTTPException(400, "Некорректный идентификатор")
    return s[:500]


def _safe_stem(name: str) -> str:
    base = Path(name).name
    stem = Path(base).stem
    stem = re.sub(r"[^\w\s\-_.]", "", stem, flags=re.UNICODE)
    stem = stem.strip("._- ") or "video"
    return stem[:200]


def _delete_pipeline_derived_artifacts(stem_key: str) -> None:
    """Транскрипт, тон и оценки — без видео и без meta (для «с начала»)."""
    delete_derived_artifacts_for_stem(stem_key)


def _job_args(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("args_json")
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_expected_speaker_count(value: Any, default: int | None = 2) -> int | None:
    if value is None or value == "":
        return default
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(out, 8))


def _job_owner_username_from_row(row: dict[str, Any]) -> str | None:
    return str(row.get("uploaded_by") or "").strip() or None


def _effective_user_running_limit(username: str | None) -> int:
    normalized = str(username or "").strip()
    if not normalized:
        return max(1, _configured_default_max_running_jobs())
    user_row = _db.get_user(normalized)
    return max(0, _effective_user_queue_limits(user_row).get("max_running_jobs", 0))


def _select_jobs_for_dispatch(queued_jobs: list[dict[str, Any]], slots: int) -> list[dict[str, Any]]:
    if slots <= 0 or not queued_jobs:
        return []
    running_counts = _db.count_running_jobs_by_user()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cycle_users: set[str] = set()
    while len(selected) < slots:
        progressed = False
        for row in queued_jobs:
            job_id = str(row.get("id") or "").strip()
            if not job_id or job_id in selected_ids:
                continue
            owner = _job_owner_username_from_row(row)
            owner_key = owner.lower() if owner else ""
            if owner_key:
                limit = _effective_user_running_limit(owner)
                current = int(running_counts.get(owner_key, 0))
                if limit <= 0 or current >= limit or owner_key in cycle_users:
                    continue
            selected.append(row)
            selected_ids.add(job_id)
            progressed = True
            if owner_key:
                running_counts[owner_key] = int(running_counts.get(owner_key, 0)) + 1
                cycle_users.add(owner_key)
            if len(selected) >= slots:
                break
        if len(selected) >= slots or not progressed:
            break
        cycle_users.clear()
    return selected


def _run_dispatched_job(job_id: str) -> None:
    try:
        job = _db.get_job(job_id)
        if not job:
            return
        if str(job.get("status") or "").strip().lower() == "cancelled":
            return
        kind = str(job.get("kind") or "").strip().lower()
        stem_key = str(job.get("stem") or "").strip()
        args = _job_args(job)
        speaker_count = _normalize_expected_speaker_count(args.get("speaker_count"), default=None)
        if speaker_count is None and stem_key:
            meta = _db.get_video_meta(stem_key)
            speaker_count = _normalize_expected_speaker_count(meta.get("expected_speaker_count"), default=2)
        with _job_cancel_lock:
            _job_cancel_events.setdefault(job_id, threading.Event())
        if kind == "pipeline":
            video_path = find_video_for_stem(stem_key)
            if not video_path or not video_path.is_file():
                _job_set(
                    job_id,
                    status="error",
                    stage="error",
                    error="Нет исходного медиафайла в 01.Video для обработки.",
                    stream_log="",
                    updated_at=_utc_now(),
                )
                return
            if bool(args.get("resume")):
                _run_pipeline_resume_job(job_id, video_path, expected_speaker_count=speaker_count)
            else:
                _run_pipeline_job(job_id, video_path, expected_speaker_count=speaker_count)
            return
        if kind == "eval_only":
            criteria_name = str(args.get("criteria_name") or "").strip() or None
            _run_eval_only_job(job_id, stem_key, criteria_name)
            return
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=f"Неподдерживаемый тип задачи: {kind or 'unknown'}",
            stream_log="",
            updated_at=_utc_now(),
        )
    finally:
        with _running_jobs_lock:
            _running_job_ids.discard(job_id)
        _dispatch_jobs()


def _dispatch_jobs() -> None:
    with _dispatch_lock:
        while True:
            with _running_jobs_lock:
                slots = max(0, _MAX_WORKERS - len(_running_job_ids))
            if slots <= 0:
                return
            queued_jobs = _db.list_dispatchable_jobs(limit=max(64, slots * 8))
            selected = _select_jobs_for_dispatch(queued_jobs, slots)
            if not selected:
                return
            launched = False
            for row in selected:
                job_id = str(row.get("id") or "").strip()
                if not job_id:
                    continue
                with _running_jobs_lock:
                    if job_id in _running_job_ids or len(_running_job_ids) >= _MAX_WORKERS:
                        continue
                    _running_job_ids.add(job_id)
                try:
                    _job_set(
                        job_id,
                        status="running",
                        stage="starting",
                        error="",
                        updated_at=_utc_now(),
                    )
                    _executor.submit(_run_dispatched_job, job_id)
                    launched = True
                except Exception as exc:
                    with _running_jobs_lock:
                        _running_job_ids.discard(job_id)
                    _job_set(
                        job_id,
                        status="error",
                        stage="error",
                        error=str(exc),
                        stream_log="",
                        updated_at=_utc_now(),
                    )
            if not launched:
                return


def _recover_jobs_after_restart() -> None:
    recovered = _db.requeue_jobs(from_statuses=("running",), stage="queued")
    if recovered:
        _audit_log(
            None,
            action="system.jobs.recovered",
            target_type="job",
            target_id="restart",
            details={"requeued_count": recovered},
        )
    _dispatch_jobs()


def _enqueue_pipeline_job_common(
    job_id: str,
    stem_key: str,
    video_path: Path,
    *,
    resume: bool,
    speaker_count: int | None = None,
) -> None:
    with _job_cancel_lock:
        _job_cancel_events[job_id] = threading.Event()
    normalized_speaker_count = _normalize_expected_speaker_count(speaker_count, default=2)
    _db.upsert_job(
        job_id,
        stem=stem_key,
        kind="pipeline",
        status="queued",
        stage="queued",
        args_json=json.dumps(
            {"resume": bool(resume), "speaker_count": normalized_speaker_count},
            ensure_ascii=False,
        ),
        video_file=video_path.name,
    )
    _db.upsert_video_meta(
        stem_key,
        filename=video_path.name,
        status="processing",
        expected_speaker_count=normalized_speaker_count,
    )
    _dispatch_jobs()


def _run_pipeline_resume_job(
    job_id: str,
    video_path: Path,
    *,
    expected_speaker_count: int | None = None,
) -> None:
    stem = video_path.stem
    try:
        with _job_cancel_lock:
            ev = _job_cancel_events.get(job_id)

        def cancel_check() -> None:
            if ev is not None and ev.is_set():
                raise PipelineCancelled()
            jx = _db.get_job(job_id)
            if jx and jx.get("status") == "cancelled":
                raise PipelineCancelled()

        j0 = _db.get_job(job_id)
        if j0 and j0.get("status") == "cancelled":
            return

        _job_set(job_id, status="running", stage="starting", updated_at=_utc_now())
        stream_cb = _job_stream_callback_throttled(job_id)

        def on_progress(phase: str) -> None:
            jp = _db.get_job(job_id)
            if jp and jp.get("status") == "cancelled":
                raise PipelineCancelled()
            if phase == "evaluating":
                _job_set(job_id, stream_log="", updated_at=_utc_now())
            _job_set(job_id, stage=phase, updated_at=_utc_now())

        criteria_slug = _resolve_bound_checklist_slug_or_raise(stem)
        tr, ev, tone = process_one_video_resume(
            video_path,
            on_progress=on_progress,
            criteria_path=Path(criteria_slug),
            db=_db,
            cancel_check=cancel_check,
            eval_stream_callback=stream_cb,
            expected_speaker_count=expected_speaker_count,
        )
        _trigger_eval_comparison_background(stem, criteria_slug, actor_username=None, attach_llm=True)
        j_done = _db.get_job(job_id)
        if j_done and j_done.get("status") == "cancelled":
            return
        _job_set(
            job_id,
            status="done",
            stage="done",
            stem=stem,
            transcript=str(tr.relative_to(ROOT)),
            evaluation=str(ev.relative_to(ROOT)) if ev else None,
            tone_file=str(tone.relative_to(ROOT)) if tone else None,
            stream_log="",
            updated_at=_utc_now(),
        )
    except PipelineCancelled:
        _job_set(
            job_id,
            status="cancelled",
            stage="cancelled",
            stream_log="",
            updated_at=_utc_now(),
        )
        _db.upsert_video_meta(stem, status="pending")
    except Exception as e:
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=str(e),
            stream_log="",
            updated_at=_utc_now(),
        )
    finally:
        with _job_cancel_lock:
            _job_cancel_events.pop(job_id, None)


def _run_pipeline_job(
    job_id: str,
    video_path: Path,
    *,
    expected_speaker_count: int | None = None,
) -> None:
    stem = video_path.stem
    try:
        with _job_cancel_lock:
            ev = _job_cancel_events.get(job_id)

        def cancel_check() -> None:
            if ev is not None and ev.is_set():
                raise PipelineCancelled()
            jx = _db.get_job(job_id)
            if jx and jx.get("status") == "cancelled":
                raise PipelineCancelled()

        j0 = _db.get_job(job_id)
        if j0 and j0.get("status") == "cancelled":
            return

        _job_set(job_id, status="running", stage="starting", updated_at=_utc_now())
        stream_cb = _job_stream_callback_throttled(job_id)

        def on_progress(phase: str) -> None:
            jp = _db.get_job(job_id)
            if jp and jp.get("status") == "cancelled":
                raise PipelineCancelled()
            if phase == "evaluating":
                _job_set(job_id, stream_log="", updated_at=_utc_now())
            _job_set(job_id, stage=phase, updated_at=_utc_now())

        criteria_slug = _resolve_bound_checklist_slug_or_raise(stem)
        tr, ev, tone = process_one_video(
            video_path,
            on_progress=on_progress,
            criteria_path=Path(criteria_slug),
            db=_db,
            cancel_check=cancel_check,
            eval_stream_callback=stream_cb,
            expected_speaker_count=expected_speaker_count,
        )
        _trigger_eval_comparison_background(stem, criteria_slug, actor_username=None, attach_llm=True)
        j_done = _db.get_job(job_id)
        if j_done and j_done.get("status") == "cancelled":
            return
        _job_set(
            job_id,
            status="done",
            stage="done",
            stem=stem,
            transcript=str(tr.relative_to(ROOT)),
            evaluation=str(ev.relative_to(ROOT)) if ev else None,
            tone_file=str(tone.relative_to(ROOT)) if tone else None,
            stream_log="",
            updated_at=_utc_now(),
        )
    except PipelineCancelled:
        _job_set(
            job_id,
            status="cancelled",
            stage="cancelled",
            stream_log="",
            updated_at=_utc_now(),
        )
        _db.upsert_video_meta(stem, status="pending")
    except Exception as e:
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=str(e),
            stream_log="",
            updated_at=_utc_now(),
        )
    finally:
        with _job_cancel_lock:
            _job_cancel_events.pop(job_id, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jobs_latest_by_stem() -> dict[str, dict]:
    """Последняя задача по каждому stem (по updated_at)."""
    return _db.latest_job_by_stem()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/auth/status")
def auth_status() -> dict[str, Any]:
    return {
        "auth_enabled": auth_enabled(),
        "auth_type": auth_mode(),
        "ad_auth_enabled": ad_auth_enabled(),
        "local_auth_enabled": local_auth_enabled(),
        "registration_allowed": _registration_allowed(),
        "is_admin": not auth_enabled(),
    }


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    out = _session_auth_payload(request)
    out["ad_auth_enabled"] = ad_auth_enabled()
    out["local_auth_enabled"] = local_auth_enabled()
    return out


@app.get("/api/upload/quota")
def api_upload_quota(request: Request) -> dict[str, Any]:
    user_row = _session_user_row(request) if auth_enabled() else None
    return _upload_quota_payload(user_row)


class _LoginBody(BaseModel):
    username: str
    password: str


class _RegisterBody(BaseModel):
    username: str
    password: str
    full_name: str
    location_id: str
    department: str


class _ProfileBody(BaseModel):
    full_name: str
    location_id: str
    department: str


class _ChangePasswordBody(BaseModel):
    current_password: str
    new_password: str
    new_password_repeat: str | None = None


class _TutorialCompleteBody(BaseModel):
    version: int | None = None


@app.get("/api/auth/locations")
def auth_locations() -> list[dict[str, Any]]:
    return [row for row in _db.list_locations() if bool(row.get("is_active", 1))]


@app.post("/api/auth/login")
async def auth_login(request: Request, body: _LoginBody) -> JSONResponse:
    mode = auth_mode()
    username = (body.username or "").strip()
    password = body.password or ""
    if mode == "none":
        raise HTTPException(400, "Вход не включён на сервере")

    if mode == "ad":
        ok, err = verify_user_password(username, password)
        if not ok:
            raise HTTPException(401, err or "Ошибка входа")
        normalized_username = username.strip()
        fetched_name = fetch_ad_display_name(username, password)
        existing = _db.get_user(normalized_username)
        user_row = _db.upsert_user(
            normalized_username,
            password_hash=None,
            display_name=fetched_name or (existing or {}).get("display_name"),
            full_name=(existing or {}).get("full_name") or fetched_name,
            auth_source=AUTH_SOURCE_AD,
            location_id=(existing or {}).get("location_id"),
            department=(existing or {}).get("department"),
            profile_completed_at=(existing or {}).get("profile_completed_at"),
            role=_effective_user_role(normalized_username, (existing or {}).get("role")),
            is_active=bool((existing or {}).get("is_active", 1)),
        )
        if not int(user_row.get("is_active") or 0):
            raise HTTPException(403, "Пользователь отключён")
    elif mode == "local":
        user, err = verify_local_user_password(_db, username, password)
        if not user:
            raise HTTPException(401, err or "Ошибка входа")
        user_row = user
    else:
        raise HTTPException(400, f"Неподдерживаемый AUTH_TYPE: {mode}")

    payload = _set_auth_session(request, user_row)
    return JSONResponse(payload)


@app.post("/api/auth/register")
async def auth_register(request: Request, body: _RegisterBody) -> JSONResponse:
    if not _registration_allowed():
        raise HTTPException(403, "Регистрация доступна только в локальном режиме")
    username = normalize_local_username(body.username)
    if not username:
        raise HTTPException(400, "Укажите логин")
    password_error = validate_local_password_strength(body.password)
    if password_error:
        raise HTTPException(400, password_error)
    if _db.get_user(username):
        raise HTTPException(409, "Пользователь уже существует")
    location_id = str(body.location_id or "").strip()
    if not location_id or not _db.get_location(location_id):
        raise HTTPException(400, "Выберите корректную локацию")
    department = _require_department_code(body.department)
    full_name = str(body.full_name or "").strip()
    if not full_name:
        raise HTTPException(400, "Укажите ФИО")
    user_row = _db.upsert_user(
        username,
        password_hash=hash_local_password(body.password),
        display_name=full_name,
        full_name=full_name,
        auth_source=AUTH_SOURCE_LOCAL,
        location_id=location_id,
        department=department,
        profile_completed_at=_utc_now(),
        role=USER_ROLE_USER,
        is_active=True,
    )
    payload = _set_auth_session(request, user_row)
    return JSONResponse(payload)


@app.put("/api/auth/profile")
async def auth_profile_put(request: Request, body: _ProfileBody) -> JSONResponse:
    user_row = _require_authenticated_user(request)
    full_name = str(body.full_name or "").strip()
    if not full_name:
        raise HTTPException(400, "Укажите ФИО")
    location_id = str(body.location_id or "").strip()
    if not location_id or not _db.get_location(location_id):
        raise HTTPException(400, "Выберите корректную локацию")
    department = _require_department_code(body.department)
    updated = _db.complete_user_profile(
        str(user_row.get("username") or ""),
        full_name=full_name,
        location_id=location_id,
        department=department,
    )
    payload = _set_auth_session(request, updated)
    return JSONResponse(payload)


@app.post("/api/auth/tutorial/complete")
async def auth_tutorial_complete(request: Request, body: _TutorialCompleteBody = _TutorialCompleteBody()) -> JSONResponse:
    if not auth_enabled():
        return JSONResponse(
            {
                "ok": True,
                "onboarding_seen_version": 0,
                "onboarding_current_version": _ONBOARDING_TOUR_VERSION,
                "onboarding_completed": False,
            }
        )
    user_row = _require_authenticated_user(request)
    version = body.version if body.version is not None else _ONBOARDING_TOUR_VERSION
    version = max(_ONBOARDING_TOUR_VERSION, int(version))
    updated = _db.set_user_onboarding_version(str(user_row.get("username") or ""), version)
    payload = _set_auth_session(request, updated)
    payload["ok"] = True
    return JSONResponse(payload)


@app.post("/api/auth/change-password")
async def auth_change_password(request: Request, body: _ChangePasswordBody) -> JSONResponse:
    user_row = _require_authenticated_user(request)
    if _normalize_auth_source(user_row.get("auth_source")) != AUTH_SOURCE_LOCAL or not local_auth_enabled():
        raise HTTPException(403, "Смена пароля доступна только для локальной учётной записи")
    username = str(user_row.get("username") or "").strip()
    if not username:
        raise HTTPException(400, "Не удалось определить пользователя")
    _, current_password_error = verify_local_user_password(_db, username, body.current_password or "")
    if current_password_error:
        raise HTTPException(400, "Неверный текущий пароль")
    new_password = str(body.new_password or "")
    if body.new_password_repeat is not None and new_password != str(body.new_password_repeat or ""):
        raise HTTPException(400, "Новый пароль и повтор не совпадают")
    if body.current_password == new_password:
        raise HTTPException(400, "Новый пароль должен отличаться от текущего")
    password_error = validate_local_password_strength(new_password)
    if password_error:
        raise HTTPException(400, password_error)
    changed = _db.set_user_password_hash(username, hash_local_password(new_password))
    if not changed:
        raise HTTPException(404, "Пользователь не найден")
    return JSONResponse({"ok": True})


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    request.session.clear()
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    training_type_slug: str = Form(""),
    department: str = Form(""),
    speaker_count: int = Form(2),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "Нет имени файла")
    ext = Path(file.filename).suffix.lower()
    if ext not in _MEDIA_UPLOAD_EXTENSIONS:
        raise HTTPException(
            400,
            f"Неподдерживаемый формат: {ext}. Допустимо: {', '.join(sorted(_MEDIA_UPLOAD_EXTENSIONS))}",
        )

    uploader = _require_completed_profile(request) if auth_enabled() else None
    training_slug = _safe_training_type_slug(training_type_slug.strip()) if training_type_slug.strip() else None
    if not training_slug:
        raise HTTPException(400, "Выберите тип тренировки перед загрузкой")
    selected_department = _require_department_code(department, error_message="Выберите отдел перед загрузкой")
    expected_speaker_count = _normalize_expected_speaker_count(speaker_count, default=2) or 2
    training_type = _get_training_type_or_none(training_slug)
    if not training_type:
        raise HTTPException(404, "Тип тренировки не найден")
    training_department = _normalize_department_code((training_type or {}).get("department"))
    if not training_department:
        raise HTTPException(400, "Для типа тренировки не указан отдел")
    if training_department != selected_department:
        raise HTTPException(400, "Тип тренировки не соответствует выбранному отделу")
    checklist_meta = _current_checklist_meta_for_training_type(training_slug)
    if not checklist_meta:
        raise HTTPException(400, "Для типа тренировки не привязан чеклист")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    _assert_upload_intake_allowed(uploader)

    temp_path = VIDEO_DIR / f".upload-{uuid.uuid4().hex}{ext}.part"
    moved_to_final = False
    try:
        sha256, file_size_bytes = await _write_upload_temp_file(file, temp_path)
        duplicate = _db.find_video_by_sha256(sha256)
        if duplicate:
            label = duplicate.get("display_title") or duplicate.get("filename") or duplicate.get("stem")
            raise HTTPException(
                409,
                f"Похожий медиафайл уже загружен: {label} ({duplicate.get('stem')}). Дубликаты блокируются.",
            )

        uploader_name = (
            str((uploader or {}).get("full_name") or (uploader or {}).get("display_name") or "").strip()
            or str((uploader or {}).get("username") or "").strip()
            or None
        )
        uploader_location = _user_location_payload(uploader)

        with _upload_intake_lock:
            _assert_upload_intake_allowed(uploader)
            duplicate = _db.find_video_by_sha256(sha256)
            if duplicate:
                label = duplicate.get("display_title") or duplicate.get("filename") or duplicate.get("stem")
                raise HTTPException(
                    409,
                    f"Похожий медиафайл уже загружен: {label} ({duplicate.get('stem')}). Дубликаты блокируются.",
                )
            dest = _next_video_destination(file.filename)
            temp_path.replace(dest)
            moved_to_final = True
            job_id = str(uuid.uuid4())
            _db.upsert_video_meta(
                dest.stem,
                filename=dest.name,
                display_title=Path(file.filename).stem,
                manager_id=(uploader or {}).get("username"),
                manager_name=uploader_name,
                location_id=(uploader_location or {}).get("id"),
                location_name=(uploader_location or {}).get("name"),
                training_type_slug=training_slug,
                uploaded_by=(uploader or {}).get("username"),
                uploaded_by_name=uploader_name,
                uploader_location_id=(uploader_location or {}).get("id"),
                uploader_location_name=(uploader_location or {}).get("name"),
                uploader_location_crm_id=(uploader_location or {}).get("crm_id"),
                training_type_name_snapshot=training_type.get("name"),
                checklist_slug_snapshot=checklist_meta.get("slug"),
                checklist_display_name_snapshot=checklist_meta.get("display_name"),
                checklist_version_snapshot=checklist_meta.get("version"),
                checklist_department_snapshot=checklist_meta.get("department"),
                expected_speaker_count=expected_speaker_count,
                file_size_bytes=file_size_bytes,
                file_sha256=sha256,
                dedupe_key=sha256,
                status="processing",
            )
            _enqueue_pipeline_job_common(
                job_id,
                dest.stem,
                dest,
                resume=False,
                speaker_count=expected_speaker_count,
            )

        return JSONResponse(
            {
                "job_id": job_id,
                "stem": dest.stem,
                "video_file": dest.name,
                "training_type_slug": training_slug,
                "checklist_slug": checklist_meta.get("slug"),
                "speaker_count": expected_speaker_count,
                "message": "Файл принят и поставлен в очередь на обработку",
            }
        )
    finally:
        await file.close()
        if not moved_to_final and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _job_payload(stem: str, jobs_by: dict[str, dict]) -> dict | None:
    job = jobs_by.get(stem)
    if not job:
        return None
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "kind": job.get("kind"),
        "error": job.get("error"),
        "updated_at": job.get("updated_at"),
        "stream_log": job.get("stream_log"),
    }


def _comparison_status_color(diff_percent: float | None) -> str:
    if diff_percent is None:
        return "off"
    diff_abs = abs(float(diff_percent))
    if diff_abs >= 40:
        return "red"
    if diff_abs >= 20:
        return "yellow"
    return "green"


def _library_status_payload(stem: str, job: dict[str, Any] | None) -> dict[str, Any]:
    if job and job.get("status") == "queued":
        return {
            "code": "queued",
            "tooltip": "Запись в очереди на обработку.",
        }
    if job and job.get("status") == "running":
        return {
            "code": "spinner",
        "tooltip": "Идёт обработка записи, оценка или сравнение.",
        }
    state = _db.get_latest_human_eval_state_for_stem(stem)
    if not state or not state.get("published_at"):
        return {
            "code": "off",
            "tooltip": "Ручная оценка ещё не опубликована.",
        }
    comparison = _db.get_latest_evaluation_comparison_for_stem(stem)
    payload = (comparison or {}).get("payload") or {}
    diff_percent = payload.get("overall_diff")
    code = _comparison_status_color(diff_percent)
    diff_text = _format_comparison_diff_text(payload)
    if code == "red":
        tooltip = f"Сильное расхождение ИИ и человека: {diff_text}"
    elif code == "yellow":
        tooltip = f"Среднее расхождение ИИ и человека: {diff_text}"
    elif code == "green":
        tooltip = f"Незначительное расхождение ИИ и человека: {diff_text}"
    else:
        tooltip = "Оценка опубликована, сравнение ещё не завершено."
    return {
        "code": code,
        "tooltip": tooltip,
        "diff_percent": diff_percent,
        "comparison_available": comparison is not None,
    }


def _library_row(
    stem: str,
    *,
    video_path: Path | None,
    jobs_by: dict[str, dict],
    meta: dict[str, Any] | None = None,
    permissions: dict[str, Any] | None = None,
) -> dict:
    tr = TRANSCRIPT_DIR / f"{stem}.json"
    tone = TRANSCRIPT_DIR / f"{stem}.tone.json"
    ev_legacy = EVALUATION_DIR / f"{stem}.json"

    video_file: str | None = None
    size_bytes = 0
    if video_path is not None and video_path.is_file():
        video_file = video_path.name
        size_bytes = video_path.stat().st_size
    elif tr.is_file():
        try:
            data = json.loads(tr.read_text(encoding="utf-8"))
            vf = data.get("video_file")
            if isinstance(vf, str) and vf:
                video_file = vf
        except (OSError, json.JSONDecodeError):
            pass

    mtimes: list[float] = []
    if video_path is not None and video_path.is_file():
        mtimes.append(video_path.stat().st_mtime)
    if tr.is_file():
        mtimes.append(tr.stat().st_mtime)
    if tone.is_file():
        mtimes.append(tone.stat().st_mtime)
    if ev_legacy.is_file():
        mtimes.append(ev_legacy.stat().st_mtime)
    for p in EVALUATION_DIR.glob(f"{stem}__*.eval.json"):
        if p.is_file():
            mtimes.append(p.stat().st_mtime)
    mtime = max(mtimes) if mtimes else 0.0

    meta = meta or _db.get_video_meta(stem)
    training_type = _get_training_type_or_none(meta.get("training_type_slug"))
    job_payload = _job_payload(stem, jobs_by)
    status_payload = _library_status_payload(stem, job_payload)
    department = _video_department_code(meta)

    return {
        "stem": stem,
        "video_file": video_file or stem,
        "filename": meta.get("filename") or (video_path.name if video_path else None),
        "has_video_file": bool(video_path and video_path.is_file()),
        "size_bytes": size_bytes,
        "mtime": mtime,
        "has_transcript": tr.is_file(),
        "has_tone": tone.is_file(),
        "has_evaluation": _stem_has_any_evaluation(stem),
        "status": meta.get("status"),
        "manager_id": meta.get("manager_id"),
        "manager_name": meta.get("manager_name"),
        "location_id": meta.get("location_id"),
        "location_name": meta.get("location_name"),
        "department": department,
        "department_label": _department_label(department),
        "interaction_date": meta.get("interaction_date"),
        "added_at": meta.get("uploaded_at"),
        "training_type_slug": meta.get("training_type_slug"),
        "training_type_name": (training_type or {}).get("name"),
        "checklist_slug_snapshot": meta.get("checklist_slug_snapshot"),
        "checklist_version_snapshot": meta.get("checklist_version_snapshot"),
        "delete_requested_at": meta.get("delete_requested_at"),
        "delete_requested_by": meta.get("delete_requested_by"),
        "deleted_at": meta.get("deleted_at"),
        "deleted_by": meta.get("deleted_by"),
        "uploaded_by": meta.get("uploaded_by"),
        "uploaded_by_name": meta.get("uploaded_by_name"),
        "tags": meta.get("tags") or [],
        "display_title": meta.get("display_title"),
        "job": job_payload,
        "status_summary": status_payload,
        "permissions": permissions or {},
    }


@app.get("/api/library")
def api_library(
    request: Request,
    include_deleted: bool = False,
    include_foreign: bool = False,
) -> list[dict]:
    """
    Записи из 01.Video плюс любые транскрипты из 02.Transcript без исходного файла в папке
    (чтобы уже обработанные файлы не пропадали из списка).
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    jobs_by = _jobs_latest_by_stem()
    access_context = _request_access_context(request)

    by_stem: dict[str, dict] = {}

    for p in list_videos(VIDEO_DIR):
        stem = p.stem
        meta = _db.get_video_meta(stem)
        permissions = _video_permissions(meta, access_context, include_foreign_scope=include_foreign)
        if not permissions.get("can_view"):
            continue
        row = _library_row(stem, video_path=p, jobs_by=jobs_by, meta=meta, permissions=permissions)
        if not include_deleted and (row.get("delete_requested_at") or row.get("deleted_at")):
            continue
        by_stem[stem] = row

    for tr_path in list_transcripts(TRANSCRIPT_DIR):
        stem = tr_path.stem
        if stem in by_stem:
            continue
        vp = find_video_for_stem(stem)
        meta = _db.get_video_meta(stem)
        permissions = _video_permissions(meta, access_context, include_foreign_scope=include_foreign)
        if not permissions.get("can_view"):
            continue
        row = _library_row(stem, video_path=vp, jobs_by=jobs_by, meta=meta, permissions=permissions)
        if not include_deleted and (row.get("delete_requested_at") or row.get("deleted_at")):
            continue
        by_stem[stem] = row

    out = sorted(by_stem.values(), key=lambda x: x.get("mtime") or 0, reverse=True)
    return out


@app.delete("/api/library/{stem}")
def api_library_delete(request: Request, stem: str) -> dict:
    """Manager: пометка на удаление. Admin: окончательное удаление."""
    stem_key = _stem_from_path_param(stem)
    meta, permissions = _require_video_write_access(request, stem_key)
    if not permissions.get("is_admin"):
        marked = _db.request_video_deletion(stem_key, request.session.get("user"))
        _audit_log(
            request,
            action="video.delete.request",
            target_type="video",
            target_id=stem_key,
            details={"mode": "mark"},
        )
        return {"ok": True, "mode": "mark", "meta": marked}

    deleted: list[str] = []

    vp = find_video_for_stem(stem_key)
    if vp and vp.is_file():
        vp.unlink()
        deleted.append(str(vp.relative_to(ROOT)))

    tr = TRANSCRIPT_DIR / f"{stem_key}.json"
    if tr.is_file():
        tr.unlink()
        deleted.append(str(tr.relative_to(ROOT)))

    tone = TRANSCRIPT_DIR / f"{stem_key}.tone.json"
    if tone.is_file():
        tone.unlink()
        deleted.append(str(tone.relative_to(ROOT)))

    ev_legacy = EVALUATION_DIR / f"{stem_key}.json"
    if ev_legacy.is_file():
        ev_legacy.unlink()
        deleted.append(str(ev_legacy.relative_to(ROOT)))

    for p in list(EVALUATION_DIR.glob(f"{stem_key}__*.eval.json")):
        if p.is_file():
            p.unlink()
            deleted.append(str(p.relative_to(ROOT)))

    for p in list(EVALUATION_DIR.glob(f"{stem_key}__*.human.json")):
        if p.is_file():
            p.unlink()
            deleted.append(str(p.relative_to(ROOT)))

    _db.delete_jobs_for_stem(stem_key)
    _db.delete_video(stem_key)

    mp = meta_json_path(stem_key)
    if mp.is_file():
        mp.unlink()
        deleted.append(str(mp.relative_to(ROOT)))

    _audit_log(
        request,
        action="video.delete.hard",
        target_type="video",
        target_id=stem_key,
        details={"deleted": deleted},
    )
    return {"ok": True, "mode": "delete", "deleted": deleted}


@app.post("/api/library/{stem}/restore")
def api_library_restore(request: Request, stem: str) -> dict:
    stem_key = _stem_from_path_param(stem)
    _require_video_write_access(request, stem_key)
    try:
        meta = _db.restore_video_from_deletion_request(stem_key)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    _audit_log(request, action="video.restore", target_type="video", target_id=stem_key)
    return {"ok": True, "meta": meta}


@app.get("/api/workspace/{stem}")
def api_workspace(stem: str, request: Request, criteria: str | None = None) -> dict:
    """Одним ответом: медиафайл, транскрипт, SER, LLM-оценка для выбранного чеклиста (query criteria)."""
    stem_key = _stem_from_path_param(stem)
    meta, permissions = _require_video_permissions(request, stem_key, include_foreign_scope=True)
    criteria_resolution_error: str | None = None
    try:
        crit_name = _resolve_bound_checklist_slug_or_raise(stem_key, criteria)
    except ValueError as e:
        crit_name = _resolve_criteria_query_param(criteria, stem_key)
        criteria_resolution_error = str(e)
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    tr_path = TRANSCRIPT_DIR / f"{stem_key}.json"
    tone_path = TRANSCRIPT_DIR / f"{stem_key}.tone.json"
    vf = find_video_for_stem(stem_key)

    transcript: dict | None = None
    tone: dict | None = None
    transcript_load_error = False
    tone_load_error = False
    if tr_path.is_file():
        try:
            transcript = json.loads(tr_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            transcript = None
            transcript_load_error = True
    if tone_path.is_file():
        try:
            tone = json.loads(tone_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            tone = None
            tone_load_error = True
    evaluation = _normalized_ai_eval(stem_key, crit_name, criteria_content)
    human_evaluation = _normalized_human_eval(stem_key, crit_name, criteria_content)
    human_state = _db.get_human_eval_state(stem_key, crit_name)
    comparison_state = _db.get_evaluation_comparison(stem_key, crit_name)
    comparison_runtime = _comparison_runtime_payload(stem_key, crit_name)
    is_admin = _session_is_admin(request)
    ai_available = evaluation is not None
    can_view_ai = is_admin or bool((comparison_state or {}).get("updated_at") or human_state.get("compared_at"))
    visible_ai = evaluation if can_view_ai else None

    jobs_by = _jobs_latest_by_stem()
    job = jobs_by.get(stem_key)
    training_type = _get_training_type_or_none(meta.get("training_type_slug"))
    checklist_stale = _checklist_stale_payload(meta)

    return {
        "stem": stem_key,
        "video_file": vf.name if vf else (transcript or {}).get("video_file"),
        "video_url": f"/api/videos/{stem_key}" if vf and vf.is_file() else None,
        "transcript": transcript,
        "tone": tone,
        "transcript_load_error": transcript_load_error,
        "tone_load_error": tone_load_error,
        "evaluation": visible_ai,
        "ai_evaluation_available": ai_available,
        "ai_hidden_reason": None if can_view_ai or not ai_available else "ИИ-оценка скрыта до публикации ручного чеклиста и сравнения.",
        "human_evaluation": human_evaluation,
        "human_eval_state": human_state,
        "comparison_state": comparison_state,
        "comparison_runtime": comparison_runtime,
        "evaluation_criteria": crit_name,
        "criteria_resolution_error": criteria_resolution_error,
        "criteria": {
            "active": _db.get_active_checklist_slug(),
            "files": _db.list_checklist_files(),
            "bound": crit_name,
            "can_manage": is_admin,
        },
        "criteria_content": criteria_content,
        "checklist_stale": checklist_stale,
        "meta": meta,
        "permissions": permissions,
        "training_type": training_type,
        "training_types": _db.list_training_types(),
        "managers": _db.list_managers(),
        "locations": _db.list_locations(),
        "auth": _session_auth_payload(request),
        "job": (
            {
                "id": job.get("id"),
                "status": job.get("status"),
                "stage": job.get("stage"),
                "kind": job.get("kind"),
                "error": job.get("error"),
                "stream_log": job.get("stream_log"),
            }
            if job
            else None
        ),
    }


@app.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    j = _db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Задача не найдена")
    stem_key = str(j.get("stem") or "").strip()
    if stem_key:
        _require_video_permissions(request, stem_key, include_foreign_scope=True)
    return j


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(request: Request, job_id: str) -> JSONResponse:
    """Кооперативная остановка пайплайна или пересчёта оценки (queued / running)."""
    jid = (job_id or "").strip()
    if not jid:
        raise HTTPException(400, "Пустой идентификатор")
    row = _db.get_job(jid)
    if not row:
        raise HTTPException(404, "Задача не найдена")
    stem_key = str(row.get("stem") or "").strip()
    if stem_key:
        _require_video_admin_write_access(request, stem_key)
    st = row.get("status")
    if st not in ("queued", "running"):
        raise HTTPException(400, "Задача уже завершена")
    with _job_cancel_lock:
        ev = _job_cancel_events.setdefault(jid, threading.Event())
        ev.set()
    # Сразу в БД — иначе UI до выхода из Whisper видит status=running и поллинг не останавливается.
    _job_set(jid, status="cancelled", stage="cancelled", updated_at=_utc_now())
    stem_q = row.get("stem")
    if isinstance(stem_q, str) and stem_q:
        meta = _db.get_video_meta(stem_q)
        if (meta.get("status") or "") == "processing":
            _db.upsert_video_meta(stem_q, status="pending")
    return JSONResponse({"ok": True})


def _assert_pipeline_can_restart(stem_key: str, vp: Path | None) -> None:
    if not vp or not vp.is_file():
        raise HTTPException(400, "Нет исходного медиафайла в 01.Video для этой записи")
    jobs_by = _jobs_latest_by_stem()
    job = jobs_by.get(stem_key)
    if not job or job.get("kind") != "pipeline":
        raise HTTPException(400, "Для этой записи нет задачи полного пайплайна")
    if job.get("status") in ("queued", "running"):
        raise HTTPException(400, "Для этой записи уже выполняется обработка")


@app.post("/api/workspace/{stem}/pipeline/resume")
def api_workspace_pipeline_resume(request: Request, stem: str) -> JSONResponse:
    """Возобновить без удаления файлов: пропуск готовых этапов (транскрипт → тон → оценка)."""
    stem_key = _stem_from_path_param(stem)
    _require_video_admin_write_access(request, stem_key)
    vp = find_video_for_stem(stem_key)
    _assert_pipeline_can_restart(stem_key, vp)
    meta = _db.get_video_meta(stem_key)
    speaker_count = _normalize_expected_speaker_count(meta.get("expected_speaker_count"), default=2)
    job_id = str(uuid.uuid4())
    _enqueue_pipeline_job_common(job_id, stem_key, vp, resume=True, speaker_count=speaker_count)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.post("/api/workspace/{stem}/pipeline/restart")
def api_workspace_pipeline_restart(request: Request, stem: str) -> JSONResponse:
    """Удалить транскрипт/тон/оценки и запустить пайплайн с нуля."""
    stem_key = _stem_from_path_param(stem)
    _, permissions = _require_video_write_access(request, stem_key)
    vp = find_video_for_stem(stem_key)
    _assert_pipeline_can_restart(stem_key, vp)
    if not permissions.get("is_admin"):
        job = _jobs_latest_by_stem().get(stem_key)
        if not job or job.get("kind") != "pipeline" or job.get("status") != "error":
            raise HTTPException(403, "Пользователь может перезапустить пайплайн только после ошибки")
    meta = _db.get_video_meta(stem_key)
    speaker_count = _normalize_expected_speaker_count(meta.get("expected_speaker_count"), default=2)
    _delete_pipeline_derived_artifacts(stem_key)
    job_id = str(uuid.uuid4())
    _enqueue_pipeline_job_common(job_id, stem_key, vp, resume=False, speaker_count=speaker_count)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/transcripts")
def api_list_transcripts(request: Request) -> list[dict]:
    _require_admin(request)
    out: list[dict] = []
    for p in list_transcripts(TRANSCRIPT_DIR):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        st = p.stat()
        out.append(
            {
                "stem": p.stem,
                "file": p.name,
                "video_file": data.get("video_file"),
                "processed_at": data.get("processed_at"),
                "duration_sec": data.get("duration_sec"),
                "segments_count": len(data.get("segments") or []),
                "mtime": st.st_mtime,
            }
        )
    out.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
    return out


@app.get("/api/transcripts/{stem}")
def api_transcript_detail(request: Request, stem: str) -> dict:
    stem_key = _stem_from_path_param(stem)
    _require_video_permissions(request, stem_key, include_foreign_scope=True)
    path = TRANSCRIPT_DIR / f"{stem_key}.json"
    if not path.is_file():
        raise HTTPException(404, "Транскрипт не найден")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_transcript_time(value: Any) -> str:
    try:
        total = max(0.0, float(value))
    except (TypeError, ValueError):
        total = 0.0
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


def _transcript_plain_text(data: dict[str, Any]) -> str:
    lines: list[str] = []
    title = str(data.get("video_file") or "transcript").strip()
    if title:
        lines.append(title)
    processed_at = str(data.get("processed_at") or "").strip()
    if processed_at:
        lines.append(f"Обработано: {processed_at}")
    model = str(data.get("whisper_model") or "").strip()
    if model:
        lines.append(f"Модель распознавания: {model}")
    speakers = data.get("speakers")
    if isinstance(speakers, list) and speakers:
        lines.append(f"Спикеры: {', '.join(str(x) for x in speakers)}")
    if lines:
        lines.append("")

    for seg in data.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        start = _format_transcript_time(seg.get("start"))
        end = _format_transcript_time(seg.get("end"))
        speaker = str(seg.get("speaker") or "SPEAKER").strip()
        role = str(seg.get("speaker_role") or "").strip()
        speaker_label = f"{speaker} ({role})" if role else speaker
        text = str(seg.get("text") or "").strip()
        lines.append(f"[{start} - {end}] {speaker_label}: {text}")
    return "\n".join(lines).strip() + "\n"


@app.get("/api/transcripts/{stem}/download")
def api_transcript_download(request: Request, stem: str, format: str = "txt") -> Response:
    _require_admin(request)
    stem_key = _stem_from_path_param(stem)
    path = TRANSCRIPT_DIR / f"{stem_key}.json"
    if not path.is_file():
        raise HTTPException(404, "Транскрипт не найден")
    fmt = (format or "txt").strip().lower()
    if fmt == "json":
        return FileResponse(
            path,
            media_type="application/json; charset=utf-8",
            filename=f"{stem_key}.transcript.json",
        )
    if fmt != "txt":
        raise HTTPException(400, "Поддерживаются форматы txt и json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Транскрипт повреждён или не читается") from exc
    filename = f"{stem_key}.transcript.txt"
    quoted = quote(filename)
    return Response(
        content=_transcript_plain_text(data),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"},
    )


@app.get("/api/videos/{stem}")
def api_stream_video(request: Request, stem: str) -> FileResponse:
    """Раздача исходного медиафайла из 01.Video (тот же stem, что у транскрипта)."""
    stem_key = _stem_from_path_param(stem)
    _require_video_permissions(request, stem_key, include_foreign_scope=True)
    video_path = find_video_for_stem(stem_key)
    if not video_path or not video_path.is_file():
        raise HTTPException(404, "Медиафайл не найден в 01.Video")
    ext = video_path.suffix.lower()
    media = _MEDIA_MIME.get(ext, "application/octet-stream")
    return FileResponse(
        video_path,
        media_type=media,
        filename=video_path.name,
    )


@app.get("/api/evaluations")
def api_list_evaluations(request: Request) -> list[dict]:
    _require_admin(request)
    out: list[dict] = []
    for p in sorted(EVALUATION_DIR.glob("*.json")):
        if not p.is_file():
            continue
        if p.name.endswith(".human.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        st = p.stat()
        name = p.name
        if "__" in name and name.endswith(".eval.json"):
            stem_part = name.split("__", 1)[0]
        else:
            stem_part = p.stem
        out.append(
            {
                "stem": stem_part,
                "file": p.name,
                "criteria_file": data.get("criteria_file"),
                "video_file": data.get("video_file"),
                "evaluated_at": data.get("evaluated_at"),
                "model": data.get("model"),
                "overall_average": data.get("overall_average"),
                "criteria_count": len(data.get("criteria") or []),
                "mtime": st.st_mtime,
            }
        )
    out.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
    return out


@app.get("/api/evaluations/{stem}")
def api_evaluation_detail(request: Request, stem: str, criteria: str | None = None) -> dict:
    _require_admin(request)
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(criteria, stem_key)
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    ev = _normalized_ai_eval(stem_key, crit_name, criteria_content)
    if not ev:
        raise HTTPException(404, "Оценка не найдена")
    return ev


def _load_yaml_list(path: Path, key: str) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict):
        return []
    items = raw.get(key)
    if not isinstance(items, list):
        return []
    return [
        {"id": str(r.get("id", "")).strip(), "name": str(r.get("name", "")).strip()}
        for r in items
        if isinstance(r, dict) and r.get("id")
    ]


def _save_yaml_list(path: Path, key: str, items: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: [{"id": i["id"], "name": i["name"]} for i in items]}
    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _load_video_meta(stem: str) -> dict[str, Any]:
    p = meta_json_path(stem)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_video_meta(stem: str, meta: dict[str, Any]) -> None:
    p = meta_json_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class VideoMetaBody(BaseModel):
    display_title: str | None = None
    manager_id: str | None = None
    manager_name: str | None = None
    location_id: str | None = None
    location_name: str | None = None
    interaction_date: str | None = None
    training_type_slug: str | None = None
    tags: list[str] = []


class ManagerBody(BaseModel):
    id: str
    name: str


class LocationBody(BaseModel):
    id: str
    name: str
    crm_id: str | None = None


class TrainingTypeBody(BaseModel):
    slug: str
    name: str
    department: str | None = None
    checklist_slug: str | None = None
    sort_order: int | None = None


@app.get("/api/managers")
def api_managers_list() -> list[dict]:
    return _db.list_managers()


@app.post("/api/managers")
def api_managers_add(request: Request, body: ManagerBody) -> dict:
    _require_admin(request)
    bid = body.id.strip()
    if not bid:
        raise HTTPException(400, "id обязателен")
    _db.add_manager(bid, body.name.strip() or bid)
    _audit_log(request, action="manager.upsert", target_type="manager", target_id=bid)
    return {"ok": True}


@app.delete("/api/managers/{mid}")
def api_managers_delete(request: Request, mid: str) -> dict:
    _require_admin(request)
    if not _db.delete_manager(mid):
        raise HTTPException(404, "Менеджер не найден")
    _audit_log(request, action="manager.delete", target_type="manager", target_id=mid)
    return {"ok": True}


@app.get("/api/locations")
def api_locations_list() -> list[dict]:
    return _db.list_locations()


@app.post("/api/locations")
def api_locations_add(request: Request, body: LocationBody) -> dict:
    _require_admin(request)
    bid = body.id.strip()
    if not bid:
        raise HTTPException(400, "id обязателен")
    _db.add_location(
        bid,
        body.name.strip() or bid,
        crm_name=body.name.strip() or bid,
        crm_id=(body.crm_id or "").strip() or None,
        is_active=True,
    )
    _audit_log(request, action="location.upsert", target_type="location", target_id=bid)
    return {"ok": True}


@app.delete("/api/locations/{lid}")
def api_locations_delete(request: Request, lid: str) -> dict:
    _require_admin(request)
    if not _db.delete_location(lid):
        raise HTTPException(404, "Локация не найдена")
    _audit_log(request, action="location.delete", target_type="location", target_id=lid)
    return {"ok": True}


@app.get("/api/workspace/{stem}/meta")
def api_workspace_meta_get(request: Request, stem: str) -> dict:
    stem_key = _stem_from_path_param(stem)
    meta, _ = _require_video_permissions(request, stem_key, include_foreign_scope=True)
    return meta


@app.put("/api/workspace/{stem}/meta")
def api_workspace_meta_put(request: Request, stem: str, body: VideoMetaBody) -> dict:
    _require_admin(request)
    stem_key = _stem_from_path_param(stem)
    disp = (body.display_title or "").strip() or None
    training_type_slug = None
    checklist_meta = None
    if body.training_type_slug and str(body.training_type_slug).strip():
        training_type_slug = _safe_training_type_slug(str(body.training_type_slug).strip())
        if not _db.training_type_exists(training_type_slug):
            raise HTTPException(404, "Тип тренировки не найден")
        checklist_meta = _current_checklist_meta_for_training_type(training_type_slug)
    meta = _db.upsert_video_meta(
        stem_key,
        display_title=disp,
        manager_id=body.manager_id,
        manager_name=body.manager_name,
        location_id=body.location_id,
        location_name=body.location_name,
        interaction_date=body.interaction_date,
        training_type_slug=training_type_slug,
        training_type_name_snapshot=(checklist_meta or {}).get("training_type", {}).get("name"),
        checklist_slug_snapshot=(checklist_meta or {}).get("slug"),
        checklist_display_name_snapshot=(checklist_meta or {}).get("display_name"),
        checklist_version_snapshot=(checklist_meta or {}).get("version"),
        checklist_department_snapshot=(checklist_meta or {}).get("department"),
        tags=body.tags,
    )
    _audit_log(request, action="video.meta.update", target_type="video", target_id=stem_key)
    return {"ok": True, "meta": meta}


@app.get("/api/training-types")
def api_training_types_list() -> list[dict[str, Any]]:
    return _db.list_training_types()


@app.post("/api/training-types")
def api_training_types_create(request: Request, body: TrainingTypeBody) -> dict:
    _require_admin(request)
    slug = _safe_training_type_slug(body.slug)
    checklist_slug = None
    if body.checklist_slug and str(body.checklist_slug).strip():
        checklist_slug = _safe_criteria_filename(str(body.checklist_slug).strip())
        if not _db.checklist_exists(checklist_slug):
            raise HTTPException(404, "Чеклист не найден")
    item = _db.upsert_training_type(
        slug,
        name=(body.name or "").strip() or slug,
        department=(body.department or "").strip() or None,
        checklist_slug=checklist_slug,
        sort_order=body.sort_order,
    )
    _audit_log(request, action="training_type.create", target_type="training_type", target_id=slug)
    return {"ok": True, "training_type": item}


@app.put("/api/training-types/{slug}")
def api_training_types_update(slug: str, request: Request, body: TrainingTypeBody) -> dict:
    _require_admin(request)
    safe_slug = _safe_training_type_slug(slug)
    if not _db.training_type_exists(safe_slug):
        raise HTTPException(404, "Тип тренировки не найден")
    checklist_slug = None
    if body.checklist_slug and str(body.checklist_slug).strip():
        checklist_slug = _safe_criteria_filename(str(body.checklist_slug).strip())
        if not _db.checklist_exists(checklist_slug):
            raise HTTPException(404, "Чеклист не найден")
    item = _db.upsert_training_type(
        safe_slug,
        name=(body.name or "").strip() or safe_slug,
        department=(body.department or "").strip() or None,
        checklist_slug=checklist_slug,
        sort_order=body.sort_order,
    )
    _audit_log(
        request,
        action="training_type.update",
        target_type="training_type",
        target_id=safe_slug,
    )
    return {"ok": True, "training_type": item}


@app.delete("/api/training-types/{slug}")
def api_training_types_delete(slug: str, request: Request) -> dict:
    _require_admin(request)
    safe_slug = _safe_training_type_slug(slug)
    if not _db.delete_training_type(safe_slug):
        raise HTTPException(404, "Тип тренировки не найден")
    _audit_log(
        request,
        action="training_type.delete",
        target_type="training_type",
        target_id=safe_slug,
    )
    return {"ok": True}


class CriteriaActiveBody(BaseModel):
    file: str


class CriterionItem(BaseModel):
    id: str
    name: str
    description: str = ""
    weight: int = 1


class CriteriaPayload(BaseModel):
    version: str = "1"
    display_name: str | None = None
    department: str | None = None
    criteria: list[CriterionItem]


class CriteriaCreateBody(BaseModel):
    filename: str
    copy_from: str | None = None


def _normalize_new_criteria_filename(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise HTTPException(400, "Укажите имя чеклиста")
    lower = s.lower()
    if lower.endswith(".yaml"):
        s = s[: -len(".yaml")]
    elif lower.endswith(".yml"):
        s = s[: -len(".yml")]
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\u0400-\u04FF]+", "", s)
    s = s.strip("._- ")
    if not s:
        raise HTTPException(400, "Не удалось сформировать имя чеклиста")
    return _safe_criteria_filename(s)


@app.get("/api/criteria")
def api_criteria_list() -> dict:
    """Список чеклистов в БД и текущий активный slug."""
    return {
        "active": _db.get_active_checklist_slug(),
        "files": _db.list_checklist_files(),
    }


def _api_criteria_create_impl(body: CriteriaCreateBody) -> dict:
    """Новый чеклист в БД (шаблон или копия существующего)."""
    fn = _normalize_new_criteria_filename(body.filename)
    if _db.checklist_exists(fn):
        raise HTTPException(409, "Чеклист с таким именем уже есть")
    if body.copy_from:
        try:
            src_slug = _safe_criteria_filename(body.copy_from.strip())
        except HTTPException as e:
            raise HTTPException(400, "Некорректное имя исходного чеклиста") from e
        if not _db.checklist_exists(src_slug):
            raise HTTPException(404, "Исходный чеклист для копирования не найден")
        src_data = _db.get_checklist_content(src_slug)
        if not src_data:
            raise HTTPException(404, "Исходный чеклист для копирования не найден")
        crits = list(src_data["criteria"])
        ver = str(src_data["version"])
    else:
        ver = "1"
        crits = [
            {
                "id": "criterion_1",
                "name": "Новый критерий",
                "description": "Опишите, что проверяет ИИ по этому пункту.",
                "weight": 1,
            }
        ]
    _db.insert_checklist(fn, ver, crits)
    return {"ok": True, "filename": fn}


@app.post("/api/criteria")
def api_criteria_post_create(request: Request, body: CriteriaCreateBody) -> dict:
    """Создать чеклист (основной URL для UI)."""
    _require_admin(request)
    result = _api_criteria_create_impl(body)
    _audit_log(
        request,
        action="checklist.create",
        target_type="checklist",
        target_id=result.get("filename"),
        details={"copy_from": body.copy_from},
    )
    return result


@app.post("/api/criteria/active")
def api_criteria_set_active(request: Request, body: CriteriaActiveBody) -> dict:
    """Сохранить активный чеклист (новые загрузки и «Обновить оценку» используют его)."""
    _require_admin(request)
    name = _safe_criteria_filename(body.file)
    if not _db.checklist_exists(name):
        raise HTTPException(404, "Чеклист не найден")
    try:
        _db.set_active_checklist_slug(name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _audit_log(request, action="checklist.set_active", target_type="checklist", target_id=name)
    return {"ok": True, "active": name}


@app.get("/api/criteria/content/{name}")
def api_criteria_get_content(name: str) -> dict:
    """Содержимое чеклиста для редактора (JSON)."""
    try:
        fn = _safe_criteria_filename(name)
    except HTTPException as e:
        raise HTTPException(404, "Чеклист не найден") from e
    data = _db.get_checklist_content(fn)
    if not data:
        raise HTTPException(404, "Чеклист не найден")
    return data


@app.put("/api/criteria/content/{name}")
def api_criteria_put_content(name: str, request: Request, body: CriteriaPayload) -> dict:
    """Сохранить чеклист из редактора в БД."""
    _require_admin(request)
    try:
        fn = _safe_criteria_filename(name)
    except HTTPException as e:
        raise HTTPException(404, "Чеклист не найден") from e
    if not _db.checklist_exists(fn):
        raise HTTPException(404, "Чеклист не найден")
    for c in body.criteria:
        if not c.id.strip():
            raise HTTPException(400, "У каждого критерия должен быть непустой id")
    rows = [
        {
            "id": c.id.strip(),
            "name": c.name.strip(),
            "description": (c.description or "").strip(),
            "weight": max(1, int(c.weight or 1)),
        }
        for c in body.criteria
    ]
    department = None
    if body.department is not None:
        raw_department = str(body.department).strip()
        department = _require_department_code(raw_department) if raw_department else None
    _db.replace_checklist(
        fn,
        (body.version or "1").strip(),
        rows,
        display_name=(body.display_name or "").strip() or None,
        department=department,
    )
    _audit_log(request, action="checklist.update", target_type="checklist", target_id=fn)
    return {"ok": True, "filename": fn}


@app.post("/api/criteria/create")
def api_criteria_create_alias(request: Request, body: CriteriaCreateBody) -> dict:
    """Алиас для совместимости (тот же сценарий, что POST /api/criteria)."""
    _require_admin(request)
    result = _api_criteria_create_impl(body)
    _audit_log(
        request,
        action="checklist.create",
        target_type="checklist",
        target_id=result.get("filename"),
        details={"copy_from": body.copy_from, "via": "alias"},
    )
    return result


@app.delete("/api/criteria/content/{name}")
def api_criteria_delete(name: str, request: Request) -> dict:
    """Удалить чеклист из БД (не единственный)."""
    _require_admin(request)
    try:
        fn = _safe_criteria_filename(name)
    except HTTPException as e:
        raise HTTPException(404, "Чеклист не найден") from e
    if _db.checklist_count() <= 1:
        raise HTTPException(400, "Нельзя удалить единственный чеклист")
    if not _db.checklist_exists(fn):
        raise HTTPException(404, "Чеклист не найден")
    try:
        deleted = _db.delete_checklist(fn)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not deleted:
        raise HTTPException(500, "Не удалось удалить")
    if _db.checklist_count():
        _db.get_active_checklist_slug()
    _audit_log(request, action="checklist.delete", target_type="checklist", target_id=fn)
    return {"ok": True}


class ReEvaluateBody(BaseModel):
    criteria: str | None = None


def _enqueue_eval_only_job(stem_key: str, criteria_name: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    with _job_cancel_lock:
        _job_cancel_events[job_id] = threading.Event()
    _db.upsert_job(
        job_id,
        stem=stem_key,
        kind="eval_only",
        status="queued",
        stage="queued",
        args_json=json.dumps({"criteria_name": criteria_name}, ensure_ascii=False),
    )
    _dispatch_jobs()
    return job_id


@app.post("/api/workspace/{stem}/re-evaluate")
def api_workspace_re_evaluate(
    request: Request,
    stem: str,
    body: ReEvaluateBody = ReEvaluateBody(),
) -> JSONResponse:
    """Повторная LLM-оценка по транскрипту; чеклист — по типу тренировки или из тела."""
    stem_key = _stem_from_path_param(stem)
    _require_video_admin_write_access(request, stem_key)
    tr = TRANSCRIPT_DIR / f"{stem_key}.json"
    if not tr.is_file():
        raise HTTPException(400, "Нет транскрипта для этой записи")

    if body.criteria and str(body.criteria).strip():
        try:
            cname = _resolve_bound_checklist_slug_or_raise(stem_key, body.criteria)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    else:
        cname = _db.get_current_checklist_slug_for_video(stem_key)
    current_meta = _db.get_checklist_meta(cname)
    meta = _db.get_video_meta(stem_key)
    if current_meta and meta:
        _db.upsert_video_meta(
            stem_key,
            checklist_slug_snapshot=current_meta.get("slug"),
            checklist_display_name_snapshot=current_meta.get("display_name"),
            checklist_version_snapshot=current_meta.get("version"),
            checklist_department_snapshot=current_meta.get("department"),
        )

    job_id = _enqueue_eval_only_job(stem_key, cname)

    return JSONResponse(
        {
            "job_id": job_id,
            "stem": stem_key,
            "message": "Пересчёт оценки запущен",
        }
    )


class HumanEvalCriterion(BaseModel):
    id: str
    name: str
    passed: bool | None = None
    weight: int | None = None
    comment: str = ""


class HumanEvalBody(BaseModel):
    criteria: list[HumanEvalCriterion]
    criteria_file: str | None = None


@app.get("/api/workspace/{stem}/human-eval")
def api_human_eval_get(request: Request, stem: str, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    _require_video_permissions(request, stem_key, include_foreign_scope=True)
    try:
        crit_name = _resolve_bound_checklist_slug_or_raise(stem_key, criteria)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    data = _normalized_human_eval(stem_key, crit_name, criteria_content)
    if data is None:
        return {"exists": False, "state": _db.get_human_eval_state(stem_key, crit_name)}
    data["exists"] = True
    data["state"] = _db.get_human_eval_state(stem_key, crit_name)
    return data


@app.put("/api/workspace/{stem}/human-eval")
def api_human_eval_put(request: Request, stem: str, body: HumanEvalBody, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    _require_video_write_access(request, stem_key)
    try:
        crit_name = _resolve_bound_checklist_slug_or_raise(stem_key, criteria)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    state = _db.get_human_eval_state(stem_key, crit_name)
    if state.get("published_at") and not _session_is_admin(request):
        raise HTTPException(409, "Чеклист уже опубликован и заблокирован")
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    crit_out = normalize_eval_criteria(
        [
            {
                "id": c.id,
                "name": c.name,
                "passed": c.passed,
                "weight": c.weight,
                "comment": c.comment,
            }
            for c in body.criteria
        ],
        list(criteria_content.get("criteria") or []),
    )
    totals = compute_eval_totals(crit_out)
    data = {
        "schema_version": 3,
        "source": "human",
        "criteria_file": body.criteria_file or crit_name,
        "criteria_version": criteria_content.get("version"),
        "criteria_snapshot": list(criteria_content.get("criteria") or []),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": crit_out,
        **totals,
    }
    _save_human_eval_for_stem(stem_key, crit_name, data)
    saved_state = _db.mark_human_eval_draft_saved(stem_key, crit_name)
    return {"ok": True, "human_evaluation": data, "state": saved_state}


@app.post("/api/workspace/{stem}/human-eval/publish")
def api_human_eval_publish(request: Request, stem: str, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    _require_video_write_access(request, stem_key)
    try:
        crit_name = _resolve_bound_checklist_slug_or_raise(stem_key, criteria)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    human_eval = _normalized_human_eval(stem_key, crit_name, criteria_content)
    if not human_eval:
        raise HTTPException(400, "Сначала сохраните ручной чеклист")
    if any(row.get("passed") is None for row in (human_eval.get("criteria") or [])):
        raise HTTPException(400, "Перед публикацией отметьте каждый пункт как да или нет")
    try:
        state = _db.publish_human_eval(stem_key, crit_name, request.session.get("user"))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    human_eval["published_at"] = state.get("published_at")
    human_eval["published_by"] = state.get("published_by")
    _save_human_eval_for_stem(stem_key, crit_name, human_eval)
    compare_triggered = _trigger_eval_comparison_background(
        stem_key,
        crit_name,
        criteria_content=criteria_content,
        actor_username=request.session.get("user"),
        attach_llm=True,
    )
    comparison_runtime = _comparison_runtime_payload(stem_key, crit_name)
    return {
        "ok": True,
        "state": state,
        "human_evaluation": human_eval,
        "comparison_triggered": compare_triggered,
        "comparison_runtime": comparison_runtime,
    }


class CompareBody(BaseModel):
    criteria: str | None = None
    force: bool = False


@app.post("/api/workspace/{stem}/compare-eval")
def api_compare_eval(request: Request, stem: str, body: CompareBody = CompareBody()) -> dict:
    """Compare AI and human evaluations for the same stem + criteria via LLM."""
    stem_key = _stem_from_path_param(stem)
    _, permissions = _require_video_permissions(request, stem_key, include_foreign_scope=True)
    try:
        crit_name = _resolve_bound_checklist_slug_or_raise(stem_key, body.criteria)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    criteria_content = _resolve_checklist_content(stem_key, crit_name)
    ai_eval = _normalized_ai_eval(stem_key, crit_name, criteria_content)
    human_eval = _normalized_human_eval(stem_key, crit_name, criteria_content)
    state = _db.get_human_eval_state(stem_key, crit_name)

    if not ai_eval:
        raise HTTPException(400, "Нет ИИ-оценки для сравнения")
    if not human_eval:
        raise HTTPException(400, "Нет ручной оценки для сравнения")
    if not _session_is_admin(request) and not state.get("published_at"):
        raise HTTPException(403, "Сравнение доступно после публикации ручного чеклиста")
    if body.force and not _session_is_admin(request):
        raise HTTPException(403, "Перезапуск сравнения доступен только администратору")
    if permissions.get("read_only") and (state or {}).get("compared_at"):
        comparison = _db.get_evaluation_comparison(stem_key, crit_name)
        if comparison and comparison.get("payload"):
            return comparison["payload"]
    if permissions.get("read_only"):
        raise HTTPException(403, "Для чужой записи доступен только просмотр уже готового сравнения")

    payload, _ = _ensure_published_eval_comparison(
        stem_key,
        crit_name,
        criteria_content=criteria_content,
        actor_username=request.session.get("user"),
        force=bool(body.force),
    )
    if not payload:
        raise HTTPException(400, "Не удалось подготовить сравнение")
    return {
        "stem": stem_key,
        "criteria_file": crit_name,
        **payload,
    }


def _collect_all_evaluations() -> list[dict[str, Any]]:
    """Read every evaluation JSON and join with video metadata."""
    rows: list[dict[str, Any]] = []
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(EVALUATION_DIR.glob("*.json")):
        if not p.is_file():
            continue
        if p.name.endswith(".human.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = p.name
        if "__" in name and name.endswith(".eval.json"):
            stem_part = name.split("__", 1)[0]
        else:
            stem_part = p.stem
        meta = _db.get_video_meta(stem_part)
        criteria_slug = str(data.get("criteria_file") or _db.resolve_checklist_slug_for_video(stem_part))
        checklist_content = _db.get_checklist_content(criteria_slug) or {"criteria": []}
        normalized = normalize_loaded_evaluation(data, list(checklist_content.get("criteria") or [])) or data
        training_type = _get_training_type_or_none(meta.get("training_type_slug"))
        row = {
            "stem": stem_part,
            "file": p.name,
            "criteria_file": normalized.get("criteria_file"),
            "video_file": normalized.get("video_file"),
            "evaluated_at": normalized.get("evaluated_at"),
            "model": normalized.get("model"),
            "overall_average": normalized.get("overall_average"),
            "earned_score": normalized.get("earned_score"),
            "max_score": normalized.get("max_score"),
            "manager_id": meta.get("manager_id"),
            "manager_name": meta.get("manager_name"),
            "location_id": meta.get("location_id"),
            "location_name": meta.get("location_name"),
            "training_type_slug": meta.get("training_type_slug"),
            "training_type_name": (training_type or {}).get("name"),
            "tags": meta.get("tags") or [],
            "criteria": normalized.get("criteria") or [],
        }
        rows.append(row)
    return rows


# ── Dashboard API ────────────────────────────────────────────────────

def _dashboard_payload(
    manager_id: str | None = None,
    location_id: str | None = None,
) -> dict[str, Any]:
    """Aggregated analytics: scores by manager, by location, overall trends."""
    evals = _collect_all_evaluations()

    if manager_id:
        evals = [e for e in evals if e.get("manager_id") == manager_id]
    if location_id:
        evals = [e for e in evals if e.get("location_id") == location_id]

    total = len(evals)
    scores = [e["overall_average"] for e in evals if e.get("overall_average") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    by_manager: dict[str, list[float]] = {}
    by_location: dict[str, list[float]] = {}
    by_date: dict[str, list[float]] = {}
    criteria_scores: dict[str, list[float]] = {}

    for e in evals:
        s = e.get("overall_average")
        if s is None:
            continue

        mid = e.get("manager_name") or e.get("manager_id") or "Не указан"
        by_manager.setdefault(mid, []).append(s)

        lid = e.get("location_name") or e.get("location_id") or "Не указана"
        by_location.setdefault(lid, []).append(s)

        dt = e.get("evaluated_at") or ""
        day = dt[:10] if len(dt) >= 10 else "unknown"
        by_date.setdefault(day, []).append(s)

        for c in e.get("criteria") or []:
            cs = c.get("weight")
            aw = c.get("awarded_weight")
            if cs is None:
                continue
            try:
                pct = (float(aw or 0) / float(cs)) * 100 if float(cs) > 0 else None
            except (TypeError, ValueError, ZeroDivisionError):
                pct = None
            if cs is not None:
                cname = c.get("name") or c.get("id") or "?"
                if pct is not None:
                    criteria_scores.setdefault(cname, []).append(pct)

    def _agg(d: dict[str, list[float]]) -> list[dict[str, Any]]:
        return sorted(
            [
                {"name": k, "count": len(v), "avg": round(sum(v) / len(v), 1), "min": min(v), "max": max(v)}
                for k, v in d.items()
            ],
            key=lambda x: -x["avg"],
        )

    return {
        "total_evaluations": total,
        "average_score": avg_score,
        "by_manager": _agg(by_manager),
        "by_location": _agg(by_location),
        "by_date": sorted(
            [{"date": k, "count": len(v), "avg": round(sum(v) / len(v), 1)} for k, v in by_date.items()],
            key=lambda x: x["date"],
        ),
        "by_criteria": _agg(criteria_scores),
        "managers": _db.list_managers(),
        "locations": _db.list_locations(),
    }


@app.get("/api/dashboard")
def api_dashboard(
    request: Request,
    manager_id: str | None = None,
    location_id: str | None = None,
) -> dict[str, Any]:
    _require_admin(request)
    return _dashboard_payload(manager_id=manager_id, location_id=location_id)


# ── Export API ───────────────────────────────────────────────────────

@app.get("/api/export/csv")
def api_export_csv(
    request: Request,
    manager_id: str | None = None,
    location_id: str | None = None,
) -> FileResponse:
    """Export evaluations as CSV."""
    _require_admin(request)
    evals = _collect_all_evaluations()
    if manager_id:
        evals = [e for e in evals if e.get("manager_id") == manager_id]
    if location_id:
        evals = [e for e in evals if e.get("location_id") == location_id]

    all_crit_names: list[str] = []
    seen: set[str] = set()
    for e in evals:
        for c in e.get("criteria") or []:
            cn = c.get("name") or c.get("id") or "?"
            if cn not in seen:
                all_crit_names.append(cn)
                seen.add(cn)

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["Файл", "Менеджер", "Локация", "Тип тренировки", "Дата оценки", "Баллы", "Макс. баллы", "Процент"] + all_crit_names
    writer.writerow(header)

    for e in evals:
        crit_map = {}
        for c in e.get("criteria") or []:
            cn = c.get("name") or c.get("id") or "?"
            passed = c.get("passed")
            crit_map[cn] = "Да" if passed is True else "Нет" if passed is False else ""
        row = [
            e.get("video_file") or e.get("stem"),
            e.get("manager_name") or e.get("manager_id") or "",
            e.get("location_name") or e.get("location_id") or "",
            e.get("training_type_name") or "",
            (e.get("evaluated_at") or "")[:19],
            e.get("earned_score") or "",
            e.get("max_score") or "",
            e.get("overall_average") or "",
        ] + [crit_map.get(cn, "") for cn in all_crit_names]
        writer.writerow(row)

    tmp = ROOT / "03.Evaluation" / "_export.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig")

    return FileResponse(
        tmp,
        media_type="text/csv; charset=utf-8",
        filename=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


class AdminUserCreateBody(BaseModel):
    username: str
    auth_source: str = AUTH_SOURCE_LOCAL
    password: str | None = None
    full_name: str | None = None
    location_id: str | None = None
    department: str | None = None
    role: str = USER_ROLE_USER
    daily_upload_limit: int | None = None
    max_queued_jobs: int | None = None
    max_running_jobs: int | None = None
    is_active: bool = True


class AdminUserUpdateBody(BaseModel):
    full_name: str | None = None
    location_id: str | None = None
    department: str | None = None
    role: str | None = None
    daily_upload_limit: int | None = None
    max_queued_jobs: int | None = None
    max_running_jobs: int | None = None
    is_active: bool | None = None


class AdminUserPasswordBody(BaseModel):
    new_password: str


class AdminJobRetryBody(BaseModel):
    mode: str = "resume"
    criteria: str | None = None


class AdminVideoProcessBody(BaseModel):
    mode: str = "resume"
    criteria: str | None = None


class AdminSettingsBody(BaseModel):
    max_workers: int | None = None
    admin_auto_refresh_seconds: int | None = None
    local_registration_enabled: bool | None = None
    max_queue_depth: int | None = None
    default_daily_upload_limit: int | None = None
    default_max_queued_jobs: int | None = None
    default_max_running_jobs: int | None = None


def _admin_user_payload(
    row: dict[str, Any],
    usage_map: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    username = str(row.get("username") or "").strip()
    location_id = str(row.get("location_id") or "").strip() or None
    location = _db.get_location(location_id) if location_id else None
    stored_role = str(row.get("role") or USER_ROLE_USER).strip().lower()
    effective_role = _effective_user_role(username, stored_role)
    limits = _effective_user_queue_limits(row)
    usage = (usage_map or {}).get(
        username.lower(),
        {"daily_uploaded_count": 0, "queued_count": 0, "running_count": 0},
    )
    daily_limit = int(limits["daily_upload_limit"])
    return {
        "username": username,
        "display_name": row.get("display_name"),
        "full_name": row.get("full_name"),
        "auth_source": row.get("auth_source"),
        "location_id": location_id,
        "location_name": (location or {}).get("crm_name") or (location or {}).get("name"),
        "department": row.get("department"),
        "department_label": _department_label(row.get("department")),
        "profile_completed_at": row.get("profile_completed_at"),
        "profile_complete": _user_profile_complete(row),
        "stored_role": stored_role,
        "effective_role": effective_role,
        "forced_admin": username.lower() in _admin_users(),
        "is_admin": effective_role == USER_ROLE_ADMIN,
        "daily_upload_limit": row.get("daily_upload_limit"),
        "max_queued_jobs": row.get("max_queued_jobs"),
        "max_running_jobs": row.get("max_running_jobs"),
        "quotas_updated_at": row.get("quotas_updated_at"),
        "effective_daily_upload_limit": daily_limit,
        "effective_max_queued_jobs": int(limits["max_queued_jobs"]),
        "effective_max_running_jobs": int(limits["max_running_jobs"]),
        "daily_uploaded_count": int(usage.get("daily_uploaded_count") or 0),
        "daily_remaining": max(0, daily_limit - int(usage.get("daily_uploaded_count") or 0)),
        "queued_count": int(usage.get("queued_count") or 0),
        "running_count": int(usage.get("running_count") or 0),
        "is_active": bool(row.get("is_active")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _admin_settings_payload() -> dict[str, Any]:
    runtime = _runtime_settings_payload()
    max_workers = runtime["max_workers"]
    settings = [
        {
            "key": OPS_MAX_WORKERS_KEY,
            "label": "Количество worker-ов обработки",
            "type": "number",
            "value": max_workers["configured"],
            "current": max_workers["current"],
            "minimum": max_workers["minimum"],
            "maximum": max_workers["maximum"],
            "help": "Определяет число параллельных background-задач обработки. При активной очереди новое значение сохраняется и применяется после завершения задач или перезапуска сервера.",
            "requires_runtime_apply": True,
            "pending_restart": not bool(max_workers["applied"]),
        },
        {
            "key": OPS_ADMIN_REFRESH_SECONDS_KEY,
            "label": "Интервал автообновления админки",
            "type": "number",
            "value": runtime["admin_auto_refresh_seconds"],
            "minimum": 3,
            "maximum": 120,
            "help": "Как часто обновлять overview, jobs и audit в интерфейсе админки.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
        {
            "key": AUTH_LOCAL_REGISTRATION_ENABLED_KEY,
            "label": "Разрешить саморегистрацию локальных пользователей",
            "type": "boolean",
            "value": runtime["local_registration_enabled"],
            "help": "Если выключено, локальный режим входа остаётся, но регистрация со страницы логина блокируется.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
        {
            "key": OPS_MAX_QUEUE_DEPTH_KEY,
            "label": "Максимальная глубина глобальной очереди",
            "type": "number",
            "value": runtime["max_queue_depth"],
            "minimum": 1,
            "maximum": _MAX_QUEUE_DEPTH_MAX,
            "help": "Если лимит ожидания достигнут, новые загрузки временно отклоняются ещё до постановки в очередь.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
        {
            "key": QUOTA_DEFAULT_DAILY_UPLOAD_LIMIT_KEY,
            "label": "Дневная квота загрузок по умолчанию",
            "type": "number",
            "value": runtime["default_daily_upload_limit"],
            "minimum": 0,
            "maximum": _DEFAULT_DAILY_UPLOAD_LIMIT_MAX,
            "help": "Сколько новых видео или аудио пользователь может загрузить за календарные сутки, если у него нет персонального override.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
        {
            "key": QUOTA_DEFAULT_MAX_QUEUED_JOBS_KEY,
            "label": "Лимит ожидающих задач по умолчанию",
            "type": "number",
            "value": runtime["default_max_queued_jobs"],
            "minimum": 0,
            "maximum": _DEFAULT_MAX_QUEUED_JOBS_MAX,
            "help": "Максимум задач пользователя в состоянии queued без персонального override.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
        {
            "key": QUOTA_DEFAULT_MAX_RUNNING_JOBS_KEY,
            "label": "Лимит одновременных задач по умолчанию",
            "type": "number",
            "value": runtime["default_max_running_jobs"],
            "minimum": 0,
            "maximum": _DEFAULT_MAX_RUNNING_JOBS_MAX,
            "help": "Сколько задач одного пользователя можно обрабатывать одновременно без персонального override.",
            "requires_runtime_apply": False,
            "pending_restart": False,
        },
    ]
    return {
        "runtime": runtime,
        "settings": settings,
    }


def _admin_reference_data() -> dict[str, Any]:
    return {
        "managers": _db.list_managers(),
        "locations": _db.list_locations(),
        "training_types": _db.list_training_types(),
        "checklists": api_criteria_list(),
        "auth": auth_status(),
    }


def _admin_overview_payload(request: Request) -> dict[str, Any]:
    videos = api_library(request, include_deleted=True, include_foreign=True)
    jobs = _db.list_jobs(limit=20)
    users = [_admin_user_payload(row) for row in _db.list_users()]
    analytics = _dashboard_payload()

    job_counts = {
        "total": _db.count_jobs(),
        "queued": _db.count_jobs(statuses=("queued",)),
        "running": _db.count_jobs(statuses=("running",)),
        "done": _db.count_jobs(statuses=("done",)),
        "error": _db.count_jobs(statuses=("error",)),
        "cancelled": _db.count_jobs(statuses=("cancelled",)),
    }
    video_counts = {
        "total": len(videos),
        "deleted": sum(1 for row in videos if row.get("deleted_at")),
        "delete_requested": sum(1 for row in videos if row.get("delete_requested_at") and not row.get("deleted_at")),
        "with_transcript": sum(1 for row in videos if row.get("has_transcript")),
        "with_evaluation": sum(1 for row in videos if row.get("has_evaluation")),
        "processing": sum(
            1
            for row in videos
            if ((row.get("job") or {}).get("status") in {"queued", "running"})
            or str(row.get("status") or "").strip().lower() == "processing"
        ),
        "orphan_transcripts": sum(
            1 for row in videos if row.get("has_transcript") and not row.get("has_video_file")
        ),
        "errors": sum(1 for row in videos if (row.get("job") or {}).get("status") == "error"),
    }
    user_counts = {
        "total": len(users),
        "active": sum(1 for row in users if row.get("is_active")),
        "inactive": sum(1 for row in users if not row.get("is_active")),
        "admins": sum(1 for row in users if row.get("is_admin")),
        "profile_complete": sum(1 for row in users if row.get("profile_complete")),
    }
    return {
        "generated_at": _utc_now(),
        "counts": {
            "videos": video_counts,
            "jobs": job_counts,
            "users": user_counts,
            "checklists": _db.checklist_count(),
            "training_types": len(_db.list_training_types()),
            "managers": len(_db.list_managers()),
            "locations": len(_db.list_locations()),
        },
        "runtime": _runtime_settings_payload(),
        "analytics": {
            "total_evaluations": analytics.get("total_evaluations"),
            "average_score": analytics.get("average_score"),
            "by_manager": (analytics.get("by_manager") or [])[:5],
            "by_location": (analytics.get("by_location") or [])[:5],
            "by_criteria": (analytics.get("by_criteria") or [])[:5],
            "by_date": (analytics.get("by_date") or [])[-7:],
        },
        "recent_jobs": jobs,
        "recent_audit": _db.list_audit_events(limit=12),
    }


def _filter_admin_videos(rows: list[dict[str, Any]], q: str | None, status: str | None) -> list[dict[str, Any]]:
    out = rows
    if q and str(q).strip():
        needle = str(q).strip().lower()
        out = [
            row
            for row in out
            if needle
            in " ".join(
                [
                    str(row.get("stem") or ""),
                    str(row.get("video_file") or ""),
                    str(row.get("display_title") or ""),
                    str(row.get("manager_name") or ""),
                    str(row.get("location_name") or ""),
                    str(row.get("training_type_name") or ""),
                    str(row.get("checklist_slug_snapshot") or ""),
                    str(row.get("uploaded_by") or ""),
                    str(row.get("uploaded_by_name") or ""),
                    " ".join(str(tag) for tag in (row.get("tags") or [])),
                ]
            ).lower()
        ]
    if status:
        code = str(status).strip().lower()
        filtered: list[dict[str, Any]] = []
        for row in out:
            job = row.get("job") or {}
            if code == "queued" and job.get("status") == "queued":
                filtered.append(row)
            elif code == "running" and job.get("status") == "running":
                filtered.append(row)
            elif code == "error" and job.get("status") == "error":
                filtered.append(row)
            elif code == "deleted" and (row.get("delete_requested_at") or row.get("deleted_at")):
                filtered.append(row)
            elif code == "ready" and row.get("has_evaluation"):
                filtered.append(row)
            elif code == "orphan" and row.get("has_transcript") and not row.get("has_video_file"):
                filtered.append(row)
            elif code == "pending" and not job and not row.get("has_evaluation"):
                filtered.append(row)
        out = filtered
    return out


@app.get("/api/admin/bootstrap")
def api_admin_bootstrap(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return {
        "me": auth_me(request),
        "reference": _admin_reference_data(),
        "settings": _admin_settings_payload(),
    }


@app.get("/api/admin/overview")
def api_admin_overview(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return _admin_overview_payload(request)


@app.get("/api/admin/reference-data")
def api_admin_reference_data(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return _admin_reference_data()


@app.get("/api/admin/jobs")
def api_admin_jobs(
    request: Request,
    limit: int = 200,
    status: str | None = None,
    kind: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    _require_admin(request)
    items = _db.list_jobs(limit=limit, status=status, kind=kind, query=q)
    return {
        "items": items,
        "counts": {
            "total": _db.count_jobs(),
            "queued": _db.count_jobs(statuses=("queued",)),
            "running": _db.count_jobs(statuses=("running",)),
            "error": _db.count_jobs(statuses=("error",)),
            "cancelled": _db.count_jobs(statuses=("cancelled",)),
        },
    }


@app.get("/api/admin/jobs/{job_id}")
def api_admin_job_detail(request: Request, job_id: str) -> dict[str, Any]:
    _require_admin(request)
    return get_job(request, job_id)


@app.post("/api/admin/jobs/{job_id}/cancel")
def api_admin_job_cancel(request: Request, job_id: str) -> JSONResponse:
    _require_admin(request)
    response = api_job_cancel(request, job_id)
    _audit_log(request, action="admin.job.cancel", target_type="job", target_id=job_id)
    return response


@app.post("/api/admin/jobs/{job_id}/retry")
def api_admin_job_retry(request: Request, job_id: str, body: AdminJobRetryBody = AdminJobRetryBody()) -> JSONResponse:
    _require_admin(request)
    row = _db.get_job(job_id)
    if not row:
        raise HTTPException(404, "Задача не найдена")
    stem_key = _stem_from_path_param(str(row.get("stem") or ""))
    mode = (body.mode or "resume").strip().lower()
    if str(row.get("kind") or "") == "pipeline":
        vp = find_video_for_stem(stem_key)
        _assert_pipeline_can_restart(stem_key, vp)
        meta = _db.get_video_meta(stem_key)
        speaker_count = _normalize_expected_speaker_count(meta.get("expected_speaker_count"), default=2)
        if mode == "restart":
            _delete_pipeline_derived_artifacts(stem_key)
            new_job_id = str(uuid.uuid4())
            _enqueue_pipeline_job_common(
                new_job_id,
                stem_key,
                vp,
                resume=False,
                speaker_count=speaker_count,
            )
        else:
            new_job_id = str(uuid.uuid4())
            _enqueue_pipeline_job_common(
                new_job_id,
                stem_key,
                vp,
                resume=True,
                speaker_count=speaker_count,
            )
    else:
        criteria_name = None
        if body.criteria and str(body.criteria).strip():
            criteria_name = _safe_criteria_filename(body.criteria)
            if not _db.checklist_exists(criteria_name):
                raise HTTPException(404, "Чеклист не найден")
        else:
            criteria_name = _db.resolve_checklist_slug_for_video(stem_key)
        new_job_id = _enqueue_eval_only_job(stem_key, criteria_name)
    _audit_log(
        request,
        action="admin.job.retry",
        target_type="job",
        target_id=job_id,
        details={"new_job_id": new_job_id, "mode": mode, "stem": stem_key},
    )
    return JSONResponse({"ok": True, "job_id": new_job_id, "stem": stem_key})


@app.get("/api/admin/users")
def api_admin_users(request: Request) -> dict[str, Any]:
    _require_admin(request)
    start_at, end_at = _utc_day_window()
    usage_map = _db.list_user_quota_usage(start_at=start_at, end_at=end_at)
    items = [_admin_user_payload(row, usage_map) for row in _db.list_users()]
    return {"items": items}


@app.post("/api/admin/users")
def api_admin_users_create(request: Request, body: AdminUserCreateBody) -> dict[str, Any]:
    _require_admin(request)
    auth_source = _normalize_auth_source(body.auth_source)
    if auth_source == AUTH_SOURCE_LOCAL and not local_auth_enabled():
        raise HTTPException(400, "Локальные пользователи доступны только при AUTH_TYPE=local")
    if auth_source == AUTH_SOURCE_AD and not ad_auth_enabled():
        raise HTTPException(400, "AD-пользователи доступны только при AUTH_TYPE=ad")
    username = (
        normalize_local_username(body.username)
        if auth_source == AUTH_SOURCE_LOCAL
        else str(body.username or "").strip()
    )
    if not username:
        raise HTTPException(400, "Укажите логин пользователя")
    if _db.get_user(username):
        raise HTTPException(409, "Пользователь уже существует")
    location_id = str(body.location_id or "").strip() or None
    if location_id and not _db.get_location(location_id):
        raise HTTPException(400, "Выберите корректную локацию")
    department = None
    if body.department is not None and str(body.department).strip():
        department = _require_department_code(body.department)
    full_name = str(body.full_name or "").strip() or None
    password_hash = None
    if auth_source == AUTH_SOURCE_LOCAL:
        password = body.password or ""
        password_error = validate_local_password_strength(password)
        if password_error:
            raise HTTPException(400, password_error)
        password_hash = hash_local_password(password)
    item = _db.upsert_user(
        username,
        password_hash=password_hash,
        display_name=full_name or username,
        full_name=full_name,
        auth_source=auth_source,
        location_id=location_id,
        department=department,
        profile_completed_at=_utc_now() if full_name and location_id and department else None,
        role=body.role,
        daily_upload_limit=body.daily_upload_limit,
        max_queued_jobs=body.max_queued_jobs,
        max_running_jobs=body.max_running_jobs,
        is_active=body.is_active,
    )
    payload = _admin_user_payload(item)
    _audit_log(
        request,
        action="admin.user.create",
        target_type="user",
        target_id=username,
        details={
            "auth_source": auth_source,
            "role": payload["effective_role"],
            "daily_upload_limit": payload["daily_upload_limit"],
            "max_queued_jobs": payload["max_queued_jobs"],
            "max_running_jobs": payload["max_running_jobs"],
        },
    )
    return {"ok": True, "user": payload}


@app.put("/api/admin/users/{username}")
def api_admin_users_update(request: Request, username: str, body: AdminUserUpdateBody) -> dict[str, Any]:
    _require_admin(request)
    existing = _db.get_user(username)
    if not existing:
        raise HTTPException(404, "Пользователь не найден")
    body_fields_set = getattr(body, "model_fields_set", getattr(body, "__fields_set__", set()))
    location_id = None
    if body.location_id is not None:
        location_id = str(body.location_id).strip()
        if location_id and not _db.get_location(location_id):
            raise HTTPException(400, "Выберите корректную локацию")
    department = None
    if body.department is not None:
        raw_department = str(body.department).strip()
        department = _require_department_code(raw_department) if raw_department else ""
    role = body.role if body.role is not None else str(existing.get("role") or USER_ROLE_USER)
    full_name = (
        str(body.full_name).strip()
        if body.full_name is not None
        else str(existing.get("full_name") or "").strip()
    )
    item = _db.upsert_user(
        username,
        password_hash=None,
        display_name=full_name or username,
        full_name=full_name,
        auth_source=str(existing.get("auth_source") or AUTH_SOURCE_LOCAL),
        location_id=location_id if body.location_id is not None else None,
        department=department if body.department is not None else None,
        profile_completed_at=existing.get("profile_completed_at"),
        role=role,
        daily_upload_limit=(
            body.daily_upload_limit
            if "daily_upload_limit" in body_fields_set
            else existing.get("daily_upload_limit")
        ),
        max_queued_jobs=(
            body.max_queued_jobs
            if "max_queued_jobs" in body_fields_set
            else existing.get("max_queued_jobs")
        ),
        max_running_jobs=(
            body.max_running_jobs
            if "max_running_jobs" in body_fields_set
            else existing.get("max_running_jobs")
        ),
        is_active=bool(existing.get("is_active")) if body.is_active is None else body.is_active,
    )
    payload = _admin_user_payload(item)
    _audit_log(
        request,
        action="admin.user.update",
        target_type="user",
        target_id=username,
        details={
            "role": payload["effective_role"],
            "is_active": payload["is_active"],
            "daily_upload_limit": payload["daily_upload_limit"],
            "max_queued_jobs": payload["max_queued_jobs"],
            "max_running_jobs": payload["max_running_jobs"],
        },
    )
    return {"ok": True, "user": payload}


@app.post("/api/admin/users/{username}/activate")
def api_admin_users_activate(request: Request, username: str) -> dict[str, Any]:
    _require_admin(request)
    if not _db.set_user_active(username, True):
        raise HTTPException(404, "Пользователь не найден")
    item = _db.get_user(username)
    _audit_log(request, action="admin.user.activate", target_type="user", target_id=username)
    return {"ok": True, "user": _admin_user_payload(item or {"username": username})}


@app.post("/api/admin/users/{username}/deactivate")
def api_admin_users_deactivate(request: Request, username: str) -> dict[str, Any]:
    _require_admin(request)
    current_user = str(request.session.get("user") or "").strip()
    if current_user and current_user == username:
        raise HTTPException(400, "Нельзя отключить текущую активную сессию администратора")
    if not _db.set_user_active(username, False):
        raise HTTPException(404, "Пользователь не найден")
    item = _db.get_user(username)
    _audit_log(request, action="admin.user.deactivate", target_type="user", target_id=username)
    return {"ok": True, "user": _admin_user_payload(item or {"username": username})}


@app.post("/api/admin/users/{username}/reset-password")
def api_admin_users_reset_password(
    request: Request,
    username: str,
    body: AdminUserPasswordBody,
) -> dict[str, Any]:
    _require_admin(request)
    user_row = _db.get_user(username)
    if not user_row:
        raise HTTPException(404, "Пользователь не найден")
    if str(user_row.get("auth_source") or AUTH_SOURCE_LOCAL) != AUTH_SOURCE_LOCAL:
        raise HTTPException(400, "Сброс пароля доступен только для локальной учётной записи")
    password_error = validate_local_password_strength(body.new_password)
    if password_error:
        raise HTTPException(400, password_error)
    _db.set_user_password_hash(username, hash_local_password(body.new_password))
    _audit_log(request, action="admin.user.reset_password", target_type="user", target_id=username)
    return {"ok": True}


@app.get("/api/admin/videos")
def api_admin_videos(
    request: Request,
    q: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    _require_admin(request)
    rows = api_library(request, include_deleted=True, include_foreign=True)
    return {"items": _filter_admin_videos(rows, q=q, status=status)}


@app.put("/api/admin/videos/{stem}")
def api_admin_video_update(request: Request, stem: str, body: VideoMetaBody) -> dict[str, Any]:
    _require_admin(request)
    return api_workspace_meta_put(request, stem, body)


@app.post("/api/admin/videos/{stem}/reprocess")
def api_admin_video_reprocess(
    request: Request,
    stem: str,
    body: AdminVideoProcessBody = AdminVideoProcessBody(),
) -> JSONResponse:
    _require_admin(request)
    mode = (body.mode or "resume").strip().lower()
    if mode == "restart":
        response = api_workspace_pipeline_restart(request, stem)
    elif mode == "re-evaluate":
        response = api_workspace_re_evaluate(
            request,
            stem,
            ReEvaluateBody(criteria=(body.criteria or "").strip() or None),
        )
    else:
        response = api_workspace_pipeline_resume(request, stem)
    _audit_log(
        request,
        action="admin.video.reprocess",
        target_type="video",
        target_id=stem,
        details={"mode": mode, "criteria": (body.criteria or "").strip() or None},
    )
    return response


@app.post("/api/admin/videos/{stem}/restore")
def api_admin_video_restore(request: Request, stem: str) -> dict[str, Any]:
    _require_admin(request)
    return api_library_restore(request, stem)


@app.delete("/api/admin/videos/{stem}")
def api_admin_video_delete(request: Request, stem: str) -> dict[str, Any]:
    _require_admin(request)
    return api_library_delete(request, stem)


@app.get("/api/admin/checklists")
def api_admin_checklists(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return {
        "criteria": api_criteria_list(),
        "training_types": _db.list_training_types(),
    }


@app.get("/api/admin/checklists/{name}")
def api_admin_checklist_content(request: Request, name: str) -> dict[str, Any]:
    _require_admin(request)
    return api_criteria_get_content(name)


@app.post("/api/admin/checklists")
def api_admin_checklist_create(request: Request, body: CriteriaCreateBody) -> dict[str, Any]:
    _require_admin(request)
    return api_criteria_post_create(request, body)


@app.post("/api/admin/checklists/active")
def api_admin_checklist_set_active(request: Request, body: CriteriaActiveBody) -> dict[str, Any]:
    _require_admin(request)
    return api_criteria_set_active(request, body)


@app.put("/api/admin/checklists/{name}")
def api_admin_checklist_update(request: Request, name: str, body: CriteriaPayload) -> dict[str, Any]:
    _require_admin(request)
    return api_criteria_put_content(name, request, body)


@app.delete("/api/admin/checklists/{name}")
def api_admin_checklist_delete(request: Request, name: str) -> dict[str, Any]:
    _require_admin(request)
    return api_criteria_delete(name, request)


@app.get("/api/admin/settings")
def api_admin_settings(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return _admin_settings_payload()


@app.put("/api/admin/settings")
def api_admin_settings_update(request: Request, body: AdminSettingsBody) -> dict[str, Any]:
    _require_admin(request)
    runtime_apply: dict[str, Any] | None = None
    if body.max_workers is not None:
        safe_workers = _coerce_int(body.max_workers, _MAX_WORKERS, _MAX_WORKERS_MIN, _MAX_WORKERS_MAX)
        _db.set_setting(OPS_MAX_WORKERS_KEY, str(safe_workers))
        runtime_apply = _apply_max_workers_setting(safe_workers)
    if body.admin_auto_refresh_seconds is not None:
        safe_refresh = _coerce_int(body.admin_auto_refresh_seconds, 10, 3, 120)
        _db.set_setting(OPS_ADMIN_REFRESH_SECONDS_KEY, str(safe_refresh))
    if body.local_registration_enabled is not None:
        _db.set_setting(
            AUTH_LOCAL_REGISTRATION_ENABLED_KEY,
            "1" if body.local_registration_enabled else "0",
        )
    if body.max_queue_depth is not None:
        safe_depth = _coerce_int(body.max_queue_depth, _MAX_QUEUE_DEPTH_DEFAULT, 1, _MAX_QUEUE_DEPTH_MAX)
        _db.set_setting(OPS_MAX_QUEUE_DEPTH_KEY, str(safe_depth))
    if body.default_daily_upload_limit is not None:
        safe_daily = _coerce_int(
            body.default_daily_upload_limit,
            _DEFAULT_DAILY_UPLOAD_LIMIT,
            0,
            _DEFAULT_DAILY_UPLOAD_LIMIT_MAX,
        )
        _db.set_setting(QUOTA_DEFAULT_DAILY_UPLOAD_LIMIT_KEY, str(safe_daily))
    if body.default_max_queued_jobs is not None:
        safe_user_queue = _coerce_int(
            body.default_max_queued_jobs,
            _DEFAULT_MAX_QUEUED_JOBS,
            0,
            _DEFAULT_MAX_QUEUED_JOBS_MAX,
        )
        _db.set_setting(QUOTA_DEFAULT_MAX_QUEUED_JOBS_KEY, str(safe_user_queue))
    if body.default_max_running_jobs is not None:
        safe_user_running = _coerce_int(
            body.default_max_running_jobs,
            _DEFAULT_MAX_RUNNING_JOBS,
            0,
            _DEFAULT_MAX_RUNNING_JOBS_MAX,
        )
        _db.set_setting(QUOTA_DEFAULT_MAX_RUNNING_JOBS_KEY, str(safe_user_running))
    _audit_log(
        request,
        action="admin.settings.update",
        target_type="settings",
        target_id="system",
        details={
            "max_workers": body.max_workers,
            "admin_auto_refresh_seconds": body.admin_auto_refresh_seconds,
            "local_registration_enabled": body.local_registration_enabled,
            "max_queue_depth": body.max_queue_depth,
            "default_daily_upload_limit": body.default_daily_upload_limit,
            "default_max_queued_jobs": body.default_max_queued_jobs,
            "default_max_running_jobs": body.default_max_running_jobs,
            "runtime_apply": runtime_apply,
        },
    )
    payload = _admin_settings_payload()
    payload["runtime_apply"] = runtime_apply
    return payload


@app.get("/api/admin/audit")
def api_admin_audit(request: Request, limit: int = 200) -> dict[str, Any]:
    _require_admin(request)
    return {"items": _db.list_audit_events(limit=limit)}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/login.html")
def login_page() -> FileResponse:
    f = STATIC_DIR / "login.html"
    if not f.is_file():
        raise HTTPException(404, "login.html не найден")
    return FileResponse(f, media_type="text/html; charset=utf-8")


@app.get("/")
def index_page() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "Соберите static/index.html")
    return FileResponse(index, media_type="text/html; charset=utf-8")


@app.get("/dashboard")
def dashboard_page(request: Request) -> FileResponse:
    _require_admin(request)
    f = STATIC_DIR / "dashboard.html"
    if not f.is_file():
        raise HTTPException(404, "dashboard.html не найден")
    return FileResponse(f, media_type="text/html; charset=utf-8")


@app.get("/admin")
def admin_page(request: Request) -> FileResponse:
    _require_admin(request)
    f = STATIC_DIR / "admin.html"
    if not f.is_file():
        raise HTTPException(404, "admin.html не найден")
    return FileResponse(f, media_type="text/html; charset=utf-8")
