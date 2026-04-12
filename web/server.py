"""
Веб-интерфейс: загрузка видео, списки транскриптов/оценок, прогресс пайплайна.
Запуск из корня проекта: uvicorn web.server:app --reload --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from src.ad_auth import (
    ad_auth_enabled,
    fetch_ad_display_name,
    require_session_secret,
    validate_ad_config_at_startup,
    verify_user_password,
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
from src.database import DB, init_db, migrate_checklists_from_config
from src.errors import PipelineCancelled

_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".m4a": "audio/mp4",
}

STATIC_DIR = Path(__file__).resolve().parent / "static"

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
        if not ad_auth_enabled():
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
        if path in ("/", "/dashboard"):
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


_MAX_WORKERS = int(os.environ.get("FA_MAX_WORKERS", "2"))
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="fa-worker")

_job_cancel_lock = threading.Lock()
_job_cancel_events: dict[str, threading.Event] = {}


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


def _resolve_criteria_query_param(raw: str | None) -> str:
    """Slug чеклиста из query; при пустом/некорректном — активный из БД."""
    fallback = _db.get_active_checklist_slug()
    if not raw or not str(raw).strip():
        return fallback
    try:
        name = _safe_criteria_filename(str(raw).strip())
    except HTTPException:
        return fallback
    if _db.checklist_exists(name):
        return name
    return fallback


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

        slug = criteria_name if criteria_name else _db.get_active_checklist_slug()
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


def _enqueue_pipeline_job_common(
    job_id: str,
    stem_key: str,
    video_path: Path,
    *,
    resume: bool,
) -> None:
    with _job_cancel_lock:
        _job_cancel_events[job_id] = threading.Event()
    _db.upsert_job(
        job_id,
        stem=stem_key,
        kind="pipeline",
        status="queued",
        stage="queued",
        video_file=video_path.name,
    )
    _db.upsert_video_meta(stem_key, filename=video_path.name, status="processing")
    if resume:
        _executor.submit(_run_pipeline_resume_job, job_id, video_path)
    else:
        _executor.submit(_run_pipeline_job, job_id, video_path)


def _run_pipeline_resume_job(job_id: str, video_path: Path) -> None:
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

        tr, ev, tone = process_one_video_resume(
            video_path,
            on_progress=on_progress,
            criteria_path=Path(_db.get_active_checklist_slug()),
            db=_db,
            cancel_check=cancel_check,
            eval_stream_callback=stream_cb,
        )
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


def _run_pipeline_job(job_id: str, video_path: Path) -> None:
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

        tr, ev, tone = process_one_video(
            video_path,
            on_progress=on_progress,
            criteria_path=Path(_db.get_active_checklist_slug()),
            db=_db,
            cancel_check=cancel_check,
            eval_stream_callback=stream_cb,
        )
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
def auth_status() -> dict[str, bool]:
    return {"ad_auth_enabled": ad_auth_enabled()}


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    return {
        "user": request.session.get("user"),
        "display_name": request.session.get("display_name"),
        "ad_auth_enabled": ad_auth_enabled(),
    }


class _LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def auth_login(request: Request, body: _LoginBody) -> JSONResponse:
    if not ad_auth_enabled():
        raise HTTPException(400, "Вход через Active Directory не включён (AD_AUTH_ENABLED).")
    ok, err = verify_user_password(body.username, body.password)
    if not ok:
        raise HTTPException(401, err or "Ошибка входа")
    name = body.username.strip()
    display_name = fetch_ad_display_name(body.username, body.password)
    request.session["user"] = name
    request.session["display_name"] = display_name
    return JSONResponse(
        {"ok": True, "user": name, "display_name": display_name}
    )


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    request.session.clear()
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "Нет имени файла")
    ext = Path(file.filename).suffix.lower()
    allowed = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".m4a"}
    if ext not in allowed:
        raise HTTPException(
            400, f"Неподдерживаемый формат: {ext}. Допустимо: {', '.join(sorted(allowed))}"
        )

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(file.filename)
    dest = VIDEO_DIR / f"{stem}{ext}"
    n = 0
    while dest.is_file():
        n += 1
        dest = VIDEO_DIR / f"{stem}_{n}{ext}"

    data = await file.read()
    dest.write_bytes(data)

    job_id = str(uuid.uuid4())
    _enqueue_pipeline_job_common(job_id, dest.stem, dest, resume=False)

    return JSONResponse(
        {
            "job_id": job_id,
            "stem": dest.stem,
            "video_file": dest.name,
            "message": "Обработка запущена",
        }
    )


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


def _library_row(
    stem: str,
    *,
    video_path: Path | None,
    jobs_by: dict[str, dict],
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

    meta = _db.get_video_meta(stem)

    return {
        "stem": stem,
        "video_file": video_file or stem,
        "has_video_file": bool(video_path and video_path.is_file()),
        "size_bytes": size_bytes,
        "mtime": mtime,
        "has_transcript": tr.is_file(),
        "has_tone": tone.is_file(),
        "has_evaluation": _stem_has_any_evaluation(stem),
        "manager_id": meta.get("manager_id"),
        "manager_name": meta.get("manager_name"),
        "location_id": meta.get("location_id"),
        "location_name": meta.get("location_name"),
        "tags": meta.get("tags") or [],
        "display_title": meta.get("display_title"),
        "job": _job_payload(stem, jobs_by),
    }


@app.get("/api/library")
def api_library() -> list[dict]:
    """
    Записи из 01.Video плюс любые транскрипты из 02.Transcript без видео в папке
    (чтобы уже обработанные файлы не пропадали из списка).
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    jobs_by = _jobs_latest_by_stem()

    by_stem: dict[str, dict] = {}

    for p in list_videos(VIDEO_DIR):
        stem = p.stem
        by_stem[stem] = _library_row(stem, video_path=p, jobs_by=jobs_by)

    for tr_path in list_transcripts(TRANSCRIPT_DIR):
        stem = tr_path.stem
        if stem in by_stem:
            continue
        vp = find_video_for_stem(stem)
        by_stem[stem] = _library_row(stem, video_path=vp, jobs_by=jobs_by)

    out = sorted(by_stem.values(), key=lambda x: x.get("mtime") or 0, reverse=True)
    return out


@app.delete("/api/library/{stem}")
def api_library_delete(stem: str) -> dict:
    """Удаляет видео в 01.Video (если есть), транскрипт, тон и все оценки для stem."""
    stem_key = _stem_from_path_param(stem)
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

    _db.delete_jobs_for_stem(stem_key)
    _db.delete_video(stem_key)

    mp = meta_json_path(stem_key)
    if mp.is_file():
        mp.unlink()
        deleted.append(str(mp.relative_to(ROOT)))

    if not deleted:
        raise HTTPException(404, "Нет данных для удаления")

    return {"ok": True, "deleted": deleted}


@app.get("/api/workspace/{stem}")
def api_workspace(stem: str, criteria: str | None = None) -> dict:
    """Одним ответом: видео, транскрипт, SER, LLM-оценка для выбранного чеклиста (query criteria)."""
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(criteria)
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
    evaluation = _load_evaluation_for_stem(stem_key, crit_name)
    human_evaluation = _load_human_eval_for_stem(stem_key, crit_name)

    jobs_by = _jobs_latest_by_stem()
    job = jobs_by.get(stem_key)

    return {
        "stem": stem_key,
        "video_file": vf.name if vf else (transcript or {}).get("video_file"),
        "video_url": f"/api/videos/{stem_key}" if vf and vf.is_file() else None,
        "transcript": transcript,
        "tone": tone,
        "transcript_load_error": transcript_load_error,
        "tone_load_error": tone_load_error,
        "evaluation": evaluation,
        "human_evaluation": human_evaluation,
        "evaluation_criteria": crit_name,
        "criteria": {
            "active": _db.get_active_checklist_slug(),
            "files": _db.list_checklist_files(),
        },
        "meta": _db.get_video_meta(stem_key),
        "managers": _db.list_managers(),
        "locations": _db.list_locations(),
        "job": (
            {
                "id": job.get("id"),
                "status": job.get("status"),
                "stage": job.get("stage"),
                "kind": job.get("kind"),
                "error": job.get("error"),
            }
            if job
            else None
        ),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    j = _db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Задача не найдена")
    return j


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str) -> JSONResponse:
    """Кооперативная остановка пайплайна или пересчёта оценки (queued / running)."""
    jid = (job_id or "").strip()
    if not jid:
        raise HTTPException(400, "Пустой идентификатор")
    row = _db.get_job(jid)
    if not row:
        raise HTTPException(404, "Задача не найдена")
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
        raise HTTPException(400, "Нет исходного файла в 01.Video для этой записи")
    jobs_by = _jobs_latest_by_stem()
    job = jobs_by.get(stem_key)
    if not job or job.get("kind") != "pipeline":
        raise HTTPException(400, "Для этой записи нет задачи полного пайплайна")
    if job.get("status") not in ("cancelled", "error"):
        raise HTTPException(400, "Доступно после остановки или ошибки обработки")


@app.post("/api/workspace/{stem}/pipeline/resume")
def api_workspace_pipeline_resume(stem: str) -> JSONResponse:
    """Возобновить без удаления файлов: пропуск готовых этапов (транскрипт → тон → оценка)."""
    stem_key = _stem_from_path_param(stem)
    vp = find_video_for_stem(stem_key)
    _assert_pipeline_can_restart(stem_key, vp)
    job_id = str(uuid.uuid4())
    _enqueue_pipeline_job_common(job_id, stem_key, vp, resume=True)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.post("/api/workspace/{stem}/pipeline/restart")
def api_workspace_pipeline_restart(stem: str) -> JSONResponse:
    """Удалить транскрипт/тон/оценки и запустить пайплайн с нуля."""
    stem_key = _stem_from_path_param(stem)
    vp = find_video_for_stem(stem_key)
    _assert_pipeline_can_restart(stem_key, vp)
    _delete_pipeline_derived_artifacts(stem_key)
    job_id = str(uuid.uuid4())
    _enqueue_pipeline_job_common(job_id, stem_key, vp, resume=False)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/transcripts")
def api_list_transcripts() -> list[dict]:
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
def api_transcript_detail(stem: str) -> dict:
    path = TRANSCRIPT_DIR / f"{stem}.json"
    if not path.is_file():
        raise HTTPException(404, "Транскрипт не найден")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/videos/{stem}")
def api_stream_video(stem: str) -> FileResponse:
    """Раздача исходного файла из 01.Video (тот же stem, что у транскрипта)."""
    stem_key = _stem_from_path_param(stem)
    video_path = find_video_for_stem(stem_key)
    if not video_path or not video_path.is_file():
        raise HTTPException(404, "Видео не найдено в 01.Video")
    ext = video_path.suffix.lower()
    media = _VIDEO_MIME.get(ext, "application/octet-stream")
    return FileResponse(
        video_path,
        media_type=media,
        filename=video_path.name,
    )


@app.get("/api/evaluations")
def api_list_evaluations() -> list[dict]:
    out: list[dict] = []
    for p in sorted(EVALUATION_DIR.glob("*.json")):
        if not p.is_file():
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
def api_evaluation_detail(stem: str, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(criteria)
    ev = _load_evaluation_for_stem(stem_key, crit_name)
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
    tags: list[str] = []


class ManagerBody(BaseModel):
    id: str
    name: str


class LocationBody(BaseModel):
    id: str
    name: str


@app.get("/api/managers")
def api_managers_list() -> list[dict]:
    return _db.list_managers()


@app.post("/api/managers")
def api_managers_add(body: ManagerBody) -> dict:
    bid = body.id.strip()
    if not bid:
        raise HTTPException(400, "id обязателен")
    _db.add_manager(bid, body.name.strip() or bid)
    return {"ok": True}


@app.delete("/api/managers/{mid}")
def api_managers_delete(mid: str) -> dict:
    if not _db.delete_manager(mid):
        raise HTTPException(404, "Менеджер не найден")
    return {"ok": True}


@app.get("/api/locations")
def api_locations_list() -> list[dict]:
    return _db.list_locations()


@app.post("/api/locations")
def api_locations_add(body: LocationBody) -> dict:
    bid = body.id.strip()
    if not bid:
        raise HTTPException(400, "id обязателен")
    _db.add_location(bid, body.name.strip() or bid)
    return {"ok": True}


@app.delete("/api/locations/{lid}")
def api_locations_delete(lid: str) -> dict:
    if not _db.delete_location(lid):
        raise HTTPException(404, "Локация не найдена")
    return {"ok": True}


@app.get("/api/workspace/{stem}/meta")
def api_workspace_meta_get(stem: str) -> dict:
    stem_key = _stem_from_path_param(stem)
    return _db.get_video_meta(stem_key)


@app.put("/api/workspace/{stem}/meta")
def api_workspace_meta_put(stem: str, body: VideoMetaBody) -> dict:
    stem_key = _stem_from_path_param(stem)
    disp = (body.display_title or "").strip() or None
    meta = _db.upsert_video_meta(
        stem_key,
        display_title=disp,
        manager_id=body.manager_id,
        manager_name=body.manager_name,
        location_id=body.location_id,
        location_name=body.location_name,
        interaction_date=body.interaction_date,
        tags=body.tags,
    )
    return {"ok": True, "meta": meta}


class CriteriaActiveBody(BaseModel):
    file: str


class CriterionItem(BaseModel):
    id: str
    name: str
    description: str = ""


class CriteriaPayload(BaseModel):
    version: str = "1"
    criteria: list[CriterionItem]


class CriteriaCreateBody(BaseModel):
    filename: str
    copy_from: str | None = None


def _normalize_new_criteria_filename(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise HTTPException(400, "Укажите имя чеклиста")
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
            }
        ]
    _db.insert_checklist(fn, ver, crits)
    return {"ok": True, "filename": fn}


@app.post("/api/criteria")
def api_criteria_post_create(body: CriteriaCreateBody) -> dict:
    """Создать чеклист (основной URL для UI)."""
    return _api_criteria_create_impl(body)


@app.post("/api/criteria/active")
def api_criteria_set_active(body: CriteriaActiveBody) -> dict:
    """Сохранить активный чеклист (новые загрузки и «Обновить оценку» используют его)."""
    name = _safe_criteria_filename(body.file)
    if not _db.checklist_exists(name):
        raise HTTPException(404, "Чеклист не найден")
    try:
        _db.set_active_checklist_slug(name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
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
def api_criteria_put_content(name: str, body: CriteriaPayload) -> dict:
    """Сохранить чеклист из редактора в БД."""
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
        {"id": c.id.strip(), "name": c.name.strip(), "description": (c.description or "").strip()}
        for c in body.criteria
    ]
    _db.replace_checklist(fn, (body.version or "1").strip(), rows)
    return {"ok": True, "filename": fn}


@app.post("/api/criteria/create")
def api_criteria_create_alias(body: CriteriaCreateBody) -> dict:
    """Алиас для совместимости (тот же сценарий, что POST /api/criteria)."""
    return _api_criteria_create_impl(body)


@app.delete("/api/criteria/content/{name}")
def api_criteria_delete(name: str) -> dict:
    """Удалить чеклист из БД (не единственный)."""
    try:
        fn = _safe_criteria_filename(name)
    except HTTPException as e:
        raise HTTPException(404, "Чеклист не найден") from e
    if _db.checklist_count() <= 1:
        raise HTTPException(400, "Нельзя удалить единственный чеклист")
    if not _db.checklist_exists(fn):
        raise HTTPException(404, "Чеклист не найден")
    if not _db.delete_checklist(fn):
        raise HTTPException(500, "Не удалось удалить")
    if _db.checklist_count():
        _db.get_active_checklist_slug()
    return {"ok": True}


class ReEvaluateBody(BaseModel):
    criteria: str | None = None


@app.post("/api/workspace/{stem}/re-evaluate")
def api_workspace_re_evaluate(
    stem: str, body: ReEvaluateBody = ReEvaluateBody()
) -> JSONResponse:
    """Повторная LLM-оценка по транскрипту; чеклист — из тела (criteria) или активный."""
    stem_key = _stem_from_path_param(stem)
    tr = TRANSCRIPT_DIR / f"{stem_key}.json"
    if not tr.is_file():
        raise HTTPException(400, "Нет транскрипта для этой записи")

    cname: str | None = None
    if body.criteria and str(body.criteria).strip():
        try:
            cname = _safe_criteria_filename(str(body.criteria).strip())
        except HTTPException:
            raise HTTPException(400, "Некорректное имя чеклиста") from None

    job_id = str(uuid.uuid4())
    with _job_cancel_lock:
        _job_cancel_events[job_id] = threading.Event()
    _db.upsert_job(
        job_id,
        stem=stem_key,
        kind="eval_only",
        status="queued",
        stage="queued",
    )

    _executor.submit(_run_eval_only_job, job_id, stem_key, cname)

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
    score: int | None = None
    comment: str = ""


class HumanEvalBody(BaseModel):
    criteria: list[HumanEvalCriterion]
    criteria_file: str | None = None


@app.get("/api/workspace/{stem}/human-eval")
def api_human_eval_get(stem: str, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(criteria)
    data = _load_human_eval_for_stem(stem_key, crit_name)
    if data is None:
        return {"exists": False}
    data["exists"] = True
    return data


@app.put("/api/workspace/{stem}/human-eval")
def api_human_eval_put(stem: str, body: HumanEvalBody, criteria: str | None = None) -> dict:
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(criteria)
    crit_out = []
    for c in body.criteria:
        crit_out.append({
            "id": c.id,
            "name": c.name,
            "score": c.score,
            "comment": c.comment,
        })
    numeric = [c["score"] for c in crit_out if c["score"] is not None]
    overall = round(sum(numeric) / len(numeric), 1) if numeric else None
    data = {
        "schema_version": 2,
        "source": "human",
        "criteria_file": body.criteria_file or crit_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "overall_average": overall,
        "criteria": crit_out,
    }
    _save_human_eval_for_stem(stem_key, crit_name, data)
    return {"ok": True, "human_evaluation": data}


class CompareBody(BaseModel):
    criteria: str | None = None


@app.post("/api/workspace/{stem}/compare-eval")
def api_compare_eval(stem: str, body: CompareBody = CompareBody()) -> dict:
    """Compare AI and human evaluations for the same stem + criteria via LLM."""
    stem_key = _stem_from_path_param(stem)
    crit_name = _resolve_criteria_query_param(body.criteria)
    ai_eval = _load_evaluation_for_stem(stem_key, crit_name)
    human_eval = _load_human_eval_for_stem(stem_key, crit_name)

    if not ai_eval:
        raise HTTPException(400, "Нет ИИ-оценки для сравнения")
    if not human_eval:
        raise HTTPException(400, "Нет ручной оценки для сравнения")

    ai_criteria = {c["id"]: c for c in (ai_eval.get("criteria") or [])}
    human_criteria = {c["id"]: c for c in (human_eval.get("criteria") or [])}

    comparison_rows: list[dict[str, Any]] = []
    all_ids = list(dict.fromkeys(
        [c["id"] for c in ai_eval.get("criteria", [])]
        + [c["id"] for c in human_eval.get("criteria", [])]
    ))
    for cid in all_ids:
        ai_c = ai_criteria.get(cid, {})
        hu_c = human_criteria.get(cid, {})
        ai_score = ai_c.get("score")
        hu_score = hu_c.get("score")
        diff: float | None = None
        diff_pct: float | None = None
        if ai_score is not None and hu_score is not None:
            diff = abs(ai_score - hu_score)
            diff_pct = diff  # scores are 0-100, diff IS the percentage points
        comparison_rows.append({
            "id": cid,
            "name": ai_c.get("name") or hu_c.get("name") or cid,
            "ai_score": ai_score,
            "ai_comment": ai_c.get("comment", ""),
            "human_score": hu_score,
            "human_comment": hu_c.get("comment", ""),
            "diff": diff,
            "diff_pct": diff_pct,
        })

    ai_overall = ai_eval.get("overall_average")
    hu_overall = human_eval.get("overall_average")
    overall_diff = (
        abs(ai_overall - hu_overall)
        if ai_overall is not None and hu_overall is not None
        else None
    )

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
            model = _os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini")
            rows_text = "\n".join(
                f"- {r['name']} (id={r['id']}): ИИ={r['ai_score']}, Человек={r['human_score']}, разница={r['diff']}"
                for r in comparison_rows
            )
            prompt = (
                "Ты аналитик контроля качества. Тебе дана таблица расхождений между ИИ-оценкой и ручной оценкой человека "
                "по чеклисту для записи делового звонка. Каждый критерий оценён по шкале 0-100.\n\n"
                f"Средний балл ИИ: {ai_overall}, средний балл Человека: {hu_overall}, общее расхождение: {overall_diff}\n\n"
                f"Критерии:\n{rows_text}\n\n"
                "Кратко (3-5 предложений на русском) проанализируй:\n"
                "1) Где расхождения значительны и почему они могут возникать\n"
                "2) Кто чаще завышает/занижает\n"
                "3) Общий вывод: насколько ИИ-оценка согласована с экспертной"
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            llm_analysis = (resp.choices[0].message.content or "").strip()
    except Exception:
        pass

    return {
        "stem": stem_key,
        "criteria_file": crit_name,
        "ai_overall": ai_overall,
        "human_overall": hu_overall,
        "overall_diff": overall_diff,
        "rows": comparison_rows,
        "llm_analysis": llm_analysis,
    }


def _collect_all_evaluations() -> list[dict[str, Any]]:
    """Read every evaluation JSON and join with video metadata."""
    rows: list[dict[str, Any]] = []
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(EVALUATION_DIR.glob("*.json")):
        if not p.is_file():
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
        row = {
            "stem": stem_part,
            "file": p.name,
            "criteria_file": data.get("criteria_file"),
            "video_file": data.get("video_file"),
            "evaluated_at": data.get("evaluated_at"),
            "model": data.get("model"),
            "overall_average": data.get("overall_average"),
            "manager_id": meta.get("manager_id"),
            "manager_name": meta.get("manager_name"),
            "location_id": meta.get("location_id"),
            "location_name": meta.get("location_name"),
            "tags": meta.get("tags") or [],
            "criteria": data.get("criteria") or [],
        }
        rows.append(row)
    return rows


# ── Dashboard API ────────────────────────────────────────────────────

@app.get("/api/dashboard")
def api_dashboard(
    manager_id: str | None = None,
    location_id: str | None = None,
) -> dict:
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
            cs = c.get("score")
            if cs is not None:
                cname = c.get("name") or c.get("id") or "?"
                criteria_scores.setdefault(cname, []).append(cs)

    def _agg(d: dict[str, list[float]]) -> list[dict]:
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


# ── Export API ───────────────────────────────────────────────────────

@app.get("/api/export/csv")
def api_export_csv(
    manager_id: str | None = None,
    location_id: str | None = None,
) -> FileResponse:
    """Export evaluations as CSV."""
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
    header = ["Файл", "Менеджер", "Локация", "Дата оценки", "Общий балл"] + all_crit_names
    writer.writerow(header)

    for e in evals:
        crit_map = {}
        for c in e.get("criteria") or []:
            cn = c.get("name") or c.get("id") or "?"
            crit_map[cn] = c.get("score")
        row = [
            e.get("video_file") or e.get("stem"),
            e.get("manager_name") or e.get("manager_id") or "",
            e.get("location_name") or e.get("location_id") or "",
            (e.get("evaluated_at") or "")[:19],
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
def dashboard_page() -> FileResponse:
    f = STATIC_DIR / "dashboard.html"
    if not f.is_file():
        raise HTTPException(404, "dashboard.html не найден")
    return FileResponse(f, media_type="text/html; charset=utf-8")
