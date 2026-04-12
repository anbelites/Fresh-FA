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
import shutil
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
    EVALUATION_DIR,
    LOCATIONS_FILE,
    MANAGERS_FILE,
    META_DIR,
    TRANSCRIPT_DIR,
    VIDEO_DIR,
    human_evaluation_json_path,
    meta_json_path,
)
from src.pipeline import (
    evaluate_only_from_transcript,
    find_video_for_stem,
    list_transcripts,
    list_videos,
    process_one_video,
)
from src.database import DB, init_db

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


@app.on_event("startup")
def _migrate_yaml_to_db() -> None:
    """One-time import of managers/locations from YAML config if DB tables are empty."""
    validate_ad_config_at_startup()
    if not _db.list_managers():
        _db.import_managers_from_yaml(_load_yaml_list(MANAGERS_FILE, "managers"))
    if not _db.list_locations():
        _db.import_locations_from_yaml(_load_yaml_list(LOCATIONS_FILE, "locations"))


import concurrent.futures

_db = DB()
_MAX_WORKERS = int(os.environ.get("FA_MAX_WORKERS", "2"))
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="fa-worker")

_ACTIVE_CRITERIA_MARKER = CONFIG_DIR / ".active_criteria"


def _safe_criteria_filename(name: str) -> str:
    s = (name or "").strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        raise HTTPException(400, "Некорректное имя файла")
    if not re.match(r"^[a-zA-Z0-9_\-\u0400-\u04FF]+\.ya?ml$", s):
        raise HTTPException(400, "Разрешены только .yaml / .yml в папке config")
    return s


def _active_criteria_path() -> Path:
    """Файл чеклиста для новых оценок и повторной генерации."""
    if _ACTIVE_CRITERIA_MARKER.is_file():
        try:
            raw = _ACTIVE_CRITERIA_MARKER.read_text(encoding="utf-8").strip()
            if raw:
                name = _safe_criteria_filename(raw)
                p = (CONFIG_DIR / name).resolve()
                if p.parent == CONFIG_DIR.resolve() and p.is_file():
                    return p
        except HTTPException:
            pass
    return CRITERIA_FILE


def _list_criteria_files() -> list[dict[str, str]]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, str]] = []
    for p in sorted(CONFIG_DIR.glob("*.yml")):
        if p.is_file():
            out.append({"name": p.name})
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        if p.is_file() and not any(x["name"] == p.name for x in out):
            out.append({"name": p.name})
    out.sort(key=lambda x: x["name"].lower())
    if not out and CRITERIA_FILE.is_file():
        out.append({"name": CRITERIA_FILE.name})
    return out


def _resolve_criteria_query_param(raw: str | None) -> str:
    """Имя файла чеклиста из query; при пустом/некорректном — активный."""
    if not raw or not str(raw).strip():
        return _active_criteria_path().name
    try:
        return _safe_criteria_filename(str(raw).strip())
    except HTTPException:
        return _active_criteria_path().name


def _load_evaluation_for_stem(stem: str, criteria_name: str) -> dict | None:
    """Оценка для пары (stem, чеклист); для criteria.yaml допускается legacy stem.json."""
    path = EVALUATION_DIR / f"{stem}__{criteria_name}.eval.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if criteria_name == CRITERIA_FILE.name:
        legacy = EVALUATION_DIR / f"{stem}.json"
        if legacy.is_file():
            return json.loads(legacy.read_text(encoding="utf-8"))
    return None


def _load_human_eval_for_stem(stem: str, criteria_name: str) -> dict | None:
    path = human_evaluation_json_path(stem, criteria_name)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
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

    try:
        if criteria_name:
            cpath = _criteria_file_path(criteria_name)
        else:
            cpath = _active_criteria_path()
        if not cpath.is_file():
            raise FileNotFoundError(str(cpath))
        _job_set(job_id, stage="evaluating", updated_at=_utc_now())
        evaluate_only_from_transcript(tr, criteria_path=cpath)
        _job_set(
            job_id,
            status="done",
            stage="done",
            stem=stem_key,
            updated_at=_utc_now(),
        )
    except Exception as e:
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=str(e),
            updated_at=_utc_now(),
        )


def _job_set(job_id: str, **updates: object) -> None:
    _db.upsert_job(job_id, **updates)


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


def _run_pipeline_job(job_id: str, video_path: Path) -> None:
    stem = video_path.stem
    _job_set(job_id, status="running", stage="starting", updated_at=_utc_now())

    def on_progress(phase: str) -> None:
        _job_set(job_id, stage=phase, updated_at=_utc_now())

    try:
        tr, ev, tone = process_one_video(
            video_path,
            on_progress=on_progress,
            criteria_path=_active_criteria_path(),
        )
        _job_set(
            job_id,
            status="done",
            stage="done",
            stem=stem,
            transcript=str(tr.relative_to(ROOT)),
            evaluation=str(ev.relative_to(ROOT)) if ev else None,
            tone_file=str(tone.relative_to(ROOT)) if tone else None,
            updated_at=_utc_now(),
        )
    except Exception as e:
        _job_set(
            job_id,
            status="error",
            stage="error",
            error=str(e),
            updated_at=_utc_now(),
        )


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
    _db.upsert_job(
        job_id,
        stem=dest.stem,
        kind="pipeline",
        status="queued",
        stage="queued",
        video_file=dest.name,
    )
    _db.upsert_video_meta(dest.stem, filename=dest.name, status="processing")

    _executor.submit(_run_pipeline_job, job_id, dest)

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
    evaluation: dict | None = None
    if tr_path.is_file():
        transcript = json.loads(tr_path.read_text(encoding="utf-8"))
    if tone_path.is_file():
        tone = json.loads(tone_path.read_text(encoding="utf-8"))
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
        "evaluation": evaluation,
        "human_evaluation": human_evaluation,
        "evaluation_criteria": crit_name,
        "criteria": {
            "active": _active_criteria_path().name,
            "files": _list_criteria_files(),
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
    meta = _db.upsert_video_meta(
        stem_key,
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


def _criteria_file_path(name: str) -> Path:
    fn = _safe_criteria_filename(name)
    return CONFIG_DIR / fn


def _normalize_new_criteria_filename(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise HTTPException(400, "Укажите имя файла")
    if not s.lower().endswith((".yaml", ".yml")):
        s += ".yaml"
    return _safe_criteria_filename(s)


def _parse_criteria_yaml_dict(raw: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    version = str(raw.get("version", "1"))
    out: list[dict[str, str]] = []
    for row in raw.get("criteria") or []:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id", "")).strip()
        if not cid:
            continue
        out.append(
            {
                "id": cid,
                "name": str(row.get("name", cid)).strip(),
                "description": str(row.get("description", "")).strip(),
            }
        )
    return version, out


def _dump_criteria_yaml(version: str, criteria: list[CriterionItem]) -> str:
    data: dict[str, Any] = {
        "version": version,
        "criteria": [
            {"id": c.id.strip(), "name": c.name.strip(), "description": c.description.strip()}
            for c in criteria
        ],
    }
    return yaml.dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    )


def _clear_active_marker_if_points_to(name: str) -> None:
    if not _ACTIVE_CRITERIA_MARKER.is_file():
        return
    try:
        if _ACTIVE_CRITERIA_MARKER.read_text(encoding="utf-8").strip() == name:
            _ACTIVE_CRITERIA_MARKER.unlink(missing_ok=True)
    except OSError:
        pass


@app.get("/api/criteria")
def api_criteria_list() -> dict:
    """Список YAML чеклистов в config/ и текущий активный файл."""
    return {
        "active": _active_criteria_path().name,
        "files": _list_criteria_files(),
    }


def _api_criteria_create_impl(body: CriteriaCreateBody) -> dict:
    """Новый YAML в config/ (пустой шаблон или копия существующего)."""
    fn = _normalize_new_criteria_filename(body.filename)
    dst = CONFIG_DIR / fn
    if dst.is_file():
        raise HTTPException(409, "Файл с таким именем уже есть")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if body.copy_from:
        src = _criteria_file_path(body.copy_from)
        if not src.is_file():
            raise HTTPException(404, "Исходный чеклист для копирования не найден")
        shutil.copy2(src, dst)
    else:
        template = _dump_criteria_yaml(
            "1",
            [
                CriterionItem(
                    id="criterion_1",
                    name="Новый критерий",
                    description="Опишите, что проверяет ИИ по этому пункту.",
                )
            ],
        )
        dst.write_text(template, encoding="utf-8")

    return {"ok": True, "filename": fn}


@app.post("/api/criteria")
def api_criteria_post_create(body: CriteriaCreateBody) -> dict:
    """Создать чеклист (основной URL для UI)."""
    return _api_criteria_create_impl(body)


@app.post("/api/criteria/active")
def api_criteria_set_active(body: CriteriaActiveBody) -> dict:
    """Сохранить активный чеклист (новые загрузки и «Обновить оценку» используют его)."""
    name = _safe_criteria_filename(body.file)
    path = CONFIG_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Файл не найден в папке config")
    _ACTIVE_CRITERIA_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_CRITERIA_MARKER.write_text(name, encoding="utf-8")
    return {"ok": True, "active": name}


@app.get("/api/criteria/content/{name}")
def api_criteria_get_content(name: str) -> dict:
    """Содержимое чеклиста для редактора (JSON)."""
    path = _criteria_file_path(name)
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise HTTPException(400, f"Не удалось прочитать YAML: {e}") from e
    if not isinstance(raw, dict):
        raise HTTPException(400, "Ожидался объект в корне YAML")
    version, criteria = _parse_criteria_yaml_dict(raw)
    files = _list_criteria_files()
    can_delete = len(files) > 1
    return {
        "filename": path.name,
        "version": version,
        "criteria": criteria,
        "can_delete": can_delete,
    }


@app.put("/api/criteria/content/{name}")
def api_criteria_put_content(name: str, body: CriteriaPayload) -> dict:
    """Сохранить чеклист из редактора."""
    path = _criteria_file_path(name)
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    for c in body.criteria:
        if not c.id.strip():
            raise HTTPException(400, "У каждого критерия должен быть непустой id")
    text = _dump_criteria_yaml(body.version, body.criteria)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise HTTPException(500, f"Не удалось записать файл: {e}") from e
    return {"ok": True, "filename": path.name}


@app.post("/api/criteria/create")
def api_criteria_create_alias(body: CriteriaCreateBody) -> dict:
    """Алиас для совместимости (тот же сценарий, что POST /api/criteria)."""
    return _api_criteria_create_impl(body)


@app.delete("/api/criteria/content/{name}")
def api_criteria_delete(name: str) -> dict:
    """Удалить файл чеклиста (не единственный в config/)."""
    path = _criteria_file_path(name)
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    files = _list_criteria_files()
    if len(files) <= 1:
        raise HTTPException(400, "Нельзя удалить единственный чеклист")
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(500, f"Не удалось удалить: {e}") from e
    _clear_active_marker_if_points_to(path.name)
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
