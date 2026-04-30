#!/usr/bin/env python3
"""
Пайплайн: 01.Video (видео/аудио) → 02.Transcript (JSON) → 03.Evaluation (JSON).

Переменные окружения (опционально):
  HF_TOKEN — токен Hugging Face (диаризация, загрузка моделей с Hub); в .env подхватывается с приоритетом над пустым env (см. load_dotenv override в run.py);
    примите условия модели pyannote/speaker-diarization-community-1 на huggingface.co
  DIARIZATION_BACKEND — auto | nemo | pyannote | mfcc; по умолчанию auto = NeMo Sortformer → pyannote → MFCC fallback
  NEMO_DIAR_MODEL — по умолчанию nvidia/diar_sortformer_4spk-v1; можно указать локальный .nemo
  NEMO_DIAR_DEVICE — auto | cuda | cpu; по умолчанию auto = CUDA при доступной GPU
  NEMO_DIAR_BATCH_SIZE — batch size для Sortformer (по умолчанию 1)
  Для NeMo см. requirements-nemo.txt; Linux: обычно нужны ffmpeg и libsndfile1
  PYANNOTE_PIPELINE — по умолчанию pyannote/speaker-diarization-community-1; для отката можно явно задать legacy 3.1 или локальный config.yaml
  PYANNOTE_ON_WINDOWS=1 — на Windows pyannote по умолчанию отключён (torchcodec); включите только если настроили FFmpeg/torchcodec
  Linux + pyannote 4: нужен системный FFmpeg (sudo apt install ffmpeg) — иначе torchcodec / libavutil.so.* not found
  SKIP_PYANNOTE=1 — не вызывать pyannote (MFCC-диаризация), на любой ОС
  PYANNOTE_NUM_SPEAKERS — если число говорящих известно (например 3: ведущий + два в диалоге), точнее кластеризация
  PYANNOTE_LABEL_SMOOTH_RADIUS — сглаживание меток pyannote по потоку слов (для community-1 по умолчанию 0; 0 = выкл.)
  PYANNOTE_SEGMENT_SMOOTH_RADIUS — доп. сглаживание по строкам транскрипта с pyannote (для community-1 по умолчанию 0; иначе 1 на CUDA, 0 на CPU)
  MFCC_SEGMENT_SPEAKER_SMOOTH_RADIUS — сглаживание по строкам без HF (по умолчанию 2; 0 = выкл.)
  PYANNOTE_MIN_SPEAKERS / PYANNOTE_MAX_SPEAKERS — границы числа спикеров
  PYANNOTE_MAX_SPEAKERS_DEFAULT — верхняя граница по умолчанию (10), если PYANNOTE_MAX_SPEAKERS не задан; 0 = не задавать
  PYANNOTE_EXCLUSIVE=1 — брать разметку без перекрывающихся реплик (по умолчанию на CUDA включается автоматически)
  PYANNOTE_MERGE_SHORT_SPEAKERS=0/1 — схлопывать короткие «фантомные» SPEAKER_* в соседний стабильный спикер
  PYANNOTE_MERGE_SHORT_SPEAKER_MAX_SEC — максимум суммарной длительности такого спикера (по умолчанию 12)
  PYANNOTE_MERGE_SHORT_SPEAKER_MAX_RUNS — максимум отдельных заходов такого спикера (по умолчанию 3)
  PYANNOTE_MERGE_MIN_DISTINCT_SPEAKERS — включать схлопывание только если разных SPEAKER_* не меньше этого числа (по умолчанию 4)
  PYANNOTE_BRIDGE_SHORT_RUN_SEC — склеивать короткую ошибочную вставку спикера между двумя одинаковыми соседями (для community-1 по умолчанию 0.25 сек)
  PYANNOTE_BRIDGE_MIN_DISTINCT_SPEAKERS — включать такую bridge-склейку только если разных спикеров не меньше этого числа (по умолчанию 4)
  PYANNOTE_ABA_FLANK_MAX_SEC — внутри одного Whisper-сегмента схлопывать короткие края паттерна A-B-A в центрального спикера (для community-1 по умолчанию 2.6 сек)
  PYANNOTE_ABA_CENTER_MIN_SEC — минимальная длительность центрального B для такой A-B-A склейки (для community-1 по умолчанию 2.0 сек)
  PYANNOTE_TAKEOVER_PREFIX_MAX_SEC — если новый сегмент явно продолжает ту же фразу, можно перетащить в его спикера короткий сомнительный префикс перед ним (для community-1 по умолчанию 2.2 сек)
  PYANNOTE_TAKEOVER_MIN_TARGET_SEC — минимальная длительность нового сегмента-«якоря» для такой takeover-склейки (для community-1 по умолчанию 1.0 сек)
  PYANNOTE_TAKEOVER_PREFIX_MAX_MEAN_WORD_PROB — takeover работает только если префикс был достаточно неуверенно распознан (для community-1 по умолчанию 0.6)
  PYANNOTE_TAKEOVER_MIN_CONF_GAIN — насколько «якорный» сегмент должен быть увереннее префикса по mean word prob (для community-1 по умолчанию 0.08)
  MFCC_DIAR_MAX_SPEAKERS — без HF_TOKEN: до скольких кластеров KMeans по сегментам Whisper (по умолчанию 8)
  MFCC_WORD_LEVEL=1 — включить MFCC по каждому слову (по умолчанию выкл. — даёт шум и дробление строк)
  MFCC_WORD_SMOOTH_RADIUS — сглаживание меток по словам (по умолчанию 4), только при MFCC_WORD_LEVEL=1
  TRANSCRIPT_PAUSE_SPLIT_SEC — пауза (сек) для дополнительного разреза строк; по умолчанию 0 (только смена спикера)
  OPENAI_API_KEY — ключ для оценки по критериям (чеклисты в SQLite)
  WHISPER_MODEL — по умолчанию large-v3 (точнее; на CPU дольше)
  WHISPER_DEVICE — если не задан: cuda при доступной GPU (CTranslate2), иначе cpu; явно: cpu | cuda
  WHISPER_COMPUTE_TYPE — если не задан: float16 на cuda, int8 на cpu; явно переопределить можно
  WHISPER_INITIAL_PROMPT — полный override доменного prompt для Whisper; если не задан, используется встроенный Fresh/авто-глоссарий
  WHISPER_DOMAIN_PROMPT — дополнительные термины к встроенному Fresh/авто-глоссарию для Whisper
  PYANNOTE_DEVICE — если не задан: cuda при torch.cuda, иначе cpu; явно: cpu | cuda
  SPEECH_EMOTION_DEVICE — если не задан или auto: GPU при torch.cuda; явно: cpu | cuda | mps
  Ошибка libcublas.so.12: nvidia-cublas-cu12 (см. requirements-cuda12.txt); на Ubuntu — venv или pip --user --break-system-packages; иначе полный CUDA 12 toolkit
  SPEECH_FAST_WPS, SPEECH_FAST_SPS, SPEECH_LOW_LOGPROB, SPEECH_LOW_WORD_PROB — пороги «тараторство/уверенность ASR» в транскрипте
  SKIP_AUDIO_TONE=1 — отключить акустику тона (F0, RMS, спектр) по WAV
  SKIP_SPEECH_EMOTION=1 — не писать SER-файл *.tone.json рядом с транскриптом
  SPEECH_EMOTION_MODEL — по умолчанию xbgoose/hubert-large…-dusha (русский Dusha); Aniemore — RESD 7 классов; superb — IEMOCAP
  SPEECH_EMOTION_TRUST_REMOTE_CODE — 0/1 (для Aniemore обычно 1; xbgoose/nikatonika — 0)
  OPENAI_BASE_URL — например https://api.deepseek.com/v1 для DeepSeek
  OPENAI_EVAL_MODEL — например deepseek-v4-pro / deepseek-v4-flash / deepseek-reasoner
  OPENAI_EVAL_REASONING_EFFORT — для DeepSeek V4: high или max (по умолчанию high)
  OPENAI_EVAL_MAX_TOKENS — лимит выхода JSON для reasoning/thinking-моделей (по умолчанию 65536)
  speaker_role в transcript теперь должен подтверждаться шагом оценки (eval), а не черновой эвристикой после ASR

  Аутентификация веб-интерфейса (опционально):
  AUTH_TYPE=none|ad|local — режим входа (по умолчанию none; для совместимости AD_AUTH_ENABLED=1 эквивалентен AUTH_TYPE=ad)
  SESSION_SECRET — обязательна длинная случайная строка (подпись cookie-сессии)
  ADMIN_USERS=user1,user2 — логины admin через .env (работает и для AD, и для local)

  Вход через Active Directory (LDAP):
  AD_LDAP_URI — ldap://dc.corp.local:389 или ldaps://… (на Windows можно не задавать — см. автообнаружение)
  AD_LDAP_AUTO_DISCOVER=1 — по умолчанию: при пустом AD_LDAP_URI на Windows взять DC и DNS-имя домена из AD (WMI + .NET)
  AD_LDAP_DISCOVER_PORT — порт для авто-URI (по умолчанию 389)
  AD_USE_STARTTLS=1 — STARTTLS к ldap://; для ручного URI по умолчанию выкл., при автообнаружении — вкл., если не задано явно
  Режим A (UPN): AD_BIND_UPN_TEMPLATE="{username}@corp.local"
  Режим B (поиск): AD_SERVICE_BIND_DN, AD_SERVICE_BIND_PASSWORD, AD_SEARCH_BASE,
    опционально AD_USER_SEARCH_FILTER="(sAMAccountName={username})", AD_SEARCH_SCOPE=BASE|SUBTREE
  AD_USER_SEARCH_BASE — при UPN-входе можно задать DN базы поиска ФИО вместо defaultNamingContext с root DSE
  AD_SKIP_DISPLAY_NAME=1 — не запрашивать displayName/ФИО в LDAP после входа
  SESSION_MAX_AGE_SEC — время жизни сессии (по умолчанию 604800)
  SESSION_COOKIE_SECURE=1 — cookie только по HTTPS
  LOCAL_AUTH_PBKDF2_ITERATIONS — число итераций PBKDF2 для локальных пользователей (по умолчанию 600000)

  python run.py serve [--host 127.0.0.1] [--port 8765] — веб-интерфейс (загрузка видео/аудио, списки, оценки)

Загрузка .env из корня проекта, если есть.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# override=True: иначе пустой HF_TOKEN из окружения IDE/шелла перекрывает .env — Hub идёт без авторизации
load_dotenv(ROOT / ".env", override=True)

from src.cuda_runtime_path import ensure_nvidia_pip_libs

ensure_nvidia_pip_libs()

from src.database import DB, init_db, migrate_checklists_from_config
from src.local_auth import hash_local_password, normalize_local_username
from src.paths import CONFIG_DIR, TRANSCRIPT_DIR, VIDEO_DIR
from src.pipeline import (
    emotion_only_from_transcript,
    evaluate_only_from_transcript,
    list_transcripts,
    list_videos,
    process_one_video,
)


def _cli_checklist_db() -> tuple[DB, Path]:
    """Инициализация SQLite и чеклистов (импорт YAML при первом запуске)."""
    init_db()
    db = DB()
    migrate_checklists_from_config(
        db,
        config_dir=CONFIG_DIR,
        active_marker=CONFIG_DIR / ".active_criteria",
        delete_source_files=True,
    )
    return db, Path(db.get_active_checklist_slug())


def main() -> None:
    parser = argparse.ArgumentParser(description="Транскрипция и оценка видео/аудио")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("process-all", help="Обработать все видео и аудио в 01.Video")
    p_all.add_argument(
        "--skip-eval",
        action="store_true",
        help="Только транскрипт, без оценки (без OPENAI_API_KEY)",
    )

    p_one = sub.add_parser("process", help="Один видео- или аудиофайл (путь к файлу)")
    p_one.add_argument("video", type=Path, help="Файл видео или аудио")
    p_one.add_argument(
        "--skip-eval",
        action="store_true",
        help="Только транскрипт",
    )

    p_ev = sub.add_parser(
        "eval-only",
        help="Только оценка по готовым JSON в 02.Transcript (нужен OPENAI_API_KEY)",
    )
    p_ev.add_argument(
        "transcript",
        type=Path,
        nargs="?",
        help="Один файл транскрипта; если не указан — все *.json в 02.Transcript",
    )

    p_em = sub.add_parser(
        "emotion-only",
        help="Только SER (*.tone.json) по готовым транскриптам и исходным файлам в 01.Video",
    )
    p_em.add_argument(
        "transcript",
        type=Path,
        nargs="?",
        help="Один JSON транскрипта; если не указан — все основные *.json в 02.Transcript",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Веб-интерфейс (FastAPI): загрузка видео/аудио, транскрипты, оценки",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Хост (по умолчанию 127.0.0.1)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Порт (по умолчанию 8765)",
    )
    p_serve.add_argument(
        "--no-reload",
        action="store_true",
        help="Отключить автоперезагрузку при правках web/server.py (по умолчанию перезагрузка включена)",
    )

    p_user_add = sub.add_parser(
        "user-add",
        help="Создать или обновить локального пользователя в SQLite",
    )
    p_user_add.add_argument("username", help="Логин локального пользователя")
    p_user_add.add_argument("--password", required=True, help="Пароль")
    p_user_add.add_argument("--display-name", default="", help="Отображаемое имя")
    p_user_add.add_argument(
        "--role",
        choices=("user", "admin"),
        default="user",
        help="Роль пользователя",
    )

    p_user_list = sub.add_parser(
        "user-list",
        help="Список локальных пользователей",
    )

    p_user_set_pw = sub.add_parser(
        "user-set-password",
        help="Сменить пароль локального пользователя",
    )
    p_user_set_pw.add_argument("username", help="Логин локального пользователя")
    p_user_set_pw.add_argument("--password", required=True, help="Новый пароль")

    p_user_toggle = sub.add_parser(
        "user-set-active",
        help="Включить или отключить локального пользователя",
    )
    p_user_toggle.add_argument("username", help="Логин локального пользователя")
    p_user_toggle.add_argument(
        "--active",
        choices=("0", "1"),
        required=True,
        help="1 — включить, 0 — отключить",
    )

    args = parser.parse_args()

    if args.cmd == "process-all":
        db, crit = _cli_checklist_db()
        videos = list_videos(VIDEO_DIR)
        if not videos:
            print(f"Нет видео или аудио в {VIDEO_DIR}", file=sys.stderr)
            sys.exit(1)
        for v in videos:
            print(f"Обработка: {v.name} …")
            tr, ev, tone = process_one_video(
                v, skip_eval=args.skip_eval, db=db, criteria_path=crit
            )
            print(f"  транскрипт: {tr}")
            if tone:
                print(f"  тон (SER):  {tone}")
            if ev:
                print(f"  оценка:     {ev}")
        return

    if args.cmd == "process":
        db, crit = _cli_checklist_db()
        tr, ev, tone = process_one_video(
            args.video, skip_eval=args.skip_eval, db=db, criteria_path=crit
        )
        print(f"Транскрипт: {tr}")
        if tone:
            print(f"Тон (SER):  {tone}")
        if ev:
            print(f"Оценка:     {ev}")
        return

    if args.cmd == "eval-only":
        db, crit = _cli_checklist_db()
        if args.transcript:
            paths = [args.transcript.resolve()]
        else:
            paths = list_transcripts(TRANSCRIPT_DIR)
        if not paths:
            print(f"Нет JSON в {TRANSCRIPT_DIR}", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            print(f"Оценка по: {p.name} …")
            evp = evaluate_only_from_transcript(p, db=db, criteria_path=crit)
            print(f"  → {evp}")
        return

    if args.cmd == "emotion-only":
        if args.transcript:
            paths = [args.transcript.resolve()]
        else:
            paths = list_transcripts(TRANSCRIPT_DIR)
        if not paths:
            print(f"Нет JSON в {TRANSCRIPT_DIR}", file=sys.stderr)
            sys.exit(1)
        failed = False
        for p in paths:
            print(f"Эмоции (SER): {p.name} …")
            try:
                outp = emotion_only_from_transcript(p)
                print(f"  → {outp}")
            except (OSError, ValueError, FileNotFoundError) as e:
                print(f"  ошибка: {e}", file=sys.stderr)
                failed = True
        if failed:
            sys.exit(1)
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "web.server:app",
            host=args.host,
            port=args.port,
            reload=not args.no_reload,
        )
        return

    if args.cmd == "user-add":
        init_db()
        db = DB()
        username = normalize_local_username(args.username)
        if not username:
            print("Пустой username", file=sys.stderr)
            sys.exit(1)
        user = db.upsert_user(
            username,
            password_hash=hash_local_password(args.password),
            display_name=(args.display_name or "").strip() or None,
            role=args.role,
            is_active=True,
        )
        print(
            f"Сохранён пользователь {user['username']} "
            f"(role={user['role']}, active={user['is_active']})"
        )
        return

    if args.cmd == "user-list":
        init_db()
        db = DB()
        users = db.list_users()
        if not users:
            print("Локальных пользователей нет")
            return
        for user in users:
            display_name = user.get("display_name") or ""
            print(
                f"{user['username']}\trole={user['role']}\tactive={user['is_active']}\t{display_name}"
            )
        return

    if args.cmd == "user-set-password":
        init_db()
        db = DB()
        username = normalize_local_username(args.username)
        existing = db.get_user(username)
        if not existing:
            print(f"Пользователь не найден: {username}", file=sys.stderr)
            sys.exit(1)
        db.upsert_user(
            username,
            password_hash=hash_local_password(args.password),
            display_name=existing.get("display_name"),
            role=str(existing.get("role") or "user"),
            is_active=bool(existing.get("is_active")),
        )
        print(f"Пароль обновлён: {username}")
        return

    if args.cmd == "user-set-active":
        init_db()
        db = DB()
        username = normalize_local_username(args.username)
        if not db.set_user_active(username, args.active == "1"):
            print(f"Пользователь не найден: {username}", file=sys.stderr)
            sys.exit(1)
        print(f"Пользователь {username}: active={args.active}")
        return


if __name__ == "__main__":
    main()
