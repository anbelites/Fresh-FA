#!/usr/bin/env python3
"""
Пайплайн: 01.Video → 02.Transcript (JSON) → 03.Evaluation (JSON).

Переменные окружения (опционально):
  HF_TOKEN — токен Hugging Face (диаризация, загрузка моделей с Hub); в .env подхватывается с приоритетом над пустым env (см. load_dotenv override в run.py);
    примите условия модели pyannote/speaker-diarization-3.1 на huggingface.co
  PYANNOTE_ON_WINDOWS=1 — на Windows pyannote по умолчанию отключён (torchcodec); включите только если настроили FFmpeg/torchcodec
  SKIP_PYANNOTE=1 — не вызывать pyannote (MFCC-диаризация), на любой ОС
  PYANNOTE_NUM_SPEAKERS — если число говорящих известно (например 3: ведущий + два в диалоге), точнее кластеризация
  PYANNOTE_LABEL_SMOOTH_RADIUS — сглаживание меток pyannote по потоку слов (по умолчанию 3; 0 = выкл.)
  PYANNOTE_SEGMENT_SMOOTH_RADIUS — доп. сглаживание по строкам транскрипта с pyannote (по умолчанию 0; 1 — осторожно)
  MFCC_SEGMENT_SPEAKER_SMOOTH_RADIUS — сглаживание по строкам без HF (по умолчанию 2; 0 = выкл.)
  PYANNOTE_MIN_SPEAKERS / PYANNOTE_MAX_SPEAKERS — границы числа спикеров
  PYANNOTE_MAX_SPEAKERS_DEFAULT — верхняя граница по умолчанию (10), если PYANNOTE_MAX_SPEAKERS не задан; 0 = не задавать
  PYANNOTE_EXCLUSIVE=1 — брать разметку без перекрывающихся реплик (иногда лучше для стыка с ASR)
  MFCC_DIAR_MAX_SPEAKERS — без HF_TOKEN: до скольких кластеров KMeans по сегментам Whisper (по умолчанию 8)
  MFCC_WORD_LEVEL=1 — включить MFCC по каждому слову (по умолчанию выкл. — даёт шум и дробление строк)
  MFCC_WORD_SMOOTH_RADIUS — сглаживание меток по словам (по умолчанию 4), только при MFCC_WORD_LEVEL=1
  TRANSCRIPT_PAUSE_SPLIT_SEC — пауза (сек) для дополнительного разреза строк; по умолчанию 0 (только смена спикера)
  OPENAI_API_KEY — ключ для оценки по критериям из config/criteria.yaml
  WHISPER_MODEL — по умолчанию small (быстрее на CPU); large-v3 точнее
  WHISPER_DEVICE — cpu или cuda
  WHISPER_MODEL — medium (дефолт в коде) / large-v3 (в .env — максимальное качество, дольше)
  WHISPER_COMPUTE_TYPE — int8 на CPU; на CUDA можно float16
  SPEECH_FAST_WPS, SPEECH_FAST_SPS, SPEECH_LOW_LOGPROB, SPEECH_LOW_WORD_PROB — пороги «тараторство/уверенность ASR» в транскрипте
  SKIP_AUDIO_TONE=1 — отключить акустику тона (F0, RMS, спектр) по WAV
  SKIP_SPEECH_EMOTION=1 — не писать SER-файл *.tone.json рядом с транскриптом
  SPEECH_EMOTION_MODEL — по умолчанию xbgoose/hubert-large…-dusha (русский Dusha); Aniemore — RESD 7 классов; superb — IEMOCAP
  SPEECH_EMOTION_TRUST_REMOTE_CODE — 0/1 (для Aniemore обычно 1; xbgoose/nikatonika — 0)
  OPENAI_BASE_URL — например https://api.deepseek.com/v1 для DeepSeek
  OPENAI_EVAL_MODEL — deepseek-reasoner (цепочка рассуждений) или deepseek-chat
  OPENAI_EVAL_MAX_TOKENS — для reasoner, лимит выхода JSON (по умолчанию 65536; у DeepSeek верхняя граница 65536)

  Вход через Active Directory (LDAP), опционально:
  AD_AUTH_ENABLED=1 — включить проверку логина для веб-интерфейса и /api (кроме /api/health и /api/auth/*)
  SESSION_SECRET — обязательна длинная случайная строка (подпись cookie-сессии)
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

  python run.py serve [--host 127.0.0.1] [--port 8765] — веб-интерфейс (загрузка видео, списки, оценки)

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

from src.paths import TRANSCRIPT_DIR, VIDEO_DIR
from src.pipeline import (
    emotion_only_from_transcript,
    evaluate_only_from_transcript,
    list_transcripts,
    list_videos,
    process_one_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Транскрипция и оценка видео")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("process-all", help="Обработать все видео в 01.Video")
    p_all.add_argument(
        "--skip-eval",
        action="store_true",
        help="Только транскрипт, без оценки (без OPENAI_API_KEY)",
    )

    p_one = sub.add_parser("process", help="Одно видео (путь к файлу)")
    p_one.add_argument("video", type=Path, help="Файл видео")
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
        help="Только SER (*.tone.json) по готовым транскриптам и видео в 01.Video",
    )
    p_em.add_argument(
        "transcript",
        type=Path,
        nargs="?",
        help="Один JSON транскрипта; если не указан — все основные *.json в 02.Transcript",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Веб-интерфейс (FastAPI): загрузка видео, транскрипты, оценки",
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

    args = parser.parse_args()

    if args.cmd == "process-all":
        videos = list_videos(VIDEO_DIR)
        if not videos:
            print(f"Нет видео в {VIDEO_DIR}", file=sys.stderr)
            sys.exit(1)
        for v in videos:
            print(f"Обработка: {v.name} …")
            tr, ev, tone = process_one_video(v, skip_eval=args.skip_eval)
            print(f"  транскрипт: {tr}")
            if tone:
                print(f"  тон (SER):  {tone}")
            if ev:
                print(f"  оценка:     {ev}")
        return

    if args.cmd == "process":
        tr, ev, tone = process_one_video(args.video, skip_eval=args.skip_eval)
        print(f"Транскрипт: {tr}")
        if tone:
            print(f"Тон (SER):  {tone}")
        if ev:
            print(f"Оценка:     {ev}")
        return

    if args.cmd == "eval-only":
        if args.transcript:
            paths = [args.transcript.resolve()]
        else:
            paths = list_transcripts(TRANSCRIPT_DIR)
        if not paths:
            print(f"Нет JSON в {TRANSCRIPT_DIR}", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            print(f"Оценка по: {p.name} …")
            evp = evaluate_only_from_transcript(p)
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


if __name__ == "__main__":
    main()
