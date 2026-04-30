"""Склейка: медиафайл → транскрипт JSON → оценка JSON."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

# Ключи совпадают с web/static/app.js STAGE_LABELS и логом в терминале
_PIPELINE_STAGE_TEXT: dict[str, str] = {
    "extract_audio": "Аудио: подготовка WAV 16 kHz из медиафайла",
    "diarization": "Диаризация: кто когда говорит (pyannote, может занять несколько минут)",
    "diarization_skip": "Диаризация: HF_TOKEN не задан — позже грубое разделение по MFCC",
    "diarization_skip_windows": "Диаризация: на Windows без pyannote (MFCC); pyannote — только с PYANNOTE_ON_WINDOWS=1",
    "whisper_load": "Загрузка модели Whisper в память",
    "asr_whisper": "Распознавание речи (Whisper — обычно самый долгий этап)",
    "segments_build": "Сборка сегментов: спикеры, слова, скорость речи, тон по кускам",
    "transcribing": "Транскрибация (устаревший общий шаг)",
    "tone": "Эмоции по аудио (SER, HuBERT/Dusha по умолчанию)",
    "evaluating": "Оценка по чеклисту (ИИ, DeepSeek и т.д.)",
    "done": "Готово",
    "error": "Ошибка",
    "starting": "Запуск пайплайна",
    "resume": "Возобновление: пропуск уже готовых этапов",
}


def _pipeline_log(phase: str) -> None:
    text = _PIPELINE_STAGE_TEXT.get(phase, phase)
    print(f"[Fresh FA] {text}", flush=True)

from src.artifacts import delete_derived_artifacts_for_stem
from src.atomic_json import try_load_transcript
from src.evaluate import evaluate_transcript, write_evaluation_json
from src.audio_extract import extract_wav_16k_mono
from src.database import DB
from src.paths import DEFAULT_CHECKLIST_SLUG, TRANSCRIPT_DIR, VIDEO_DIR, evaluation_json_path
from src.speech_emotion import build_speech_emotion_sidecar, write_tone_json
from src.transcribe import transcribe_video_to_structure, write_transcript_json


def stem_for_outputs(video_path: Path) -> str:
    return video_path.stem


def _sync_eval_roles_into_transcript(
    transcript_data: dict[str, Any], eval_data: dict[str, Any]
) -> bool:
    """
    Роли из evaluation — основной источник истины для transcript/UI.
    Возвращает True, если transcript_data изменён.
    """
    changed = False
    raw_roles = eval_data.get("speaker_roles")
    roles = raw_roles if isinstance(raw_roles, dict) and raw_roles else None

    if roles is None:
        for key in (
            "speaker_roles",
            "speaker_roles_method",
            "speaker_roles_confidence",
            "speaker_roles_reasoning",
            "employee_speaker",
        ):
            if key in transcript_data:
                transcript_data.pop(key, None)
                changed = True
        for seg in transcript_data.get("segments") or []:
            if isinstance(seg, dict) and "speaker_role" in seg:
                seg.pop("speaker_role", None)
                changed = True
        return changed

    norm_roles = {
        str(spk).strip(): str(role).strip().upper()
        for spk, role in roles.items()
        if str(spk or "").strip() and str(role or "").strip().upper() in ("EMPLOYEE", "CLIENT")
    }
    if not norm_roles:
        return changed

    if transcript_data.get("speaker_roles") != norm_roles:
        transcript_data["speaker_roles"] = norm_roles
        changed = True

    if transcript_data.get("speaker_roles_method") != "eval_llm":
        transcript_data["speaker_roles_method"] = "eval_llm"
        changed = True

    confidence = eval_data.get("speaker_roles_confidence")
    if isinstance(confidence, dict):
        if transcript_data.get("speaker_roles_confidence") != confidence:
            transcript_data["speaker_roles_confidence"] = confidence
            changed = True
    elif "speaker_roles_confidence" in transcript_data:
        transcript_data.pop("speaker_roles_confidence", None)
        changed = True

    reasoning = eval_data.get("speaker_roles_reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        if transcript_data.get("speaker_roles_reasoning") != reasoning.strip():
            transcript_data["speaker_roles_reasoning"] = reasoning.strip()
            changed = True
    elif "speaker_roles_reasoning" in transcript_data:
        transcript_data.pop("speaker_roles_reasoning", None)
        changed = True

    employee = eval_data.get("employee_speaker")
    employee_norm = str(employee).strip() if employee else ""
    if employee_norm:
        if transcript_data.get("employee_speaker") != employee_norm:
            transcript_data["employee_speaker"] = employee_norm
            changed = True
    elif "employee_speaker" in transcript_data:
        transcript_data.pop("employee_speaker", None)
        changed = True

    for seg in transcript_data.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        role = norm_roles.get(str(seg.get("speaker", "")).strip())
        if role:
            if seg.get("speaker_role") != role:
                seg["speaker_role"] = role
                changed = True
        elif "speaker_role" in seg:
            seg.pop("speaker_role", None)
            changed = True

    return changed


def find_video_for_stem(stem: str, video_dir: Path | None = None) -> Path | None:
    """Ищет медиафайл с тем же stem, что у транскрипта."""
    base = video_dir or VIDEO_DIR
    for ext in (
        ".MOV",
        ".mov",
        ".mp4",
        ".MP4",
        ".mkv",
        ".m4v",
        ".m4a",
        ".mp3",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".wma",
        ".amr",
        ".avi",
        ".webm",
    ):
        p = base / f"{stem}{ext}"
        if p.is_file():
            return p
    for p in base.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def emotion_only_from_transcript(
    transcript_path: Path,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> Path:
    """
    SER по готовому JSON транскрипта: тот же медиафайл в 01.Video, сегменты из JSON.
    Пишет stem.tone.json и обновляет поле speech_emotion_sidecar в транскрипте.
    """
    import tempfile

    import numpy as np
    import soundfile as sf

    transcript_path = transcript_path.resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(str(transcript_path))
    if transcript_path.name.endswith(".tone.json"):
        raise ValueError("Укажите основной файл транскрипта (*.json), не *.tone.json")

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    stem = transcript_path.stem
    segments = data.get("segments") or []
    if not segments:
        raise ValueError("В транскрипте нет segments")

    video = find_video_for_stem(stem)
    if not video:
        raise FileNotFoundError(
            f"Не найден медиафайл для stem «{stem}» в {VIDEO_DIR}"
        )

    vf = data.get("video_file") or video.name

    if cancel_check:
        cancel_check()

    with tempfile.TemporaryDirectory(prefix="fa_tone_") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_wav_16k_mono(video, wav)
        if cancel_check:
            cancel_check()
        y_audio, sr_audio = sf.read(str(wav), always_2d=False, dtype="float32")
        if y_audio.ndim > 1:
            y_audio = np.mean(y_audio, axis=1)
        if sr_audio != 16000:
            raise ValueError(f"Ожидался WAV 16 kHz, получено {sr_audio} Hz")

    tone_data = build_speech_emotion_sidecar(y_audio, sr_audio, segments, vf)
    tone_path = TRANSCRIPT_DIR / f"{stem}.tone.json"
    write_tone_json(tone_data, tone_path)

    data["speech_emotion_sidecar"] = f"{stem}.tone.json"
    write_transcript_json(data, transcript_path)
    return tone_path


def process_one_video(
    video_path: Path,
    *,
    criteria_path: Path | None = None,
    db: DB | None = None,
    skip_eval: bool = False,
    on_progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    eval_stream_callback: Callable[[str], None] | None = None,
    expected_speaker_count: int | None = None,
) -> tuple[Path, Path | None, Path | None]:
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))

    print(f"[Fresh FA] Обработка: {video_path.name}", flush=True)

    def ping(phase: str) -> None:
        if cancel_check:
            cancel_check()
        _pipeline_log(phase)
        if on_progress:
            on_progress(phase)

    stem = stem_for_outputs(video_path)
    transcript_path = TRANSCRIPT_DIR / f"{stem}.json"
    crit = criteria_path or Path(DEFAULT_CHECKLIST_SLUG)
    eval_path = evaluation_json_path(stem, crit)
    tone_path = TRANSCRIPT_DIR / f"{stem}.tone.json"

    data, tone_data = transcribe_video_to_structure(
        video_path,
        on_progress=ping,
        cancel_check=cancel_check,
        expected_speaker_count=expected_speaker_count,
    )
    write_transcript_json(data, transcript_path)
    if tone_data is not None:
        write_tone_json(tone_data, tone_path)
    else:
        tone_path = None

    if skip_eval:
        ping("done")
        return transcript_path, None, tone_path

    ping("evaluating")
    ev = evaluate_transcript(
        data,
        criteria_path=crit,
        transcript_path=transcript_path,
        cancel_check=cancel_check,
        db=db,
        stream_callback=eval_stream_callback,
    )
    if _sync_eval_roles_into_transcript(data, ev):
        write_transcript_json(data, transcript_path)
    write_evaluation_json(ev, eval_path)
    ping("done")
    return transcript_path, eval_path, tone_path


def process_one_video_resume(
    video_path: Path,
    *,
    criteria_path: Path | None = None,
    db: DB | None = None,
    on_progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    eval_stream_callback: Callable[[str], None] | None = None,
    expected_speaker_count: int | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """
    После остановки пайплайна: не повторять уже завершённые шаги.
    Нет транскрипта — полный прогон; есть транскрипт, нет тона — только SER;
    есть транскрипт и тон — только оценка (если для активного чеклиста ещё нет файла оценки).
    Битый или неподходящий JSON транскрипта — удаление производных файлов и полный прогон с нуля.
    """
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))

    def ping(phase: str) -> None:
        if cancel_check:
            cancel_check()
        _pipeline_log(phase)
        if on_progress:
            on_progress(phase)

    stem = stem_for_outputs(video_path)
    transcript_path = TRANSCRIPT_DIR / f"{stem}.json"
    tone_path = TRANSCRIPT_DIR / f"{stem}.tone.json"
    crit = criteria_path or Path(DEFAULT_CHECKLIST_SLUG)
    eval_path = evaluation_json_path(stem, crit)

    if not transcript_path.is_file():
        print(f"[Fresh FA] Возобновление: нет транскрипта — полный пайплайн", flush=True)
        return process_one_video(
            video_path,
            criteria_path=criteria_path,
            db=db,
            on_progress=on_progress,
            cancel_check=cancel_check,
            eval_stream_callback=eval_stream_callback,
            expected_speaker_count=expected_speaker_count,
        )

    data = try_load_transcript(transcript_path)
    if data is None:
        print(
            "[Fresh FA] Возобновление: транскрипт повреждён или неподходит — "
            "сброс производных и полный пайплайн",
            flush=True,
        )
        delete_derived_artifacts_for_stem(stem)
        return process_one_video(
            video_path,
            criteria_path=criteria_path,
            db=db,
            on_progress=on_progress,
            cancel_check=cancel_check,
            eval_stream_callback=eval_stream_callback,
            expected_speaker_count=expected_speaker_count,
        )

    ping("resume")

    if not tone_path.is_file():
        ping("tone")
        emotion_only_from_transcript(transcript_path, cancel_check=cancel_check)
        data = try_load_transcript(transcript_path)
        if data is None:
            raise RuntimeError(
                "Транскрипт недоступен после SER — удалите повреждённый файл вручную или запустите «с начала»."
            )

    if eval_path.is_file():
        ping("done")
        return transcript_path, eval_path, tone_path if tone_path.is_file() else None

    ping("evaluating")
    ev = evaluate_transcript(
        data,
        criteria_path=crit,
        transcript_path=transcript_path,
        cancel_check=cancel_check,
        db=db,
        stream_callback=eval_stream_callback,
    )
    if _sync_eval_roles_into_transcript(data, ev):
        write_transcript_json(data, transcript_path)
    write_evaluation_json(ev, eval_path)
    ping("done")
    return transcript_path, eval_path, tone_path if tone_path.is_file() else None


def evaluate_only_from_transcript(
    transcript_path: Path,
    *,
    criteria_path: Path | None = None,
    db: DB | None = None,
    cancel_check: Callable[[], None] | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> Path:
    """Только LLM-оценка по готовому JSON транскрипта (без повторного распознавания)."""
    transcript_path = transcript_path.resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(str(transcript_path))
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    stem = transcript_path.stem
    crit = criteria_path or Path(DEFAULT_CHECKLIST_SLUG)
    eval_path = evaluation_json_path(stem, crit)
    ev = evaluate_transcript(
        data,
        criteria_path=crit,
        transcript_path=transcript_path,
        cancel_check=cancel_check,
        db=db,
        stream_callback=stream_callback,
    )
    if _sync_eval_roles_into_transcript(data, ev):
        write_transcript_json(data, transcript_path)
    write_evaluation_json(ev, eval_path)
    return eval_path


def list_transcripts(transcript_dir: Path | None = None) -> list[Path]:
    base = transcript_dir or TRANSCRIPT_DIR
    return sorted(
        p
        for p in base.glob("*.json")
        if p.is_file() and not p.name.endswith(".tone.json")
    )


def list_videos(video_dir: Path) -> list[Path]:
    exts = {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v",
        ".m4a",
        ".mp3",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".wma",
        ".amr",
    }
    out: list[Path] = []
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out
