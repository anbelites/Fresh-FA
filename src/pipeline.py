"""Склейка: видео → транскрипт JSON → оценка JSON."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

# Ключи совпадают с web/static/app.js STAGE_LABELS и логом в терминале
_PIPELINE_STAGE_TEXT: dict[str, str] = {
    "extract_audio": "Аудио: извлечение WAV 16 kHz из видео",
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
}


def _pipeline_log(phase: str) -> None:
    text = _PIPELINE_STAGE_TEXT.get(phase, phase)
    print(f"[Fresh FA] {text}", flush=True)

from src.evaluate import evaluate_transcript, write_evaluation_json
from src.audio_extract import extract_wav_16k_mono
from src.paths import CRITERIA_FILE, EVALUATION_DIR, TRANSCRIPT_DIR, VIDEO_DIR, evaluation_json_path
from src.speech_emotion import build_speech_emotion_sidecar, write_tone_json
from src.transcribe import transcribe_video_to_structure, write_transcript_json


def stem_for_outputs(video_path: Path) -> str:
    return video_path.stem


def find_video_for_stem(stem: str, video_dir: Path | None = None) -> Path | None:
    """Ищет файл видео с тем же stem, что у транскрипта."""
    base = video_dir or VIDEO_DIR
    for ext in (
        ".MOV",
        ".mov",
        ".mp4",
        ".MP4",
        ".mkv",
        ".m4v",
        ".m4a",
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


def emotion_only_from_transcript(transcript_path: Path) -> Path:
    """
    SER по готовому JSON транскрипта: то же видео в 01.Video, сегменты из JSON.
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
            f"Не найдено видео для stem «{stem}» в {VIDEO_DIR}"
        )

    vf = data.get("video_file") or video.name

    with tempfile.TemporaryDirectory(prefix="fa_tone_") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_wav_16k_mono(video, wav)
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
    skip_eval: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path | None, Path | None]:
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))

    print(f"[Fresh FA] Обработка: {video_path.name}", flush=True)

    def ping(phase: str) -> None:
        _pipeline_log(phase)
        if on_progress:
            on_progress(phase)

    stem = stem_for_outputs(video_path)
    transcript_path = TRANSCRIPT_DIR / f"{stem}.json"
    crit = criteria_path or CRITERIA_FILE
    eval_path = evaluation_json_path(stem, crit)
    tone_path = TRANSCRIPT_DIR / f"{stem}.tone.json"

    data, tone_data = transcribe_video_to_structure(video_path, on_progress=ping)
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
    )
    write_evaluation_json(ev, eval_path)
    ping("done")
    return transcript_path, eval_path, tone_path


def evaluate_only_from_transcript(
    transcript_path: Path,
    *,
    criteria_path: Path | None = None,
) -> Path:
    """Только LLM-оценка по готовому JSON транскрипта (без повторного распознавания)."""
    transcript_path = transcript_path.resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(str(transcript_path))
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    stem = transcript_path.stem
    crit = criteria_path or CRITERIA_FILE
    eval_path = evaluation_json_path(stem, crit)
    ev = evaluate_transcript(
        data,
        criteria_path=crit,
        transcript_path=transcript_path,
    )
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
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".m4a"}
    out: list[Path] = []
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out
