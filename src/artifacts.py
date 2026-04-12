"""Удаление производных файлов пайплайна по stem (транскрипт, тон, оценки)."""
from __future__ import annotations

from src.paths import EVALUATION_DIR, TRANSCRIPT_DIR


def delete_derived_artifacts_for_stem(stem: str) -> None:
    """Транскрипт, тон и оценки — без видео и без meta (как «с начала» / битый resume)."""
    tr = TRANSCRIPT_DIR / f"{stem}.json"
    if tr.is_file():
        tr.unlink()
    tone = TRANSCRIPT_DIR / f"{stem}.tone.json"
    if tone.is_file():
        tone.unlink()
    ev_legacy = EVALUATION_DIR / f"{stem}.json"
    if ev_legacy.is_file():
        ev_legacy.unlink()
    for p in list(EVALUATION_DIR.glob(f"{stem}__*.eval.json")):
        if p.is_file():
            p.unlink()
    for p in list(EVALUATION_DIR.glob(f"{stem}__*.human.json")):
        if p.is_file():
            p.unlink()
