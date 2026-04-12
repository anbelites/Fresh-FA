"""Атомарная запись JSON и безопасная загрузка транскрипта."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: dict[str, Any], *, indent: int = 2) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.name + ".",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def try_load_transcript(path: Path) -> dict[str, Any] | None:
    """
    Загружает JSON транскрипта или возвращает None при отсутствии файла,
    битом JSON или неподходящей структуре (нет segments как списка).
    """
    path = path.resolve()
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    if "segments" not in data or not isinstance(data["segments"], list):
        return None
    return data
