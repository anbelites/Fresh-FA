"""Загрузка критериев оценки из YAML или из БД."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.paths import CRITERIA_FILE

if TYPE_CHECKING:
    from src.database import DB


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    description: str
    weight: int = 1


def _normalize_weight(raw: object) -> int:
    try:
        return max(1, int(raw or 1))
    except (TypeError, ValueError):
        return 1


def load_criteria(path: Path | None = None) -> tuple[str, list[Criterion]]:
    """Чтение YAML с диска (резерв для миграции и утилит)."""
    p = path or CRITERIA_FILE
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    version = str(raw.get("version", "1"))
    items: list[Criterion] = []
    for row in raw.get("criteria", []):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id", "")).strip()
        if not cid:
            continue
        items.append(
            Criterion(
                id=cid,
                name=str(row.get("name", cid)).strip(),
                description=str(row.get("description", "")).strip(),
                weight=_normalize_weight(row.get("weight", 1)),
            )
        )
    return version, items


def load_criteria_from_db(db: DB, slug: str) -> tuple[str, list[Criterion]]:
    """Критерии чеклиста по slug (короткое имя в БД, напр. criteria)."""
    data = db.get_checklist_content(slug)
    if not data:
        raise FileNotFoundError(f"Чеклист не найден в БД: {slug}")
    version = str(data.get("version", "1"))
    items: list[Criterion] = []
    for row in data.get("criteria") or []:
        cid = str(row.get("id", "")).strip()
        if not cid:
            continue
        items.append(
            Criterion(
                id=cid,
                name=str(row.get("name", cid)).strip(),
                description=str(row.get("description", "") or "").strip(),
                weight=_normalize_weight(row.get("weight", 1)),
            )
        )
    if not items:
        raise ValueError(f"В чеклисте «{slug}» нет ни одного критерия")
    return version, items


def criteria_to_prompt_block(criteria: list[Criterion]) -> str:
    lines: list[str] = []
    for c in criteria:
        lines.append(f"- **{c.id}** ({c.name}, вес {c.weight}): {c.description}")
    return "\n".join(lines)
