"""Загрузка критериев оценки из YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.paths import CRITERIA_FILE


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    description: str


def load_criteria(path: Path | None = None) -> tuple[str, list[Criterion]]:
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
            )
        )
    return version, items


def criteria_to_prompt_block(criteria: list[Criterion]) -> str:
    lines: list[str] = []
    for c in criteria:
        lines.append(f"- **{c.id}** ({c.name}): {c.description}")
    return "\n".join(lines)
