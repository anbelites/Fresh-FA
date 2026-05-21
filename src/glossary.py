"""Shared glossary formatting for ASR and LLM prompts."""
from __future__ import annotations

from typing import Any

from src.database import DB
from src.glossary_seed import GLOSSARY_SEED


def _seed_entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(GLOSSARY_SEED):
        out.append(
            {
                "id": str(item.get("id") or ""),
                "category": str(item.get("category") or ""),
                "term": str(item.get("term") or ""),
                "variants": list(item.get("variants") or []),
                "definition": str(item.get("definition") or ""),
                "whisper_hint": str(item.get("whisper_hint") or ""),
                "llm_hint": str(item.get("llm_hint") or ""),
                "use_for_whisper": bool(item.get("use_for_whisper", True)),
                "use_for_llm": bool(item.get("use_for_llm", True)),
                "is_active": bool(item.get("is_active", True)),
                "sort_order": int(item.get("sort_order", idx)),
            }
        )
    return out


def load_active_glossary_entries() -> list[dict[str, Any]]:
    db: DB | None = None
    try:
        db = DB()
        rows = db.list_glossary_entries(include_inactive=False)
        return rows or [row for row in _seed_entries() if row.get("is_active")]
    except Exception:
        return [row for row in _seed_entries() if row.get("is_active")]
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _split_hint(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace("\n", ",").split(",") if part.strip()]


def _unique_terms(parts: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def format_glossary_for_whisper(entries: list[dict[str, Any]] | None = None) -> str:
    rows = entries if entries is not None else load_active_glossary_entries()
    priority_terms: list[str] = []
    terms: list[str] = []
    for row in rows:
        if not row.get("is_active", True) or not row.get("use_for_whisper", True):
            continue
        hint = str(row.get("whisper_hint") or "").strip()
        target = priority_terms if str(row.get("category") or "") == "priority_whisper_terms" else terms
        if hint:
            target.extend(_split_hint(hint))
        else:
            target.append(str(row.get("term") or "").strip())
            target.extend(str(v).strip() for v in (row.get("variants") or []) if str(v or "").strip())
    priority = ", ".join(_unique_terms([x for x in priority_terms if x]))
    compact = ", ".join(_unique_terms([x for x in terms if x])[:80])
    if not compact and not priority:
        return ""
    priority_sentence = (
        f"Критически важные фирменные термины: {priority}. "
        if priority
        else ""
    )
    return (
        "Это запись разговора в автосалоне Fresh про покупку, продажу, оценку, "
        "диагностику и обслуживание автомобилей. "
        f"{priority_sentence}"
        "В речи могут встречаться фирменные "
        f"и автомобильные термины. Правильно распознавай и сохраняй написание терминов: {compact}."
    )


def format_glossary_for_eval(entries: list[dict[str, Any]] | None = None) -> str:
    rows = entries if entries is not None else load_active_glossary_entries()
    lines: list[str] = []
    for row in rows:
        if not row.get("is_active", True) or not row.get("use_for_llm", True):
            continue
        term = str(row.get("term") or "").strip()
        if not term:
            continue
        variants = [str(v).strip() for v in (row.get("variants") or []) if str(v or "").strip()]
        definition = str(row.get("definition") or "").strip()
        llm_hint = str(row.get("llm_hint") or "").strip()
        parts = []
        if variants:
            parts.append(f"варианты: {', '.join(variants)}")
        if definition:
            parts.append(definition)
        if llm_hint and llm_hint != definition:
            parts.append(llm_hint)
        detail = "; ".join(parts) if parts else "термин глоссария Fresh"
        lines.append(f"- {term}: {detail}")
    if not lines:
        return ""
    return (
        "Глоссарий Fresh для трактовки сокращений и терминов. "
        "Если в транскрипте термин распознан с ошибкой, но контекст очевиден, "
        "учитывай нормализованный смысл из глоссария:\n"
        + "\n".join(lines)
    )
