"""Shared helpers for weighted yes/no checklist evaluations."""
from __future__ import annotations

from typing import Any


def normalize_weight(raw: Any) -> int:
    try:
        return max(1, int(raw or 1))
    except (TypeError, ValueError):
        return 1


def normalize_passed(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "да", "done"}:
        return True
    if text in {"0", "false", "no", "n", "нет"}:
        return False
    return None


def parse_legacy_score(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return max(0, min(100, int(raw)))
    except (TypeError, ValueError):
        return None


def awarded_weight(weight: int, passed: bool | None) -> int:
    return normalize_weight(weight) if passed is True else 0


def compute_eval_totals(criteria_rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_score = sum(normalize_weight(row.get("weight", 1)) for row in criteria_rows)
    earned_score = sum(
        awarded_weight(normalize_weight(row.get("weight", 1)), normalize_passed(row.get("passed")))
        for row in criteria_rows
    )
    percent = round((earned_score / max_score) * 100, 1) if max_score > 0 else None
    return {
        "earned_score": earned_score,
        "max_score": max_score,
        "overall_average": percent,
    }


def criteria_definitions_from_payload(
    payload: dict[str, Any] | None,
    fallback_criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if payload and isinstance(payload.get("criteria_snapshot"), list):
        rows = []
        for row in payload.get("criteria_snapshot") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id", "")).strip()
            if not cid:
                continue
            rows.append(
                {
                    "id": cid,
                    "name": str(row.get("name", cid)).strip() or cid,
                    "description": str(row.get("description", "") or "").strip(),
                    "weight": normalize_weight(row.get("weight", 1)),
                }
            )
        if rows:
            return rows
    return fallback_criteria


def normalize_eval_criteria(
    raw_criteria: Any,
    checklist_criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw_criteria, list):
        for row in raw_criteria:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id", "")).strip()
            if cid:
                source_by_id[cid] = row
    elif isinstance(raw_criteria, dict):
        for cid_raw, row in raw_criteria.items():
            cid = str(cid_raw or "").strip()
            if cid and isinstance(row, dict):
                source_by_id[cid] = row

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for definition in checklist_criteria:
        cid = str(definition.get("id", "")).strip()
        if not cid:
            continue
        seen.add(cid)
        row = source_by_id.get(cid, {})
        legacy_score = parse_legacy_score(row.get("score"))
        passed = normalize_passed(row.get("passed"))
        if passed is None and legacy_score is not None:
            passed = legacy_score >= 50
        weight = normalize_weight(definition.get("weight", row.get("weight", 1)))
        normalized.append(
            {
                "id": cid,
                "name": str(definition.get("name", row.get("name", cid))).strip() or cid,
                "description": str(definition.get("description", row.get("description", "")) or "").strip(),
                "weight": weight,
                "passed": passed,
                "comment": str(row.get("comment", "") or "").strip(),
                "evidence_segments": row.get("evidence_segments") if isinstance(row.get("evidence_segments"), list) else [],
                "awarded_weight": awarded_weight(weight, passed),
                **({"legacy_score": legacy_score} if legacy_score is not None else {}),
            }
        )

    for cid, row in source_by_id.items():
        if cid in seen:
            continue
        legacy_score = parse_legacy_score(row.get("score"))
        passed = normalize_passed(row.get("passed"))
        if passed is None and legacy_score is not None:
            passed = legacy_score >= 50
        weight = normalize_weight(row.get("weight", 1))
        normalized.append(
            {
                "id": cid,
                "name": str(row.get("name", cid)).strip() or cid,
                "description": str(row.get("description", "") or "").strip(),
                "weight": weight,
                "passed": passed,
                "comment": str(row.get("comment", "") or "").strip(),
                "evidence_segments": row.get("evidence_segments") if isinstance(row.get("evidence_segments"), list) else [],
                "awarded_weight": awarded_weight(weight, passed),
                **({"legacy_score": legacy_score} if legacy_score is not None else {}),
            }
        )

    return normalized


def normalize_loaded_evaluation(
    payload: dict[str, Any] | None,
    checklist_criteria: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not payload:
        return None
    out = dict(payload)
    criteria_defs = criteria_definitions_from_payload(out, checklist_criteria)
    criteria = normalize_eval_criteria(out.get("criteria"), criteria_defs)
    out["criteria"] = criteria
    out["criteria_snapshot"] = criteria_defs
    out.update(compute_eval_totals(criteria))
    try:
        schema_version = int(out.get("schema_version", 0) or 0)
    except (TypeError, ValueError):
        schema_version = 0
    out["schema_version"] = max(3, schema_version)
    return out
