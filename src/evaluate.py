"""Оценка транскрипта по критериям из YAML через OpenAI-совместимый Chat API."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.criteria_loader import Criterion, criteria_to_prompt_block, load_criteria
from src.paths import CRITERIA_FILE
from src.speech_emotion import speech_emotion_context_for_eval


def _parse_json_from_message(content: str) -> dict[str, Any]:
    """Достаёт JSON из ответа (иногда оборачивают в ```json)."""
    text = (content or "").strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
    return json.loads(text)


def _client() -> OpenAI:
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY", "")
    kw: dict[str, Any] = {"api_key": key or "missing"}
    if base:
        kw["base_url"] = base
    return OpenAI(**kw)


def _model() -> str:
    return os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini")


def _is_reasoner_model(name: str) -> bool:
    return "reasoner" in name.lower()


def build_eval_prompt(transcript_text: str, criteria: list[Criterion]) -> tuple[str, str]:
    system = (
        "Ты эксперт по оценке речи в деловом и сервисном общении. "
        "У каждой строки транскрипта могут быть пометки: «аудио:» (акустические признаки — F0σ, RMS, флаги тона), «SER:» (метка эмоции по аудиомодели). "
        "Также есть блоки «Акустика по спикерам» и «Эмоциональная окраска по аудио» в конце. Используй их как дополнительные сигналы к тексту. "
        "ВАЖНО про SER: модель обучена на актёрских эмоциях, поэтому «нейтральный» — это обычный разговорный тон, а НЕ безразличие. "
        "Для оценки вовлечённости и доброжелательности сотрудника больше опирайся на: "
        "(1) смысл реплик, (2) акустические признаки (expressive_pitch_variation = живая интонация, "
        "bright_timbral = яркий тембр, высокий F0σ = выразительность; monotone_pitch = монотонность), "
        "(3) SER-метку как вспомогательный сигнал (не единственный). "
        "Все критерии относятся только к **сотруднику** (менеджеру, консультанту), не к клиенту. "
        "Метки SPEAKER_01, SPEAKER_02 и т.д. — технические; порядок не гарантирует роль. "
        "Перед оценкой мысленно определи: какой говорящий — сотрудник (приветствие от компании, вопросы по товару/сделке, презентация), а какой — клиент (ответы, запросы «для себя»). "
        "Не приписывай репликам клиента оценку по критериям сотрудника; в evidence_segments для таких критериев указывай только интервалы [t0–t1], где в строке указан **сотрудник** (тот, кого ты определил как оцениваемого). "
        "Отвечай только валидным JSON по схеме из запроса. Комментарии на русском, кратко и по делу. "
        "Оценки — целые числа от 0 до 100 (чем выше, тем лучше соответствие критерию). "
        "Если по фрагменту нельзя судить (обрезано, нет реплик сотрудника), ставь null в score и объясни в comment. "
        "Для каждого критерия обязательно укажи evidence_segments: массив объектов "
        '{"start": <сек>, "end": <сек>} — отрезки из транскрипта, на которых основана оценка '
        "(можно несколько, если признак проявляется в разных местах; границы возьми из строк [t0–t1] транскрипта, не выдумывай таймкоды). "
        "Если score null или нечего привязать — []."
    )
    crit_block = criteria_to_prompt_block(criteria)
    ex_a = criteria[0].id if criteria else "criterion_a"
    ex_b = criteria[1].id if len(criteria) > 1 else "criterion_b"
    user = f"""Транскрипт диалога (с указанием говорящих по сегментам):

{transcript_text}

---

Сначала определи по смыслу реплик, кто из SPEAKER_XX — сотрудник (его оцениваем по чеклисту), а кто — клиент. Критерии ниже — только про сотрудника; evidence — только отрезки с речью сотрудника, если критерий про его действия.

Критерии для оценки:
{crit_block}

Верни JSON-объект: ровно по одному ключу на каждый id из списка выше. Значение — объект:
{{ "score": <число 0-100 или null>, "comment": "<строка>", "evidence_segments": [ {{"start": 12.5, "end": 18.4}}, ... ] }}

evidence_segments — только реальные интервалы из транскрипта (секунды); пустой массив если нечего привязать.

Пример структуры для двух первых критериев (повтори для всех id из списка):
{{ "{ex_a}": {{ "score": 85, "comment": "...", "evidence_segments": [{{"start": 1.2, "end": 6.0}}] }}, "{ex_b}": {{ "score": 70, "comment": "...", "evidence_segments": [] }} }}"""
    return system, user


def audio_tone_context_for_eval(transcript_json: dict[str, Any]) -> str:
    """Краткий блок акустики для промпта (агрегаты по спикерам)."""
    by_spk = transcript_json.get("audio_tone_by_speaker")
    if not isinstance(by_spk, dict) or not by_spk:
        return ""
    lines: list[str] = [
        "",
        "--- Акустика по спикерам (по записи: высота/громкость/тембр — ориентиры, не эмоции):",
    ]
    for spk, data in sorted(by_spk.items()):
        if not isinstance(data, dict):
            continue
        lines.append(
            f"  {spk}: средняя высота тона ≈ {data.get('mean_f0_hz')} Hz, "
            f"разброс F0 ≈ {data.get('f0_std_hz')} Hz, "
            f"энергия RMS ≈ {data.get('rms_energy_mean')}, "
            f"спектр. центроид ≈ {data.get('spectral_centroid_mean_hz')} Hz"
        )
    lines.append(
        "  (более высокий разброс F0 — выразительнее интонация; монотонность — малый разброс; "
        "центроид выше — «ярче» тембр.)"
    )
    return "\n".join(lines)


def _normalize_evidence_segments(raw: Any) -> list[dict[str, float]]:
    """Таймкоды-обоснования из ответа модели."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            t0 = float(item.get("start"))
            t1 = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if t1 < t0 or t0 < 0:
            continue
        out.append({"start": round(t0, 3), "end": round(t1, 3)})
    return out


def transcript_to_linear_text(
    transcript_json: dict[str, Any],
    tone_segments: list[dict[str, Any]] | None = None,
) -> str:
    from src.speech_emotion import ser_label_for_segment

    lines: list[str] = []
    _tone_segs = tone_segments or []
    for seg in transcript_json.get("segments", []):
        sp = seg.get("speaker", "?")
        role = seg.get("speaker_role")
        if role:
            sp = f"{sp}({role})"
        t0 = seg.get("start", 0)
        t1 = seg.get("end", 0)
        text = seg.get("text", "").strip()

        extras: list[str] = []

        dv = seg.get("delivery") or {}
        dv_flags = dv.get("flags") or []
        if dv_flags:
            extras.append(f"ASR: {', '.join(dv_flags)}")

        at = seg.get("audio_tone") or {}
        if isinstance(at, dict) and not at.get("skipped"):
            at_flags = at.get("flags") or []
            at_parts: list[str] = []
            if at_flags:
                at_parts.append(", ".join(at_flags))
            f0_std = at.get("f0_std_hz")
            if f0_std is not None:
                at_parts.append(f"F0σ={f0_std}Hz")
            rms = at.get("rms_energy_mean")
            if rms is not None:
                at_parts.append(f"RMS={rms}")
            if at_parts:
                extras.append(f"аудио: {'; '.join(at_parts)}")

        if _tone_segs:
            ser_label = ser_label_for_segment(float(t0), _tone_segs)
            if ser_label:
                extras.append(f"SER: {ser_label}")

        suffix = ""
        if extras:
            suffix = " | " + " | ".join(extras)
        lines.append(f"[{t0:.2f}–{t1:.2f}] {sp}: {text}{suffix}")
    return "\n".join(lines)


def evaluate_transcript(
    transcript_data: dict[str, Any],
    criteria_path: Path | None = None,
    *,
    transcript_path: Path | None = None,
) -> dict[str, Any]:
    cp = criteria_path or CRITERIA_FILE
    if not os.environ.get("OPENAI_API_KEY"):
        version, criteria = load_criteria(criteria_path)
        return {
            "schema_version": 2,
            "criteria_version": version,
            "criteria_file": cp.name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": None,
            "video_file": transcript_data.get("video_file"),
            "overall_average": None,
            "error": "OPENAI_API_KEY не задан — оценка пропущена. Задайте ключ в окружении.",
            "criteria": [
                {
                    "id": c.id,
                    "name": c.name,
                    "score": None,
                    "comment": "Оценка недоступна без API.",
                    "evidence_segments": [],
                }
                for c in criteria
            ],
        }

    version, criteria = load_criteria(criteria_path)

    tone_data: dict[str, Any] | None = None
    if transcript_path is not None:
        tp = Path(transcript_path)
        side = tp.parent / f"{tp.stem}.tone.json"
        if side.is_file():
            try:
                tone_data = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                tone_data = None

    tone_segments = (tone_data.get("segments") or []) if tone_data else []
    linear = transcript_to_linear_text(transcript_data, tone_segments=tone_segments)
    audio_ctx = audio_tone_context_for_eval(transcript_data)
    ser_ctx = speech_emotion_context_for_eval(tone_data) if tone_data else ""
    system, user = build_eval_prompt(linear + audio_ctx + ser_ctx, criteria)

    client = _client()
    model_name = _model()
    reasoner = _is_reasoner_model(model_name)

    create_kw: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if reasoner:
        # deepseek-reasoner: длинный CoT + JSON; DeepSeek API ограничивает max_tokens (часто max 65536).
        _raw = int(os.environ.get("OPENAI_EVAL_MAX_TOKENS", "65536"))
        create_kw["max_tokens"] = max(1, min(_raw, 65536))
    else:
        create_kw["temperature"] = float(os.environ.get("OPENAI_EVAL_TEMPERATURE", "0.2"))

    response = client.chat.completions.create(**create_kw)
    msg = response.choices[0].message
    raw = msg.content or "{}"
    reasoning_trace = getattr(msg, "reasoning_content", None)
    parse_err: str | None = None
    try:
        parsed = _parse_json_from_message(raw)
    except json.JSONDecodeError as e:
        parsed = {}
        parse_err = str(e)

    criteria_out: list[dict[str, Any]] = []
    for c in criteria:
        block = parsed.get(c.id)
        score: int | None = None
        comment = ""
        evidence_segments: list[dict[str, float]] = []
        if isinstance(block, dict):
            s = block.get("score")
            if s is None:
                score = None
            else:
                try:
                    score = max(0, min(100, int(s)))
                except (TypeError, ValueError):
                    score = None
            comment = str(block.get("comment", "")).strip()
            evidence_segments = _normalize_evidence_segments(block.get("evidence_segments"))
        criteria_out.append(
            {
                "id": c.id,
                "name": c.name,
                "score": score,
                "comment": comment,
                "evidence_segments": evidence_segments,
            }
        )

    numeric_scores = [x["score"] for x in criteria_out if x["score"] is not None]
    overall = (
        round(sum(numeric_scores) / len(numeric_scores), 1) if numeric_scores else None
    )

    out: dict[str, Any] = {
        "schema_version": 2,
        "criteria_version": version,
        "criteria_file": cp.name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "video_file": transcript_data.get("video_file"),
        "criteria": criteria_out,
        "overall_average": overall,
    }
    if reasoning_trace:
        out["reasoning_trace"] = reasoning_trace
    if parse_err:
        out["evaluation_parse_error"] = parse_err
        out["raw_model_content_preview"] = (raw[:2000] + "…") if len(raw) > 2000 else raw
    return out


def write_evaluation_json(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
