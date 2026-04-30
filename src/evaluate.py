"""Оценка транскрипта по критериям из YAML через OpenAI-совместимый Chat API."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.criteria_loader import Criterion, criteria_to_prompt_block, load_criteria, load_criteria_from_db
from src.database import DB
from src.eval_schema import awarded_weight, compute_eval_totals, normalize_passed, parse_legacy_score
from src.paths import DEFAULT_CHECKLIST_SLUG
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
    return os.environ.get("OPENAI_EVAL_MODEL", "deepseek-v4-pro")


def _is_deepseek_v4_model(name: str) -> bool:
    return str(name or "").strip().lower().startswith("deepseek-v4-")


def _is_reasoner_model(name: str) -> bool:
    low = str(name or "").strip().lower()
    return "reasoner" in low or _is_deepseek_v4_model(low)


def _delta_reasoning_and_content(delta: Any) -> tuple[str | None, str | None]:
    """Извлекает фрагменты из delta (OpenAI SDK / совместимые провайдеры)."""
    if delta is None:
        return None, None
    rc = getattr(delta, "reasoning_content", None)
    ct = getattr(delta, "content", None)
    if rc is None and ct is None and hasattr(delta, "model_dump"):
        try:
            d = delta.model_dump()
            rc = d.get("reasoning_content")
            ct = d.get("content")
        except Exception:
            pass
    return (
        rc if isinstance(rc, str) else None,
        ct if isinstance(ct, str) else None,
    )


def _message_reasoning_and_content(msg: Any) -> tuple[str | None, str]:
    rc = getattr(msg, "reasoning_content", None)
    if rc is None and hasattr(msg, "model_dump"):
        try:
            d = msg.model_dump()
            r2 = d.get("reasoning_content")
            if isinstance(r2, str):
                rc = r2
        except Exception:
            pass
    raw = getattr(msg, "content", None) or "{}"
    if not isinstance(raw, str):
        raw = "{}"
    return (rc if isinstance(rc, str) else None), raw


def _stream_completion_chunks(
    client: OpenAI,
    create_kw: dict[str, Any],
    *,
    cancel_check: Callable[[], None] | None,
    stream_callback: Callable[[str], None],
) -> tuple[str, str | None]:
    """Стриминг чата; возвращает (content, reasoning_trace или None). response_format убирается — не все провайдеры совместимы со stream."""
    kw = dict(create_kw)
    kw["stream"] = True
    kw.pop("response_format", None)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    stream = client.chat.completions.create(**kw)
    for chunk in stream:
        if cancel_check:
            cancel_check()
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        rc, ct = _delta_reasoning_and_content(delta)
        if rc:
            reasoning_parts.append(rc)
            stream_callback("".join(reasoning_parts))
        if ct:
            content_parts.append(ct)
            if not reasoning_parts:
                stream_callback("Текст ответа модели:\n" + "".join(content_parts))
    raw = "".join(content_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else None
    disp = reasoning if reasoning else raw
    if disp:
        stream_callback(disp)
    return raw, reasoning


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
        "Перед оценкой явно определи: какой говорящий — сотрудник (приветствие от компании, вопросы по товару/сделке, презентация), а какой — клиент (ответы, запросы «для себя»). "
        "Не приписывай репликам клиента оценку по критериям сотрудника; в evidence_segments для таких критериев указывай только интервалы [t0–t1], где в строке указан **сотрудник** (тот, кого ты определил как оцениваемого). "
        "Отвечай только валидным JSON по схеме из запроса. Комментарии на русском, кратко и по делу. "
        "Для каждого критерия верни только passed=true/false/null и краткий comment. "
        "Вес критерия указан в описании, но ты не вычисляешь итоговые баллы. "
        "Если по фрагменту нельзя уверенно судить (обрезано, нет реплик сотрудника, нет подтверждения действия), ставь false в passed и прямо пиши в comment, что информации недостаточно. "
        "Если критерий включает визуальную составляющую видео (например улыбка, визуальный осмотр, показ отчёта, демонстрация состояния автомобиля, жесты, контакт глаз или другой визуально наблюдаемый элемент), "
        "сначала попробуй восстановить её по речевым маркерам в разговоре: словам сотрудника и клиента, явным упоминаниям показа, осмотра, демонстрации, реакции собеседника и другим текстовым индикаторам. "
        "Если по разговору визуальную часть надёжно подтвердить нельзя, не придумывай её: ставь false в passed и явно пиши в comment, что визуальную составляющую по разговору подтвердить не удалось. "
        "Для каждого критерия обязательно укажи evidence_segments: массив объектов "
        '{"start": <сек>, "end": <сек>} — отрезки из транскрипта, на которых основана оценка '
        "(можно несколько, если признак проявляется в разных местах; границы возьми из строк [t0–t1] транскрипта, не выдумывай таймкоды). "
        "Если passed null или нечего привязать — []."
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

Верни JSON-объект с полями:
- "speaker_roles": объект вида {{"SPEAKER_01": "EMPLOYEE", "SPEAKER_02": "CLIENT"}}
- "employee_speaker": строка с id сотрудника, например "SPEAKER_02"
- "speaker_roles_confidence": объект уверенности по каждому спикеру в диапазоне 0..1
- "speaker_roles_reasoning": краткое объяснение, почему именно этот спикер выбран как сотрудник
- "criteria": объект: ровно по одному ключу на каждый id из списка выше. Значение — объект:
  {{ "passed": <true|false|null>, "comment": "<строка>", "evidence_segments": [ {{"start": 12.5, "end": 18.4}}, ... ] }}

evidence_segments — только реальные интервалы из транскрипта (секунды); пустой массив если нечего привязать.

Пример структуры для двух первых критериев (повтори для всех id из списка):
{{ "speaker_roles": {{"SPEAKER_01": "CLIENT", "SPEAKER_02": "EMPLOYEE"}}, "employee_speaker": "SPEAKER_02", "speaker_roles_confidence": {{"SPEAKER_01": 0.91, "SPEAKER_02": 0.94}}, "speaker_roles_reasoning": "SPEAKER_02 приветствует от имени компании, уточняет потребности и ведет сделку.", "criteria": {{ "{ex_a}": {{ "passed": true, "comment": "...", "evidence_segments": [{{"start": 1.2, "end": 6.0}}] }}, "{ex_b}": {{ "passed": false, "comment": "...", "evidence_segments": [] }} }} }}"""
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


def _normalize_role_value(raw: Any) -> str | None:
    v = str(raw or "").strip().upper()
    if v in ("EMPLOYEE", "CLIENT"):
        return v
    return None


def _normalize_speaker_roles(
    raw: Any, allowed_speakers: set[str]
) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    for spk, role in raw.items():
        spk_s = str(spk or "").strip()
        if not spk_s or (allowed_speakers and spk_s not in allowed_speakers):
            continue
        role_s = _normalize_role_value(role)
        if role_s:
            out[spk_s] = role_s
    return out or None


def _normalize_employee_speaker(
    raw: Any, speaker_roles: dict[str, str] | None, allowed_speakers: set[str]
) -> str | None:
    v = str(raw or "").strip()
    if v and (not allowed_speakers or v in allowed_speakers):
        return v
    if speaker_roles:
        for spk, role in speaker_roles.items():
            if role == "EMPLOYEE":
                return spk
    return None


def _normalize_role_confidence(
    raw: Any, allowed_speakers: set[str]
) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for spk, score in raw.items():
        spk_s = str(spk or "").strip()
        if not spk_s or (allowed_speakers and spk_s not in allowed_speakers):
            continue
        try:
            out[spk_s] = round(max(0.0, min(1.0, float(score))), 4)
        except (TypeError, ValueError):
            continue
    return out or None


def transcript_to_linear_text(
    transcript_json: dict[str, Any],
    tone_segments: list[dict[str, Any]] | None = None,
) -> str:
    from src.speech_emotion import ser_label_for_segment

    lines: list[str] = []
    _tone_segs = tone_segments or []
    for seg in transcript_json.get("segments", []):
        sp = seg.get("speaker", "?")
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
    cancel_check: Callable[[], None] | None = None,
    db: DB | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    cp = criteria_path or Path(DEFAULT_CHECKLIST_SLUG)

    def _load_crit() -> tuple[str, list[Criterion]]:
        if db is not None:
            return load_criteria_from_db(db, cp.name)
        if not cp.is_file():
            raise FileNotFoundError(
                f"Чеклист не найден: {cp} (ожидается запись в БД или YAML в config/)"
            )
        return load_criteria(cp)

    if cancel_check:
        cancel_check()
    if not os.environ.get("OPENAI_API_KEY"):
        version, criteria = _load_crit()
        criteria_out = [
            {
                "id": c.id,
                "name": c.name,
                "weight": c.weight,
                "passed": False,
                "comment": "Оценка недоступна без API.",
                "evidence_segments": [],
                "awarded_weight": 0,
            }
            for c in criteria
        ]
        return {
            "schema_version": 3,
            "criteria_version": version,
            "criteria_file": cp.name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": None,
            "video_file": transcript_data.get("video_file"),
            "speaker_roles": None,
            "employee_speaker": None,
            "error": "OPENAI_API_KEY не задан — оценка пропущена. Задайте ключ в окружении.",
            "criteria": criteria_out,
            "criteria_snapshot": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "weight": c.weight,
                }
                for c in criteria
            ],
            **compute_eval_totals(criteria_out),
        }

    version, criteria = _load_crit()

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
        # DeepSeek thinking models: длинный CoT + JSON; держим совместимый лимит выхода.
        _raw = int(os.environ.get("OPENAI_EVAL_MAX_TOKENS", "65536"))
        create_kw["max_tokens"] = max(1, min(_raw, 65536))
        if _is_deepseek_v4_model(model_name):
            effort = str(os.environ.get("OPENAI_EVAL_REASONING_EFFORT", "high") or "high").strip().lower()
            if effort not in {"high", "max"}:
                effort = "high"
            create_kw["reasoning_effort"] = effort
            create_kw["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        create_kw["temperature"] = float(os.environ.get("OPENAI_EVAL_TEMPERATURE", "0.2"))

    if cancel_check:
        cancel_check()

    raw: str
    reasoning_trace: str | None = None
    if stream_callback:
        try:
            raw, reasoning_trace = _stream_completion_chunks(
                client,
                create_kw,
                cancel_check=cancel_check,
                stream_callback=stream_callback,
            )
        except Exception:
            if cancel_check:
                cancel_check()
            response = client.chat.completions.create(**create_kw)
            msg = response.choices[0].message
            reasoning_trace, raw = _message_reasoning_and_content(msg)
            acc: list[str] = []
            if reasoning_trace and str(reasoning_trace).strip():
                acc.append(str(reasoning_trace).strip())
            if raw and str(raw).strip():
                acc.append(str(raw).strip())
            if acc:
                stream_callback("\n\n---\n\n".join(acc))
    else:
        response = client.chat.completions.create(**create_kw)
        msg = response.choices[0].message
        reasoning_trace, raw = _message_reasoning_and_content(msg)

    raw = raw if (raw or "").strip() else "{}"
    parse_err: str | None = None
    try:
        parsed = _parse_json_from_message(raw)
    except json.JSONDecodeError as e:
        parsed = {}
        parse_err = str(e)

    allowed_speakers = {
        str(x).strip()
        for x in (
            transcript_data.get("speakers")
            or [seg.get("speaker") for seg in transcript_data.get("segments", [])]
        )
        if str(x or "").strip()
    }
    speaker_roles = _normalize_speaker_roles(parsed.get("speaker_roles"), allowed_speakers)
    employee_speaker = _normalize_employee_speaker(
        parsed.get("employee_speaker"), speaker_roles, allowed_speakers
    )
    speaker_roles_confidence = _normalize_role_confidence(
        parsed.get("speaker_roles_confidence"), allowed_speakers
    )
    speaker_roles_reasoning = str(parsed.get("speaker_roles_reasoning", "")).strip() or None

    criteria_root = parsed.get("criteria")
    if not isinstance(criteria_root, dict):
        criteria_root = parsed

    criteria_out: list[dict[str, Any]] = []
    for c in criteria:
        block = criteria_root.get(c.id)
        comment = ""
        evidence_segments: list[dict[str, float]] = []
        passed: bool | None = None
        if isinstance(block, dict):
            passed = normalize_passed(block.get("passed"))
            if passed is None:
                legacy_score = parse_legacy_score(block.get("score"))
                if legacy_score is not None:
                    passed = legacy_score >= 50
            if passed is None:
                passed = False
            comment = str(block.get("comment", "")).strip()
            if passed is False and not comment:
                comment = "Недостаточно информации для подтверждения критерия."
            evidence_segments = _normalize_evidence_segments(block.get("evidence_segments"))
        criteria_out.append(
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "weight": c.weight,
                "passed": passed,
                "comment": comment,
                "evidence_segments": evidence_segments,
                "awarded_weight": awarded_weight(c.weight, passed),
            }
        )

    totals = compute_eval_totals(criteria_out)

    out: dict[str, Any] = {
        "schema_version": 3,
        "criteria_version": version,
        "criteria_file": cp.name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "video_file": transcript_data.get("video_file"),
        "speaker_roles": speaker_roles,
        "employee_speaker": employee_speaker,
        "criteria": criteria_out,
        "criteria_snapshot": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "weight": c.weight,
            }
            for c in criteria
        ],
        **totals,
    }
    if speaker_roles_confidence:
        out["speaker_roles_confidence"] = speaker_roles_confidence
    if speaker_roles_reasoning:
        out["speaker_roles_reasoning"] = speaker_roles_reasoning
    if reasoning_trace:
        out["reasoning_trace"] = reasoning_trace
    if parse_err:
        out["evaluation_parse_error"] = parse_err
        out["raw_model_content_preview"] = (raw[:2000] + "…") if len(raw) > 2000 else raw
    return out


def write_evaluation_json(data: dict[str, Any], out_path: Path) -> None:
    from src.atomic_json import atomic_write_json

    atomic_write_json(out_path, data, indent=2)
