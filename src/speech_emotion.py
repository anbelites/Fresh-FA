"""
Распознавание эмоциональной окраски речи по сегментам аудио (Transformers + SER).

Режим выбирается через SPEECH_EMOTION_MODE:
  "categorical" (по умолчанию) — классический SER (HuBERT-Dusha и т.п.)
  "valence_arousal"           — непрерывная шкала valence/arousal (emotion2vec и др.)

Для categorical:
  SPEECH_EMOTION_MODEL=xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned  # по умолчанию
  SPEECH_EMOTION_MODEL=Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition  # RESD
  SPEECH_EMOTION_MODEL=superb/wav2vec2-base-superb-er   # IEMOCAP (англ.)
  SPEECH_EMOTION_MODEL=nikatonika/wavlm-finetune-natural-balance  # 7 классов RU

Для valence_arousal:
  SPEECH_EMOTION_VA_MODEL=audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  # по умолчанию
  Выход: valence, arousal, dominance (0..1). Подходит для естественной речи.

Калибровка/trust_remote_code — только для categorical.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# До pipeline/transformers: иначе HF_TOKEN из .env может не попасть в huggingface_hub (порядок импортов, override).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

# --- Режимы по умолчанию (не удалять — переключение через env) ---
DEFAULT_SPEECH_EMOTION_MODEL_RU = (
    "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
)
"""Русская SER (Dusha): neutral, angry, positive, sad, other — без отдельного класса «страх»."""

ANIEMORE_RESD_MODEL = "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition"
"""RESD (7 классов), можно вернуть через SPEECH_EMOTION_MODEL."""

LEGACY_SPEECH_EMOTION_MODEL_EN = "superb/wav2vec2-base-superb-er"
"""IEMOCAP: neu / hap / sad / ang."""

DEFAULT_VA_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
"""Valence/Arousal/Dominance regression — natural speech, language-agnostic."""

_pipeline = None
_pipeline_model_id: str | None = None

_va_processor = None
_va_model_obj = None
_va_model_id: str | None = None


def _ser_mode() -> str:
    return os.environ.get("SPEECH_EMOTION_MODE", "categorical").strip().lower()


def default_speech_emotion_model() -> str:
    if _ser_mode() == "valence_arousal":
        return os.environ.get("SPEECH_EMOTION_VA_MODEL", DEFAULT_VA_MODEL)
    return os.environ.get("SPEECH_EMOTION_MODEL", DEFAULT_SPEECH_EMOTION_MODEL_RU)


def _get_device_arg() -> int | str:
    v = os.environ.get("SPEECH_EMOTION_DEVICE", "").strip().lower()
    if v == "cuda" or v == "gpu":
        return 0
    if v == "mps":
        return "mps"
    return -1


def _use_trust_remote_code(model_id: str) -> bool:
    """Только у моделей с кастомным классификатором на карточке HF (например Aniemore)."""
    ovr = os.environ.get("SPEECH_EMOTION_TRUST_REMOTE_CODE", "").strip().lower()
    if ovr in ("0", "false", "no"):
        return False
    if ovr in ("1", "true", "yes"):
        return True
    if model_id.startswith("superb/"):
        return False
    if model_id.startswith("xbgoose/") or model_id.startswith("nikatonika/"):
        return False
    if "Aniemore" in model_id or "aniemore" in model_id.lower():
        return True
    return False


def _get_classifier():
    global _pipeline, _pipeline_model_id
    from transformers import pipeline

    model_id = default_speech_emotion_model()
    if _pipeline is not None and _pipeline_model_id == model_id:
        return _pipeline

    trust = _use_trust_remote_code(model_id)
    kw: dict[str, Any] = {
        "model": model_id,
        "device": _get_device_arg(),
    }
    if trust:
        kw["trust_remote_code"] = True

    _pipeline = pipeline("audio-classification", **kw)
    _pipeline_model_id = model_id
    return _pipeline


def _entropy_norm(scores: dict[str, float]) -> float:
    """0 = одна метка, 1 = почти равномерное распределение (softmax «не уверен»)."""
    arr = np.array(list(scores.values()), dtype=np.float64)
    s = float(arr.sum())
    if s <= 0:
        return 1.0
    p = arr / s
    p = np.clip(p, 1e-12, 1.0)
    n = len(p)
    if n <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return h / float(np.log(n))


def _calibrate_ser_output(scores: dict[str, float]) -> dict[str, Any]:
    """
    Softmax по 7 классам часто почти плоский: argmax может быть шумом, если 1-й и 2-й класс
    отличаются меньше чем на min_margin — тогда uncertain (без top_score как «уверенности»).

    Порог по энтропии по умолчанию выключен: для типичного выхода RESD энтропия почти всегда
    близка к максимуму, и жёсткий порог помечал бы всё как uncertain.
    """
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    model_top_label, model_top_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    margin = float(model_top_score) - float(second_score)
    ent = _entropy_norm(scores)

    calibrate = os.environ.get("SPEECH_EMOTION_CALIBRATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    min_margin = float(os.environ.get("SPEECH_EMOTION_MIN_MARGIN", "0.002"))
    max_ent_raw = os.environ.get("SPEECH_EMOTION_MAX_ENTROPY_NORM", "1.0").strip()
    max_ent = float(max_ent_raw) if max_ent_raw else 1.0

    use_entropy = max_ent < 1.0
    ambiguous = calibrate and (
        (margin < min_margin) or (use_entropy and (ent > max_ent))
    )
    rounded_scores = {k: round(v, 4) for k, v in scores.items()}
    base_meta = {
        "model_top_label": str(model_top_label),
        "model_top_score": round(float(model_top_score), 4),
        "label_margin": round(margin, 4),
        "entropy_norm": round(ent, 4),
        "scores": rounded_scores,
    }
    if ambiguous:
        return {
            "top_label": "uncertain",
            **base_meta,
        }
    return {
        "top_label": str(model_top_label),
        "top_score": round(float(model_top_score), 4),
        **base_meta,
    }


def _get_va_model():
    """Load valence/arousal regression model (audeering wav2vec2)."""
    global _va_processor, _va_model_obj, _va_model_id
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

    model_id = os.environ.get("SPEECH_EMOTION_VA_MODEL", DEFAULT_VA_MODEL)
    if _va_model_obj is not None and _va_model_id == model_id:
        return _va_processor, _va_model_obj

    _va_processor = Wav2Vec2Processor.from_pretrained(model_id)
    _va_model_obj = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    _va_model_obj.eval()
    _va_model_id = model_id
    return _va_processor, _va_model_obj


def _classify_chunk_va(
    y: np.ndarray, sr: int, max_sec: float = 30.0
) -> dict[str, Any] | None:
    """Valence/Arousal/Dominance regression via wav2vec2."""
    import torch

    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.15 * sr):
        return None
    max_n = int(max_sec * sr)
    if len(y) > max_n:
        start = (len(y) - max_n) // 2
        y = y[start : start + max_n]

    processor, model = _get_va_model()

    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().cpu().numpy()

    arousal = float(logits[0]) if len(logits) > 0 else 0.5
    dominance = float(logits[1]) if len(logits) > 1 else 0.5
    valence = float(logits[2]) if len(logits) > 2 else 0.5

    arousal = max(0.0, min(1.0, arousal))
    valence = max(0.0, min(1.0, valence))
    dominance = max(0.0, min(1.0, dominance))

    if arousal > 0.6 and valence > 0.55:
        label = "engaged_positive"
    elif arousal > 0.6 and valence < 0.4:
        label = "agitated_negative"
    elif arousal < 0.35 and valence < 0.4:
        label = "low_negative"
    elif arousal < 0.35:
        label = "calm"
    else:
        label = "neutral"

    return {
        "top_label": label,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "mode": "valence_arousal",
    }


def _classify_chunk(
    y: np.ndarray, sr: int, classifier, max_sec: float = 30.0
) -> dict[str, Any] | None:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.15 * sr):
        return None
    max_n = int(max_sec * sr)
    if len(y) > max_n:
        start = (len(y) - max_n) // 2
        y = y[start : start + max_n]
    preds = classifier({"raw": y, "sampling_rate": sr}, top_k=None)
    scores: dict[str, float] = {}
    for p in preds:
        scores[str(p["label"])] = float(p["score"])
    return _calibrate_ser_output(scores)


def _aggregate_by_speaker(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Доли итоговых меток (после калибровки) по длительности сегментов — не среднее по «плоскому» softmax."""
    by_spk: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        sp = row.get("speaker") or "?"
        t0 = float(row.get("start", 0))
        t1 = float(row.get("end", 0))
        w = max(t1 - t0, 0.05)
        lab = str(row.get("top_label") or "?")
        by_spk[sp].append((w, lab))

    out: dict[str, Any] = {}
    for spk, pairs in sorted(by_spk.items()):
        tw = sum(w for w, _ in pairs)
        merged: dict[str, float] = defaultdict(float)
        for w, lab in pairs:
            merged[lab] += w
        for k in merged:
            merged[k] /= tw
        dominant = max(merged, key=lambda k: merged[k]) if merged else None
        out[spk] = {
            "dominant_label": dominant,
            "label_shares": {k: round(v, 4) for k, v in sorted(merged.items())},
        }
    return out


def _sidecar_note(model_id: str) -> str:
    if "superb" in model_id and "Aniemore" not in model_id:
        return (
            "Классы IEMOCAP (англ.): neu, hap, sad, ang. Для русской речи — грубый ориентир; "
            f"лучше {DEFAULT_SPEECH_EMOTION_MODEL_RU} или {ANIEMORE_RESD_MODEL}."
        )
    if "xbgoose/hubert-large" in model_id and "dusha" in model_id.lower():
        return (
            "HuBERT, дообучение на русском Dusha: neutral, angry, positive, sad, other. "
            "Нет отдельного класса «страх» (снижает ложные срабатывания по сравнению с RESD). "
            "Ориентир по аудио, не подменяет смысл реплик."
        )
    if "Aniemore" in model_id or "aniemore" in model_id.lower():
        return (
            "RESD: anger, disgust, enthusiasm, fear, happiness, neutral, sadness — "
            "при плоском softmax возможна метка uncertain. Альтернатива: "
            f"{DEFAULT_SPEECH_EMOTION_MODEL_RU}."
        )
    if "nikatonika" in model_id.lower():
        return (
            "WavLM, баланс классов по русскому корпусу: Angry, Disgusted, Happy, Neutral, Sad, Scared, Surprised."
        )
    return "См. карточку модели на Hugging Face для списка классов."


_POSITIVE_TEXT_HINTS = {
    "отлично", "замечательно", "прекрасно", "здорово", "супер", "класс",
    "рад", "рада", "поздравляю", "благодарю", "спасибо", "молодец",
    "восхитительно", "великолепно", "потрясающе", "чудесно",
}
_NEGATIVE_TEXT_HINTS = {
    "плохо", "ужас", "кошмар", "жаль", "к сожалению", "проблема",
    "жалоба", "недоволен", "недовольна", "разочарован",
}

_VALENCE_FROM_LABEL: dict[str, float] = {
    "positive": 0.75, "Happy": 0.8, "Surprised": 0.5,
    "neutral": 0.5, "Neutral": 0.5, "other": 0.5, "uncertain": 0.5,
    "sad": 0.25, "Sad": 0.25, "angry": 0.2, "Angry": 0.2,
    "Disgusted": 0.15, "Scared": 0.3,
}


def _estimate_arousal_valence(
    chunk_y: np.ndarray, sr: int, ser_label: str, text: str,
) -> dict[str, float]:
    """Heuristic arousal/valence from acoustic features + SER label + text keywords."""
    import librosa

    arousal = 0.5
    valence = _VALENCE_FROM_LABEL.get(ser_label, 0.5)

    if len(chunk_y) >= int(0.3 * sr):
        rms_val = float(np.sqrt(np.mean(chunk_y ** 2)))
        arousal_rms = min(1.0, rms_val / 0.08)

        f0, _, _ = librosa.pyin(chunk_y, fmin=60, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) > 2:
            f0_std = float(np.std(f0_valid))
            arousal_f0 = min(1.0, f0_std / 60.0)
        else:
            arousal_f0 = 0.4
        arousal = 0.5 * arousal_rms + 0.5 * arousal_f0

    text_lower = text.lower()
    text_words = set(text_lower.split())
    if text_words & _POSITIVE_TEXT_HINTS:
        valence = min(1.0, valence + 0.2)
        arousal = min(1.0, arousal + 0.1)
    if text_words & _NEGATIVE_TEXT_HINTS:
        valence = max(0.0, valence - 0.2)

    return {
        "arousal": round(float(np.clip(arousal, 0, 1)), 3),
        "valence": round(float(np.clip(valence, 0, 1)), 3),
    }


def _interpret_arousal_valence(a: float, v: float) -> str:
    """Human-readable engagement label from arousal + valence."""
    if a >= 0.55 and v >= 0.6:
        return "engaged_positive"
    if a >= 0.55 and v < 0.4:
        return "engaged_negative"
    if a < 0.35 and v >= 0.55:
        return "calm_positive"
    if a < 0.35 and v < 0.4:
        return "disengaged"
    return "neutral_moderate"


def build_speech_emotion_sidecar(
    y: np.ndarray,
    sr: int,
    segments: list[dict[str, Any]],
    video_file: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    mode = _ser_mode()
    use_dedicated_va = mode == "valence_arousal"
    model_id = model_id or default_speech_emotion_model()
    rows: list[dict[str, Any]] = []

    clf = None
    if use_dedicated_va:
        try:
            _get_va_model()
        except Exception as e:
            return {
                "schema_version": 2,
                "video_file": video_file,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "model": model_id,
                "mode": mode,
                "error": f"Не удалось загрузить VA-модель: {e}",
                "segments": [],
                "by_speaker": {},
            }
    else:
        try:
            clf = _get_classifier()
        except Exception as e:
            return {
                "schema_version": 2,
                "video_file": video_file,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "model": model_id,
                "mode": mode,
                "error": f"Не удалось загрузить модель: {e}",
                "segments": [],
                "by_speaker": {},
            }

    use_heuristic_va = (
        not use_dedicated_va
        and os.environ.get("SER_VALENCE_AROUSAL", "1").strip().lower()
        not in ("0", "false", "no")
    )

    for seg in segments:
        t0 = float(seg.get("start", 0))
        t1 = float(seg.get("end", 0))
        spk = seg.get("speaker", "?")
        text = seg.get("text", "")
        i0 = max(0, int(t0 * sr))
        i1 = min(len(y), int(t1 * sr))
        chunk = y[i0:i1]
        base = {
            "start": round(t0, 3),
            "end": round(t1, 3),
            "speaker": spk,
        }

        if use_dedicated_va:
            pred = _classify_chunk_va(chunk, sr)
        else:
            pred = _classify_chunk(chunk, sr, clf)

        if pred is None:
            rows.append({**base, "skipped": True, "reason": "segment_too_short"})
        else:
            if use_heuristic_va and not use_dedicated_va:
                ser_label = pred.get("top_label", "neutral")
                va = _estimate_arousal_valence(chunk, sr, ser_label, text)
                pred["arousal"] = va["arousal"]
                pred["valence"] = va["valence"]
                pred["engagement"] = _interpret_arousal_valence(va["arousal"], va["valence"])
            rows.append({**base, **pred})

    note = (
        "Valence/Arousal/Dominance regression (wav2vec2). "
        "arousal>0.6+valence>0.55 = engaged_positive; arousal<0.35 = calm."
    ) if use_dedicated_va else _sidecar_note(model_id)

    return {
        "schema_version": 2,
        "video_file": video_file,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "mode": mode,
        "valence_arousal": use_dedicated_va or use_heuristic_va,
        "note": note,
        "segments": rows,
        "by_speaker": _aggregate_by_speaker(rows),
    }


def write_tone_json(data: dict[str, Any], out_path: Path) -> None:
    from src.atomic_json import atomic_write_json

    atomic_write_json(out_path, data, indent=2)


SER_LABEL_RU: dict[str, str] = {
    "uncertain": "неопределённо",
    "neutral": "нейтральный",
    "angry": "раздражение",
    "positive": "позитив",
    "sad": "грусть",
    "other": "прочее",
    "Angry": "раздражение",
    "Disgusted": "отвращение",
    "Happy": "радость",
    "Neutral": "нейтральный",
    "Sad": "грусть",
    "Scared": "страх",
    "Surprised": "удивление",
    "engaged_positive": "вовлечён/позитив",
    "agitated_negative": "возбуждён/негатив",
    "low_negative": "вялый/негатив",
    "calm": "спокойный",
    "engaged_negative": "вовлечён/негатив",
    "calm_positive": "спокойный/позитив",
    "disengaged": "не вовлечён",
    "neutral_moderate": "нейтральный/умеренно",
}


def ser_label_for_segment(t0: float, tone_segments: list[dict[str, Any]]) -> str | None:
    """Find SER label for a transcript segment by matching start time."""
    for ts in tone_segments:
        if ts.get("skipped"):
            continue
        ts_start = ts.get("start")
        if ts_start is not None and abs(float(ts_start) - t0) < 0.6:
            engagement = ts.get("engagement")
            if engagement:
                return SER_LABEL_RU.get(str(engagement), str(engagement))
            raw = ts.get("top_label")
            if raw:
                return SER_LABEL_RU.get(str(raw), str(raw))
    return None


def speech_emotion_context_for_eval(tone_data: dict[str, Any]) -> str:
    """Per-speaker summary + per-segment SER labels for the LLM evaluator."""
    if tone_data.get("error") and not tone_data.get("by_speaker"):
        return ""
    by_spk = tone_data.get("by_speaker") or {}
    segments = tone_data.get("segments") or []
    if not by_spk and not segments:
        return ""

    lines: list[str] = [
        "",
        "--- Эмоциональная окраска по аудио (SER, предобученная модель, ориентир):",
        "  ВАЖНО: SER-модель обучена на актёрских эмоциях; «neutral» часто означает обычный разговорный тон.",
        "  Для оценки вовлечённости и доброжелательности больше опирайся на акустические признаки",
        "  (expressive_pitch_variation, bright_timbral) и смысл реплик, а не только на метку SER.",
    ]

    for spk, data in sorted(by_spk.items()):
        dom = data.get("dominant_label")
        dom_s = SER_LABEL_RU.get(str(dom), dom)
        shares = data.get("label_shares") or {}
        top3 = sorted(shares.items(), key=lambda x: -x[1])[:4]
        brief = ", ".join(f"{k} {v:.2f}" for k, v in top3)
        lines.append(f"  {spk}: преобладает «{dom_s}»; доли по сегментам: {brief}")

    non_neutral = [
        s for s in segments
        if not s.get("skipped") and s.get("top_label") not in ("neutral", "Neutral", None)
    ]
    if non_neutral:
        lines.append("")
        lines.append("  Сегменты с не-нейтральной SER-меткой:")
        for s in non_neutral[:30]:
            t0 = s.get("start", 0)
            t1 = s.get("end", 0)
            spk = s.get("speaker", "?")
            lab = SER_LABEL_RU.get(str(s.get("top_label", "")), s.get("top_label", ""))
            score = s.get("top_score")
            score_s = f" ({score:.0%})" if score is not None else ""
            lines.append(f"    [{t0:.1f}–{t1:.1f}] {spk}: {lab}{score_s}")

    return "\n".join(lines)
