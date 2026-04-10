"""
Эвристики «тараторство / бубнеж / неразборчивость» поверх Whisper.

Whisper не выдаёт отдельную метку «тараторит» — используем:
- скорость речи (слова и оценка слогов/сек по гласным в русском тексте);
- avg_logprob сегмента и среднее/минимальную probability по словам (чем ниже — тем хуже уверенность ASR).

Пороги настраиваются через переменные окружения (см. ниже).
"""
from __future__ import annotations

import os
from typing import Any

_RU_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def _float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def est_syllables_ru(text: str) -> int:
    """Грубая оценка числа слогов по гласным (для русского)."""
    return sum(1 for c in text if c in _RU_VOWELS)


def analyze_segment(
    *,
    text: str,
    t0: float,
    t1: float,
    avg_logprob: float,
    compression_ratio: float,
    no_speech_prob: float,
    word_probs: list[float] | None,
) -> dict[str, Any]:
    dur = max(t1 - t0, 0.05)
    words_list = [w for w in text.split() if w.strip()]
    n_words = len(words_list)
    n_chars = len(text.replace("\n", " "))
    syll = est_syllables_ru(text)

    wps = n_words / dur
    sps = syll / dur if syll > 0 else 0.0
    cps = n_chars / dur if n_chars > 0 else 0.0

    mean_wp = None
    min_wp = None
    if word_probs:
        mean_wp = sum(word_probs) / len(word_probs)
        min_wp = min(word_probs)

    thr_wps = _float_env("SPEECH_FAST_WPS", 3.4)
    thr_sps = _float_env("SPEECH_FAST_SPS", 6.5)
    thr_logp = _float_env("SPEECH_LOW_LOGPROB", -0.38)
    thr_wprob = _float_env("SPEECH_LOW_WORD_PROB", 0.42)
    thr_nsp = _float_env("SPEECH_HIGH_NO_SPEECH", 0.35)

    fast = wps >= thr_wps or sps >= thr_sps
    low_seg = avg_logprob < thr_logp
    low_words = mean_wp is not None and mean_wp < thr_wprob
    min_word_bad = min_wp is not None and min_wp < (thr_wprob - 0.1)
    noisy = no_speech_prob > thr_nsp

    flags: list[str] = []
    if fast:
        flags.append("fast_pace")
    if low_seg:
        flags.append("low_asr_logprob")
    if low_words:
        flags.append("low_word_confidence")
    if min_word_bad:
        flags.append("some_words_uncertain")
    if noisy:
        flags.append("possible_non_speech_noise")

    # Сильный сигнал «тараторит и плохо легло в распознавание»
    if fast and (low_seg or low_words):
        flags.append("fast_and_unclear_asr")

    return {
        "duration_sec": round(dur, 3),
        "words_per_sec": round(wps, 2),
        "syllables_per_sec_est": round(sps, 2),
        "chars_per_sec": round(cps, 2),
        "avg_logprob": round(avg_logprob, 4),
        "compression_ratio": round(compression_ratio, 3),
        "no_speech_prob": round(no_speech_prob, 4),
        "mean_word_probability": None if mean_wp is None else round(mean_wp, 4),
        "min_word_probability": None if min_wp is None else round(min_wp, 4),
        "flags": sorted(set(flags)),
        "thresholds_used": {
            "fast_wps_gte": thr_wps,
            "fast_sps_gte": thr_sps,
            "low_logprob_lt": thr_logp,
            "low_mean_word_prob_lt": thr_wprob,
            "high_no_speech_gt": thr_nsp,
        },
    }


def delivery_summary_note() -> str:
    return (
        "Поля delivery.* — эвристики: быстрый темп (слова/слоги в сек), низкая уверенность Whisper. "
        "Подкрутите пороги SPEECH_FAST_WPS, SPEECH_FAST_SPS, SPEECH_LOW_LOGPROB, SPEECH_LOW_WORD_PROB. "
        "Точное «бубнеж» без дообучения модели не различить; для лучшего распознавания попробуйте WHISPER_MODEL=large-v3."
    )
