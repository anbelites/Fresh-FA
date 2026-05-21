"""Транскрипция (faster-whisper) + диаризация говорящих (pyannote, при HF_TOKEN)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import subprocess
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import multiprocessing
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty
from typing import Any

from src.cuda_runtime_path import ensure_nvidia_pip_libs

ensure_nvidia_pip_libs()

from src.audio_extract import extract_wav_16k_mono
from src.errors import PipelineCancelled
from src.glossary import format_glossary_for_whisper
from src.asr_quality import (
    quality_summary_text,
    rescue_windows_from_report,
    transcript_quality_report,
)
from src.audio_tone import (
    aggregate_tone_by_speaker,
    audio_tone_summary_note,
    segment_audio_tone,
)
from src.nemo_diarize import (
    load_diarization_rows_nemo,
    nemo_device_name,
    nemo_diarization_backend_enabled,
    nemo_is_installed,
    nemo_model_name,
)
from src.speech_delivery import analyze_segment, delivery_summary_note


def _ct2_cuda_available() -> bool:
    """CTranslate2 (faster-whisper) видит CUDA без загрузки PyTorch."""
    try:
        import ctranslate2 as ct2

        return ct2.get_cuda_device_count() > 0
    except Exception:
        return False


def _device() -> str:
    v = os.environ.get("WHISPER_DEVICE", "").strip()
    if v:
        return v
    return "cuda" if _ct2_cuda_available() else "cpu"


def _compute_type() -> str:
    v = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip()
    if v:
        return v
    return "float16" if _device() == "cuda" else "int8"


def _model_name() -> str:
    return os.environ.get("WHISPER_MODEL", "large-v3")


def _language() -> str | None:
    v = os.environ.get("WHISPER_LANGUAGE", "ru").strip()
    return v if v else None


def _initial_prompt() -> str | None:
    """Domain vocabulary hint for Whisper (improves recognition of names, brands, terms)."""
    v = os.environ.get("WHISPER_INITIAL_PROMPT", "").strip()
    if v:
        return v
    glossary_prompt = format_glossary_for_whisper()
    extra = os.environ.get("WHISPER_DOMAIN_PROMPT", "").strip()
    if extra and glossary_prompt:
        return f"{glossary_prompt}\nДополнительный глоссарий: {extra}"
    if extra:
        return f"Дополнительный глоссарий для распознавания речи: {extra}"
    return glossary_prompt or None


def _initial_prompt_source() -> str:
    if os.environ.get("WHISPER_INITIAL_PROMPT", "").strip():
        return "WHISPER_INITIAL_PROMPT"
    if os.environ.get("WHISPER_DOMAIN_PROMPT", "").strip():
        return "database_glossary+WHISPER_DOMAIN_PROMPT"
    return "database_glossary"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _diarization_backend_preference() -> str:
    raw = os.environ.get("DIARIZATION_BACKEND", "auto").strip().lower()
    if raw in ("nemo", "nemo_sortformer"):
        return "nemo"
    if raw == "pyannote":
        return "pyannote"
    if raw == "mfcc":
        return "mfcc"
    return "auto"


def _diarization_backend_candidates(hf: str | None) -> list[str]:
    pref = _diarization_backend_preference()
    if pref == "nemo":
        return ["nemo", "pyannote", "mfcc"]
    if pref == "pyannote":
        return ["pyannote", "mfcc"]
    if pref == "mfcc":
        return ["mfcc"]
    candidates: list[str] = []
    if nemo_diarization_backend_enabled():
        candidates.append("nemo")
    candidates.append("pyannote")
    candidates.append("mfcc")
    return candidates


def _diarization_pipeline_path_or_id() -> Path | str:
    """Hub id или локальный config.yaml для pyannote; по умолчанию community-1."""
    root = Path(__file__).resolve().parent.parent
    override = os.environ.get("PYANNOTE_PIPELINE", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if p.suffix in (".yaml", ".yml") and p.is_file():
            return p
        return override
    local = root / "config" / "pyannote" / "speaker-diarization-community-1.yaml"
    if local.is_file():
        return local
    return "pyannote/speaker-diarization-community-1"


def _pyannote_pipeline_name() -> str:
    return str(_diarization_pipeline_path_or_id()).strip().lower()


def _is_pyannote_community_one() -> bool:
    name = _pyannote_pipeline_name()
    return "community-1" in name or "community_1" in name


def _pyannote_pipeline_kwargs(expected_speaker_count: int | None = None) -> dict[str, int]:
    """Подсказки по числу говорящих (если известно) — уменьшает путаницу кластеров."""
    if expected_speaker_count is not None:
        try:
            count = int(expected_speaker_count)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return {"num_speakers": max(1, min(count, 8))}
    out: dict[str, int] = {}
    for key, env_name in (
        ("num_speakers", "PYANNOTE_NUM_SPEAKERS"),
        ("min_speakers", "PYANNOTE_MIN_SPEAKERS"),
        ("max_speakers", "PYANNOTE_MAX_SPEAKERS"),
    ):
        v = os.environ.get(env_name, "").strip()
        if v:
            try:
                out[key] = int(v)
            except ValueError:
                pass
    if "min_speakers" not in out:
        d = os.environ.get("PYANNOTE_MIN_SPEAKERS_DEFAULT", "2").strip()
        if d and d != "0":
            try:
                out["min_speakers"] = int(d)
            except ValueError:
                out["min_speakers"] = 2
    if "max_speakers" not in out:
        d = os.environ.get("PYANNOTE_MAX_SPEAKERS_DEFAULT", "2").strip()
        if d and d != "0":
            try:
                out["max_speakers"] = int(d)
            except ValueError:
                out["max_speakers"] = 2
    return out


def _mfcc_max_clusters() -> int:
    try:
        return max(2, int(os.environ.get("MFCC_DIAR_MAX_SPEAKERS", "8")))
    except ValueError:
        return 8


def _pause_split_sec() -> float:
    """Пауза между словами (сек): дополнительная новая строка при длинной тишине. 0 = только смена спикера."""
    v = os.environ.get("TRANSCRIPT_PAUSE_SPLIT_SEC", "0").strip()
    if not v or v == "0":
        return 0.0
    try:
        return max(0.0, float(v))
    except ValueError:
        return 0.0


def _mfcc_word_level_enabled() -> bool:
    """По умолчанию выкл.: KMeans по словам даёт шум и дробит одного говорящего на несколько меток."""
    return os.environ.get("MFCC_WORD_LEVEL", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _should_run_pyannote(hf: str | None) -> bool:
    """
    Pyannote 4.x тянет torchcodec; на Windows без «полного» FFmpeg shared DLL часто не грузятся
    (OSError / libtorchcodec_*.dll). По умолчанию на win32 pyannote не вызываем — MFCC.
    Включить вручную: PYANNOTE_ON_WINDOWS=1 (и настроить FFmpeg по доке torchcodec).
    Полностью отключить pyannote: SKIP_PYANNOTE=1.
    """
    if not hf:
        return False
    if os.environ.get("SKIP_PYANNOTE", "").lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "win32":
        return os.environ.get("PYANNOTE_ON_WINDOWS", "").lower() in (
            "1",
            "true",
            "yes",
        )
    return True


def _should_run_nemo(hf: str | None) -> bool:
    if not nemo_diarization_backend_enabled():
        return False
    model_ref = nemo_model_name()
    model_path = Path(model_ref)
    if model_path.is_file():
        return True
    return bool(hf)


def _pyannote_inference_device():
    """CPU/CUDA для pyannote: авто — CUDA при доступности, иначе CPU."""
    import torch

    v = os.environ.get("PYANNOTE_DEVICE", "").strip().lower()
    if v in ("cpu",):
        return torch.device("cpu")
    if v in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pyannote_device_type() -> str:
    return str(_pyannote_inference_device().type)


def _use_pyannote_exclusive() -> bool:
    """
    Exclusive diarization обычно устойчивее стыкуется с ASR и уменьшает ложные
    короткие переключения. На CUDA включаем по умолчанию; env может переопределить.
    """
    v = os.environ.get("PYANNOTE_EXCLUSIVE", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return _pyannote_device_type() == "cuda"


def _pyannote_segment_smooth_radius() -> int:
    """
    На GPU по умолчанию слегка сглаживаем одиночные ложные скачки спикера между
    соседними строками; на CPU оставляем старое поведение без сглаживания.
    """
    raw = os.environ.get("PYANNOTE_SEGMENT_SMOOTH_RADIUS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    if _is_pyannote_community_one():
        return 0
    return 1 if _pyannote_device_type() == "cuda" else 0


def _pyannote_label_smooth_radius() -> int:
    raw = os.environ.get("PYANNOTE_LABEL_SMOOTH_RADIUS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    return 0 if _is_pyannote_community_one() else 3


def _pyannote_bridge_short_run_sec() -> float:
    raw = os.environ.get("PYANNOTE_BRIDGE_SHORT_RUN_SEC", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 0.25 if _is_pyannote_community_one() else 2.8


def _pyannote_bridge_min_distinct_speakers() -> int:
    raw = os.environ.get("PYANNOTE_BRIDGE_MIN_DISTINCT_SPEAKERS", "").strip()
    if raw:
        try:
            return max(2, int(raw))
        except ValueError:
            return 4
    return 4


def _load_diarization_rows(
    wav_path: Path,
    hf_token: str,
    *,
    expected_speaker_count: int | None = None,
) -> list[tuple[float, float, str]]:
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(
        _diarization_pipeline_path_or_id(), token=hf_token
    )
    pipeline = pipeline.to(_pyannote_inference_device())
    kwargs = _pyannote_pipeline_kwargs(expected_speaker_count)
    with torch.inference_mode():
        raw_out = pipeline({"audio": str(wav_path)}, **kwargs)
    if hasattr(raw_out, "speaker_diarization"):
        use_exclusive = _use_pyannote_exclusive()
        if use_exclusive and getattr(raw_out, "exclusive_speaker_diarization", None):
            diarization = raw_out.exclusive_speaker_diarization
        else:
            diarization = raw_out.speaker_diarization
    else:
        diarization = raw_out
    rows: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        rows.append((float(turn.start), float(turn.end), str(speaker)))
    return rows


def _friendly_pyannote_error(exc: Exception) -> str:
    msg = str(exc).strip()
    low = msg.lower()
    pipeline_ref = str(_diarization_pipeline_path_or_id())
    if (
        "speaker-diarization-community-1" in pipeline_ref
        and ("gated" in low or "private" in low or "accept user conditions" in low)
    ):
        return (
            "Pyannote community-1 недоступен для текущего HF_TOKEN: нужно один раз принять условия "
            "на https://hf.co/pyannote/speaker-diarization-community-1. "
            "Временный откат: PYANNOTE_PIPELINE=pyannote/speaker-diarization-3.1."
        )
    return msg


def _remap_speakers_sequential(raw_labels: list[str]) -> list[str]:
    """Приводит любые метки к SPEAKER_01, SPEAKER_02, … по порядку появления."""
    mapping: dict[str, str] = {}
    out: list[str] = []
    for r in raw_labels:
        if r not in mapping:
            mapping[r] = f"SPEAKER_{len(mapping) + 1:02d}"
        out.append(mapping[r])
    return out


def _mfcc_kmeans_speakers(
    wav_path: Path,
    segment_bounds: list[tuple[float, float]],
    y_sr: tuple[Any, int] | None = None,
) -> list[str]:  # y_sr: (waveform ndarray, sample_rate)
    """
    Локальная «диаризация» без Hugging Face: один вектор MFCC на сегмент Whisper,
    KMeans (k=min(2, n)). Хуже pyannote, но даёт SPEAKER_01/02 вместо UNKNOWN.
    """
    import librosa
    import numpy as np
    from sklearn.cluster import KMeans

    if y_sr is not None:
        y, sr = y_sr
    else:
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    feats: list[np.ndarray] = []
    min_samp = int(0.08 * sr)
    for t0, t1 in segment_bounds:
        i0 = max(0, int(t0 * sr))
        i1 = max(i0 + min_samp, int(t1 * sr))
        chunk = y[i0:i1]
        if chunk.size < min_samp:
            chunk = np.pad(chunk, (0, min_samp - chunk.size))
        n_fft = min(2048, max(256, len(chunk)))
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20, n_fft=n_fft)
        feats.append(np.mean(mfcc, axis=1))
    n = len(feats)
    if n == 0:
        return []
    if n == 1:
        return ["SPEAKER_01"]
    X = np.stack(feats)
    k = min(_mfcc_max_clusters(), n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return [f"SPEAKER_{int(labels[i]) + 1:02d}" for i in range(n)]


def _speaker_for_interval(
    t0: float,
    t1: float,
    diar_rows: list[tuple[float, float, str]],
) -> str:
    best = "SPEAKER_UNKNOWN"
    best_ov = 0.0
    for d0, d1, spk in diar_rows:
        ov = max(0.0, min(t1, d1) - max(t0, d0))
        if ov > best_ov:
            best_ov = ov
            best = spk
    if best_ov > 0:
        return best
    mid = (t0 + t1) / 2.0
    for d0, d1, spk in diar_rows:
        if d0 <= mid <= d1:
            return spk
    return best


def _speaker_at_time(t: float, diar_rows: list[tuple[float, float, str]]) -> str:
    """Кто говорит в момент t (по дорожке pyannote)."""
    for d0, d1, spk in diar_rows:
        if d0 <= t < d1:
            return spk
    return _speaker_for_interval(t, t, diar_rows)


def _speaker_for_word_voted(
    w: Any,
    diar_rows: list[tuple[float, float, str]],
) -> str:
    """Выбор спикера для слова: сначала по overlap, затем по 3 точкам внутри слова."""
    w0 = float(w.start)
    w1 = float(w.end)
    if w1 <= w0:
        return _speaker_for_interval(w0, w1, diar_rows)
    overlap_votes: Counter[str] = Counter()
    for d0, d1, spk in diar_rows:
        ov = max(0.0, min(w1, d1) - max(w0, d0))
        if ov > 0:
            overlap_votes[spk] += ov
    if overlap_votes:
        return overlap_votes.most_common(1)[0][0]
    samples = [
        w0 + 0.15 * (w1 - w0),
        w0 + 0.5 * (w1 - w0),
        w0 + 0.85 * (w1 - w0),
    ]
    votes = [_speaker_at_time(t, diar_rows) for t in samples]
    return Counter(votes).most_common(1)[0][0]


def _join_whisper_words(wlist: list[Any]) -> str:
    """Склеивает токены faster-whisper (пробелы уже внутри токенов)."""
    return ("".join(str(getattr(w, "word", "") or "") for w in wlist)).strip()


def _text_for_word_slice(orig_seg: Any, wlist: list[Any]) -> str:
    """
    Preserve Whisper's formatted segment text when diarization keeps the full segment.
    For real splits, fall back to word tokens because only the slice belongs to this speaker.
    """
    if not wlist:
        return str(getattr(orig_seg, "text", "") or "").strip()
    seg_words = list(getattr(orig_seg, "words", None) or [])
    if len(wlist) == len(seg_words):
        text = str(getattr(orig_seg, "text", "") or "").strip()
        if text:
            return text
    return _join_whisper_words(wlist)


def _pyannote_within_segment_aba_flank_max_sec() -> float:
    raw = os.environ.get("PYANNOTE_ABA_FLANK_MAX_SEC", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 2.6 if _is_pyannote_community_one() else 0.0


def _pyannote_within_segment_aba_center_min_sec() -> float:
    raw = os.environ.get("PYANNOTE_ABA_CENTER_MIN_SEC", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 2.0 if _is_pyannote_community_one() else 0.0


def _pyannote_soft_takeover_prefix_max_sec() -> float:
    raw = os.environ.get("PYANNOTE_TAKEOVER_PREFIX_MAX_SEC", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 2.2 if _is_pyannote_community_one() else 0.0


def _pyannote_soft_takeover_min_target_sec() -> float:
    raw = os.environ.get("PYANNOTE_TAKEOVER_MIN_TARGET_SEC", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 1.0 if _is_pyannote_community_one() else 0.0


def _pyannote_soft_takeover_prefix_max_mean_word_prob() -> float:
    raw = os.environ.get("PYANNOTE_TAKEOVER_PREFIX_MAX_MEAN_WORD_PROB", "").strip()
    if raw:
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError:
            return 0.0
    return 0.6 if _is_pyannote_community_one() else 0.0


def _pyannote_soft_takeover_min_conf_gain() -> float:
    raw = os.environ.get("PYANNOTE_TAKEOVER_MIN_CONF_GAIN", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    return 0.08 if _is_pyannote_community_one() else 0.0


def _soft_text_continuation(left_text: str, right_text: str) -> bool:
    left = (left_text or "").strip()
    right = (right_text or "").strip()
    if not left or not right:
        return False
    if left.endswith((",", ":", ";", "-", "—", "(")):
        return True
    first = right[0]
    if first.islower() or first.isdigit():
        return True
    return left[-1] not in ".!?"


def _strict_text_continuation(left_text: str, right_text: str) -> bool:
    left = (left_text or "").strip()
    right = (right_text or "").strip()
    if not left or not right:
        return False
    if left.endswith((",", ":", ";", "-", "—", "(")):
        return True
    first = right[0]
    if first.islower() or first.isdigit():
        return left[-1] not in ".!?"
    return False


def _segment_mean_word_probability(seg: dict[str, Any]) -> float:
    delivery = seg.get("delivery")
    if isinstance(delivery, dict):
        raw = delivery.get("mean_word_probability")
        try:
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            pass
    words = seg.get("words")
    if isinstance(words, list) and words:
        vals: list[float] = []
        for w in words:
            if not isinstance(w, dict):
                continue
            raw = w.get("probability")
            try:
                vals.append(float(raw))
            except (TypeError, ValueError):
                continue
        if vals:
            return sum(vals) / len(vals)
    return 0.0


def _set_segment_speaker(seg: dict[str, Any], speaker: str) -> None:
    seg["speaker"] = speaker
    for w in seg.get("words") or []:
        if isinstance(w, dict):
            w["speaker"] = speaker


def _reassign_soft_continuation_prefix_to_next_speaker(
    segments_out: list[dict[str, Any]],
) -> None:
    prefix_max = _pyannote_soft_takeover_prefix_max_sec()
    min_target_sec = _pyannote_soft_takeover_min_target_sec()
    prefix_max_prob = _pyannote_soft_takeover_prefix_max_mean_word_prob()
    min_conf_gain = _pyannote_soft_takeover_min_conf_gain()
    if (
        prefix_max <= 0
        or min_target_sec <= 0
        or prefix_max_prob <= 0
        or len(segments_out) < 2
    ):
        return

    for idx in range(1, len(segments_out)):
        curr = segments_out[idx]
        curr_spk = str(curr.get("speaker", "") or "")
        prev_spk = str(segments_out[idx - 1].get("speaker", "") or "")
        if not curr_spk or not prev_spk or curr_spk == prev_spk:
            continue

        curr_start = float(curr.get("start", 0.0) or 0.0)
        curr_end = float(curr.get("end", 0.0) or 0.0)
        curr_dur = max(0.0, curr_end - curr_start)
        if curr_dur < min_target_sec:
            continue

        start_idx = idx - 1
        while start_idx > 0:
            prev = segments_out[start_idx]
            prev_prev = segments_out[start_idx - 1]
            if str(prev_prev.get("speaker", "") or "") != prev_spk:
                break
            gap = max(
                0.0,
                float(prev.get("start", 0.0) or 0.0)
                - float(prev_prev.get("end", 0.0) or 0.0),
            )
            prefix_start = float(prev_prev.get("start", 0.0) or 0.0)
            if gap > 0.15 or curr_end - prefix_start > prefix_max + curr_dur:
                break
            start_idx -= 1

        prefix = segments_out[start_idx:idx]
        prefix_start = float(prefix[0].get("start", 0.0) or 0.0)
        prefix_end = float(prefix[-1].get("end", 0.0) or 0.0)
        prefix_dur = max(0.0, prefix_end - prefix_start)
        if prefix_dur <= 0 or prefix_dur > prefix_max:
            continue

        gap = max(0.0, curr_start - prefix_end)
        if gap > 0.15:
            continue

        prefix_text = " ".join(str(seg.get("text", "") or "").strip() for seg in prefix).strip()
        curr_text = str(curr.get("text", "") or "").strip()
        if not _strict_text_continuation(prefix_text, curr_text):
            continue

        prefix_probs = [_segment_mean_word_probability(seg) for seg in prefix]
        prefix_prob = sum(prefix_probs) / len(prefix_probs) if prefix_probs else 0.0
        curr_prob = _segment_mean_word_probability(curr)
        if prefix_prob > prefix_max_prob or curr_prob < prefix_prob + min_conf_gain:
            continue

        for seg in prefix:
            _set_segment_speaker(seg, curr_spk)


def _merge_same_utterance_aba_within_segment(
    local_specs: list[tuple[Any, float, float, str, list[Any]]],
) -> list[tuple[Any, float, float, str, list[Any]]]:
    flank_max = _pyannote_within_segment_aba_flank_max_sec()
    center_min = _pyannote_within_segment_aba_center_min_sec()
    if flank_max <= 0 or center_min <= 0 or len(local_specs) < 3:
        return local_specs

    merged = list(local_specs)
    changed = True
    while changed and len(merged) >= 3:
        changed = False
        out: list[tuple[Any, float, float, str, list[Any]]] = []
        i = 0
        while i < len(merged):
            if i + 2 >= len(merged):
                out.extend(merged[i:])
                break
            a = merged[i]
            b = merged[i + 1]
            c = merged[i + 2]
            seg_a, a0, a1, spk_a, wa = a
            seg_b, b0, b1, spk_b, wb = b
            seg_c, c0, c1, spk_c, wc = c
            a_dur = max(0.0, a1 - a0)
            b_dur = max(0.0, b1 - b0)
            c_dur = max(0.0, c1 - c0)
            left_text = _join_whisper_words(wa)
            mid_text = _join_whisper_words(wb)
            right_text = _join_whisper_words(wc)
            if (
                seg_a is seg_b is seg_c
                and spk_a == spk_c
                and spk_a != spk_b
                and a_dur <= flank_max
                and c_dur <= flank_max
                and b_dur >= center_min
                and _soft_text_continuation(left_text, mid_text)
                and _soft_text_continuation(mid_text, right_text)
            ):
                out.append((seg_a, a0, c1, spk_b, wa + wb + wc))
                i += 3
                changed = True
                continue
            out.append(a)
            i += 1
        merged = out
    return merged


def _expand_segments_from_words(
    seg_list: list[Any],
    speaker_for_word: Callable[[Any], str],
    speaker_for_empty_segment: Callable[[Any], str],
    pause_sec: float,
) -> list[tuple[Any, float, float, str, list[Any]]]:
    """
    Режет сегменты Whisper по смене спикера и (опционально) по длинной паузе между словами.
    """
    specs: list[tuple[Any, float, float, str, list[Any]]] = []
    for seg in seg_list:
        t0 = float(seg.start)
        t1 = float(seg.end)
        words = getattr(seg, "words", None)
        if not words:
            spk = speaker_for_empty_segment(seg)
            specs.append((seg, t0, t1, spk, []))
            continue

        current_spk: str | None = None
        bucket: list[Any] = []
        prev_end: float | None = None
        local_specs: list[tuple[Any, float, float, str, list[Any]]] = []

        def flush() -> None:
            nonlocal bucket, current_spk
            if not bucket or current_spk is None:
                return
            wt0 = float(bucket[0].start)
            wt1 = float(bucket[-1].end)
            local_specs.append((seg, wt0, wt1, current_spk, bucket[:]))

        for w in words:
            w0 = float(w.start)
            w1 = float(w.end)
            spk = speaker_for_word(w)
            gap = (w0 - prev_end) if prev_end is not None else 0.0
            pause_break = pause_sec > 0 and prev_end is not None and gap > pause_sec

            if current_spk is None:
                current_spk = spk
                bucket = [w]
            elif pause_break:
                flush()
                current_spk = spk
                bucket = [w]
            elif spk == current_spk:
                bucket.append(w)
            else:
                flush()
                current_spk = spk
                bucket = [w]
            prev_end = w1
        flush()
        specs.extend(_merge_same_utterance_aba_within_segment(local_specs))
    return specs


def _smooth_speaker_string_labels(labels: list[str], radius: int) -> list[str]:
    """Модальная метка в окне — соседние слова/сегменты тянут к одному SPEAKER_* при кратких ошибках."""
    n = len(labels)
    if n == 0 or radius <= 0:
        return list(labels)
    out: list[str] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append(Counter(labels[lo:hi]).most_common(1)[0][0])
    return out


def _apply_segment_speaker_smoothing(
    segments_out: list[dict[str, Any]],
    radius: int,
) -> None:
    """После сборки JSON: сгладить последовательность спикеров по строкам и обновить words[].speaker."""
    if radius <= 0 or len(segments_out) < 2:
        return
    seq = [str(s.get("speaker", "") or "") for s in segments_out]
    sm = _smooth_speaker_string_labels(seq, radius=radius)
    for i, row in enumerate(segments_out):
        spk = sm[i]
        row["speaker"] = spk
        for w in row.get("words") or []:
            if isinstance(w, dict):
                w["speaker"] = spk


def _merge_fragmented_pyannote_speakers(segments_out: list[dict[str, Any]]) -> None:
    """
    Склеивает короткие «фантомные» SPEAKER_* у pyannote в соседний стабильный спикер.
    Это снижает дробление одного реального голоса на множество кластеров.
    """
    raw = os.environ.get("PYANNOTE_MERGE_SHORT_SPEAKERS", "").strip().lower()
    if raw in ("0", "false", "no"):
        return
    if raw not in ("", "1", "true", "yes"):
        return
    if len(segments_out) < 3:
        return

    speakers = [str(s.get("speaker", "") or "") for s in segments_out]
    uniq = sorted({sp for sp in speakers if sp})
    try:
        min_distinct = max(
            3, int(os.environ.get("PYANNOTE_MERGE_MIN_DISTINCT_SPEAKERS", "4"))
        )
    except ValueError:
        min_distinct = 4
    if len(uniq) < min_distinct:
        return

    try:
        max_total_sec = float(
            os.environ.get("PYANNOTE_MERGE_SHORT_SPEAKER_MAX_SEC", "12")
        )
    except ValueError:
        max_total_sec = 12.0
    try:
        max_runs = max(1, int(os.environ.get("PYANNOTE_MERGE_SHORT_SPEAKER_MAX_RUNS", "3")))
    except ValueError:
        max_runs = 3

    durations: dict[str, float] = {}
    run_counts: Counter[str] = Counter()
    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < len(segments_out):
        spk = speakers[i]
        j = i + 1
        run_dur = max(
            0.0,
            float(segments_out[i].get("end", 0.0)) - float(segments_out[i].get("start", 0.0)),
        )
        while j < len(segments_out) and speakers[j] == spk:
            run_dur += max(
                0.0,
                float(segments_out[j].get("end", 0.0))
                - float(segments_out[j].get("start", 0.0)),
            )
            j += 1
        durations[spk] = durations.get(spk, 0.0) + run_dur
        run_counts[spk] += 1
        runs.append((i, j, spk))
        i = j

    fragmented = {
        spk
        for spk, total_sec in durations.items()
        if spk and total_sec <= max_total_sec and run_counts[spk] <= max_runs
    }
    if not fragmented:
        return

    for run_idx, (start, end, spk) in enumerate(runs):
        if spk not in fragmented:
            continue
        prev_spk = runs[run_idx - 1][2] if run_idx > 0 else ""
        next_spk = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else ""
        replacement = ""
        if prev_spk and prev_spk == next_spk and prev_spk != spk:
            replacement = prev_spk
        elif prev_spk and prev_spk != spk and prev_spk not in fragmented:
            replacement = prev_spk
        elif next_spk and next_spk != spk and next_spk not in fragmented:
            replacement = next_spk
        if not replacement:
            continue
        for idx in range(start, end):
            segments_out[idx]["speaker"] = replacement
            for w in segments_out[idx].get("words") or []:
                if isinstance(w, dict):
                    w["speaker"] = replacement


def _bridge_short_pyannote_runs(segments_out: list[dict[str, Any]]) -> None:
    """
    Склеивает короткий ложный run между двумя одинаковыми спикерами: A B A -> A A A.
    Полезно, когда pyannote на границе реплик даёт короткую ошибочную вставку.
    """
    max_sec = _pyannote_bridge_short_run_sec()
    if max_sec <= 0 or len(segments_out) < 3:
        return
    uniq = {
        str(seg.get("speaker", "") or "")
        for seg in segments_out
        if str(seg.get("speaker", "") or "")
    }
    if len(uniq) < _pyannote_bridge_min_distinct_speakers():
        return

    for _ in range(3):
        changed = False
        runs: list[tuple[int, int, str, float]] = []
        i = 0
        while i < len(segments_out):
            spk = str(segments_out[i].get("speaker", "") or "")
            j = i + 1
            dur = max(
                0.0,
                float(segments_out[i].get("end", 0.0))
                - float(segments_out[i].get("start", 0.0)),
            )
            while j < len(segments_out) and str(segments_out[j].get("speaker", "") or "") == spk:
                dur += max(
                    0.0,
                    float(segments_out[j].get("end", 0.0))
                    - float(segments_out[j].get("start", 0.0)),
                )
                j += 1
            runs.append((i, j, spk, dur))
            i = j

        for idx in range(1, len(runs) - 1):
            start, end, spk, dur = runs[idx]
            prev_spk = runs[idx - 1][2]
            next_spk = runs[idx + 1][2]
            if not spk or spk == prev_spk or spk == next_spk:
                continue
            if prev_spk != next_spk:
                continue
            if dur > max_sec:
                continue
            for seg_idx in range(start, end):
                segments_out[seg_idx]["speaker"] = prev_spk
                for w in segments_out[seg_idx].get("words") or []:
                    if isinstance(w, dict):
                        w["speaker"] = prev_spk
            changed = True
        if not changed:
            break


def _majority_smooth_int_clusters(labels: list[int], radius: int = 4) -> list[int]:
    """Сглаживание меток кластера по окну — один голос меньше «дребезжит» между SPEAKER_*."""
    n = len(labels)
    if n == 0:
        return []
    out: list[int] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append(Counter(labels[lo:hi]).most_common(1)[0][0])
    return out


def _expand_pyannote_word_segments(
    seg_list: list[Any],
    diar_rows: list[tuple[float, float, str]],
) -> list[tuple[Any, float, float, str, list[Any]]]:
    """
    Pyannote + голосование по слову; сглаживание меток по потоку слов (PYANNOTE_LABEL_SMOOTH_RADIUS);
    опционально разрез по паузе (TRANSCRIPT_PAUSE_SPLIT_SEC).
    """
    pause = _pause_split_sec()
    flat: list[tuple[Any, Any]] = []
    for seg in seg_list:
        for w in getattr(seg, "words", None) or []:
            flat.append((seg, w))

    if not flat:
        specs: list[tuple[Any, float, float, str, list[Any]]] = []
        for seg in seg_list:
            t0, t1 = float(seg.start), float(seg.end)
            spk = _speaker_for_interval(t0, t1, diar_rows)
            specs.append((seg, t0, t1, spk, []))
        return specs

    raw = [_speaker_for_word_voted(w, diar_rows) for _, w in flat]
    rad = _pyannote_label_smooth_radius()
    smooth_lbl = _smooth_speaker_string_labels(raw, radius=rad) if rad > 0 else raw
    word_to_spk = {id(w): smooth_lbl[i] for i, (_, w) in enumerate(flat)}

    def spk_w(w: Any) -> str:
        return word_to_spk[id(w)]

    return _expand_segments_from_words(
        seg_list,
        spk_w,
        lambda s: _speaker_for_interval(float(s.start), float(s.end), diar_rows),
        pause,
    )


def _expand_mfcc_word_segments(
    seg_list: list[Any],
    y: Any,
    sr: int,
) -> list[tuple[Any, float, float, str, list[Any]]]:
    """
    Локальная диаризация: MFCC на каждом слове + KMeans (до MFCC_DIAR_MAX_SPEAKERS кластеров),
    затем тот же разрез по смене метки и паузе, что и для pyannote.
    """
    import librosa
    import numpy as np
    from sklearn.cluster import KMeans

    def _vec_for_interval(t0: float, t1: float) -> np.ndarray:
        i0 = max(0, int(t0 * sr))
        i1 = max(i0 + int(0.08 * sr), int(t1 * sr))
        chunk = y[i0:i1]
        min_samp = int(0.08 * sr)
        if chunk.size < min_samp:
            chunk = np.pad(chunk, (0, min_samp - chunk.size))
        n_fft = min(2048, max(256, len(chunk)))
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20, n_fft=n_fft)
        return np.mean(mfcc, axis=1)

    flat: list[tuple[Any, Any]] = []
    feats: list[np.ndarray] = []
    for seg in seg_list:
        for w in getattr(seg, "words", None) or []:
            w0 = float(w.start)
            w1 = float(w.end)
            feats.append(_vec_for_interval(w0, w1))
            flat.append((seg, w))

    if not feats:
        return []

    n = len(feats)
    k = min(_mfcc_max_clusters(), n)
    X = np.stack(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    lab_list = [int(x) for x in km.labels_]
    try:
        rad = max(2, int(os.environ.get("MFCC_WORD_SMOOTH_RADIUS", "4")))
    except ValueError:
        rad = 4
    lab_smooth = _majority_smooth_int_clusters(lab_list, radius=rad)
    word_to_raw = {id(w): f"mfcc_c{lab_smooth[i]}" for i, (_, w) in enumerate(flat)}

    def _spk_w(w: Any) -> str:
        return word_to_raw[id(w)]

    def _spk_seg(seg: Any) -> str:
        t0, t1 = float(seg.start), float(seg.end)
        v = _vec_for_interval(t0, t1).reshape(1, -1)
        cid = int(km.predict(v)[0])
        return f"mfcc_c{cid}"

    pause = _pause_split_sec()
    return _expand_segments_from_words(seg_list, _spk_w, _spk_seg, pause)


import re as _re

_GREETING_PATTERNS = _re.compile(
    r"(здравствуйте|добрый\s+(день|вечер|утро)|приветствую|добро\s+пожаловать"
    r"|компани[яи]|автосалон|салон|дилер|менеджер|консультант|меня\s+зовут)",
    _re.IGNORECASE,
)
_CLIENT_PATTERNS = _re.compile(
    r"(хочу\s+(купить|посмотреть|узнать)|интересует|подскажите|сколько\s+стоит"
    r"|мне\s+нужн[оа]|для\s+себя|ищу\s+авто)",
    _re.IGNORECASE,
)


def _identify_employee_speaker(segments: list[dict[str, Any]]) -> dict[str, str]:
    """Heuristic role assignment: employee (greets, presents) vs client.

    Returns mapping like {"SPEAKER_01": "EMPLOYEE", "SPEAKER_02": "CLIENT"}.
    """
    speakers = sorted({s.get("speaker", "") for s in segments if s.get("speaker")})
    if len(speakers) < 2:
        return {}

    score: dict[str, float] = {sp: 0.0 for sp in speakers}

    for i, seg in enumerate(segments):
        sp = seg.get("speaker", "")
        if sp not in score:
            continue
        text = seg.get("text", "")

        if _GREETING_PATTERNS.search(text):
            score[sp] += 3.0
        if _CLIENT_PATTERNS.search(text):
            score[sp] -= 2.0

        if i < 3:
            score[sp] += 1.0

    dur: dict[str, float] = {sp: 0.0 for sp in speakers}
    for seg in segments:
        sp = seg.get("speaker", "")
        if sp in dur:
            dur[sp] += max(seg.get("end", 0) - seg.get("start", 0), 0)
    if dur:
        most_talking = max(dur, key=lambda k: dur[k])
        score[most_talking] += 1.0

    ranked = sorted(score, key=lambda k: -score[k])
    employee = ranked[0]
    roles: dict[str, str] = {}
    for sp in speakers:
        roles[sp] = "EMPLOYEE" if sp == employee else "CLIENT"
    return roles


def _apply_speaker_roles(
    segments: list[dict[str, Any]], roles: dict[str, str]
) -> None:
    """Add 'speaker_role' field to each segment."""
    for seg in segments:
        sp = seg.get("speaker", "")
        if sp in roles:
            seg["speaker_role"] = roles[sp]


def _speaker_alignment_report(
    segments: list[dict[str, Any]],
    *,
    expected_speaker_count: int | None,
) -> dict[str, Any]:
    speakers = [str(seg.get("speaker") or "") for seg in segments if str(seg.get("speaker") or "")]
    unique = sorted(set(speakers))
    switches = sum(1 for left, right in zip(speakers, speakers[1:]) if left != right)
    short_fragments = 0
    words_total = 0
    words_with_timestamps = 0
    by_speaker: dict[str, float] = {}
    for seg in segments:
        spk = str(seg.get("speaker") or "")
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        dur = max(0.0, end - start)
        if spk:
            by_speaker[spk] = by_speaker.get(spk, 0.0) + dur
        if dur < _float_env("FA_ASR_SPEAKER_SHORT_FRAGMENT_SEC", 0.8):
            short_fragments += 1
        words = seg.get("words") or []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                words_total += 1
                if word.get("start") is not None and word.get("end") is not None:
                    words_with_timestamps += 1
    expected_delta = None
    if expected_speaker_count:
        expected_delta = len(unique) - int(expected_speaker_count)
    words_pct = words_with_timestamps / max(1, words_total) * 100.0
    switch_rate = switches / max(1, len(segments) - 1)
    status = "pass"
    if words_pct < 95.0 or (expected_delta is not None and expected_delta >= 2) or switch_rate > 0.65:
        status = "warn"
    return {
        "schema_version": 1,
        "status": status,
        "speakers_count": len(unique),
        "expected_speaker_count": expected_speaker_count,
        "speaker_count_delta": expected_delta,
        "speaker_switches": switches,
        "speaker_switch_rate": round(switch_rate, 4),
        "short_speaker_fragment_count": short_fragments,
        "words_count": words_total,
        "words_with_timestamps_pct": round(words_pct, 2),
        "duration_by_speaker": {key: round(val, 3) for key, val in sorted(by_speaker.items())},
    }


def _use_transcribe_subprocess() -> bool:
    """
    Распознавание в отдельном процессе: при отмене задачи родитель делает terminate(),
    и загрузка CPU прекращается (иначе faster-whisper может долго не выходить в Python
    между сегментами). На Linux тоже включено по умолчанию: так модель Whisper и её
    CUDA-память освобождаются после каждого ASR-прохода, а веб-сервер не копит VRAM.
    Выкл.: FA_TRANSCRIBE_SUBPROCESS=0.
    """
    v = os.environ.get("FA_TRANSCRIBE_SUBPROCESS", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    return True


def _whisper_chunk_length_kw() -> dict[str, Any]:
    """
    По умолчанию не передаём chunk_length — используется дефолт faster-whisper (~30 с).
    Явно задать окно: FA_WHISPER_CHUNK_LENGTH_SEC=20 (редко нужно; влияет на качество).
    """
    raw = os.environ.get("FA_WHISPER_CHUNK_LENGTH_SEC", "").strip()
    if not raw or raw in ("0", "default"):
        return {}
    try:
        n = int(raw)
        if 8 <= n <= 60:
            return {"chunk_length": n}
    except ValueError:
        pass
    return {}


def _whisper_condition_on_previous_text() -> bool:
    """
    Не переносим предыдущий ASR-текст в следующее окно по умолчанию: на длинных
    звонках Whisper иначе может зациклить уверенно распознанную фразу до конца аудио.
    """
    raw = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False


@dataclass(frozen=True)
class AsrProfile:
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _asr_profile_name() -> str:
    raw = os.environ.get("FA_ASR_PROFILE", "balanced").strip().lower()
    if raw == "windowed":
        return "windowed_strict"
    if raw in ("accurate", "rescue", "balanced", "windowed_strict", "windowed_enhanced"):
        return raw
    return "balanced"


def _asr_profile(profile_name: str | None = None) -> AsrProfile:
    name = (profile_name or _asr_profile_name()).strip().lower()
    base: dict[str, Any] = {
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": _whisper_condition_on_previous_text(),
    }
    base.update(_whisper_chunk_length_kw())
    if name == "accurate":
        base.update(
            {
                "beam_size": _int_env("FA_ASR_ACCURATE_BEAM_SIZE", 3),
                "best_of": _int_env("FA_ASR_ACCURATE_BEST_OF", 3),
                "temperature": [0.0, 0.2, 0.4],
                "compression_ratio_threshold": _float_env(
                    "FA_ASR_ACCURATE_COMPRESSION_RATIO_THRESHOLD", 2.4
                ),
                "log_prob_threshold": _float_env("FA_ASR_ACCURATE_LOG_PROB_THRESHOLD", -1.0),
                "no_speech_threshold": _float_env("FA_ASR_ACCURATE_NO_SPEECH_THRESHOLD", 0.6),
                "hallucination_silence_threshold": _float_env(
                    "FA_ASR_ACCURATE_HALLUCINATION_SILENCE_THRESHOLD", 2.0
                ),
                "vad_parameters": {
                    "min_silence_duration_ms": _int_env(
                        "FA_ASR_ACCURATE_VAD_MIN_SILENCE_MS", 450
                    ),
                    "speech_pad_ms": _int_env("FA_ASR_ACCURATE_VAD_SPEECH_PAD_MS", 500),
                },
            }
        )
        return AsrProfile(name="accurate", options=base, description="higher quality full pass")
    if name == "rescue":
        base.update(
            {
                "beam_size": _int_env("FA_ASR_RESCUE_BEAM_SIZE", 3),
                "best_of": _int_env("FA_ASR_RESCUE_BEST_OF", 3),
                "temperature": [0.0, 0.2],
                "vad_filter": os.environ.get("FA_ASR_RESCUE_VAD_FILTER", "0").strip().lower()
                in ("1", "true", "yes", "on"),
                "compression_ratio_threshold": _float_env(
                    "FA_ASR_RESCUE_COMPRESSION_RATIO_THRESHOLD", 2.8
                ),
                "log_prob_threshold": _float_env("FA_ASR_RESCUE_LOG_PROB_THRESHOLD", -1.6),
                "no_speech_threshold": _float_env("FA_ASR_RESCUE_NO_SPEECH_THRESHOLD", 0.8),
                "condition_on_previous_text": False,
            }
        )
        return AsrProfile(name="rescue", options=base, description="short-window gap recovery")
    if name == "windowed":
        name = "windowed_strict"
    if name == "windowed_strict":
        base.update(
            {
                "beam_size": _int_env("FA_ASR_WINDOWED_STRICT_BEAM_SIZE", 3),
                "best_of": _int_env("FA_ASR_WINDOWED_STRICT_BEST_OF", 3),
                "temperature": [0.0],
                "vad_filter": os.environ.get("FA_ASR_WINDOWED_STRICT_VAD_FILTER", "1").strip().lower()
                in ("1", "true", "yes", "on"),
                "compression_ratio_threshold": _float_env(
                    "FA_ASR_WINDOWED_STRICT_COMPRESSION_RATIO_THRESHOLD", 2.4
                ),
                "log_prob_threshold": _float_env("FA_ASR_WINDOWED_STRICT_LOG_PROB_THRESHOLD", -0.9),
                "no_speech_threshold": _float_env("FA_ASR_WINDOWED_STRICT_NO_SPEECH_THRESHOLD", 0.55),
                "hallucination_silence_threshold": _float_env(
                    "FA_ASR_WINDOWED_STRICT_HALLUCINATION_SILENCE_THRESHOLD", 1.6
                ),
                "condition_on_previous_text": False,
                "vad_parameters": {
                    "min_silence_duration_ms": _int_env(
                        "FA_ASR_WINDOWED_STRICT_VAD_MIN_SILENCE_MS", 350
                    ),
                    "speech_pad_ms": _int_env("FA_ASR_WINDOWED_STRICT_VAD_SPEECH_PAD_MS", 250),
                },
            }
        )
        return AsrProfile(
            name="windowed_strict",
            options=base,
            description="speech-island windowed primary pass",
        )
    if name == "windowed_enhanced":
        base.update(
            {
                "beam_size": _int_env("FA_ASR_WINDOWED_ENHANCED_BEAM_SIZE", 3),
                "best_of": _int_env("FA_ASR_WINDOWED_ENHANCED_BEST_OF", 3),
                "temperature": [0.0, 0.2],
                "vad_filter": os.environ.get("FA_ASR_WINDOWED_ENHANCED_VAD_FILTER", "1").strip().lower()
                in ("1", "true", "yes", "on"),
                "compression_ratio_threshold": _float_env(
                    "FA_ASR_WINDOWED_ENHANCED_COMPRESSION_RATIO_THRESHOLD", 2.8
                ),
                "log_prob_threshold": _float_env("FA_ASR_WINDOWED_ENHANCED_LOG_PROB_THRESHOLD", -1.25),
                "no_speech_threshold": _float_env("FA_ASR_WINDOWED_ENHANCED_NO_SPEECH_THRESHOLD", 0.7),
                "hallucination_silence_threshold": _float_env(
                    "FA_ASR_WINDOWED_ENHANCED_HALLUCINATION_SILENCE_THRESHOLD", 1.8
                ),
                "condition_on_previous_text": False,
            }
        )
        return AsrProfile(
            name="windowed_enhanced",
            options=base,
            description="voice-enhanced windowed difficult-audio pass",
        )
    return AsrProfile(name="balanced", options=base, description="stable default full pass")


def _asr_profile_metadata(profile: AsrProfile, transcribe_kw: dict[str, Any]) -> dict[str, Any]:
    stored = {
        key: value
        for key, value in transcribe_kw.items()
        if key not in {"initial_prompt"}
    }
    if "initial_prompt" in transcribe_kw:
        stored["initial_prompt_present"] = True
    return {
        "name": profile.name,
        "description": profile.description,
        "transcribe_options": stored,
    }


def _should_try_accurate_on_fail() -> bool:
    raw = os.environ.get("FA_ASR_ACCURATE_RERUN_ON_FAIL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _should_run_rescue() -> bool:
    raw = os.environ.get("FA_ASR_RESCUE_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _windowed_fallback_enabled() -> bool:
    raw = os.environ.get("FA_ASR_WINDOWED_FALLBACK_TO_BALANCED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _candidate_router_enabled() -> bool:
    raw = os.environ.get("FA_ASR_CANDIDATE_ROUTER", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _skip_diarization_for_validation() -> bool:
    raw = os.environ.get("FA_SKIP_DIARIZATION", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _rescue_min_mean_word_probability() -> float:
    return _float_env("FA_ASR_RESCUE_MIN_MEAN_WORD_PROB", 0.35)


def _gpu_guard_enabled() -> bool:
    raw = os.environ.get("FA_ASR_GPU_GUARD", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _gpu_snapshot() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return None
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def _wait_for_gpu_capacity(stage: str, *, device: str, cancel_check: Callable[[], None] | None) -> None:
    if device != "cuda" or not _gpu_guard_enabled():
        return
    max_util = _int_env("FA_ASR_GPU_MAX_UTIL_BEFORE_START", 35)
    max_mem = _int_env("FA_ASR_GPU_MAX_USED_MB_BEFORE_START", 6000)
    wait_sec = _float_env("FA_ASR_GPU_WAIT_SEC", 300.0)
    poll_sec = max(1.0, _float_env("FA_ASR_GPU_POLL_SEC", 5.0))
    deadline = time.time() + max(0.0, wait_sec)
    last: tuple[int, int] | None = None
    while True:
        if cancel_check:
            cancel_check()
        snap = _gpu_snapshot()
        if snap is None:
            return
        last = snap
        util, mem = snap
        if util <= max_util and mem <= max_mem:
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"GPU busy before ASR {stage}: util={util}%, mem={mem} MiB "
                f"(limits: util<={max_util}%, mem<={max_mem} MiB)."
            )
        time.sleep(poll_sec)


def _segment_to_candidate(seg: Any, *, offset_sec: float, source_window: dict[str, float]) -> dict[str, Any]:
    start = float(seg.start) + offset_sec
    end = float(seg.end) + offset_sec
    words_out: list[dict[str, Any]] = []
    probs: list[float] = []
    for w in getattr(seg, "words", None) or []:
        p = float(getattr(w, "probability", 0.0) or 0.0)
        probs.append(p)
        words_out.append(
            {
                "start": round(float(w.start) + offset_sec, 3),
                "end": round(float(w.end) + offset_sec, 3),
                "word": str(getattr(w, "word", "") or ""),
                "probability": round(p, 4),
            }
        )
    text = str(getattr(seg, "text", "") or "").strip()
    avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
    compression = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
    no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
    mean_word_prob = sum(probs) / len(probs) if probs else None
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
        "words": words_out,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression,
        "no_speech_prob": no_speech,
        "mean_word_probability": mean_word_prob,
        "source_window": source_window,
        "asr_source": "rescue",
    }


def _candidate_mean_prob(candidate: dict[str, Any]) -> float | None:
    raw = candidate.get("mean_word_probability")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _candidate_overlaps_existing(
    candidate: dict[str, Any],
    existing: list[Any],
    *,
    max_overlap_sec: float = 0.7,
) -> bool:
    c0 = float(candidate.get("start", 0.0) or 0.0)
    c1 = float(candidate.get("end", 0.0) or 0.0)
    if c1 <= c0:
        return True
    for seg in existing:
        if isinstance(seg, dict):
            s0 = float(seg.get("start", 0.0) or 0.0)
            s1 = float(seg.get("end", 0.0) or 0.0)
        else:
            s0 = float(getattr(seg, "start", 0.0) or 0.0)
            s1 = float(getattr(seg, "end", 0.0) or 0.0)
        overlap = max(0.0, min(c1, s1) - max(c0, s0))
        if overlap > max_overlap_sec:
            return True
    return False


def _candidate_passes_rescue_rules(candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("text") or "").strip()
    if len(text) < 4:
        return False
    mean_prob = _candidate_mean_prob(candidate)
    if mean_prob is not None and mean_prob < _rescue_min_mean_word_probability():
        return False
    try:
        if float(candidate.get("compression_ratio") or 0.0) > _float_env(
            "FA_ASR_RESCUE_MAX_COMPRESSION_RATIO", 3.0
        ):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _asr_voice_filter(profile_name: str) -> str:
    if profile_name == "windowed_strict":
        env_name = "FA_ASR_WINDOWED_STRICT_AUDIO_FILTER"
        default = "0"
    elif profile_name in ("windowed", "windowed_enhanced"):
        env_name = "FA_ASR_WINDOWED_ENHANCED_AUDIO_FILTER"
        default = "highpass=f=120,lowpass=f=3800,dynaudnorm=f=150:g=20:p=0.95,volume=8dB"
    else:
        env_name = "FA_ASR_RESCUE_AUDIO_FILTER"
        default = "highpass=f=120,lowpass=f=3800,dynaudnorm=f=150:g=20:p=0.95,volume=8dB"
    legacy = os.environ.get("FA_ASR_WINDOWED_AUDIO_FILTER", "").strip()
    if profile_name in ("windowed", "windowed_enhanced") and legacy:
        return legacy
    return os.environ.get(env_name, default).strip()


def _write_asr_clip(
    wav: Path,
    out_wav: Path,
    start: float,
    end: float,
    *,
    profile_name: str,
) -> None:
    from src.audio_extract import _ffmpeg_executable

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    audio_filter = _asr_voice_filter(profile_name)
    cmd = [
        _ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(0.1, end - start):.3f}",
        "-i",
        str(wav),
    ]
    if audio_filter and audio_filter.lower() not in ("0", "none", "off"):
        cmd.extend(["-af", audio_filter])
    cmd.extend(
        [
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _write_rescue_clip(wav: Path, out_wav: Path, start: float, end: float) -> None:
    _write_asr_clip(wav, out_wav, start, end, profile_name="rescue")


def _rescue_subwindows(window: dict[str, float]) -> list[dict[str, float]]:
    gap_start = float(window.get("gap_start", window["start"]))
    gap_end = float(window.get("gap_end", window["end"]))
    if gap_end <= gap_start:
        return [window]
    right_pad = _float_env("FA_ASR_RESCUE_INNER_RIGHT_PAD_SEC", 0.5)
    span = _float_env("FA_ASR_RESCUE_SLIDING_WINDOW_SEC", 28.0)
    step = _float_env("FA_ASR_RESCUE_SLIDING_STEP_SEC", 18.0)
    windows: list[dict[str, float]] = []

    first = {"start": max(0.0, gap_start), "end": gap_end + right_pad}
    windows.append(first)
    if first["end"] - first["start"] > span:
        cursor = max(0.0, gap_start)
        while cursor < gap_end:
            end = min(gap_end + right_pad, cursor + span)
            windows.append({"start": cursor, "end": end})
            if end >= gap_end + right_pad:
                break
            cursor += max(1.0, step)

    out: list[dict[str, float]] = []
    for item in windows:
        start = max(0.0, float(item["start"]))
        end = float(item["end"])
        if end - start >= 1.0:
            out.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "parent_start": float(window["start"]),
                    "parent_end": float(window["end"]),
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                }
            )
    return out


def _run_whisper(
    wav: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    profile: AsrProfile,
    lang: str | None,
    prompt: str | None,
    cancel_check: Callable[[], None] | None,
    use_subprocess: bool | None = None,
) -> tuple[list[Any], str, dict[str, Any]]:
    _wait_for_gpu_capacity(profile.name, device=device, cancel_check=cancel_check)
    transcribe_kw = dict(profile.options)
    if lang:
        transcribe_kw["language"] = lang
    if prompt:
        transcribe_kw["initial_prompt"] = prompt

    if use_subprocess if use_subprocess is not None else _use_transcribe_subprocess():
        seg_list, detected_lang = _transcribe_segments_subprocess(
            wav, model_name, device, compute_type, transcribe_kw, cancel_check
        )
        return seg_list, detected_lang or lang or "ru", transcribe_kw

    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_gen, info = model.transcribe(str(wav), **transcribe_kw)
    detected_lang = (info.language or lang or "ru") if info else (lang or "ru")
    seg_list = []
    for seg in segments_gen:
        if cancel_check:
            cancel_check()
        seg_list.append(seg)
    return seg_list, detected_lang, transcribe_kw


def _preview_transcript_for_quality(
    seg_list: list[Any],
    *,
    duration_sec: float,
    expected_speaker_count: int | None,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for seg in seg_list:
        word_probs: list[float] | None = None
        words = getattr(seg, "words", None)
        if words:
            word_probs = [
                float(getattr(word, "probability", 0.0) or 0.0)
                for word in words
            ]
        text = str(getattr(seg, "text", "") or "").strip()
        t0 = float(getattr(seg, "start", 0.0) or 0.0)
        t1 = float(getattr(seg, "end", 0.0) or 0.0)
        item: dict[str, Any] = {
            "start": round(t0, 3),
            "end": round(t1, 3),
            "speaker": "SPEAKER_01",
            "text": text,
            "words": [
                {
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "word": str(getattr(word, "word", "") or "").strip(),
                    "probability": round(float(getattr(word, "probability", 0.0) or 0.0), 4),
                }
                for word in (words or [])
            ],
            "delivery": analyze_segment(
                text=text,
                t0=t0,
                t1=t1,
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                compression_ratio=float(getattr(seg, "compression_ratio", 0.0) or 0.0),
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                word_probs=word_probs,
            ),
        }
        asr_source = getattr(seg, "asr_source", "")
        if asr_source:
            item["asr_source"] = asr_source
        if getattr(seg, "asr_window_id", None) is not None:
            item["asr_window_id"] = getattr(seg, "asr_window_id")
        if getattr(seg, "asr_window", None):
            item["asr_window"] = getattr(seg, "asr_window")
        segments.append(item)
    return transcript_quality_report(
        {
            "duration_sec": duration_sec,
            "segments": segments,
            "expected_speaker_count": expected_speaker_count,
        },
        expected_speaker_count=expected_speaker_count,
        duration_sec=duration_sec,
    )


def _quality_risk(quality: dict[str, Any]) -> float:
    try:
        return float(quality.get("risk_score") or 0.0)
    except (TypeError, ValueError):
        return 999.0


def _quality_max_gap(quality: dict[str, Any]) -> float:
    summary = quality.get("summary") or {}
    try:
        return float(summary.get("max_gap_sec") or 0.0)
    except (TypeError, ValueError):
        return 999.0


def _quality_coverage(quality: dict[str, Any]) -> float:
    summary = quality.get("summary") or {}
    try:
        return float(summary.get("coverage_pct") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_quality_better_or_equal(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    risk_margin = _float_env("FA_ASR_ROUTER_MAX_RISK_REGRESSION", 4.0)
    coverage_margin = _float_env("FA_ASR_ROUTER_MAX_COVERAGE_DROP", 2.0)
    gap_margin = _float_env("FA_ASR_ROUTER_MAX_GAP_INCREASE", 1.5)
    cand_risk = _quality_risk(candidate)
    base_risk = _quality_risk(baseline)
    if cand_risk > base_risk + risk_margin:
        return False
    if _quality_coverage(candidate) < _quality_coverage(baseline) - coverage_margin:
        return False
    if _quality_max_gap(candidate) > _quality_max_gap(baseline) + gap_margin:
        return False
    return cand_risk <= base_risk or candidate.get("status") in ("pass", "warn")


def _rescue_decode_windows(
    wav: Path,
    windows: list[dict[str, float]],
    *,
    model_name: str,
    device: str,
    compute_type: str,
    lang: str | None,
    prompt: str | None,
    cancel_check: Callable[[], None] | None,
    existing_segments: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not windows:
        return [], {"attempted": False, "windows": []}

    profile = _asr_profile("rescue")
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fa_asr_rescue_") as tmp:
        tmpdir = Path(tmp)
        expanded_windows: list[dict[str, float]] = []
        for window in windows:
            expanded_windows.extend(_rescue_subwindows(window))
        for idx, window in enumerate(expanded_windows):
            if cancel_check:
                cancel_check()
            start = float(window["start"])
            end = float(window["end"])
            clip = tmpdir / f"rescue_{idx:03d}.wav"
            _write_rescue_clip(wav, clip, start, end)
            seg_list, _detected, transcribe_kw = _run_whisper(
                clip,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                profile=profile,
                lang=lang,
                prompt=prompt,
                cancel_check=cancel_check,
                use_subprocess=True,
            )
            window_candidates = [
                _segment_to_candidate(seg, offset_sec=start, source_window=window)
                for seg in seg_list
            ]
            accepted_for_window: list[dict[str, Any]] = []
            for candidate in window_candidates:
                if _candidate_overlaps_existing(candidate, existing_segments):
                    continue
                if _candidate_overlaps_existing(candidate, accepted):
                    continue
                if not _candidate_passes_rescue_rules(candidate):
                    continue
                accepted.append(candidate)
                accepted_for_window.append(candidate)
            attempts.append(
                {
                    "window": window,
                    "profile": _asr_profile_metadata(profile, transcribe_kw),
                    "candidates": len(window_candidates),
                    "accepted": len(accepted_for_window),
                    "accepted_ranges": [
                        {"start": item["start"], "end": item["end"]}
                        for item in accepted_for_window
                    ],
                }
            )

    return accepted, {
        "attempted": True,
        "profile": profile.name,
        "source_windows": windows,
        "windows": attempts,
        "accepted_segments": len(accepted),
    }


def _windowed_window_sec() -> float:
    return max(8.0, _float_env("FA_ASR_WINDOWED_WINDOW_SEC", 28.0))


def _windowed_overlap_sec() -> float:
    return max(0.0, _float_env("FA_ASR_WINDOWED_OVERLAP_SEC", 3.0))


def _speech_islands_enabled() -> bool:
    raw = os.environ.get("FA_ASR_SPEECH_ISLANDS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _speech_islands(
    y_audio: Any,
    sr_audio: int,
    *,
    duration_sec: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if not _speech_islands_enabled() or sr_audio <= 0 or duration_sec <= 0:
        window = {"index": 0.0, "start": 0.0, "end": round(duration_sec, 3), "speech_ratio": 1.0}
        return [window], {"enabled": False, "islands_count": 1}

    import numpy as np

    frame_sec = max(0.02, _float_env("FA_ASR_SPEECH_FRAME_SEC", 0.1))
    hop = max(1, int(frame_sec * sr_audio))
    min_speech_sec = _float_env("FA_ASR_SPEECH_MIN_ISLAND_SEC", 0.35)
    merge_gap_sec = _float_env("FA_ASR_SPEECH_MERGE_GAP_SEC", 3.2)
    pad_sec = _float_env("FA_ASR_SPEECH_PAD_SEC", 1.2)
    max_island_sec = _float_env("FA_ASR_SPEECH_MAX_ISLAND_SEC", _windowed_window_sec())
    threshold_ratio = _float_env("FA_ASR_SPEECH_RMS_THRESHOLD_RATIO", 0.08)
    min_coverage_pct = _float_env("FA_ASR_SPEECH_MIN_COVERAGE_PCT", 74.0)

    y = np.asarray(y_audio, dtype=np.float32)
    if y.size == 0:
        return [], {"enabled": True, "islands_count": 0, "reason": "empty_audio"}

    rms_rows: list[tuple[float, float, float]] = []
    for start_idx in range(0, len(y), hop):
        chunk = y[start_idx : start_idx + hop]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        start = start_idx / sr_audio
        end = min(duration_sec, (start_idx + chunk.size) / sr_audio)
        rms_rows.append((start, end, rms))
    if not rms_rows:
        return [], {"enabled": True, "islands_count": 0, "reason": "no_frames"}

    vals = np.array([row[2] for row in rms_rows], dtype=np.float32)
    noise = float(np.percentile(vals, 20))
    peak = float(np.percentile(vals, 95))
    threshold = max(noise * 1.8, peak * threshold_ratio, 1e-6)
    raw: list[list[float]] = []
    for start, end, rms in rms_rows:
        if rms < threshold:
            continue
        if not raw or start - raw[-1][1] > merge_gap_sec:
            raw.append([start, end])
        else:
            raw[-1][1] = end

    padded: list[list[float]] = []
    for start, end in raw:
        if end - start < min_speech_sec:
            continue
        start = max(0.0, start - pad_sec)
        end = min(duration_sec, end + pad_sec)
        if padded and start - padded[-1][1] <= merge_gap_sec:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])

    islands: list[dict[str, float]] = []
    idx = 0
    overlap = min(_windowed_overlap_sec(), max(0.0, max_island_sec - 1.0))
    step = max(1.0, max_island_sec - overlap)
    for start, end in padded:
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_island_sec)
            if chunk_end - cursor >= min_speech_sec:
                span_rows = [row for row in rms_rows if row[0] < chunk_end and row[1] > cursor]
                active = [row for row in span_rows if row[2] >= threshold]
                speech_ratio = len(active) / max(1, len(span_rows))
                mean_rms = sum(row[2] for row in span_rows) / max(1, len(span_rows))
                islands.append(
                    {
                        "index": float(idx),
                        "start": round(cursor, 3),
                        "end": round(chunk_end, 3),
                        "speech_ratio": round(float(speech_ratio), 4),
                        "rms": round(float(mean_rms), 6),
                    }
                )
                idx += 1
            if chunk_end >= end:
                break
            cursor += step

    covered = sum(float(item["end"]) - float(item["start"]) for item in islands)
    coverage_pct = round(covered / duration_sec * 100.0, 2) if duration_sec > 0 else 0.0
    report = {
        "enabled": True,
        "islands_count": len(islands),
        "covered_sec": round(covered, 3),
        "coverage_pct": coverage_pct,
        "rms_noise_floor": round(noise, 6),
        "rms_peak": round(peak, 6),
        "rms_threshold": round(threshold, 6),
        "frame_sec": frame_sec,
        "merge_gap_sec": merge_gap_sec,
        "pad_sec": pad_sec,
    }
    if not islands or coverage_pct < min_coverage_pct:
        fixed = _windowed_windows(duration_sec)
        for window in fixed:
            start = float(window["start"])
            end = float(window["end"])
            span_rows = [row for row in rms_rows if row[0] < end and row[1] > start]
            active = [row for row in span_rows if row[2] >= threshold]
            window["speech_ratio"] = round(len(active) / max(1, len(span_rows)), 4)
            window["rms"] = round(sum(row[2] for row in span_rows) / max(1, len(span_rows)), 6)
        report["fallback"] = "fixed_windows_low_speech_coverage"
        report["fallback_windows_count"] = len(fixed)
        return fixed, report
    return islands, report


def _windowed_windows(duration_sec: float) -> list[dict[str, float]]:
    if duration_sec <= 0:
        return []
    window_sec = _windowed_window_sec()
    overlap_sec = min(_windowed_overlap_sec(), max(0.0, window_sec - 1.0))
    step = max(1.0, window_sec - overlap_sec)
    windows: list[dict[str, float]] = []
    start = 0.0
    idx = 0
    while start < duration_sec:
        end = min(duration_sec, start + window_sec)
        if end - start >= 1.0:
            windows.append(
                {
                    "index": float(idx),
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
            idx += 1
        if end >= duration_sec:
            break
        start += step
    return windows


def _candidate_overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    l0 = float(left.get("start", 0.0) or 0.0)
    l1 = float(left.get("end", 0.0) or 0.0)
    r0 = float(right.get("start", 0.0) or 0.0)
    r1 = float(right.get("end", 0.0) or 0.0)
    return max(0.0, min(l1, r1) - max(l0, r0))


def _candidate_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = _candidate_overlap_seconds(left, right)
    if overlap <= 0:
        return 0.0
    left_dur = max(0.01, float(left.get("end", 0.0) or 0.0) - float(left.get("start", 0.0) or 0.0))
    right_dur = max(0.01, float(right.get("end", 0.0) or 0.0) - float(right.get("start", 0.0) or 0.0))
    return overlap / min(left_dur, right_dur)


def _candidate_text_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    l_text = str(left.get("text") or "").strip().lower()
    r_text = str(right.get("text") or "").strip().lower()
    if not l_text or not r_text:
        return 0.0
    return SequenceMatcher(None, l_text, r_text).ratio()


def _candidate_score(candidate: dict[str, Any]) -> float:
    mean_prob = _candidate_mean_prob(candidate)
    avg_logprob = float(candidate.get("avg_logprob", 0.0) or 0.0)
    compression = float(candidate.get("compression_ratio", 0.0) or 0.0)
    dur = max(0.01, float(candidate.get("end", 0.0) or 0.0) - float(candidate.get("start", 0.0) or 0.0))
    score = avg_logprob
    if mean_prob is not None:
        score += mean_prob
    score -= max(0.0, compression - 2.0) * 0.15
    score += min(dur, 8.0) * 0.01
    return score


def _candidate_has_low_confidence(candidate: dict[str, Any]) -> bool:
    mean_prob = _candidate_mean_prob(candidate)
    if mean_prob is not None and mean_prob < _float_env("FA_ASR_WINDOWED_MIN_MEAN_WORD_PROB", 0.35):
        return True
    try:
        if float(candidate.get("compression_ratio") or 0.0) > _float_env(
            "FA_ASR_WINDOWED_MAX_COMPRESSION_RATIO", 3.2
        ):
            return True
    except (TypeError, ValueError):
        return True
    text = str(candidate.get("text") or "").strip()
    return len(text) < 2


def _merge_windowed_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_overlap = _float_env("FA_ASR_WINDOWED_MAX_OVERLAP_SEC", 0.9)
    min_overlap_ratio = _float_env("FA_ASR_WINDOWED_DUP_OVERLAP_RATIO", 0.55)
    min_text_similarity = _float_env("FA_ASR_WINDOWED_DUP_TEXT_SIMILARITY", 0.62)
    ordered = sorted(candidates, key=lambda item: (float(item["start"]), float(item["end"])))
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in ordered:
        if _candidate_has_low_confidence(candidate):
            dropped.append(
                {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "reason": "low_confidence",
                }
            )
            continue
        replacement_idx: int | None = None
        should_drop = False
        for idx, existing in enumerate(accepted):
            overlap = _candidate_overlap_seconds(candidate, existing)
            if overlap <= max_overlap:
                continue
            if (
                _candidate_overlap_ratio(candidate, existing) < min_overlap_ratio
                and _candidate_text_similarity(candidate, existing) < min_text_similarity
            ):
                continue
            if _candidate_score(candidate) > _candidate_score(existing):
                replacement_idx = idx
            else:
                should_drop = True
            break
        if replacement_idx is not None:
            dropped.append(
                {
                    "start": accepted[replacement_idx]["start"],
                    "end": accepted[replacement_idx]["end"],
                    "replacement_start": candidate["start"],
                    "replacement_end": candidate["end"],
                    "reason": "lower_confidence_overlap",
                }
            )
            accepted[replacement_idx] = candidate
        elif should_drop:
            dropped.append(
                {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "reason": "lower_confidence_overlap",
                }
            )
        else:
            accepted.append(candidate)
    accepted.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return accepted, {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "dropped": len(dropped),
        "dropped_overlaps": dropped[:40],
    }


def _decode_windowed_primary(
    wav: Path,
    *,
    duration_sec: float,
    y_audio: Any | None = None,
    sr_audio: int | None = None,
    profile_name: str = "windowed_strict",
    model_name: str,
    device: str,
    compute_type: str,
    lang: str | None,
    prompt: str | None,
    cancel_check: Callable[[], None] | None,
) -> tuple[list[Any], str, dict[str, Any], dict[str, Any]]:
    profile = _asr_profile(profile_name)
    speech_report: dict[str, Any] = {"enabled": False}
    if y_audio is not None and sr_audio:
        windows, speech_report = _speech_islands(y_audio, sr_audio, duration_sec=duration_sec)
    else:
        windows = _windowed_windows(duration_sec)
    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    detected_lang = lang or "ru"
    last_kw: dict[str, Any] = dict(profile.options)
    with tempfile.TemporaryDirectory(prefix="fa_asr_windowed_") as tmp:
        tmpdir = Path(tmp)
        for idx, window in enumerate(windows):
            if cancel_check:
                cancel_check()
            start = float(window["start"])
            end = float(window["end"])
            clip = tmpdir / f"window_{idx:03d}.wav"
            _write_asr_clip(wav, clip, start, end, profile_name=profile.name)
            seg_list, detected, transcribe_kw = _run_whisper(
                clip,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                profile=profile,
                lang=detected_lang or lang,
                prompt=prompt,
                cancel_check=cancel_check,
                use_subprocess=True,
            )
            detected_lang = detected or detected_lang or lang or "ru"
            last_kw = transcribe_kw
            accepted_in_window = 0
            for seg in seg_list:
                candidate = _segment_to_candidate(
                    seg,
                    offset_sec=start,
                    source_window={
                        "start": start,
                        "end": end,
                        "speech_ratio": float(window.get("speech_ratio", 1.0)),
                        "rms": float(window.get("rms", 0.0)),
                    },
                )
                candidate["asr_source"] = "windowed"
                candidate["asr_window_id"] = idx
                candidate["asr_window"] = {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speech_ratio": float(window.get("speech_ratio", 1.0)),
                    "rms": float(window.get("rms", 0.0)),
                }
                if str(candidate.get("text") or "").strip():
                    candidates.append(candidate)
                    accepted_in_window += 1
            attempts.append(
                {
                    "window_id": idx,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speech_ratio": float(window.get("speech_ratio", 1.0)),
                    "rms": float(window.get("rms", 0.0)),
                    "segments": len(seg_list),
                    "accepted_candidates": accepted_in_window,
                }
            )

    merged, merge_report = _merge_windowed_candidates(candidates)
    report = {
        "attempted": True,
        "profile": profile.name,
        "window_sec": _windowed_window_sec(),
        "overlap_sec": _windowed_overlap_sec(),
        "speech_islands": speech_report,
        "windows_count": len(windows),
        "windows": attempts,
        "merge": merge_report,
        "audio_filter": _asr_voice_filter(profile.name),
    }
    return [_WindowedSegment(item) for item in merged], detected_lang, last_kw, report


class _RescuedSegment:
    def __init__(self, candidate: dict[str, Any]) -> None:
        self.start = float(candidate.get("start", 0.0) or 0.0)
        self.end = float(candidate.get("end", 0.0) or 0.0)
        self.text = str(candidate.get("text") or "")
        self.avg_logprob = float(candidate.get("avg_logprob", 0.0) or 0.0)
        self.compression_ratio = float(candidate.get("compression_ratio", 0.0) or 0.0)
        self.no_speech_prob = float(candidate.get("no_speech_prob", 0.0) or 0.0)
        self.words = [_RescuedWord(word) for word in (candidate.get("words") or [])]
        self.asr_source = "rescue"
        self.source_window = candidate.get("source_window")
        self.asr_window_id = candidate.get("asr_window_id")
        self.asr_window = candidate.get("asr_window")


class _RescuedWord:
    def __init__(self, word: dict[str, Any]) -> None:
        self.start = float(word.get("start", 0.0) or 0.0)
        self.end = float(word.get("end", 0.0) or 0.0)
        self.word = str(word.get("word") or "")
        self.probability = float(word.get("probability", 0.0) or 0.0)


class _WindowedSegment(_RescuedSegment):
    def __init__(self, candidate: dict[str, Any]) -> None:
        super().__init__(candidate)
        self.asr_source = "windowed"


def _fw_transcribe_worker(
    wav_path_str: str,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kw: dict[str, Any],
    out_q: Any,
    err_q: Any,
) -> None:
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments_gen, info = model.transcribe(wav_path_str, **transcribe_kw)
        lang = (info.language or "ru") if info else "ru"
        out_q.put(("meta", {"language": lang}))
        for seg in segments_gen:
            out_q.put(("seg", seg))
        out_q.put(("finished", None))
    except Exception as e:
        err_q.put(repr(e))


def _transcribe_segments_subprocess(
    wav: Path,
    model_name: str,
    device: str,
    compute_type: str,
    transcribe_kw: dict[str, Any],
    cancel_check: Callable[[], None] | None,
) -> tuple[list[Any], str]:
    ctx = multiprocessing.get_context("spawn")
    out_q = ctx.Queue()
    err_q = ctx.Queue()
    proc = ctx.Process(
        target=_fw_transcribe_worker,
        args=(str(wav), model_name, device, compute_type, transcribe_kw, out_q, err_q),
    )
    proc.start()
    seg_list: list[Any] = []
    detected_lang = "ru"
    finished = False
    try:
        while not finished:
            if cancel_check:
                try:
                    cancel_check()
                except PipelineCancelled:
                    proc.terminate()
                    for _ in range(80):
                        if not proc.is_alive():
                            break
                        time.sleep(0.1)
                    raise
            try:
                kind, payload = out_q.get(timeout=0.35)
            except Empty:
                if not proc.is_alive():
                    if not err_q.empty():
                        raise RuntimeError(err_q.get())
                    if finished:
                        break
                    raise RuntimeError(
                        "Процесс распознавания завершился без результата (см. лог сервера)"
                    )
                continue
            if kind == "meta":
                detected_lang = (payload.get("language") or "ru") if payload else "ru"
            elif kind == "seg":
                seg_list.append(payload)
            elif kind == "finished":
                proc.join(timeout=300)
                finished = True
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=8)
    return seg_list, detected_lang


def _run_diarization_after_asr(
    wav: Path,
    *,
    hf: str | None,
    expected_speaker_count: int | None,
    ping: Callable[[str], None],
    cancel_check: Callable[[], None] | None,
) -> tuple[
    list[tuple[float, float, str]],
    str,
    str,
    str | None,
    list[str],
    str | None,
    str | None,
]:
    diar_rows: list[tuple[float, float, str]] = []
    diar_error = ""
    diar_backend = "none"
    diar_model: str | None = None
    backend_failures: list[str] = []
    skipped_pyannote: str | None = None
    skipped_nemo: str | None = None

    for backend in _diarization_backend_candidates(hf):
        if backend == "nemo":
            if not _should_run_nemo(hf):
                ping("diarization_skip")
                if Path(nemo_model_name()).is_file():
                    skipped_nemo = "disabled"
                else:
                    skipped_nemo = "no_hf_token"
                continue
            if not nemo_is_installed():
                backend_failures.append(
                    "NeMo Sortformer не установлен; поставьте зависимости из requirements-nemo.txt."
                )
                continue
            ping("diarization")
            try:
                if cancel_check:
                    cancel_check()
                diar_rows = load_diarization_rows_nemo(wav)
                diar_backend = "nemo"
                diar_model = nemo_model_name()
                break
            except Exception as e:
                backend_failures.append(f"NeMo Sortformer: {e}")
                diar_rows = []
        elif backend == "pyannote":
            if _should_run_pyannote(hf):
                ping("diarization")
                try:
                    if cancel_check:
                        cancel_check()
                    diar_rows = _load_diarization_rows(
                        wav,
                        hf or "",
                        expected_speaker_count=expected_speaker_count,
                    )
                    diar_backend = "pyannote"
                    diar_model = str(_diarization_pipeline_path_or_id())
                    break
                except Exception as e:
                    backend_failures.append(f"pyannote: {_friendly_pyannote_error(e)}")
                    diar_rows = []
            elif hf:
                if sys.platform == "win32":
                    ping("diarization_skip_windows")
                    skipped_pyannote = "windows"
                else:
                    ping("diarization_skip")
                    skipped_pyannote = "skip_env"
                    backend_failures.append(
                        "Pyannote отключён (SKIP_PYANNOTE=1) — используется fallback."
                    )
            else:
                ping("diarization_skip")
                skipped_pyannote = "no_hf_token"
        elif backend == "mfcc":
            break

    diar_error = " | ".join(x for x in backend_failures if x)
    return (
        diar_rows,
        diar_error,
        diar_backend,
        diar_model,
        backend_failures,
        skipped_pyannote,
        skipped_nemo,
    )


def transcribe_video_to_structure(
    video_path: Path,
    on_progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    expected_speaker_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    def ping(phase: str) -> None:
        if cancel_check:
            cancel_check()
        if on_progress:
            on_progress(phase)

    video_path = video_path.resolve()
    device = _device()
    compute_type = _compute_type()
    model_name = _model_name()
    lang = _language()

    with tempfile.TemporaryDirectory(prefix="fa_transcribe_") as tmp:
        wav = Path(tmp) / "audio.wav"
        if cancel_check:
            cancel_check()
        extract_wav_16k_mono(video_path, wav)
        ping("extract_audio")

        import librosa

        y_audio, sr_audio = librosa.load(str(wav), sr=16000, mono=True)
        audio_duration_sec = float(len(y_audio) / sr_audio) if sr_audio else 0.0

        hf = _hf_token()
        diar_rows: list[tuple[float, float, str]] = []
        diar_error = ""
        diar_backend = "none"
        diar_model: str | None = None
        backend_failures: list[str] = []
        skipped_pyannote: str | None = None
        skipped_nemo: str | None = None

        prompt = _initial_prompt()
        primary_profile = _asr_profile()
        asr_rescue_report: dict[str, Any] = {"attempted": False, "windows": []}
        asr_failures: list[dict[str, str]] = []
        extra_asr_profile_history: list[dict[str, Any]] = []
        ping("whisper_load")
        ping("asr_whisper")
        windowed_report: dict[str, Any] = {"attempted": False}
        if primary_profile.name in ("windowed_strict", "windowed_enhanced"):
            try:
                seg_list, detected_lang, transcribe_kw, windowed_report = _decode_windowed_primary(
                    wav,
                    duration_sec=audio_duration_sec,
                    y_audio=y_audio,
                    sr_audio=sr_audio,
                    profile_name=primary_profile.name,
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    lang=lang,
                    prompt=prompt,
                    cancel_check=cancel_check,
                )
            except Exception as exc:
                asr_failures.append({"stage": primary_profile.name, "error": str(exc)})
                if not _windowed_fallback_enabled():
                    raise
                primary_profile = _asr_profile("balanced")
                windowed_report = {
                    "attempted": True,
                    "failed": True,
                    "fallback": "balanced",
                    "error": str(exc),
                }
                seg_list, detected_lang, transcribe_kw = _run_whisper(
                    wav,
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    profile=primary_profile,
                    lang=lang,
                    prompt=prompt,
                    cancel_check=cancel_check,
                )
        else:
            seg_list, detected_lang, transcribe_kw = _run_whisper(
                wav,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                profile=primary_profile,
                lang=lang,
                prompt=prompt,
                cancel_check=cancel_check,
            )
            if _candidate_router_enabled():
                router_profile = _asr_profile("windowed_strict")
                try:
                    router_segments, router_lang, router_kw, router_report = _decode_windowed_primary(
                        wav,
                        duration_sec=audio_duration_sec,
                        y_audio=y_audio,
                        sr_audio=sr_audio,
                        profile_name=router_profile.name,
                        model_name=model_name,
                        device=device,
                        compute_type=compute_type,
                        lang=detected_lang or lang,
                        prompt=prompt,
                        cancel_check=cancel_check,
                    )
                    router_quality = _preview_transcript_for_quality(
                        router_segments,
                        duration_sec=audio_duration_sec,
                        expected_speaker_count=expected_speaker_count,
                    )
                    base_quality = _preview_transcript_for_quality(
                        seg_list,
                        duration_sec=audio_duration_sec,
                        expected_speaker_count=expected_speaker_count,
                    )
                    extra_asr_profile_history.append(_asr_profile_metadata(router_profile, router_kw))
                    windowed_report = {
                        **router_report,
                        "candidate_router": True,
                        "selected": False,
                        "balanced_quality": base_quality,
                        "windowed_quality": router_quality,
                    }
                    if _candidate_quality_better_or_equal(router_quality, base_quality):
                        seg_list = router_segments
                        detected_lang = router_lang or detected_lang
                        transcribe_kw = router_kw
                        primary_profile = router_profile
                        windowed_report["selected"] = True
                except Exception as exc:
                    asr_failures.append({"stage": "candidate_router_windowed_strict", "error": str(exc)})
        detected_lang = detected_lang or lang or "ru"
        asr_profile_history: list[dict[str, Any]] = [
            _asr_profile_metadata(primary_profile, transcribe_kw)
        ]
        asr_profile_history.extend(extra_asr_profile_history)
        primary_quality = _preview_transcript_for_quality(
            seg_list,
            duration_sec=audio_duration_sec,
            expected_speaker_count=expected_speaker_count,
        )
        asr_premerge_quality_history: list[dict[str, Any]] = [
            {
                "stage": primary_profile.name,
                "quality": primary_quality,
                "summary": quality_summary_text(primary_quality),
            }
        ]

        if (
            primary_profile.name not in ("windowed_strict", "windowed_enhanced")
            and primary_quality.get("status") == "fail"
            and _should_try_accurate_on_fail()
        ):
            accurate_profile = _asr_profile("accurate")
            ping("asr_whisper")
            try:
                accurate_segments, accurate_lang, accurate_kw = _run_whisper(
                    wav,
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    profile=accurate_profile,
                    lang=detected_lang or lang,
                    prompt=prompt,
                    cancel_check=cancel_check,
                )
                accurate_quality = _preview_transcript_for_quality(
                    accurate_segments,
                    duration_sec=audio_duration_sec,
                    expected_speaker_count=expected_speaker_count,
                )
                asr_profile_history.append(_asr_profile_metadata(accurate_profile, accurate_kw))
                asr_premerge_quality_history.append(
                    {
                        "stage": "accurate",
                        "quality": accurate_quality,
                        "summary": quality_summary_text(accurate_quality),
                    }
                )
                if float(accurate_quality.get("risk_score") or 0.0) <= float(
                    primary_quality.get("risk_score") or 999.0
                ):
                    seg_list = accurate_segments
                    detected_lang = accurate_lang or detected_lang
                    transcribe_kw = accurate_kw
                    primary_profile = accurate_profile
                    primary_quality = accurate_quality
            except Exception as exc:
                asr_failures.append({"stage": "accurate", "error": str(exc)})

        rescue_windows = rescue_windows_from_report(
            primary_quality,
            duration_sec=audio_duration_sec,
        )
        if rescue_windows and _should_run_rescue():
            ping("asr_whisper")
            try:
                rescue_candidates, asr_rescue_report = _rescue_decode_windows(
                    wav,
                    rescue_windows,
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    lang=detected_lang or lang,
                    prompt=prompt,
                    cancel_check=cancel_check,
                    existing_segments=seg_list,
                )
                if rescue_candidates:
                    seg_list = sorted(
                        [*seg_list, *[_RescuedSegment(item) for item in rescue_candidates]],
                        key=lambda seg: (float(seg.start), float(seg.end)),
                    )
                    rescued_quality = _preview_transcript_for_quality(
                        seg_list,
                        duration_sec=audio_duration_sec,
                        expected_speaker_count=expected_speaker_count,
                    )
                    asr_premerge_quality_history.append(
                        {
                            "stage": "rescue_merged",
                            "quality": rescued_quality,
                            "summary": quality_summary_text(rescued_quality),
                        }
                    )
                    primary_quality = rescued_quality
            except Exception as exc:
                asr_rescue_report = {
                    "attempted": True,
                    "source_windows": rescue_windows,
                    "windows": [{"window": window} for window in rescue_windows],
                    "accepted_segments": 0,
                    "error": str(exc),
                }
                asr_failures.append({"stage": "rescue", "error": str(exc)})
        if _skip_diarization_for_validation():
            diar_error = "Диаризация пропущена через FA_SKIP_DIARIZATION=1."
            backend_failures = [diar_error]
        else:
            (
                diar_rows,
                diar_error,
                diar_backend,
                diar_model,
                backend_failures,
                skipped_pyannote,
                skipped_nemo,
            ) = _run_diarization_after_asr(
                wav,
                hf=hf,
                expected_speaker_count=expected_speaker_count,
                ping=ping,
                cancel_check=cancel_check,
            )
        ping("segments_build")
        bounds = [(float(s.start), float(s.end)) for s in seg_list]
        skip_mfcc = os.environ.get("SKIP_MFCC_SPEAKERS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        skip_audio_tone = os.environ.get("SKIP_AUDIO_TONE", "").lower() in (
            "1",
            "true",
            "yes",
        )

        raw_speakers: list[str] = []
        diar_method = "none"
        expanded_word_specs: list[tuple[Any, float, float, str, list[Any]]] | None = None

        if diar_rows:
            expanded_word_specs = _expand_pyannote_word_segments(seg_list, diar_rows)
            raw_speakers = [s[3] for s in expanded_word_specs]
            diar_method = "nemo_sortformer" if diar_backend == "nemo" else "pyannote"
            norm_speakers = _remap_speakers_sequential(raw_speakers)
        elif bounds and not skip_mfcc:
            if _mfcc_word_level_enabled() and any(
                getattr(s, "words", None) for s in seg_list
            ):
                try:
                    expanded_word_specs = _expand_mfcc_word_segments(
                        seg_list, y_audio, sr_audio
                    )
                except Exception:
                    expanded_word_specs = None
            if expanded_word_specs:
                raw_speakers = [s[3] for s in expanded_word_specs]
                diar_method = "mfcc_word_kmeans"
                diar_backend = "mfcc"
                diar_model = "librosa+sklearn"
                norm_speakers = _remap_speakers_sequential(raw_speakers)
            else:
                try:
                    mfcc_labels = _mfcc_kmeans_speakers(
                        wav, bounds, y_sr=(y_audio, sr_audio)
                    )
                    if len(mfcc_labels) != len(bounds):
                        raise ValueError("mfcc length mismatch")
                    norm_speakers = mfcc_labels
                    diar_method = "mfcc_kmeans"
                    diar_backend = "mfcc"
                    diar_model = "librosa+sklearn"
                except Exception:
                    norm_speakers = ["SPEAKER_01"] * len(bounds)
                    diar_method = "mfcc_kmeans_failed"
                    diar_backend = "mfcc"
                    diar_model = "librosa+sklearn"
        else:
            norm_speakers = ["SPEAKER_01"] * len(bounds)
            diar_method = "none"
            if bounds:
                diar_backend = "none"

        segments_out: list[dict[str, Any]] = []
        if expanded_word_specs is not None:
            for idx, spec in enumerate(expanded_word_specs):
                orig_seg, t0, t1, _, wlist = spec
                spk = norm_speakers[idx] if idx < len(norm_speakers) else "SPEAKER_01"
                text = _text_for_word_slice(orig_seg, wlist)
                item: dict[str, Any] = {
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "speaker": spk,
                    "text": text,
                }
                asr_source = getattr(orig_seg, "asr_source", "")
                if asr_source:
                    item["asr_source"] = asr_source
                if getattr(orig_seg, "source_window", None):
                    item["asr_source_window"] = getattr(orig_seg, "source_window")
                if getattr(orig_seg, "asr_window_id", None) is not None:
                    item["asr_window_id"] = getattr(orig_seg, "asr_window_id")
                if getattr(orig_seg, "asr_window", None):
                    item["asr_window"] = getattr(orig_seg, "asr_window")
                word_probs: list[float] | None = None
                if wlist:
                    wl: list[dict[str, Any]] = []
                    word_probs = []
                    for w in wlist:
                        w0 = float(w.start)
                        w1 = float(w.end)
                        p = float(getattr(w, "probability", 0.0) or 0.0)
                        word_probs.append(p)
                        wl.append(
                            {
                                "start": round(w0, 3),
                                "end": round(w1, 3),
                                "word": str(getattr(w, "word", "") or "").strip(),
                                "probability": round(p, 4),
                                "speaker": spk,
                            }
                        )
                    item["words"] = wl
                item["delivery"] = analyze_segment(
                    text=text,
                    t0=t0,
                    t1=t1,
                    avg_logprob=float(getattr(orig_seg, "avg_logprob", 0.0)),
                    compression_ratio=float(getattr(orig_seg, "compression_ratio", 0.0)),
                    no_speech_prob=float(getattr(orig_seg, "no_speech_prob", 0.0)),
                    word_probs=word_probs,
                )
                if not skip_audio_tone:
                    item["audio_tone"] = segment_audio_tone(y_audio, sr_audio, t0, t1)
                segments_out.append(item)
        else:
            for idx, seg in enumerate(seg_list):
                t0 = float(seg.start)
                t1 = float(seg.end)
                text = (seg.text or "").strip()
                spk = norm_speakers[idx] if idx < len(norm_speakers) else "SPEAKER_01"

                item = {
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "speaker": spk,
                    "text": text,
                }
                asr_source = getattr(seg, "asr_source", "")
                if asr_source:
                    item["asr_source"] = asr_source
                if getattr(seg, "source_window", None):
                    item["asr_source_window"] = getattr(seg, "source_window")
                if getattr(seg, "asr_window_id", None) is not None:
                    item["asr_window_id"] = getattr(seg, "asr_window_id")
                if getattr(seg, "asr_window", None):
                    item["asr_window"] = getattr(seg, "asr_window")
                words = getattr(seg, "words", None)
                word_probs = None
                if words:
                    wl: list[dict[str, Any]] = []
                    word_probs = []
                    for w in words:
                        w0 = float(w.start)
                        w1 = float(w.end)
                        p = float(getattr(w, "probability", 0.0) or 0.0)
                        word_probs.append(p)
                        wl.append(
                            {
                                "start": round(w0, 3),
                                "end": round(w1, 3),
                                "word": str(getattr(w, "word", "") or "").strip(),
                                "probability": round(p, 4),
                                "speaker": spk,
                            }
                        )
                    item["words"] = wl
                item["delivery"] = analyze_segment(
                    text=text,
                    t0=t0,
                    t1=t1,
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
                    compression_ratio=float(getattr(seg, "compression_ratio", 0.0)),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    word_probs=word_probs,
                )
                if not skip_audio_tone:
                    item["audio_tone"] = segment_audio_tone(y_audio, sr_audio, t0, t1)
                segments_out.append(item)

        # Сглаживание «дребезга» одного голоса между разными SPEAKER_* (особенно MFCC по сегментам).
        if diar_method in ("mfcc_kmeans", "mfcc_word_kmeans"):
            try:
                r_mfcc = int(os.environ.get("MFCC_SEGMENT_SPEAKER_SMOOTH_RADIUS", "2"))
            except ValueError:
                r_mfcc = 2
            if r_mfcc > 0:
                _apply_segment_speaker_smoothing(segments_out, radius=r_mfcc)
        elif diar_method in ("pyannote", "nemo_sortformer"):
            r_seg = _pyannote_segment_smooth_radius()
            if r_seg > 0:
                _apply_segment_speaker_smoothing(segments_out, radius=r_seg)
            _reassign_soft_continuation_prefix_to_next_speaker(segments_out)
            _merge_fragmented_pyannote_speakers(segments_out)
            _bridge_short_pyannote_runs(segments_out)

        duration = max((s["end"] for s in segments_out), default=0.0)
        speakers = sorted({s["speaker"] for s in segments_out})
        speaker_alignment = _speaker_alignment_report(
            segments_out,
            expected_speaker_count=expected_speaker_count,
        )
        final_asr_quality = transcript_quality_report(
            {
                "duration_sec": round(audio_duration_sec or float(duration), 3),
                "segments": segments_out,
                "expected_speaker_count": expected_speaker_count,
            },
            expected_speaker_count=expected_speaker_count,
            duration_sec=audio_duration_sec or float(duration),
        )

        flagged = sum(
            1
            for s in segments_out
            if any(
                x in s.get("delivery", {}).get("flags", [])
                for x in ("fast_pace", "fast_and_unclear_asr")
            )
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "speech_delivery_version": 1,
            "video_file": str(video_path.name),
            "video_path": str(video_path),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "whisper_model": model_name,
            "whisper_device": device,
            "whisper_compute_type": compute_type,
            "asr_profile": primary_profile.name,
            "asr_profiles": asr_profile_history,
            "asr_premerge_quality_history": asr_premerge_quality_history,
            "asr_windowed": windowed_report,
            "asr_rescue_applied": bool(asr_rescue_report.get("accepted_segments")),
            "asr_rescue": asr_rescue_report,
            "asr_failures": asr_failures,
            "asr_quality": final_asr_quality,
            "asr_quality_summary": quality_summary_text(final_asr_quality),
            "whisper_initial_prompt_source": _initial_prompt_source(),
            "whisper_condition_on_previous_text": transcribe_kw.get(
                "condition_on_previous_text"
            ),
            "language": detected_lang,
            "diarization": diar_method
            in ("nemo_sortformer", "pyannote", "mfcc_kmeans", "mfcc_word_kmeans"),
            "diarization_backend": diar_backend,
            "diarization_method": diar_method,
            "diarization_model": diar_model,
            "speaker_alignment_quality": speaker_alignment,
            "expected_speaker_count": expected_speaker_count,
            "duration_sec": round(float(duration), 3),
            "speakers": speakers,
            "pyannote_device": _pyannote_device_type() if diar_method == "pyannote" else None,
            "nemo_device": nemo_device_name() if diar_backend == "nemo" else None,
            "segments": segments_out,
            "delivery_segments_fast_or_rushed": flagged,
            "delivery_note": delivery_summary_note(),
        }
        if diar_method in ("nemo_sortformer", "pyannote", "mfcc_word_kmeans"):
            result["diarization_speaker_alignment"] = "words"
        if not skip_audio_tone:
            result["speech_audio_tone_version"] = 1
            result["audio_tone_by_speaker"] = aggregate_tone_by_speaker(segments_out)
            result["audio_tone_note"] = audio_tone_summary_note()
        else:
            result["audio_tone_note"] = "Акустический анализ тона отключён (SKIP_AUDIO_TONE=1)."
        if diar_error:
            result["diarization_error"] = diar_error
        if backend_failures:
            result["diarization_fallbacks"] = backend_failures

        if skipped_pyannote == "windows":
            result["diarization_note"] = (
                "На Windows pyannote по умолчанию отключён (torchcodec/FFmpeg на этой ОС часто несовместимы). "
                "Используется локальная диаризация (MFCC). Чтобы попытаться pyannote: PYANNOTE_ON_WINDOWS=1 в .env "
                "(нужны совместимые FFmpeg DLL; см. документацию torchcodec)."
            )
        elif skipped_pyannote == "skip_env":
            result["diarization_note"] = diar_error
        elif skipped_nemo == "no_hf_token" and diar_method.startswith("mfcc"):
            result["diarization_note"] = (
                "NeMo Sortformer пропущен: для модели с Hugging Face нужен HF_TOKEN либо локальный .nemo-файл. "
                "Использован локальный MFCC fallback."
            )
        elif diar_method == "nemo_sortformer":
            result["diarization_note"] = "Основной backend: NVIDIA NeMo Sortformer."
        elif diar_method == "pyannote" and backend_failures:
            result["diarization_note"] = (
                "NeMo Sortformer недоступен; использован pyannote fallback."
            )
        elif diar_method == "mfcc_word_kmeans":
            if not hf:
                result["diarization_note"] = (
                    "HF_TOKEN не задан — спикеры оценены локально: MFCC по каждому слову + KMeans и разрез по паузам. "
                    "Для нормальной диаризации задайте HF_TOKEN или локальный NeMo .nemo и перезапустите транскрипцию."
                )
            else:
                result["diarization_note"] = (
                    "NeMo/pyannote не дали разметку (см. diarization_error). "
                    "Использован локальный MFCC по словам и разрез по паузам между словами."
                )
        elif diar_method == "mfcc_kmeans":
            if not hf:
                result["diarization_note"] = (
                    "HF_TOKEN не задан — говорящие разделены локально (MFCC + KMeans по сегментам Whisper); "
                    "для более точной диаризации укажите HF_TOKEN либо локальную/Hub-модель NeMo."
                )
            elif not diar_rows:
                result["diarization_note"] = (
                    "NeMo/pyannote не вернули разметку; использован локальный MFCC (грубее модели)."
                )
        elif diar_method == "mfcc_kmeans_failed":
            result["diarization_note"] = (
                "Не удалось разделить говорящих даже грубым методом (все сегменты помечены как SPEAKER_01)."
            )
        elif diar_method == "none" and bounds:
            result["diarization_note"] = (
                "Спикеры не разделены: нет сегментов, включён SKIP_MFCC_SPEAKERS или отключён MFCC fallback."
            )

        skip_speech_emotion = os.environ.get("SKIP_SPEECH_EMOTION", "").lower() in (
            "1",
            "true",
            "yes",
        )
        tone_sidecar: dict[str, Any] | None = None
        if not skip_speech_emotion:
            ping("tone")
            from src.speech_emotion import build_speech_emotion_sidecar

            tone_sidecar = build_speech_emotion_sidecar(
                y_audio, sr_audio, segments_out, str(video_path.name)
            )
            if tone_sidecar is not None:
                result["speech_emotion_sidecar"] = f"{video_path.stem}.tone.json"

        return result, tone_sidecar


def write_transcript_json(data: dict[str, Any], out_path: Path) -> None:
    from src.atomic_json import atomic_write_json

    atomic_write_json(out_path, data, indent=2)
