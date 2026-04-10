"""
Акустические признаки «тона» по WAV (librosa): не эмоции и не вежливость,
а измеримые прокси — высота (F0), громкость, «яркость» спектра.

Используется вместе с текстом и delivery.* при оценке речи.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np


def _float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def segment_audio_tone(y: np.ndarray, sr: int, t0: float, t1: float) -> dict[str, Any]:
    import librosa

    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    chunk = y[i0:i1]
    min_len = int(_float_env("AUDIO_TONE_MIN_SEGMENT_SEC", 0.18) * sr)
    if len(chunk) < min_len:
        return {"skipped": True, "reason": "segment_too_short"}

    rms = librosa.feature.rms(y=chunk)[0]
    rms_mean = float(np.mean(rms) + 1e-10)
    rms_std = float(np.std(rms))

    cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    zcr = librosa.feature.zero_crossing_rate(y=chunk)[0]
    zcr_mean = float(np.mean(zcr))

    fmin_hz = _float_env("AUDIO_TONE_FMIN_HZ", 80.0)
    fmax_hz = _float_env("AUDIO_TONE_FMAX_HZ", 450.0)

    f0 = librosa.pyin(
        chunk,
        sr=sr,
        fmin=fmin_hz,
        fmax=fmax_hz,
        frame_length=2048,
        hop_length=256,
    )[0]
    voiced = f0[~np.isnan(f0)]
    mean_f0 = float(np.nanmean(f0)) if voiced.size else None
    std_f0 = float(np.nanstd(f0)) if voiced.size else None

    thr_f0_std = _float_env("AUDIO_TONE_MONOTONE_F0_STD_HZ", 18.0)
    thr_f0_var = _float_env("AUDIO_TONE_EXPRESSIVE_F0_STD_HZ", 42.0)
    thr_cent = _float_env("AUDIO_TONE_BRIGHT_CENTROID_HZ", 2400.0)

    flags: list[str] = []
    dur = (t1 - t0) or 0.05
    if std_f0 is not None:
        if std_f0 < thr_f0_std and dur > 0.6:
            flags.append("monotone_pitch")
        if std_f0 > thr_f0_var:
            flags.append("expressive_pitch_variation")
    if cent_mean >= thr_cent:
        flags.append("bright_timbral")
    if cent_mean < 1500:
        flags.append("warm_dark_timbral")

    rms_rel = rms_mean  # абсолютный; сравнение по файлу — в aggregate
    return {
        "mean_f0_hz": None if mean_f0 is None else round(mean_f0, 1),
        "f0_std_hz": None if std_f0 is None else round(std_f0, 1),
        "rms_energy_mean": round(rms_rel, 5),
        "rms_energy_std": round(rms_std, 5),
        "spectral_centroid_mean_hz": round(cent_mean, 1),
        "spectral_centroid_std_hz": round(cent_std, 1),
        "zero_crossing_rate_mean": round(zcr_mean, 4),
        "flags": sorted(set(flags)),
    }


def aggregate_tone_by_speaker(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Средние по спикеру (взвешенные длительностью сегмента)."""
    acc: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for seg in segments:
        sp = seg.get("speaker") or "?"
        at = seg.get("audio_tone")
        if not isinstance(at, dict) or at.get("skipped"):
            continue
        t0 = float(seg.get("start", 0))
        t1 = float(seg.get("end", 0))
        w = max(t1 - t0, 0.05)
        acc[sp].append((w, at))

    out: dict[str, Any] = {}
    for sp, pairs in sorted(acc.items()):
        tw = sum(w for w, _ in pairs)
        if tw <= 0:
            continue

        def wavg(key: str) -> float | None:
            vals = [(w, a.get(key)) for w, a in pairs]
            nums = [(w, float(v)) for w, v in vals if v is not None]
            if not nums:
                return None
            return round(sum(w * v for w, v in nums) / tw, 2)

        out[sp] = {
            "mean_f0_hz": wavg("mean_f0_hz"),
            "f0_std_hz": wavg("f0_std_hz"),
            "rms_energy_mean": wavg("rms_energy_mean"),
            "spectral_centroid_mean_hz": wavg("spectral_centroid_mean_hz"),
            "note": (
                "Средние по сегментам с акустикой; коррелируют с тембром/интонацией, "
                "не с «вежливостью» напрямую."
            ),
        }
    return out


def audio_tone_summary_note() -> str:
    return (
        "audio_tone.* — признаки по сырому аудио (F0, громкость, спектральный центроид). "
        "Полезно вместе с текстом; не путать с распознаванием эмоций."
    )
