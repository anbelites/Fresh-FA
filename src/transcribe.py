"""Транскрипция (faster-whisper) + диаризация говорящих (pyannote, при HF_TOKEN)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.audio_extract import extract_wav_16k_mono
from src.audio_tone import (
    aggregate_tone_by_speaker,
    audio_tone_summary_note,
    segment_audio_tone,
)
from src.speech_delivery import analyze_segment, delivery_summary_note


def _device() -> str:
    return os.environ.get("WHISPER_DEVICE", "cpu")


def _compute_type() -> str:
    return os.environ.get("WHISPER_COMPUTE_TYPE", "int8")


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
    default = os.environ.get("WHISPER_DOMAIN_PROMPT", "").strip()
    if default:
        return default
    return None


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _diarization_pipeline_path_or_id() -> Path | str:
    """Hub id или локальный config.yaml (см. config/pyannote/speaker-diarization-3.1.yaml)."""
    root = Path(__file__).resolve().parent.parent
    override = os.environ.get("PYANNOTE_PIPELINE", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if p.suffix in (".yaml", ".yml") and p.is_file():
            return p
        return override
    local = root / "config" / "pyannote" / "speaker-diarization-3.1.yaml"
    if local.is_file():
        return local
    return "pyannote/speaker-diarization-3.1"


def _pyannote_pipeline_kwargs() -> dict[str, int]:
    """Подсказки по числу говорящих (если известно) — уменьшает путаницу кластеров."""
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


def _load_diarization_rows(
    wav_path: Path, hf_token: str
) -> list[tuple[float, float, str]]:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        _diarization_pipeline_path_or_id(), token=hf_token
    )
    kwargs = _pyannote_pipeline_kwargs()
    raw_out = pipeline({"audio": str(wav_path)}, **kwargs)
    if hasattr(raw_out, "speaker_diarization"):
        use_exclusive = os.environ.get("PYANNOTE_EXCLUSIVE", "").lower() in (
            "1",
            "true",
            "yes",
        )
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
    """Голосование по 3 точкам внутри слова — стабильнее на границах реплик."""
    w0 = float(w.start)
    w1 = float(w.end)
    if w1 <= w0:
        return _speaker_for_interval(w0, w1, diar_rows)
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

        def flush() -> None:
            nonlocal bucket, current_spk
            if not bucket or current_spk is None:
                return
            wt0 = float(bucket[0].start)
            wt1 = float(bucket[-1].end)
            specs.append((seg, wt0, wt1, current_spk, bucket[:]))

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
    try:
        rad = int(os.environ.get("PYANNOTE_LABEL_SMOOTH_RADIUS", "3"))
    except ValueError:
        rad = 3
    rad = max(0, rad)
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


def transcribe_video_to_structure(
    video_path: Path,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from faster_whisper import WhisperModel

    def ping(phase: str) -> None:
        if on_progress:
            on_progress(phase)

    video_path = video_path.resolve()
    device = _device()
    compute_type = _compute_type()
    model_name = _model_name()
    lang = _language()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="fa_transcribe_") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_wav_16k_mono(video_path, wav)
        ping("extract_audio")

        import librosa

        y_audio, sr_audio = librosa.load(str(wav), sr=16000, mono=True)

        hf = _hf_token()
        diar_rows: list[tuple[float, float, str]] = []
        diar_error = ""
        skipped_pyannote: str | None = None
        if _should_run_pyannote(hf):
            ping("diarization")
            try:
                diar_rows = _load_diarization_rows(wav, hf)
            except Exception as e:
                diar_error = str(e)
        elif hf:
            if sys.platform == "win32":
                ping("diarization_skip_windows")
                skipped_pyannote = "windows"
                diar_error = ""
            else:
                ping("diarization_skip")
                skipped_pyannote = "skip_env"
                diar_error = "Pyannote отключён (SKIP_PYANNOTE=1) — используется MFCC."
        else:
            ping("diarization_skip")
            diar_error = "HF_TOKEN не задан — говорящие не различаются."

        ping("whisper_load")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        transcribe_kw: dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": True,
        }
        if lang:
            transcribe_kw["language"] = lang
        prompt = _initial_prompt()
        if prompt:
            transcribe_kw["initial_prompt"] = prompt

        ping("asr_whisper")
        segments_gen, info = model.transcribe(str(wav), **transcribe_kw)
        detected_lang = (info.language or lang or "ru") if info else (lang or "ru")

        seg_list = list(segments_gen)
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
            diar_method = "pyannote"
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
                except Exception:
                    norm_speakers = ["SPEAKER_01"] * len(bounds)
                    diar_method = "mfcc_kmeans_failed"
        else:
            norm_speakers = ["SPEAKER_01"] * len(bounds)
            diar_method = "none"

        segments_out: list[dict[str, Any]] = []
        if expanded_word_specs is not None:
            for idx, spec in enumerate(expanded_word_specs):
                orig_seg, t0, t1, _, wlist = spec
                spk = norm_speakers[idx] if idx < len(norm_speakers) else "SPEAKER_01"
                if wlist:
                    text = _join_whisper_words(wlist)
                else:
                    text = (orig_seg.text or "").strip()
                item: dict[str, Any] = {
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "speaker": spk,
                    "text": text,
                }
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
        elif diar_method == "pyannote":
            try:
                r_seg = int(os.environ.get("PYANNOTE_SEGMENT_SMOOTH_RADIUS", "0"))
            except ValueError:
                r_seg = 0
            if r_seg > 0:
                _apply_segment_speaker_smoothing(segments_out, radius=r_seg)

        speaker_roles = _identify_employee_speaker(segments_out)
        if speaker_roles:
            _apply_speaker_roles(segments_out, speaker_roles)

        duration = max((s["end"] for s in segments_out), default=0.0)
        speakers = sorted({s["speaker"] for s in segments_out})

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
            "language": detected_lang,
            "diarization": diar_method
            in ("pyannote", "mfcc_kmeans", "mfcc_word_kmeans"),
            "diarization_method": diar_method,
            "duration_sec": round(float(duration), 3),
            "speakers": speakers,
            "speaker_roles": speaker_roles if speaker_roles else None,
            "segments": segments_out,
            "delivery_segments_fast_or_rushed": flagged,
            "delivery_note": delivery_summary_note(),
        }
        if diar_method in ("pyannote", "mfcc_word_kmeans"):
            result["diarization_speaker_alignment"] = "words"
        if not skip_audio_tone:
            result["speech_audio_tone_version"] = 1
            result["audio_tone_by_speaker"] = aggregate_tone_by_speaker(segments_out)
            result["audio_tone_note"] = audio_tone_summary_note()
        else:
            result["audio_tone_note"] = "Акустический анализ тона отключён (SKIP_AUDIO_TONE=1)."
        if hf and not diar_rows and diar_error:
            result["diarization_error"] = diar_error

        if skipped_pyannote == "windows":
            result["diarization_note"] = (
                "На Windows pyannote по умолчанию отключён (torchcodec/FFmpeg на этой ОС часто несовместимы). "
                "Используется локальная диаризация (MFCC). Чтобы попытаться pyannote: PYANNOTE_ON_WINDOWS=1 в .env "
                "(нужны совместимые FFmpeg DLL; см. документацию torchcodec)."
            )
        elif skipped_pyannote == "skip_env":
            result["diarization_note"] = diar_error
        elif diar_method == "mfcc_word_kmeans":
            if not hf:
                result["diarization_note"] = (
                    "HF_TOKEN не задан — спикеры оценены локально: MFCC по каждому слову + KMeans и разрез по паузам. "
                    "Для нормальной диаризации задайте HF_TOKEN (pyannote) в .env и перезапустите транскрипцию."
                )
            else:
                result["diarization_note"] = (
                    "Pyannote не дал разметку (см. diarization_error). "
                    "Использован локальный MFCC по словам и разрез по паузам между словами."
                )
        elif diar_method == "mfcc_kmeans":
            if not hf:
                result["diarization_note"] = (
                    "HF_TOKEN не задан — говорящие разделены локально (MFCC + KMeans по сегментам Whisper); "
                    "для более точной диаризации укажите HF_TOKEN (pyannote)."
                )
            elif hf and not diar_rows:
                result["diarization_note"] = (
                    "Pyannote не вернул разметку; использован локальный MFCC (грубее модели)."
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
