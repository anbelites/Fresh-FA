"""Quality gates for ASR transcripts.

The goal is to make transcript completeness observable before downstream
evaluation trusts the text.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[\wА-Яа-яЁё]+", re.U)


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


def asr_quality_thresholds() -> dict[str, float | int]:
    return {
        "gap_warn_sec": _float_env("ASR_QUALITY_GAP_WARN_SEC", 6.0),
        "gap_fail_sec": _float_env("ASR_QUALITY_GAP_FAIL_SEC", 12.0),
        "gap_rescue_sec": _float_env("ASR_RESCUE_GAP_SEC", 8.0),
        "coverage_warn_lt": _float_env("ASR_QUALITY_COVERAGE_WARN_LT", 82.0),
        "coverage_fail_lt": _float_env("ASR_QUALITY_COVERAGE_FAIL_LT", 70.0),
        "low_mean_word_prob_lt": _float_env("ASR_QUALITY_LOW_MEAN_WORD_PROB_LT", 0.55),
        "low_word_prob_lt": _float_env("ASR_QUALITY_LOW_WORD_PROB_LT", 0.5),
        "low_logprob_lt": _float_env("ASR_QUALITY_LOW_LOGPROB_LT", -0.38),
        "high_compression_gt": _float_env("ASR_QUALITY_HIGH_COMPRESSION_GT", 2.4),
        "high_no_speech_gt": _float_env("ASR_QUALITY_HIGH_NO_SPEECH_GT", 0.35),
        "tiny_segment_sec": _float_env("ASR_QUALITY_TINY_SEGMENT_SEC", 0.6),
        "max_warn_score": _float_env("ASR_QUALITY_MAX_WARN_SCORE", 18.0),
        "max_fail_score": _float_env("ASR_QUALITY_MAX_FAIL_SCORE", 32.0),
        "repeat_ngram_min_count": _int_env("ASR_QUALITY_REPEAT_NGRAM_MIN_COUNT", 4),
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((a, b) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(a, b) for a, b in merged]


def _duration_from_segments(segments: list[dict[str, Any]]) -> float:
    return max((_as_float(seg.get("end")) for seg in segments), default=0.0)


def _windowed_primary_summary(segments: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    windowed_segments = [
        seg for seg in segments if str(seg.get("asr_source") or "") == "windowed"
    ]
    rescue_segments = [
        seg for seg in segments if str(seg.get("asr_source") or "") == "rescue"
    ]
    windows: dict[str, dict[str, Any]] = {}
    for seg in windowed_segments:
        raw_id = seg.get("asr_window_id")
        window_id = str(raw_id) if raw_id is not None else "unknown"
        bucket = windows.setdefault(
            window_id,
            {
                "window_id": raw_id,
                "segments_count": 0,
                "word_probs": [],
            },
        )
        bucket["segments_count"] += 1
        words = seg.get("words") or []
        if isinstance(words, list):
            for word in words:
                if isinstance(word, dict) and "probability" in word:
                    bucket["word_probs"].append(_as_float(word.get("probability")))

    window_rows: list[dict[str, Any]] = []
    all_probs: list[float] = []
    for bucket in windows.values():
        vals = bucket.pop("word_probs", [])
        all_probs.extend(vals)
        if vals:
            bucket["mean_word_probability"] = round(sum(vals) / len(vals), 4)
            bucket["words_count"] = len(vals)
        else:
            bucket["mean_word_probability"] = None
            bucket["words_count"] = 0
        window_rows.append(bucket)

    return {
        "enabled": bool(windowed_segments),
        "segments_count": len(windowed_segments),
        "rescue_segments_count": len(rescue_segments),
        "remaining_gaps_count": len(gaps),
        "remaining_max_gap_sec": round(max((gap["duration_sec"] for gap in gaps), default=0.0), 3),
        "mean_word_probability": round(sum(all_probs) / len(all_probs), 4) if all_probs else None,
        "windows_count": len(window_rows),
        "windows": sorted(window_rows, key=lambda row: str(row.get("window_id")))[:60],
    }


def transcript_quality_report(
    transcript: dict[str, Any],
    *,
    expected_speaker_count: int | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    thresholds = asr_quality_thresholds()
    segments = [seg for seg in (transcript.get("segments") or []) if isinstance(seg, dict)]
    duration = float(duration_sec or transcript.get("duration_sec") or _duration_from_segments(segments))

    intervals = [
        (_as_float(seg.get("start")), _as_float(seg.get("end")))
        for seg in segments
        if _as_float(seg.get("end")) > _as_float(seg.get("start"))
    ]
    merged = _merge_intervals(intervals)
    covered = sum(end - start for start, end in merged)
    coverage_pct = (covered / duration * 100.0) if duration > 0 else 0.0

    gaps: list[dict[str, Any]] = []
    gap_warn = float(thresholds["gap_warn_sec"])
    if duration > 0 and merged:
        if merged[0][0] > gap_warn:
            gaps.append(
                {
                    "start": 0.0,
                    "end": round(merged[0][0], 3),
                    "duration_sec": round(merged[0][0], 3),
                    "kind": "start",
                }
            )
        for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
            gap = next_start - prev_end
            if gap > gap_warn:
                gaps.append(
                    {
                        "start": round(prev_end, 3),
                        "end": round(next_start, 3),
                        "duration_sec": round(gap, 3),
                        "kind": "middle",
                    }
                )
        if duration - merged[-1][1] > gap_warn:
            gaps.append(
                {
                    "start": round(merged[-1][1], 3),
                    "end": round(duration, 3),
                    "duration_sec": round(duration - merged[-1][1], 3),
                    "kind": "end",
                }
            )
    elif duration > gap_warn:
        gaps.append(
            {
                "start": 0.0,
                "end": round(duration, 3),
                "duration_sec": round(duration, 3),
                "kind": "all",
            }
        )

    low_logprob_segments: list[dict[str, Any]] = []
    low_confidence_segments: list[dict[str, Any]] = []
    high_compression_segments: list[dict[str, Any]] = []
    high_no_speech_segments: list[dict[str, Any]] = []
    tiny_segments: list[dict[str, Any]] = []
    low_word_count = 0
    word_count = 0

    for idx, seg in enumerate(segments):
        start = _as_float(seg.get("start"))
        end = _as_float(seg.get("end"))
        seg_duration = max(0.0, end - start)
        if seg_duration < float(thresholds["tiny_segment_sec"]):
            tiny_segments.append({"index": idx, "start": round(start, 3), "end": round(end, 3)})

        words = seg.get("words") or []
        vals: list[float] = []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                if "probability" in word:
                    word_count += 1
                    prob = _as_float(word.get("probability"))
                    vals.append(prob)
                    if prob < float(thresholds["low_word_prob_lt"]):
                        low_word_count += 1

        delivery = seg.get("delivery") or {}
        mean_prob = delivery.get("mean_word_probability")
        if mean_prob is None and vals:
            mean_prob = sum(vals) / len(vals)
        avg_logprob = _as_float(delivery.get("avg_logprob"))
        compression = _as_float(delivery.get("compression_ratio"))
        no_speech = _as_float(delivery.get("no_speech_prob"))

        base = {
            "index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": seg.get("speaker"),
            "text": str(seg.get("text") or "")[:160],
        }
        if avg_logprob < float(thresholds["low_logprob_lt"]):
            low_logprob_segments.append({**base, "avg_logprob": round(avg_logprob, 4)})
        if mean_prob is not None and _as_float(mean_prob) < float(thresholds["low_mean_word_prob_lt"]):
            low_confidence_segments.append({**base, "mean_word_probability": round(_as_float(mean_prob), 4)})
        if compression > float(thresholds["high_compression_gt"]):
            high_compression_segments.append({**base, "compression_ratio": round(compression, 3)})
        if no_speech > float(thresholds["high_no_speech_gt"]):
            high_no_speech_segments.append({**base, "no_speech_prob": round(no_speech, 4)})

    texts = [_normalize_text(str(seg.get("text") or "")) for seg in segments]
    duplicate_segments = [
        {"text": text[:160], "count": count}
        for text, count in Counter(t for t in texts if len(t) >= 18).most_common()
        if count >= 2
    ][:8]

    tokens: list[str] = []
    for text in texts:
        tokens.extend(text.split())
    repeated_ngrams: list[dict[str, Any]] = []
    min_count = int(thresholds["repeat_ngram_min_count"])
    for n in (4, 5, 6, 7, 8):
        counts = Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))
        for ngram, count in counts.items():
            if count >= min_count:
                repeated_ngrams.append({"text": " ".join(ngram)[:160], "count": count, "n": n})
    repeated_ngrams = sorted(repeated_ngrams, key=lambda item: (item["count"], item["n"]), reverse=True)[:8]

    speakers = sorted(
        {
            str(seg.get("speaker"))
            for seg in segments
            if str(seg.get("speaker") or "").strip()
        }
    )
    expected = expected_speaker_count
    if expected is None:
        raw_expected = transcript.get("expected_speaker_count")
        try:
            expected = int(raw_expected) if raw_expected is not None else None
        except (TypeError, ValueError):
            expected = None
    extra_speakers = max(0, len(speakers) - int(expected)) if expected else 0

    score = 0.0
    if coverage_pct < float(thresholds["coverage_warn_lt"]):
        score += (float(thresholds["coverage_warn_lt"]) - coverage_pct) * 0.8
    if coverage_pct < float(thresholds["coverage_fail_lt"]):
        score += (float(thresholds["coverage_fail_lt"]) - coverage_pct) * 1.6
    score += min(26.0, len([gap for gap in gaps if gap["duration_sec"] >= float(thresholds["gap_fail_sec"])]) * 10.0)
    score += min(16.0, len(gaps) * 2.0)
    score += min(18.0, len(low_logprob_segments) * 1.2)
    score += min(18.0, len(low_confidence_segments) * 1.6)
    score += min(15.0, (low_word_count / max(1, word_count)) * 60.0)
    score += min(16.0, len(high_compression_segments) * 2.0)
    score += min(12.0, len(high_no_speech_segments) * 2.0)
    score += min(15.0, len(repeated_ngrams) * 3.0 + len(duplicate_segments) * 2.0)
    score += min(12.0, len(tiny_segments) / max(1, len(segments)) * 35.0)
    score += extra_speakers * 7.0

    hard_fail = (
        coverage_pct < float(thresholds["coverage_fail_lt"])
        or any(gap["duration_sec"] >= float(thresholds["gap_fail_sec"]) for gap in gaps)
        or bool(repeated_ngrams)
        or extra_speakers >= 2
    )

    if hard_fail and score >= float(thresholds["max_fail_score"]):
        status = "fail"
    elif score >= float(thresholds["max_warn_score"]):
        status = "warn"
    else:
        status = "pass"

    return {
        "schema_version": 1,
        "status": status,
        "risk_score": round(score, 2),
        "thresholds": thresholds,
        "summary": {
            "segments_count": len(segments),
            "duration_sec": round(duration, 3),
            "covered_sec": round(covered, 3),
            "coverage_pct": round(coverage_pct, 2),
            "gaps_count": len(gaps),
            "max_gap_sec": round(max((gap["duration_sec"] for gap in gaps), default=0.0), 3),
            "low_logprob_segments_count": len(low_logprob_segments),
            "low_confidence_segments_count": len(low_confidence_segments),
            "low_word_probability_count": low_word_count,
            "words_count": word_count,
            "high_compression_segments_count": len(high_compression_segments),
            "high_no_speech_segments_count": len(high_no_speech_segments),
            "repeated_ngrams_count": len(repeated_ngrams),
            "duplicate_segments_count": len(duplicate_segments),
            "tiny_segments_count": len(tiny_segments),
            "speakers_count": len(speakers),
            "expected_speaker_count": expected,
            "extra_speakers_count": extra_speakers,
        },
        "gaps": gaps,
        "low_logprob_segments": low_logprob_segments[:25],
        "low_confidence_segments": low_confidence_segments[:25],
        "high_compression_segments": high_compression_segments[:25],
        "high_no_speech_segments": high_no_speech_segments[:25],
        "duplicate_segments": duplicate_segments,
        "repeated_ngrams": repeated_ngrams,
        "tiny_segments": tiny_segments[:25],
        "windowed_primary": _windowed_primary_summary(segments, gaps),
    }


def rescue_windows_from_report(
    quality: dict[str, Any],
    *,
    duration_sec: float,
) -> list[dict[str, float]]:
    thresholds = quality.get("thresholds") or asr_quality_thresholds()
    gap_min = float(thresholds.get("gap_rescue_sec", 8.0))
    pad = _float_env("ASR_RESCUE_PAD_SEC", 2.0)
    max_window = _float_env("ASR_RESCUE_MAX_WINDOW_SEC", 45.0)
    windows: list[dict[str, float]] = []
    for gap in quality.get("gaps") or []:
        try:
            gap_start = float(gap.get("start"))
            gap_end = float(gap.get("end"))
            gap_dur = float(gap.get("duration_sec"))
        except (TypeError, ValueError):
            continue
        if gap_dur < gap_min:
            continue
        start = max(0.0, gap_start - pad)
        end = min(float(duration_sec), gap_end + pad)
        if end - start > max_window:
            midpoint = (gap_start + gap_end) / 2.0
            start = max(0.0, midpoint - max_window / 2.0)
            end = min(float(duration_sec), start + max_window)
        if end - start >= 1.0:
            windows.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "gap_start": round(gap_start, 3),
                    "gap_end": round(gap_end, 3),
                }
            )
    return _dedupe_windows(windows)


def _dedupe_windows(windows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: (float(w["start"]), float(w["end"])))
    merged: list[dict[str, float]] = []
    for window in ordered:
        start = float(window["start"])
        end = float(window["end"])
        gap_start = float(window.get("gap_start", start))
        gap_end = float(window.get("gap_end", end))
        if not merged or start > float(merged[-1]["end"]) + 0.2:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                }
            )
        else:
            merged[-1]["end"] = max(float(merged[-1]["end"]), end)
            merged[-1]["gap_start"] = min(float(merged[-1]["gap_start"]), gap_start)
            merged[-1]["gap_end"] = max(float(merged[-1]["gap_end"]), gap_end)
    return [
        {
            "start": round(float(item["start"]), 3),
            "end": round(float(item["end"]), 3),
            "gap_start": round(float(item["gap_start"]), 3),
            "gap_end": round(float(item["gap_end"]), 3),
        }
        for item in merged
    ]


def quality_summary_text(quality: dict[str, Any]) -> str:
    summary = quality.get("summary") or {}
    status = str(quality.get("status") or "unknown")
    return (
        f"ASR quality={status}, risk={quality.get('risk_score')}, "
        f"coverage={summary.get('coverage_pct')}%, "
        f"gaps={summary.get('gaps_count')} (max {summary.get('max_gap_sec')}s), "
        f"low_conf={summary.get('low_confidence_segments_count')}, "
        f"low_logprob={summary.get('low_logprob_segments_count')}, "
        f"repeats={summary.get('repeated_ngrams_count')}"
    )
