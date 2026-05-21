#!/usr/bin/env python3
"""Validate opt-in windowed ASR without overwriting production transcripts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from src.asr_quality import transcript_quality_report
from src.paths import META_DIR, TRANSCRIPT_DIR
from src.transcribe import transcribe_video_to_structure


DEFAULT_VIDEOS = [
    "IMG_0925",
    "VID_20260512_134806",
    "16166897256988",
]


OUTLIER_VIDEOS = [
    "VID_20260512_110433",
    "15671825337033",
    "IMG_8992",
    "IMG_2125",
    "осмотр КП",
]


def _video_path(name: str) -> Path:
    raw = Path(name)
    if raw.is_file():
        return raw.resolve()
    for suffix in (".mp4", ".mov", ".MOV", ".MP4", ".m4a", ".wav"):
        candidate = ROOT / "01.Video" / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found for {name}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None
    quality = transcript_quality_report(data)
    summary = quality.get("summary") or {}
    return {
        "status": quality.get("status"),
        "risk_score": quality.get("risk_score"),
        "coverage_pct": summary.get("coverage_pct"),
        "max_gap_sec": summary.get("max_gap_sec"),
        "gaps_count": summary.get("gaps_count"),
        "low_confidence_segments_count": summary.get("low_confidence_segments_count"),
        "low_logprob_segments_count": summary.get("low_logprob_segments_count"),
        "repeated_ngrams_count": summary.get("repeated_ngrams_count"),
        "speakers_count": summary.get("speakers_count"),
        "speaker_alignment": data.get("speaker_alignment_quality"),
        "windowed_primary": quality.get("windowed_primary"),
    }


def _is_worse(old: dict[str, Any] | None, new: dict[str, Any] | None) -> bool:
    if not old or not new:
        return False
    old_gap = float(old.get("max_gap_sec") or 0.0)
    new_gap = float(new.get("max_gap_sec") or 0.0)
    old_cov = float(old.get("coverage_pct") or 0.0)
    new_cov = float(new.get("coverage_pct") or 0.0)
    old_risk = float(old.get("risk_score") or 0.0)
    new_risk = float(new.get("risk_score") or 0.0)
    return new_gap > old_gap + 2.0 or new_cov < old_cov - 3.0 or new_risk > old_risk + 8.0


def _first_texts(data: dict[str, Any] | None, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg in (data or {}).get("segments") or []:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker"),
                "text": text,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _suspicious_texts(data: dict[str, Any] | None, limit: int = 12) -> list[dict[str, Any]]:
    quality = transcript_quality_report(data or {})
    out: list[dict[str, Any]] = []
    for key in ("gaps", "low_confidence_segments", "low_logprob_segments", "repeated_ngrams"):
        for item in quality.get(key) or []:
            if isinstance(item, dict):
                out.append({"kind": key, **item})
                if len(out) >= limit:
                    return out
    return out


def _semantic_review(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_segments_old": _first_texts(old),
        "first_segments_new": _first_texts(new),
        "suspicious_old": _suspicious_texts(old),
        "suspicious_new": _suspicious_texts(new),
        "speaker_alignment_new": new.get("speaker_alignment_quality"),
        "notes": [
            "Review first_segments_new for dialogue completeness and domain terms.",
            "Review suspicious_new for gaps, low confidence, low logprob, and repeats.",
            "Do not accept a candidate if speaker_alignment_new is worse than baseline.",
        ],
    }


def _validate_one(video: Path, out_dir: Path) -> dict[str, Any]:
    stem = video.stem
    baseline_path = TRANSCRIPT_DIR / f"{stem}.json"
    baseline = _load_json(baseline_path)
    data, tone_data = transcribe_video_to_structure(video)
    out_path = out_dir / f"{stem}.windowed.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if tone_data is not None:
        (out_dir / f"{stem}.windowed.tone.json").write_text(
            json.dumps(tone_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    old_summary = _summary(baseline)
    new_summary = _summary(data)
    row = {
        "stem": stem,
        "video": str(video),
        "baseline_path": str(baseline_path) if baseline_path.is_file() else None,
        "windowed_path": str(out_path),
        "old": old_summary,
        "new": new_summary,
        "worse_than_baseline": _is_worse(old_summary, new_summary),
        "asr_profile": data.get("asr_profile"),
        "asr_rescue_applied": data.get("asr_rescue_applied"),
        "asr_failures": data.get("asr_failures") or [],
        "semantic_review": _semantic_review(baseline, data),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run opt-in windowed ASR validation safely")
    parser.add_argument("videos", nargs="*", default=DEFAULT_VIDEOS)
    parser.add_argument("--include-outliers", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=META_DIR / "asr_windowed_validation")
    args = parser.parse_args()

    os.environ["FA_ASR_PROFILE"] = os.environ.get("FA_ASR_PROFILE", "windowed_strict")
    os.environ.setdefault("FA_TRANSCRIBE_SUBPROCESS", "1")
    os.environ.setdefault("FA_ASR_GPU_GUARD", "1")
    os.environ.setdefault("FA_ASR_WINDOWED_FALLBACK_TO_BALANCED", "1")
    if args.skip_diarization:
        os.environ["FA_SKIP_DIARIZATION"] = "1"

    videos = list(args.videos)
    if args.include_outliers:
        videos.extend(OUTLIER_VIDEOS)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "src" / "transcribe.py", out_dir / "_transcribe_snapshot.py")
    rows: list[dict[str, Any]] = []
    for name in videos:
        video = _video_path(name)
        print(f"[windowed-validation] {video.name}", flush=True)
        rows.append(_validate_one(video, out_dir))
        report_path = out_dir / "summary.json"
        report_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        old = row.get("old") or {}
        new = row.get("new") or {}
        print(
            f"{row['stem']}: old risk={old.get('risk_score')} gap={old.get('max_gap_sec')} "
            f"coverage={old.get('coverage_pct')} -> new risk={new.get('risk_score')} "
            f"gap={new.get('max_gap_sec')} coverage={new.get('coverage_pct')} "
            f"worse={row['worse_than_baseline']}"
        )


if __name__ == "__main__":
    main()
