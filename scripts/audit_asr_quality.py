#!/usr/bin/env python3
"""Read-only ASR quality audit for existing transcript JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr_quality import quality_summary_text, transcript_quality_report
from src.paths import TRANSCRIPT_DIR


def _transcript_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(
        path
        for path in base.glob("*.json")
        if path.is_file() and not path.name.endswith(".tone.json")
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "load_error": str(exc),
            "asr_quality": {
                "status": "fail",
                "risk_score": 999,
                "summary": {},
                "gaps": [],
            },
        }
    return data if isinstance(data, dict) else None


def _row_for(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    quality = transcript_quality_report(data)
    summary = quality.get("summary") or {}
    return {
        "file": path.name,
        "status": quality.get("status"),
        "risk_score": quality.get("risk_score"),
        "coverage_pct": summary.get("coverage_pct"),
        "max_gap_sec": summary.get("max_gap_sec"),
        "gaps_count": summary.get("gaps_count"),
        "low_confidence_segments_count": summary.get("low_confidence_segments_count"),
        "low_logprob_segments_count": summary.get("low_logprob_segments_count"),
        "repeated_ngrams_count": summary.get("repeated_ngrams_count"),
        "speakers_count": summary.get("speakers_count"),
        "expected_speaker_count": summary.get("expected_speaker_count"),
        "asr_rescue_applied": data.get("asr_rescue_applied"),
        "transcript_needs_review": data.get("transcript_needs_review"),
        "summary": quality_summary_text(quality),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank transcript JSON files by ASR quality risk")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=TRANSCRIPT_DIR,
        help="Transcript JSON or directory (default: 02.Transcript)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Show only top N risky rows")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in _transcript_paths(args.path):
        data = _load_json(path)
        if not data:
            continue
        rows.append(_row_for(path, data))

    rows.sort(key=lambda row: float(row.get("risk_score") or 0), reverse=True)
    shown = rows[: args.limit] if args.limit and args.limit > 0 else rows

    for row in shown:
        print(
            f"{row['risk_score']:>6} {row['status']:<4} "
            f"max_gap={row['max_gap_sec']}s coverage={row['coverage_pct']}% "
            f"low_conf={row['low_confidence_segments_count']} "
            f"repeats={row['repeated_ngrams_count']} {row['file']}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
