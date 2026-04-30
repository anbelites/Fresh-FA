"""NeMo Sortformer diarization backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.cuda_runtime_path import ensure_nvidia_pip_libs

ensure_nvidia_pip_libs()

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def nemo_diarization_backend_enabled() -> bool:
    v = os.environ.get("DIARIZATION_BACKEND", "auto").strip().lower()
    return v in ("auto", "nemo", "nemo_sortformer")


def nemo_device_name() -> str:
    v = os.environ.get("NEMO_DIAR_DEVICE", "").strip().lower()
    if v in ("cpu", "cuda"):
        return v
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def nemo_model_name() -> str:
    return os.environ.get("NEMO_DIAR_MODEL", "nvidia/diar_sortformer_4spk-v1").strip()


def nemo_batch_size() -> int:
    raw = os.environ.get("NEMO_DIAR_BATCH_SIZE", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def nemo_is_installed() -> bool:
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: F401

        return True
    except Exception:
        return False


def _load_sortformer_model() -> Any:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model_id = nemo_model_name()
    device = nemo_device_name()
    key = (model_id, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    p = Path(model_id)
    if p.is_file() and p.suffix == ".nemo":
        model = SortformerEncLabelModel.restore_from(
            restore_path=str(p),
            map_location=device,
            strict=False,
        )
    else:
        model = SortformerEncLabelModel.from_pretrained(model_id)
        model = model.to(torch.device(device))
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _normalize_speaker_label(raw: Any) -> str:
    if isinstance(raw, bool):
        return "spk_1" if raw else "spk_0"
    if isinstance(raw, int):
        return f"spk_{raw}"
    s = str(raw or "").strip()
    if not s:
        return "spk_unknown"
    return s


def _normalize_segment_item(item: Any) -> tuple[float, float, str] | None:
    if isinstance(item, str):
        parts = item.strip().split()
        if len(parts) >= 3:
            try:
                start_f = float(parts[0])
                end_f = float(parts[1])
            except ValueError:
                return None
            if end_f <= start_f:
                return None
            return (start_f, end_f, _normalize_speaker_label(" ".join(parts[2:])))
        return None
    if isinstance(item, dict):
        t0 = item.get("start", item.get("begin", item.get("offset")))
        t1 = item.get("end", item.get("stop", item.get("duration")))
        spk = item.get("speaker", item.get("speaker_id", item.get("label")))
        if item.get("duration") is not None and item.get("end") is None and t0 is not None:
            try:
                t1 = float(t0) + float(item["duration"])
            except (TypeError, ValueError):
                pass
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        t0, t1, spk = item[0], item[1], item[2]
    else:
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        label = getattr(item, "speaker", None)
        if label is None:
            label = getattr(item, "label", None)
        if start is None or end is None or label is None:
            return None
        t0, t1, spk = start, end, label
    try:
        start_f = float(t0)
        end_f = float(t1)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        return None
    return (start_f, end_f, _normalize_speaker_label(spk))


def _flatten_predicted_segments(raw: Any) -> list[tuple[float, float, str]]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            nested = raw[0]
            norm = [_normalize_segment_item(x) for x in nested]
            return [x for x in norm if x is not None]
        norm = [_normalize_segment_item(x) for x in raw]
        rows = [x for x in norm if x is not None]
        if rows:
            return rows
    item = _normalize_segment_item(raw)
    return [item] if item is not None else []


def _merge_adjacent_rows(
    rows: list[tuple[float, float, str]], gap_sec: float = 0.16
) -> list[tuple[float, float, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda x: (x[0], x[1], x[2]))
    merged: list[list[Any]] = [[ordered[0][0], ordered[0][1], ordered[0][2]]]
    for start, end, spk in ordered[1:]:
        prev = merged[-1]
        if spk == prev[2] and start <= float(prev[1]) + gap_sec:
            prev[1] = max(float(prev[1]), end)
        else:
            merged.append([start, end, spk])
    return [(float(s), float(e), str(spk)) for s, e, spk in merged]


def load_diarization_rows_nemo(wav_path: Path) -> list[tuple[float, float, str]]:
    model = _load_sortformer_model()
    predicted = model.diarize(audio=[str(wav_path)], batch_size=nemo_batch_size())
    rows = _flatten_predicted_segments(predicted)
    if not rows:
        raise RuntimeError("NeMo Sortformer не вернул ни одного сегмента диаризации")
    return _merge_adjacent_rows(rows)
