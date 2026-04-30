"""Извлечение WAV из видео или аудио через ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ffmpeg_executable() -> str:
    """Путь к ffmpeg: сначала PATH, иначе бинарник из imageio-ffmpeg (Windows без установки ffmpeg)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise FileNotFoundError(
        "Не найден ffmpeg: установите его (https://ffmpeg.org) и добавьте в PATH, "
        "или установите зависимость: pip install imageio-ffmpeg"
    )


def extract_wav_16k_mono(video_path: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    exe = _ffmpeg_executable()
    cmd = [
        exe,
        "-y",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
