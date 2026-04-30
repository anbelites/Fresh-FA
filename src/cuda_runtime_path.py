"""
Подхват CUDA runtime из pip (nvidia-cublas-cu12 и др.) в LD_LIBRARY_PATH.

Иначе при GPU часто: «libcublas.so.12 is not found», хотя библиотека лежит в site-packages.
Отключить: FA_SKIP_NVIDIA_LD_PATCH=1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PATCHED = False


def ensure_nvidia_pip_libs() -> None:
    """Добавить в LD_LIBRARY_PATH каталоги с *.so под site-packages/nvidia/."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    if sys.platform != "linux":
        return
    if os.environ.get("FA_SKIP_NVIDIA_LD_PATCH", "").strip() == "1":
        return
    try:
        import site
        import sysconfig

        roots: list[Path] = []
        for key in ("purelib", "platlib"):
            try:
                p = sysconfig.get_path(key)
                if p:
                    roots.append(Path(p))
            except Exception:
                pass
        try:
            u = site.getusersitepackages()
            if isinstance(u, str):
                roots.append(Path(u))
            else:
                roots.extend(Path(x) for x in u)
        except Exception:
            pass

        seen_dirs: set[str] = set()
        for root in roots:
            nvidia = root / "nvidia"
            if not nvidia.is_dir():
                continue
            for p in nvidia.rglob("*"):
                if not p.is_file():
                    continue
                name = p.name
                if not name.startswith("lib") or ".so" not in name:
                    continue
                d = str(p.parent.resolve())
                if d not in seen_dirs:
                    seen_dirs.add(d)

        if not seen_dirs:
            return

        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [x for x in cur.split(":") if x]
        for d in sorted(seen_dirs):
            if d not in parts:
                parts.insert(0, d)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    except Exception:
        pass
