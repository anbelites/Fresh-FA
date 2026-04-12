"""Автообнаружение LDAP к AD на машине, присоединённой к домену (Windows)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_powershell_file(script_path: Path, timeout: float = 25.0) -> tuple[int, str, str]:
    args = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
    ]
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    r = subprocess.run(args, **kwargs)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    return r.returncode, out, err


def discover_ldap_windows() -> tuple[str, str] | None:
    """
    На присоединённом к AD сервере Windows определяет URI контроллера и DNS-имя домена.

    Возвращает (ldap://dc-fqdn:389, domain.example.com) или None, если не Windows,
    машина не в домене, или не удалось выполнить запрос.
    """
    if sys.platform != "win32":
        return None

    # Одна строка ASCII: OK|dc-fqdn|dns-domain (без JSON — меньше проблем с кодировкой консоли).
    script = r"""
$ErrorActionPreference = 'Stop'
try {
    $cs = Get-CimInstance -ClassName Win32_ComputerSystem
    if (-not $cs.PartOfDomain) { Write-Output 'NOT_JOINED'; exit 0 }
    $d = [System.DirectoryServices.ActiveDirectory.Domain]::GetCurrentDomain()
    $dc = $d.FindDomainController().Name
    $dom = $d.Name
    if ([string]::IsNullOrWhiteSpace($dc) -or [string]::IsNullOrWhiteSpace($dom)) { Write-Output 'EMPTY'; exit 0 }
    Write-Output ('OK|' + $dc + '|' + $dom)
    exit 0
} catch {
    Write-Output ('ERR|' + $_.Exception.Message)
    exit 0
}
"""
    fd, tmp_path = tempfile.mkstemp(suffix=".ps1", prefix="fa_ldap_discover_")
    os.close(fd)
    try:
        Path(tmp_path).write_text(script.strip() + "\n", encoding="utf-8")
        _code, out, _err = _run_powershell_file(Path(tmp_path))
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not out:
        return None
    first = out.splitlines()[0].strip()
    if first == "NOT_JOINED" or first == "EMPTY" or first.startswith("ERR|"):
        return None
    if not first.startswith("OK|"):
        return None
    parts = first.split("|", 2)
    if len(parts) != 3:
        return None
    dc = parts[1].strip()
    dom = parts[2].strip()
    if not dc or not dom:
        return None
    port = os.environ.get("AD_LDAP_DISCOVER_PORT", "389").strip() or "389"
    uri = f"ldap://{dc}:{port}"
    return (uri, dom.lower())
