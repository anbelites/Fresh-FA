"""Проверка логина/пароля через Active Directory (LDAP / LDAPS)."""
from __future__ import annotations

import os
import re
import sys

from ldap3 import BASE, SUBTREE, Connection, Server
from ldap3.core.exceptions import LDAPBindError, LDAPException, LDAPSocketOpenError
from ldap3.utils.conv import escape_filter_chars

# Кэш после первого успешного разрешения URI/домена (в т.ч. автообнаружение на Windows).
_disc_cache: dict[str, str | bool] | None = None


def _ldap_auto_discover_allowed() -> bool:
    """По умолчанию включено на Windows, если AD_LDAP_URI не задан."""
    v = os.environ.get("AD_LDAP_AUTO_DISCOVER", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    return True


def _refresh_disc_cache() -> None:
    """Заполняет _disc_cache: явный AD_LDAP_URI или авто на доменном Windows."""
    global _disc_cache
    if _disc_cache is not None:
        return

    explicit = os.environ.get("AD_LDAP_URI", "").strip()
    if explicit:
        _disc_cache = {"ldap_uri": explicit, "auto_discovered": False}
        return

    if sys.platform == "win32" and _ldap_auto_discover_allowed():
        from src.ad_ldap_discover import discover_ldap_windows

        found = discover_ldap_windows()
        if found:
            uri, dns = found
            _disc_cache = {
                "ldap_uri": uri,
                "dns_domain": dns,
                "auto_discovered": True,
            }
            return
        raise RuntimeError(
            "AD_LDAP_URI не задан и автообнаружение LDAP не удалось "
            "(нужна машина в домене AD и доступ к WMI/.NET). "
            "Укажите AD_LDAP_URI вручную или проверьте выполнение PowerShell."
        )

    if sys.platform == "win32":
        raise RuntimeError(
            "AD_LDAP_URI не задан, а AD_LDAP_AUTO_DISCOVER отключён. "
            "Укажите AD_LDAP_URI или включите AD_LDAP_AUTO_DISCOVER=1."
        )

    raise RuntimeError(
        "AD_LDAP_URI не задан. На Linux задайте ldap:// или ldaps:// вручную."
    )


def auth_mode() -> str:
    raw = os.environ.get("AUTH_TYPE", "").strip().lower()
    if raw in {"none", "ad", "local"}:
        return raw
    if os.environ.get("AD_AUTH_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        return "ad"
    return "none"


def auth_enabled() -> bool:
    return auth_mode() != "none"


def ad_auth_enabled() -> bool:
    return auth_mode() == "ad"


def local_auth_enabled() -> bool:
    return auth_mode() == "local"


def require_session_secret() -> str:
    """Секрет подписи cookie-сессии; при включённой auth обязателен SESSION_SECRET в .env."""
    import secrets

    raw = os.environ.get("SESSION_SECRET", "").strip()
    if auth_enabled():
        if len(raw) < 16:
            raise RuntimeError(
                "AUTH_TYPE!=none: задайте в .env SESSION_SECRET (не короче 16 символов)."
            )
        return raw
    return raw or secrets.token_hex(32)


def _ldap_uri() -> str:
    if not ad_auth_enabled():
        return os.environ.get("AD_LDAP_URI", "").strip()
    _refresh_disc_cache()
    assert _disc_cache is not None
    return str(_disc_cache["ldap_uri"])


def _auto_discovered() -> bool:
    if not ad_auth_enabled():
        return False
    _refresh_disc_cache()
    assert _disc_cache is not None
    return bool(_disc_cache.get("auto_discovered"))


def _use_ssl() -> bool:
    return _ldap_uri().lower().startswith("ldaps:")


def _use_starttls() -> bool:
    if _use_ssl():
        return False
    v = os.environ.get("AD_USE_STARTTLS", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    # По умолчанию: STARTTLS для ldap:// при автообнаружении на Windows
    if _auto_discovered():
        return True
    return False


def _bind_upn_template() -> str:
    explicit = os.environ.get("AD_BIND_UPN_TEMPLATE", "").strip()
    if explicit:
        return explicit
    _refresh_disc_cache()
    assert _disc_cache is not None
    dns = _disc_cache.get("dns_domain")
    if isinstance(dns, str) and dns.strip():
        return "{username}@" + dns.strip().lower()
    raise RuntimeError(
        "Задайте AD_BIND_UPN_TEMPLATE (например {username}@corp.local) "
        "или включите автообнаружение на Windows при пустом AD_LDAP_URI."
    )


def _ldap_bind_principal(username: str) -> tuple[str, str | None]:
    """
    Строка для LDAP bind в AD:
    - вид DOMAIN\\sAMAccountName (как CRM\\a.beloborodov) — используем как есть;
    - иначе UPN из AD_BIND_UPN_TEMPLATE и {username}.
    """
    u = (username or "").strip()
    if not u:
        return "", "Введите имя пользователя"
    if "\\" in u:
        return u, None
    try:
        template = _bind_upn_template()
    except RuntimeError as e:
        return "", str(e)
    try:
        return template.format(username=u), None
    except KeyError as e:
        return "", f"В AD_BIND_UPN_TEMPLATE не хватает плейсхолдера: {e}"


def _bind_sequence(
    server: Server, user_dn: str, password: str
) -> tuple[bool, str | None]:
    """Попытка bind; (True, None) или (False, сообщение для пользователя)."""
    uri = _ldap_uri()
    try:
        conn = Connection(
            server,
            user=user_dn,
            password=password,
            auto_bind=False,
            receive_timeout=int(os.environ.get("AD_LDAP_TIMEOUT_SEC", "15")),
        )
        try:
            conn.open()
        except (LDAPSocketOpenError, OSError) as e:
            return False, _ldap_connect_hint(uri, str(e))
        # ldap3: open() при успехе возвращает None — нельзя писать «if not conn.open()».
        if getattr(conn, "closed", True):
            le = getattr(conn, "last_error", None) or getattr(
                conn, "connection_error", None
            )
            return False, _ldap_connect_hint(uri, le)

        if _use_starttls():
            try:
                tls_ok = conn.start_tls()
            except LDAPException as e:
                try:
                    conn.unbind()
                except Exception:
                    pass
                return (
                    False,
                    f"STARTTLS не удалось: {e}. "
                    "Попробуйте в .env: AD_USE_STARTTLS=0 или используйте ldaps://…:636",
                )
            if not tls_ok:
                err = conn.last_error or "start_tls failed"
                try:
                    conn.unbind()
                except Exception:
                    pass
                return (
                    False,
                    f"STARTTLS: {err}. "
                    "Попробуйте AD_USE_STARTTLS=0 или ldaps://хост:636",
                )

        if not conn.bind():
            err = conn.last_error or "bind failed"
            conn.unbind()
            return False, _friendly_bind_error(err)
        conn.unbind()
        return True, None
    except LDAPBindError:
        return False, "Неверное имя пользователя или пароль"
    except LDAPSocketOpenError as e:
        return False, _ldap_connect_hint(uri, str(e))
    except LDAPException as e:
        return False, f"LDAP: {e}"
    except OSError as e:
        return False, _ldap_connect_hint(uri, str(e))


def _ldap_connect_hint(uri: str, detail: object | None) -> str:
    """Пояснение при недоступности LDAP (сеть/порт)."""
    d = str(detail).strip() if detail else ""
    base = (
        f"Нет соединения с LDAP ({uri}). "
        "Соединение открывается с машины, где запущено приложение Python, а не из браузера."
    )
    if d:
        base += f" Деталь: {d[:400]}"
    base += (
        " Проверьте: доступен ли контроллер по сети, открыт ли порт "
        "(обычно 389 для ldap://, 636 для ldaps://), брандмауэр. "
        "Команда на сервере приложения: "
        "Test-NetConnection <IP> -Port 389"
    )
    return base


def _friendly_bind_error(err: str) -> str:
    e = (err or "").lower()
    if "invalidcredentials" in e or "invalid credentials" in e:
        return "Неверное имя пользователя или пароль"
    if "52e" in e or "525" in e:  # типичные коды AD при неверной паре логин/пароль
        return "Неверное имя пользователя или пароль"
    return str(err)[:200]


def _verify_via_upn(username: str, password: str) -> tuple[bool, str | None]:
    uri = _ldap_uri()
    if not uri:
        return False, "Не задан AD_LDAP_URI"

    principal, err = _ldap_bind_principal(username)
    if err:
        return False, err

    server = Server(uri, connect_timeout=10)
    return _bind_sequence(server, principal, password)


def _verify_via_search(username: str, password: str) -> tuple[bool, str | None]:
    """Сервисная учётка ищет DN пользователя, затем проверка пароля bind-ом."""
    uri = _ldap_uri()
    service_dn = os.environ.get("AD_SERVICE_BIND_DN", "").strip()
    service_pw = os.environ.get("AD_SERVICE_BIND_PASSWORD", "")
    search_base = os.environ.get("AD_SEARCH_BASE", "").strip()
    filt_tmpl = os.environ.get(
        "AD_USER_SEARCH_FILTER", "(sAMAccountName={username})"
    ).strip()

    if not uri or not service_dn or not search_base:
        return False, "Для поиска по каталогу нужны AD_LDAP_URI, AD_SERVICE_BIND_DN, AD_SEARCH_BASE"

    safe = escape_filter_chars(username)
    search_filter = filt_tmpl.replace("{username}", safe)

    server = Server(uri, connect_timeout=10)
    user_dn: str | None = None
    try:
        conn = Connection(
            server,
            user=service_dn,
            password=service_pw,
            auto_bind=False,
            receive_timeout=int(os.environ.get("AD_LDAP_TIMEOUT_SEC", "15")),
        )
        try:
            conn.open()
        except (LDAPSocketOpenError, OSError):
            return False, "Не удалось подключиться к LDAP (сервисная учётка)"
        if getattr(conn, "closed", True):
            return False, "Не удалось подключиться к LDAP (сервисная учётка)"
        if _use_starttls():
            conn.start_tls()
        if not conn.bind():
            conn.unbind()
            return False, "Сервисная учётка AD: неверный пароль или DN"
        scope = SUBTREE
        if os.environ.get("AD_SEARCH_SCOPE", "").strip().upper() == "BASE":
            scope = BASE
        conn.search(
            search_base,
            search_filter,
            search_scope=scope,
            attributes=["1.1"],
            size_limit=1,
        )
        if not conn.entries:
            conn.unbind()
            return False, "Пользователь не найден в каталоге"
        user_dn = str(conn.entries[0].entry_dn)
        conn.unbind()
    except LDAPException as e:
        return False, f"LDAP (поиск): {e}"

    if not user_dn:
        return False, "Пустой DN пользователя"

    return _bind_sequence(server, user_dn, password)


def verify_user_password(username: str, password: str) -> tuple[bool, str | None]:
    """
    Проверка учётных данных в AD.
    Режим 1 (простой): AD_BIND_UPN_TEMPLATE — bind как {username}@domain.
    Режим 2: заданы AD_SERVICE_BIND_DN + AD_SEARCH_BASE — поиск DN, затем bind.
    """
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return False, "Введите имя пользователя и пароль"

    if not re.match(r"^[\w.\-@\\]{1,256}$", username):
        return False, "Некорректные символы в имени пользователя"

    try:
        if _ldap_uri() and os.environ.get("AD_SERVICE_BIND_DN", "").strip():
            return _verify_via_search(username, password)
        return _verify_via_upn(username, password)
    except RuntimeError as e:
        return False, str(e)[:800]


def _entry_attr_first(entry, name: str) -> str | None:
    try:
        if name not in entry:
            return None
        v = entry[name].value
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _format_display_from_entry(entry) -> str | None:
    d = _entry_attr_first(entry, "displayName")
    if d:
        return d
    gn = _entry_attr_first(entry, "givenName") or ""
    sn = _entry_attr_first(entry, "sn") or ""
    if gn.strip() or sn.strip():
        return f"{gn.strip()} {sn.strip()}".strip()
    c = _entry_attr_first(entry, "cn")
    if c:
        return c
    return None


def _user_ldap_filter(raw: str) -> str:
    """Фильтр одной учётной записи пользователя в AD."""
    u = (raw or "").strip()
    if "\\" in u:
        sam = u.split("\\", 1)[1].strip()
        return f"(sAMAccountName={escape_filter_chars(sam)})"
    if "@" in u:
        return f"(userPrincipalName={escape_filter_chars(u)})"
    return f"(sAMAccountName={escape_filter_chars(u)})"


def fetch_ad_display_name(username: str, password: str) -> str | None:
    """
    Подтягивает ФИО (displayName или givenName+sn) из AD после успешного входа.
    При AD_SKIP_DISPLAY_NAME=1 не вызывает LDAP.
    """
    if os.environ.get("AD_SKIP_DISPLAY_NAME", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return None
    if os.environ.get("AD_SERVICE_BIND_DN", "").strip():
        return _fetch_display_name_via_service(username)
    return _fetch_display_name_via_user_bind(username, password)


def _fetch_display_name_via_service(username: str) -> str | None:
    """Поиск по каталогу под сервисной УЗ (тот же фильтр, что и при входе)."""
    uri = _ldap_uri()
    service_dn = os.environ.get("AD_SERVICE_BIND_DN", "").strip()
    service_pw = os.environ.get("AD_SERVICE_BIND_PASSWORD", "")
    search_base = os.environ.get("AD_SEARCH_BASE", "").strip()
    filt_tmpl = os.environ.get(
        "AD_USER_SEARCH_FILTER", "(sAMAccountName={username})"
    ).strip()
    if not uri or not service_dn or not search_base:
        return None
    safe = escape_filter_chars(username)
    filt = filt_tmpl.replace("{username}", safe)
    attrs = ["displayName", "givenName", "sn", "cn"]
    server = Server(uri, connect_timeout=10)
    try:
        conn = Connection(
            server,
            user=service_dn,
            password=service_pw,
            auto_bind=False,
            receive_timeout=int(os.environ.get("AD_LDAP_TIMEOUT_SEC", "15")),
        )
        try:
            conn.open()
        except (LDAPSocketOpenError, OSError):
            return None
        if getattr(conn, "closed", True):
            return None
        if _use_starttls():
            try:
                if not conn.start_tls():
                    conn.unbind()
                    return None
            except LDAPException:
                try:
                    conn.unbind()
                except Exception:
                    pass
                return None
        if not conn.bind():
            try:
                conn.unbind()
            except Exception:
                pass
            return None
        scope = SUBTREE
        if os.environ.get("AD_SEARCH_SCOPE", "").strip().upper() == "BASE":
            scope = BASE
        conn.search(
            search_base,
            filt,
            search_scope=scope,
            attributes=attrs,
            size_limit=3,
        )
        if not conn.entries:
            conn.unbind()
            return None
        out = _format_display_from_entry(conn.entries[0])
        conn.unbind()
        return out
    except (LDAPException, OSError):
        return None


def _fetch_display_name_via_user_bind(username: str, password: str) -> str | None:
    """Bind тем же пользователем, root DSE → defaultNamingContext, поиск учётки."""
    principal, err = _ldap_bind_principal(username)
    if err or not principal:
        return None
    uri = _ldap_uri()
    if not uri:
        return None
    explicit_base = os.environ.get("AD_USER_SEARCH_BASE", "").strip()
    attrs = ["displayName", "givenName", "sn", "cn"]
    server = Server(uri, connect_timeout=10)
    conn = Connection(
        server,
        user=principal,
        password=password,
        auto_bind=False,
        receive_timeout=int(os.environ.get("AD_LDAP_TIMEOUT_SEC", "15")),
    )
    try:
        try:
            conn.open()
        except (LDAPSocketOpenError, OSError):
            return None
        if getattr(conn, "closed", True):
            return None
        if _use_starttls():
            try:
                if not conn.start_tls():
                    try:
                        conn.unbind()
                    except Exception:
                        pass
                    return None
            except LDAPException:
                try:
                    conn.unbind()
                except Exception:
                    pass
                return None
        if not conn.bind():
            try:
                conn.unbind()
            except Exception:
                pass
            return None

        base = explicit_base
        if not base:
            conn.search("", "(objectClass=*)", BASE, attributes=["defaultNamingContext"])
            if not conn.entries:
                conn.unbind()
                return None
            ent0 = conn.entries[0]
            if "defaultNamingContext" not in ent0:
                conn.unbind()
                return None
            dnc = ent0["defaultNamingContext"].value
            if isinstance(dnc, list):
                dnc = dnc[0] if dnc else None
            if not dnc:
                conn.unbind()
                return None
            base = str(dnc).strip()

        flt = _user_ldap_filter(username)
        conn.search(base, flt, SUBTREE, attributes=attrs, size_limit=5)
        if not conn.entries:
            conn.unbind()
            return None
        out = _format_display_from_entry(conn.entries[0])
        conn.unbind()
        return out
    except (LDAPException, OSError):
        try:
            conn.unbind()
        except Exception:
            pass
        return None


def validate_ad_config_at_startup() -> None:
    """Вызывать при старте приложения, если включён AD. LDAP не опрашиваем — только обязательные поля ОС."""
    if not ad_auth_enabled():
        return
    explicit_uri = os.environ.get("AD_LDAP_URI", "").strip()
    if sys.platform != "win32" and not explicit_uri:
        raise RuntimeError(
            "AD_AUTH_ENABLED=1: задайте AD_LDAP_URI (ldap:// или ldaps://). "
            "Автообнаружение DC доступно только на Windows в домене."
        )
    if os.environ.get("AD_SERVICE_BIND_DN", "").strip():
        if not os.environ.get("AD_SEARCH_BASE", "").strip():
            raise RuntimeError(
                "AD_AUTH_ENABLED=1: для режима поиска LDAP задайте AD_SEARCH_BASE."
            )
