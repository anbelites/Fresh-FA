"""Local user auth helpers backed by SQLite password hashes."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Any

from src.database import DB

_PBKDF2_ALGO = "sha256"
_PBKDF2_ITERATIONS = int(os.environ.get("LOCAL_AUTH_PBKDF2_ITERATIONS", "600000"))
_PASSWORD_MIN_LENGTH = int(os.environ.get("LOCAL_AUTH_PASSWORD_MIN_LENGTH", "10"))
LOCAL_PASSWORD_POLICY_TEXT = (
    f"Пароль должен быть не короче {_PASSWORD_MIN_LENGTH} символов, содержать строчную и "
    "прописную букву, цифру и спецсимвол, без пробелов."
)


def normalize_local_username(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_local_password_strength(password: str) -> str | None:
    if not isinstance(password, str) or not password:
        return "Пароль не может быть пустым"
    if len(password) < _PASSWORD_MIN_LENGTH:
        return f"Пароль должен быть не короче {_PASSWORD_MIN_LENGTH} символов"
    if any(ch.isspace() for ch in password):
        return "Пароль не должен содержать пробелы"
    if not any(ch.islower() for ch in password):
        return "Пароль должен содержать хотя бы одну строчную букву"
    if not any(ch.isupper() for ch in password):
        return "Пароль должен содержать хотя бы одну прописную букву"
    if not any(ch.isdigit() for ch in password):
        return "Пароль должен содержать хотя бы одну цифру"
    if not any(not ch.isalnum() for ch in password):
        return "Пароль должен содержать хотя бы один спецсимвол"
    return None


def hash_local_password(password: str, *, salt: bytes | None = None) -> str:
    validation_error = validate_local_password_strength(password)
    if validation_error:
        raise ValueError(validation_error)
    raw_salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO,
        password.encode("utf-8"),
        raw_salt,
        _PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.b64encode(raw_salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_{_PBKDF2_ALGO}${_PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_local_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt_b64, digest_b64 = str(stored_hash or "").split("$", 3)
    except ValueError:
        return False
    if scheme != f"pbkdf2_{_PBKDF2_ALGO}":
        return False
    try:
        iterations = max(1, int(iterations_raw))
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except (ValueError, base64.binascii.Error):
        return False
    actual = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO,
        (password or "").encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def verify_local_user_password(
    db: DB,
    username: str,
    password: str,
) -> tuple[dict[str, Any] | None, str | None]:
    uname = normalize_local_username(username)
    if not uname or not password:
        return None, "Введите имя пользователя и пароль"
    user = db.get_user(uname)
    if not user:
        return None, "Пользователь не найден"
    if not int(user.get("is_active") or 0):
        return None, "Пользователь отключён"
    stored_hash = str(user.get("password_hash") or "")
    if not stored_hash or not verify_local_password(password, stored_hash):
        return None, "Неверное имя пользователя или пароль"
    return user, None
