"""
dev_config.py – Developer-only configuration helpers for Agri Doctor
=====================================================================
Provides:

  authenticate_developer(password)  → bool
  is_developer_authenticated()      → bool
  logout_developer()                → None
  save_env_credentials(api_key, project_id, endpoint) → SaveResult
  load_env_values()                 → dict[str, str]
  rotate_dev_key(old_password, new_password) → bool

Security model
--------------
* Developer access is gated by a bcrypt-hashed password stored in
  DEV_ADMIN_PASSWORD_HASH (env var or .env).  The plaintext password is
  *never* stored.  A default hash for the password "changeme" ships as the
  fallback so the app works out-of-the-box; production deployments MUST
  override DEV_ADMIN_PASSWORD_HASH in .env or Streamlit Cloud secrets.

* IBM credential values written to .env are additionally encrypted at rest
  using Fernet (AES-128-CBC + HMAC-SHA256).  The encryption key is derived
  from DEV_ENCRYPTION_KEY (env var).  If that variable is absent a fresh
  key is auto-generated and appended to .env on first use, keeping the
  setup self-contained.

* Authentication state is kept only in st.session_state under a namespaced
  key so it is scoped to the browser session and expires with it.

* All writes are atomic (write-to-temp then os.replace) to avoid partial
  .env states on a crash.

Dependencies: bcrypt, cryptography (both available via requirements.txt).
Fallback: if bcrypt is missing the module uses hashlib.sha256 (less secure,
warns the developer); if cryptography is missing values are stored as plain
env-var strings (still kept out of the UI).
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavyweight crypto deps (graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    import bcrypt as _bcrypt
    _HAS_BCRYPT = True
except ImportError:  # pragma: no cover
    _HAS_BCRYPT = False
    warnings.warn(
        "dev_config: 'bcrypt' not installed — developer password verification "
        "falls back to SHA-256 (less secure). Install bcrypt for production.",
        stacklevel=1,
    )

try:
    from cryptography.fernet import Fernet, InvalidToken
    _HAS_FERNET = True
except ImportError:  # pragma: no cover
    _HAS_FERNET = False
    warnings.warn(
        "dev_config: 'cryptography' not installed — credential values will be "
        "stored as plain text in .env (not encrypted at rest). "
        "Install cryptography for production.",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DOTENV_PATH = Path(__file__).parent / ".env"

# Session-state key for developer auth flag
_SK_DEV_AUTH    = "_dev_authenticated"
_SK_DEV_AUTH_TS = "_dev_auth_timestamp"

# How long (seconds) a developer session stays authenticated without activity
_DEV_SESSION_TTL = 3600   # 1 hour

# Default bcrypt hash for password "changeme"  (cost factor 12)
# MUST be overridden via DEV_ADMIN_PASSWORD_HASH in .env / Streamlit secrets.
_DEFAULT_BCRYPT_HASH = (
    b"$2b$12$5c9ZS.0ZG/INIPE2e9MbqejH3JeT0Ydibr.qANFep7pkfD2ST37Om"
)
# SHA-256 hex of "changeme" — used only when bcrypt is absent
_DEFAULT_SHA256_HASH = (
    "057ba03d6c44104863dc7361fe4578965d1887360f90a0895882e58a6248fc86"
)

# Env var names the config page may read/write
_ENV_VARS = {
    "api_key":    "IBM_CLOUD_API_KEY",
    "project_id": "IBM_PROJECT_ID",
    "endpoint":   "IBM_ENDPOINT_URL",
}

# Prefix added to encrypted values stored in .env so the reader can detect them
_ENC_PREFIX = "ENC:"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SaveResult:
    ok:      bool
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ss_get(key: str, default=None):
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def _ss_set(key: str, value) -> None:
    try:
        import streamlit as st
        st.session_state[key] = value
    except Exception:
        pass


def _ss_del(key: str) -> None:
    try:
        import streamlit as st
        st.session_state.pop(key, None)
    except Exception:
        pass


def _get_password_hash() -> bytes | str:
    """Return the stored developer password hash (bcrypt bytes or sha256 hex)."""
    raw = os.environ.get("DEV_ADMIN_PASSWORD_HASH", "")
    if raw:
        return raw.encode() if _HAS_BCRYPT else raw
    # Try st.secrets
    try:
        import streamlit as st
        secret = st.secrets.get("DEV_ADMIN_PASSWORD_HASH", "")
        if secret:
            return secret.encode() if _HAS_BCRYPT else secret
    except Exception:
        pass
    return _DEFAULT_BCRYPT_HASH if _HAS_BCRYPT else _DEFAULT_SHA256_HASH


def _verify_password(password: str) -> bool:
    stored = _get_password_hash()
    if _HAS_BCRYPT:
        try:
            pw_bytes = password.encode("utf-8")
            if isinstance(stored, str):
                stored = stored.encode()
            return _bcrypt.checkpw(pw_bytes, stored)
        except Exception:
            return False
    else:
        # SHA-256 fallback
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return secrets.compare_digest(digest, stored if isinstance(stored, str) else stored.decode())


def _get_or_create_fernet_key() -> bytes | None:
    """Return the Fernet encryption key, creating and persisting one if absent."""
    if not _HAS_FERNET:
        return None
    raw = os.environ.get("DEV_ENCRYPTION_KEY", "")
    if not raw:
        try:
            import streamlit as st
            raw = st.secrets.get("DEV_ENCRYPTION_KEY", "")
        except Exception:
            pass
    if raw:
        return raw.encode() if isinstance(raw, str) else raw
    # Auto-generate a key and append it to .env
    new_key = Fernet.generate_key()
    _append_env_line("DEV_ENCRYPTION_KEY", new_key.decode())
    os.environ["DEV_ENCRYPTION_KEY"] = new_key.decode()
    return new_key


def _fernet() -> "Fernet | None":
    key = _get_or_create_fernet_key()
    if key is None:
        return None
    return Fernet(key)


def _encrypt(value: str) -> str:
    """Encrypt a credential value; return _ENC_PREFIX + base64 ciphertext."""
    f = _fernet()
    if f is None:
        return value  # plain-text fallback
    return _ENC_PREFIX + f.encrypt(value.encode()).decode()


def _decrypt(raw: str) -> str:
    """Decrypt a value previously encrypted by _encrypt()."""
    if not raw.startswith(_ENC_PREFIX):
        return raw  # plain-text value (legacy or no-crypto mode)
    f = _fernet()
    if f is None:
        return raw  # can't decrypt without the library
    try:
        return f.decrypt(raw[len(_ENC_PREFIX):].encode()).decode()
    except Exception:
        return ""


def _read_dotenv_lines() -> list[str]:
    if not _DOTENV_PATH.exists():
        return []
    return _DOTENV_PATH.read_text(encoding="utf-8").splitlines()


def _write_dotenv_lines(lines: list[str]) -> None:
    """Atomically write lines to .env."""
    content = "\n".join(lines) + "\n"
    fd, tmp = tempfile.mkstemp(dir=_DOTENV_PATH.parent, prefix=".env.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, _DOTENV_PATH)
    except Exception:
        os.unlink(tmp)
        raise


def _set_env_var(key: str, value: str) -> None:
    """Update or insert KEY=VALUE in .env, then reload into os.environ."""
    lines    = _read_dotenv_lines()
    pattern  = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replaced = False
    new_lines: list[str] = []
    for line in lines:
        if pattern.match(line):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{key}={value}")
    _write_dotenv_lines(new_lines)
    os.environ[key] = _decrypt(value)   # keep os.environ in sync (decrypted)


def _append_env_line(key: str, value: str) -> None:
    """Append a KEY=VALUE line to .env (used for auto-generated keys)."""
    lines = _read_dotenv_lines()
    lines.append(f"{key}={value}")
    _write_dotenv_lines(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def authenticate_developer(password: str) -> bool:
    """
    Verify the developer password and mark the session as authenticated.
    Returns True on success.
    """
    if _verify_password(password):
        _ss_set(_SK_DEV_AUTH, True)
        _ss_set(_SK_DEV_AUTH_TS, time.time())
        return True
    return False


def is_developer_authenticated() -> bool:
    """Return True if the current session holds a valid developer auth token."""
    authed = _ss_get(_SK_DEV_AUTH, False)
    if not authed:
        return False
    ts = _ss_get(_SK_DEV_AUTH_TS, 0.0)
    if (time.time() - ts) > _DEV_SESSION_TTL:
        logout_developer()
        return False
    return True


def logout_developer() -> None:
    """Clear developer authentication from the current session."""
    _ss_del(_SK_DEV_AUTH)
    _ss_del(_SK_DEV_AUTH_TS)


def save_env_credentials(
    api_key:    str = "",
    project_id: str = "",
    endpoint:   str = "",
) -> SaveResult:
    """
    Encrypt and persist credential values to .env, then reload them into
    os.environ so the running process picks them up immediately (without
    restarting the server).

    Only non-empty values are written; pass an empty string to leave an
    existing .env entry unchanged.
    """
    written: list[str] = []
    try:
        if api_key.strip():
            _set_env_var(_ENV_VARS["api_key"],    _encrypt(api_key.strip()))
            written.append("IBM_CLOUD_API_KEY")
        if project_id.strip():
            _set_env_var(_ENV_VARS["project_id"], _encrypt(project_id.strip()))
            written.append("IBM_PROJECT_ID")
        if endpoint.strip():
            _set_env_var(_ENV_VARS["endpoint"],   _encrypt(endpoint.strip()))
            written.append("IBM_ENDPOINT_URL")
    except OSError as exc:
        return SaveResult(ok=False, message=f"Failed to write .env: {exc}")

    if not written:
        return SaveResult(ok=False, message="No values provided — nothing was saved.")

    # Reload .env so credential_manager.resolve_credentials() picks up the new values
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
    except ImportError:
        pass

    return SaveResult(
        ok=True,
        message=f"Saved and reloaded: {', '.join(written)}",
        details={"written_keys": written},
    )


def load_env_values() -> dict[str, str]:
    """
    Read the current .env values for the credential keys and return them
    *decrypted*, for pre-populating the config form.

    Returns a dict with keys: 'api_key', 'project_id', 'endpoint'.
    Missing / placeholder values are returned as empty strings.
    """
    result: dict[str, str] = {"api_key": "", "project_id": "", "endpoint": ""}
    if not _DOTENV_PATH.exists():
        return result

    pattern = re.compile(r"^\s*([A-Z_]+)\s*=\s*(.*)\s*$")
    raw: dict[str, str] = {}
    for line in _read_dotenv_lines():
        m = pattern.match(line)
        if m:
            raw[m.group(1)] = m.group(2).strip().strip('"').strip("'")

    for field_name, env_key in _ENV_VARS.items():
        v = raw.get(env_key, "")
        result[field_name] = _decrypt(v) if v else ""

    return result


def rotate_dev_key(old_password: str, new_password: str) -> bool:
    """
    Change the developer password.  Hashes the new password and writes the
    new hash to DEV_ADMIN_PASSWORD_HASH in .env.

    Returns True on success.  Requires old_password to be correct.
    """
    if not _verify_password(old_password):
        return False
    if len(new_password) < 10:
        return False
    if _HAS_BCRYPT:
        new_hash = _bcrypt.hashpw(new_password.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode()
    else:
        new_hash = hashlib.sha256(new_password.encode("utf-8")).hexdigest()
    _set_env_var("DEV_ADMIN_PASSWORD_HASH", new_hash)
    os.environ["DEV_ADMIN_PASSWORD_HASH"] = new_hash
    return True
