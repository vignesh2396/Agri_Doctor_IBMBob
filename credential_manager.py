"""
credential_manager.py – Runtime credential management for Agri Doctor
======================================================================
Provides a single source of truth for IBM Cloud credentials used across
the application. Resolves credentials from three sources in priority order:

  Priority 1 (highest) : Session-level values entered via the website UI
  Priority 2           : Streamlit Cloud secrets  (st.secrets)
  Priority 3 (fallback): Environment variables loaded from the .env file

Credentials never persist beyond the browser session; the .env file is
never modified at runtime.

Public API
----------
  resolve_credentials()         → CredentialSet (what the pipeline actually uses)
  validate_credentials(creds)   → ValidationResult (IAM + connectivity probe)
  store_session_credentials(…)  → writes to st.session_state
  clear_session_credentials()   → resets to .env fallback
  get_credential_source()       → "session" | "env" | "none"
  KNOWN_REGIONS                 → dict of region name → base endpoint URL
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Known IBM watsonx regions (endpoint base URLs)
# ---------------------------------------------------------------------------
KNOWN_REGIONS: dict[str, str] = {
    "Frankfurt (eu-de)":     "https://eu-de.ml.cloud.ibm.com",
    "Dallas (us-south)":     "https://us-south.ml.cloud.ibm.com",
    "Tokyo (jp-tok)":        "https://jp-tok.ml.cloud.ibm.com",
    "Sydney (au-syd)":       "https://au-syd.ml.cloud.ibm.com",
}

# Default fallback values (used only when env + session are both absent)
_DEFAULT_ENDPOINT   = "https://eu-de.ml.cloud.ibm.com"
_DEFAULT_PROJECT_ID = "38de3042-7dea-4687-9452-979220f75f7b"

# Env-variable placeholder strings that should be treated as "not set"
_PLACEHOLDERS = {
    "your_ibm_cloud_api_key_here",
    "your_project_id_here",
    "https://your-region.ml.cloud.ibm.com",
}

# Session-state keys (prefixed to avoid collisions)
_SK_API_KEY    = "_cred_api_key"
_SK_PROJECT_ID = "_cred_project_id"
_SK_ENDPOINT   = "_cred_endpoint"
_SK_VALIDATED  = "_cred_validated"      # True / False / None (not yet checked)
_SK_VAL_MSG    = "_cred_val_message"
_SK_VAL_TS     = "_cred_val_timestamp"

# How long (seconds) a successful validation result is considered fresh
_VALIDATION_CACHE_TTL = 300   # 5 minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CredentialSet:
    """Resolved, ready-to-use credentials."""
    api_key:    str
    project_id: str
    endpoint:   str            # base URL, e.g. https://eu-de.ml.cloud.ibm.com
    source:     str            # "session" | "env" | "default"

    @property
    def inference_url(self) -> str:
        return f"{self.endpoint}/ml/v1/text/chat?version=2024-05-31"

    @property
    def is_complete(self) -> bool:
        """True when all three credential fields are non-empty."""
        return bool(self.api_key and self.project_id and self.endpoint)


@dataclass
class ValidationResult:
    """Outcome of a live credential validation probe."""
    ok:         bool
    message:    str
    error_type: str = ""       # "invalid_key" | "invalid_project" | "network" | "timeout" | ""
    latency_ms: int = 0
    details:    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(val: str | None) -> str:
    """Strip whitespace and return empty string for None / placeholder values."""
    if not val:
        return ""
    v = val.strip()
    return "" if v in _PLACEHOLDERS else v


def _env(key: str) -> str:
    """Read an env var, cleaning out placeholders."""
    return _clean(os.environ.get(key, ""))


def _is_valid_endpoint(url: str) -> bool:
    """Check the URL is a well-formed https IBM watsonx endpoint."""
    try:
        p = urlparse(url)
        return (
            p.scheme == "https"
            and p.netloc.endswith("ml.cloud.ibm.com")
        )
    except Exception:
        return False


def _is_plausible_api_key(key: str) -> bool:
    """Basic length + character sanity check (not a network call)."""
    return 20 <= len(key) <= 256 and not key.startswith(("'", '"'))


def _is_plausible_project_id(pid: str) -> bool:
    """UUID-shaped project ID check."""
    return bool(re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        pid, re.IGNORECASE
    ))


# ---------------------------------------------------------------------------
# Session state helpers (safe to call outside Streamlit via try/except)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _st_secret(key: str) -> str:
    """Read a value from st.secrets (Streamlit Cloud), or return '' if unavailable."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        return _clean(str(val)) if val else ""
    except Exception:
        return ""


def resolve_credentials() -> CredentialSet:
    """
    Return the best available credential set.

    Resolution order:
      1. Session-level values stored via store_session_credentials()
      2. Streamlit Cloud secrets  (st.secrets — set via Streamlit Cloud dashboard)
      3. Environment variables  IBM_CLOUD_API_KEY / IBM_PROJECT_ID / IBM_ENDPOINT_URL
      4. Hard-coded defaults for project_id and endpoint (api_key stays empty)
    """
    # ── 1. Session values ──────────────────────────────────────────────────
    s_key = _clean(_ss_get(_SK_API_KEY, ""))
    s_pid = _clean(_ss_get(_SK_PROJECT_ID, ""))
    s_ep  = _clean(_ss_get(_SK_ENDPOINT, ""))

    if s_key or s_pid or s_ep:
        # At least one field was set via the UI — use session for all three,
        # falling back to secrets/env/default for any missing piece.
        api_key    = s_key or _st_secret("IBM_CLOUD_API_KEY") or _env("IBM_CLOUD_API_KEY")
        project_id = s_pid or _st_secret("IBM_PROJECT_ID")    or _env("IBM_PROJECT_ID") or _DEFAULT_PROJECT_ID
        endpoint   = s_ep  or _st_secret("IBM_ENDPOINT_URL")  or _env("IBM_ENDPOINT_URL") or _DEFAULT_ENDPOINT
        return CredentialSet(
            api_key=api_key,
            project_id=project_id,
            endpoint=endpoint,
            source="session",
        )

    # ── 2. Streamlit Cloud secrets ─────────────────────────────────────────
    sc_key = _st_secret("IBM_CLOUD_API_KEY")
    sc_pid = _st_secret("IBM_PROJECT_ID")  or _env("IBM_PROJECT_ID") or _DEFAULT_PROJECT_ID
    sc_ep  = _st_secret("IBM_ENDPOINT_URL") or _env("IBM_ENDPOINT_URL") or _DEFAULT_ENDPOINT
    if sc_key:
        return CredentialSet(
            api_key=sc_key,
            project_id=sc_pid,
            endpoint=sc_ep,
            source="env",   # treated same as env for UI badge purposes
        )

    # ── 3. Environment variables (.env file) ───────────────────────────────
    e_key = _env("IBM_CLOUD_API_KEY")
    e_pid = _env("IBM_PROJECT_ID") or _DEFAULT_PROJECT_ID
    e_ep  = _env("IBM_ENDPOINT_URL") or _DEFAULT_ENDPOINT
    if e_key:
        return CredentialSet(
            api_key=e_key,
            project_id=e_pid,
            endpoint=e_ep,
            source="env",
        )

    # ── 4. No API key available anywhere ──────────────────────────────────
    return CredentialSet(
        api_key="",
        project_id=e_pid,
        endpoint=e_ep,
        source="default",
    )


def get_credential_source() -> str:
    """Return 'session', 'env', or 'none'."""
    creds = resolve_credentials()
    if not creds.api_key:
        return "none"
    return creds.source


def store_session_credentials(
    api_key:    str = "",
    project_id: str = "",
    endpoint:   str = "",
) -> None:
    """
    Persist user-provided values to session state only.
    Pass an empty string for any field to clear that field (fall back to .env).
    """
    _ss_set(_SK_API_KEY,    _clean(api_key))
    _ss_set(_SK_PROJECT_ID, _clean(project_id))
    _ss_set(_SK_ENDPOINT,   _clean(endpoint))
    # Invalidate any cached validation result when credentials change
    _ss_set(_SK_VALIDATED, None)
    _ss_set(_SK_VAL_MSG, "")
    _ss_set(_SK_VAL_TS, 0.0)


def clear_session_credentials() -> None:
    """Remove all session-level credentials, reverting to .env fallback."""
    for k in (_SK_API_KEY, _SK_PROJECT_ID, _SK_ENDPOINT,
              _SK_VALIDATED, _SK_VAL_MSG, _SK_VAL_TS):
        _ss_del(k)


def validate_credentials(creds: CredentialSet) -> ValidationResult:
    """
    Probe IBM Cloud IAM and optionally the watsonx endpoint to verify
    credentials are live and correct.

    Checks performed (in order):
      1. Local format checks  (fast, no network)
      2. IAM token fetch      (verifies API key is valid and active)
      3. Endpoint reachability (verifies endpoint URL is reachable)

    Results are cached in session state for _VALIDATION_CACHE_TTL seconds
    so repeated UI reruns don't hammer the IAM service.
    """
    if not creds.api_key:
        return ValidationResult(
            ok=False,
            message="No API key provided. Enter your IBM Cloud API key.",
            error_type="invalid_key",
        )

    # ── Return cached result if still fresh ───────────────────────────────
    cached_ok  = _ss_get(_SK_VALIDATED)
    cached_msg = _ss_get(_SK_VAL_MSG, "")
    cached_ts  = _ss_get(_SK_VAL_TS, 0.0)
    if cached_ok is not None and (time.time() - cached_ts) < _VALIDATION_CACHE_TTL:
        return ValidationResult(ok=cached_ok, message=cached_msg)

    # ── 1. Local format checks ─────────────────────────────────────────────
    if not _is_plausible_api_key(creds.api_key):
        return ValidationResult(
            ok=False,
            message=(
                "API key format looks invalid "
                f"(length {len(creds.api_key)}, must be 20–256 characters, no surrounding quotes)."
            ),
            error_type="invalid_key",
        )

    if not _is_plausible_project_id(creds.project_id):
        return ValidationResult(
            ok=False,
            message=(
                f"Project ID '{creds.project_id[:8]}…' does not look like a valid UUID. "
                "Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            ),
            error_type="invalid_project",
        )

    if not _is_valid_endpoint(creds.endpoint):
        return ValidationResult(
            ok=False,
            message=(
                f"Endpoint URL '{creds.endpoint}' is not a recognised IBM watsonx URL. "
                "It must be https://<region>.ml.cloud.ibm.com"
            ),
            error_type="network",
        )

    # ── 2. IAM token probe ─────────────────────────────────────────────────
    t0 = time.time()
    try:
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        auth  = IAMAuthenticator(creds.api_key)
        token = auth.token_manager.get_token()
        if not token:
            raise ValueError("Empty token returned by IAM service.")
        latency = int((time.time() - t0) * 1000)
    except Exception as exc:
        err = str(exc)
        # Classify the error
        if any(x in err.lower() for x in ("401", "403", "invalid", "unauthorized")):
            etype = "invalid_key"
            msg   = (
                "IBM Cloud API key is invalid or has been revoked. "
                "Generate a new key at IBM Cloud → Manage → Access (IAM) → API keys."
            )
        elif any(x in err.lower() for x in ("timeout", "timed out", "read timed out")):
            etype = "timeout"
            msg   = (
                "Connection to IBM IAM timed out. "
                "Check your internet connection and try again."
            )
        elif any(x in err.lower() for x in ("connection", "network", "resolve", "nodename")):
            etype = "network"
            msg   = (
                "Could not reach IBM Cloud IAM service. "
                "Verify your internet connection or corporate proxy settings."
            )
        else:
            etype = "invalid_key"
            msg   = f"Could not obtain IBM IAM token: {err[:200]}"

        result = ValidationResult(ok=False, message=msg, error_type=etype,
                                  latency_ms=int((time.time() - t0) * 1000))
        _ss_set(_SK_VALIDATED, False)
        _ss_set(_SK_VAL_MSG, msg)
        _ss_set(_SK_VAL_TS, time.time())
        return result

    # ── 3. Quick endpoint reachability check (HEAD, 10 s timeout) ─────────
    try:
        probe_url = f"{creds.endpoint}/ml/v1/foundation_model_specs?version=2024-05-31&limit=1"
        resp = requests.head(
            probe_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            allow_redirects=True,
        )
        # 200, 400, 404 all mean the endpoint is live; only 401/403 indicate
        # a project/region mismatch; connection errors mean unreachable.
        if resp.status_code in (401, 403):
            msg = (
                f"Endpoint '{creds.endpoint}' rejected the request (HTTP {resp.status_code}). "
                "Verify the region is correct for your IBM Cloud account."
            )
            result = ValidationResult(ok=False, message=msg,
                                      error_type="invalid_project",
                                      latency_ms=latency)
            _ss_set(_SK_VALIDATED, False)
            _ss_set(_SK_VAL_MSG, msg)
            _ss_set(_SK_VAL_TS, time.time())
            return result
    except requests.exceptions.Timeout:
        msg = (
            f"Endpoint '{creds.endpoint}' did not respond within 10 seconds. "
            "The region may be temporarily unavailable — try again shortly."
        )
        result = ValidationResult(ok=False, message=msg,
                                  error_type="timeout", latency_ms=10_000)
        _ss_set(_SK_VALIDATED, False)
        _ss_set(_SK_VAL_MSG, msg)
        _ss_set(_SK_VAL_TS, time.time())
        return result
    except requests.exceptions.ConnectionError:
        msg = (
            f"Cannot reach '{creds.endpoint}'. "
            "Check your internet connection or whether the region is correct."
        )
        result = ValidationResult(ok=False, message=msg,
                                  error_type="network", latency_ms=0)
        _ss_set(_SK_VALIDATED, False)
        _ss_set(_SK_VAL_MSG, msg)
        _ss_set(_SK_VAL_TS, time.time())
        return result
    except Exception:
        pass   # Other probe errors are non-fatal; IAM already passed

    # ── All checks passed ──────────────────────────────────────────────────
    msg = (
        f"Credentials validated ✓  "
        f"(IAM token obtained in {latency} ms, "
        f"endpoint {creds.endpoint} reachable)"
    )
    _ss_set(_SK_VALIDATED, True)
    _ss_set(_SK_VAL_MSG, msg)
    _ss_set(_SK_VAL_TS, time.time())
    return ValidationResult(ok=True, message=msg, latency_ms=latency)
