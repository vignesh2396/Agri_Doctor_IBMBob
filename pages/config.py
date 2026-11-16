"""
pages/config.py – Developer Configuration Page for Agri Doctor
===============================================================
Access-controlled page for viewing and updating IBM watsonx credentials
and application configuration.  Farmers / end users are not directed here
and have no way to reach it from the main application UI.

Authentication
--------------
Entry requires the developer password stored (bcrypt-hashed) in
DEV_ADMIN_PASSWORD_HASH (.env or Streamlit Cloud secret).  The default
password is "changeme" — change it via the Password Management section
after first login.

Persistence
-----------
Saved values are encrypted (Fernet/AES) and written to .env.  The running
process reloads the new values immediately so changes take effect without a
server restart.
"""

import os
from pathlib import Path

import streamlit as st

# ── Project-local modules ──────────────────────────────────────────────────
from credential_manager import (
    KNOWN_REGIONS,
    CredentialSet,
    ValidationResult,
    clear_session_credentials,
    resolve_credentials,
    store_session_credentials,
    validate_credentials,
)
from dev_config import (
    SaveResult,
    authenticate_developer,
    is_developer_authenticated,
    load_env_values,
    logout_developer,
    rotate_dev_key,
    save_env_credentials,
)

# ── .env loader (refresh on every rerun) ──────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _DOTENV_PATH = Path(__file__).parent.parent / ".env"
    _load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚙️ Developer Config – Agri Doctor",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Custom CSS — minimal, security-focused styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Header banner */
    .dev-banner {
        background: #1e293b;
        color: #f1f5f9;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .dev-banner h2 { margin: 0; font-size: 1.25rem; }
    .dev-banner .badge {
        background: #dc2626;
        color: white;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    /* Section cards */
    .config-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        background: #f8fafc;
    }
    /* Obfuscated key preview */
    .key-preview {
        font-family: monospace;
        font-size: 0.85rem;
        color: #475569;
        background: #f1f5f9;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: obfuscate a credential value for safe display
# ---------------------------------------------------------------------------

def _obfuscate(value: str, show_chars: int = 4) -> str:
    if not value:
        return "*(not set)*"
    if len(value) <= show_chars * 2:
        return "●" * len(value)
    return value[:show_chars] + "●" * (len(value) - show_chars * 2) + value[-show_chars:]


# ---------------------------------------------------------------------------
# Auth gate — renders login form; returns early if not authenticated
# ---------------------------------------------------------------------------

def _render_login() -> None:
    st.markdown(
        '<div class="dev-banner">'
        '<span style="font-size:1.5rem">🔒</span>'
        '<div><h2>Developer Configuration</h2>'
        '<span style="color:#94a3b8;font-size:0.85rem">Agri Doctor · Restricted Access</span></div>'
        '<span class="badge">DEV ONLY</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This page is restricted to **developers and administrators** only. "
        "Enter the developer password to continue."
    )

    with st.form("dev_login_form", clear_on_submit=True):
        password = st.text_input(
            "Developer Password",
            type="password",
            placeholder="Enter developer password",
        )
        submitted = st.form_submit_button("🔓 Unlock", use_container_width=True, type="primary")

    if submitted:
        if not password:
            st.error("Please enter a password.", icon="🔒")
        elif authenticate_developer(password):
            st.success("Access granted. Loading configuration…", icon="✅")
            st.rerun()
        else:
            st.error("Incorrect password. Access denied.", icon="🚫")

    st.divider()
    st.caption(
        "⚠️ Default password is **changeme** — update it immediately after first login. "
        "To change it, set `DEV_ADMIN_PASSWORD_HASH` in your `.env` file or "
        "Streamlit Cloud secrets (bcrypt hash, cost factor ≥ 12)."
    )
    st.page_link("app.py", label="← Back to Agri Doctor", icon="🌿")


# ---------------------------------------------------------------------------
# Main config UI — only shown when authenticated
# ---------------------------------------------------------------------------

def _render_config() -> None:
    # ── Header ────────────────────────────────────────────────────────────
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<div class="dev-banner">'
            '<span style="font-size:1.5rem">🔧</span>'
            '<div><h2>Developer Configuration</h2>'
            '<span style="color:#94a3b8;font-size:0.85rem">Agri Doctor · Restricted Access</span></div>'
            '<span class="badge">AUTHENTICATED</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_logout:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔓 Logout", use_container_width=True):
            logout_developer()
            clear_session_credentials()
            st.rerun()

    st.caption(
        "Changes saved here are **encrypted and written to `.env`** and take "
        "effect immediately without restarting the server."
    )

    # ── 1. Current Status ─────────────────────────────────────────────────
    st.subheader("1 · Current Credential Status")
    active = resolve_credentials()
    env_vals = load_env_values()

    status_col, detail_col = st.columns(2)
    with status_col:
        src_label = {
            "session": ("🟢", "Session override"),
            "env":     ("🔵", ".env / Secrets"),
            "default": ("🟡", "No API key set"),
        }.get(active.source, ("⚪", active.source))
        st.metric("Active Source", src_label[1], delta=None)
        if active.is_complete:
            st.success("Credentials complete — pipeline is operational.", icon="✅")
        else:
            st.error("API key missing — pipeline cannot run.", icon="🚫")

    with detail_col:
        st.markdown("**Current effective values**")
        st.markdown(f"API Key: `{_obfuscate(active.api_key)}`")
        st.markdown(f"Project ID: `{_obfuscate(active.project_id, show_chars=8)}`")
        st.markdown(f"Endpoint: `{active.endpoint or '*(not set)*'}`")
        if env_vals.get("api_key"):
            st.markdown(
                f"<span style='font-size:0.8rem;color:#64748b'>"
                f".env stored: `{_obfuscate(env_vals['api_key'])}` (encrypted at rest)</span>",
                unsafe_allow_html=True,
            )

    # ── 2. Edit & Save Credentials ────────────────────────────────────────
    st.divider()
    st.subheader("2 · Edit & Save Credentials")
    st.caption(
        "Values are encrypted (AES-128-CBC / Fernet) before writing to `.env`. "
        "Leave a field blank to keep the existing value unchanged."
    )

    with st.form("credentials_form"):
        new_api_key = st.text_input(
            "IBM Cloud API Key",
            type="password",
            placeholder="Paste new key to update (leave blank to keep current)",
            help="Manage → Access (IAM) → API keys on IBM Cloud.",
        )

        region_names   = list(KNOWN_REGIONS.keys()) + ["Custom URL…"]
        current_ep     = env_vals.get("endpoint") or active.endpoint
        matched_region = next(
            (r for r, url in KNOWN_REGIONS.items() if url == current_ep),
            "Custom URL…",
        )
        region_choice = st.selectbox(
            "Region / Endpoint",
            options=region_names,
            index=region_names.index(matched_region),
            help="Select the IBM watsonx region for your project.",
        )
        if region_choice == "Custom URL…":
            new_endpoint = st.text_input(
                "Custom Endpoint URL",
                value=current_ep if matched_region == "Custom URL…" else "",
                placeholder="https://your-region.ml.cloud.ibm.com",
            )
        else:
            new_endpoint = KNOWN_REGIONS[region_choice]
            st.caption(f"→ `{new_endpoint}`")

        new_project_id = st.text_input(
            "Project ID",
            value=env_vals.get("project_id", ""),
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="IBM watsonx.ai → Projects → Manage → General.",
        )

        col_save, col_validate = st.columns(2)
        with col_save:
            save_btn = st.form_submit_button(
                "💾 Encrypt & Save to .env",
                use_container_width=True,
                type="primary",
            )
        with col_validate:
            validate_btn = st.form_submit_button(
                "🔍 Validate Credentials",
                use_container_width=True,
            )

    # Handle Save
    if save_btn:
        result: SaveResult = save_env_credentials(
            api_key    = new_api_key,
            project_id = new_project_id,
            endpoint   = new_endpoint if (new_api_key or new_project_id or region_choice != matched_region) else "",
        )
        if result.ok:
            st.success(f"✅ {result.message}", icon="💾")
            # Also push into session so the next resolve sees the update immediately
            if new_api_key:
                store_session_credentials(
                    api_key    = new_api_key,
                    project_id = new_project_id or active.project_id,
                    endpoint   = new_endpoint   or active.endpoint,
                )
            st.rerun()
        else:
            st.error(f"❌ {result.message}", icon="🚫")

    # Handle Validate
    if validate_btn:
        test_key = new_api_key or active.api_key
        test_pid = new_project_id or active.project_id
        test_ep  = new_endpoint   or active.endpoint
        test_creds = CredentialSet(
            api_key    = test_key,
            project_id = test_pid,
            endpoint   = test_ep,
            source     = "session",
        )
        with st.spinner("Probing IBM Cloud IAM & endpoint…"):
            vr: ValidationResult = validate_credentials(test_creds)
        if vr.ok:
            st.success(vr.message, icon="✅")
        elif vr.error_type == "invalid_key":
            st.error(f"❌ Invalid API key — {vr.message}")
        elif vr.error_type == "invalid_project":
            st.error(f"❌ Invalid project/region — {vr.message}")
        elif vr.error_type in ("network", "timeout"):
            st.warning(f"⚠️ Connectivity issue — {vr.message}")
        else:
            st.error(f"❌ {vr.message}")

    # ── 3. Additional Configuration ───────────────────────────────────────
    st.divider()
    st.subheader("3 · Additional Configuration")

    with st.expander("🤖 Model & Inference Settings (read-only)", expanded=False):
        st.caption("These values are compiled into the application. Edit `app.py` to change them.")
        st.code(
            "MODEL_ID               = ibm/granite-4-h-small\n"
            "MAX_TOKENS_PATHOLOGIST = 900\n"
            "MAX_TOKENS_EXTENSION   = 1100\n"
            "TEMPERATURE            = 0.3",
            language=None,
        )

    # ── 4. Password Management ────────────────────────────────────────────
    st.divider()
    st.subheader("4 · Password Management")
    st.caption(
        "Change the developer password.  New password must be at least 10 characters."
    )
    with st.form("password_form", clear_on_submit=True):
        old_pw  = st.text_input("Current Password",  type="password", placeholder="Current password")
        new_pw  = st.text_input("New Password",       type="password", placeholder="Minimum 10 characters")
        new_pw2 = st.text_input("Confirm New Password", type="password", placeholder="Repeat new password")
        change_btn = st.form_submit_button("🔑 Change Password", use_container_width=True)

    if change_btn:
        if not old_pw or not new_pw or not new_pw2:
            st.error("All three password fields are required.", icon="🔒")
        elif new_pw != new_pw2:
            st.error("New passwords do not match.", icon="🔒")
        elif len(new_pw) < 10:
            st.error("New password must be at least 10 characters.", icon="🔒")
        else:
            if rotate_dev_key(old_pw, new_pw):
                st.success(
                    "Password changed and saved to `.env` as a bcrypt hash. "
                    "You will need to use the new password on next login.",
                    icon="✅",
                )
                logout_developer()
                st.rerun()
            else:
                st.error("Current password is incorrect.", icon="🚫")

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.page_link("app.py", label="← Back to Agri Doctor", icon="🌿")
    st.caption(
        "⚠️ Credentials are encrypted at rest using Fernet (AES-128-CBC + HMAC-SHA256). "
        "This page is not shown in the farmer-facing navigation."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if is_developer_authenticated():
        _render_config()
    else:
        _render_login()


main()
