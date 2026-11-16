"""
pages/config.py – Configuration Page for Agri Doctor
=====================================================
Allows viewing and updating IBM watsonx credentials and saving them to .env.
Credential values are encrypted at rest (Fernet/AES) in the .env file.
Changes take effect immediately without restarting the server.
"""

from pathlib import Path

import streamlit as st

from credential_manager import (
    KNOWN_REGIONS,
    CredentialSet,
    ValidationResult,
    resolve_credentials,
    store_session_credentials,
    validate_credentials,
)
from dev_config import (
    SaveResult,
    load_env_values,
    save_env_credentials,
)

# ── Reload .env on every rerun ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _DOTENV_PATH = Path(__file__).parent.parent / ".env"
    _load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚙️ Configuration – Agri Doctor",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .config-banner {
        background: #1e293b;
        color: #f1f5f9;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .config-banner h2 { margin: 0; font-size: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _obfuscate(value: str, show_chars: int = 4) -> str:
    if not value:
        return "*(not set)*"
    if len(value) <= show_chars * 2:
        return "●" * len(value)
    return value[:show_chars] + "●" * (len(value) - show_chars * 2) + value[-show_chars:]

# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="config-banner">'
    '<span style="font-size:1.5rem">🔧</span>'
    '<div><h2>Configuration</h2>'
    '<span style="color:#94a3b8;font-size:0.85rem">Agri Doctor · IBM watsonx Credentials</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── 1. Current Status ─────────────────────────────────────────────────────
st.subheader("1 · Current Credential Status")

active   = resolve_credentials()
env_vals = load_env_values()

status_col, detail_col = st.columns(2)
with status_col:
    src_label = {
        "session": "🟢 Session override",
        "env":     "🔵 .env / Secrets",
        "default": "🟡 No API key set",
    }.get(active.source, active.source)
    st.metric("Active Source", src_label)
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
        st.caption(f".env stored: `{_obfuscate(env_vals['api_key'])}` (encrypted at rest)")

# ── 2. Edit & Save Credentials ────────────────────────────────────────────
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
            "💾 Save to .env",
            use_container_width=True,
            type="primary",
        )
    with col_validate:
        validate_btn = st.form_submit_button(
            "🔍 Validate Credentials",
            use_container_width=True,
        )

if save_btn:
    result: SaveResult = save_env_credentials(
        api_key    = new_api_key,
        project_id = new_project_id,
        endpoint   = new_endpoint if (new_api_key or new_project_id or region_choice != matched_region) else "",
    )
    if result.ok:
        st.success(f"✅ {result.message}", icon="💾")
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
    test_creds = CredentialSet(
        api_key    = new_api_key    or active.api_key,
        project_id = new_project_id or active.project_id,
        endpoint   = new_endpoint   or active.endpoint,
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

# ── 3. Model Settings (read-only) ─────────────────────────────────────────
st.divider()
with st.expander("🤖 Model & Inference Settings (read-only)", expanded=False):
    st.caption("These values are compiled into the application. Edit `app.py` to change them.")
    st.code(
        "MODEL_ID               = ibm/granite-4-h-small\n"
        "MAX_TOKENS_PATHOLOGIST = 900\n"
        "MAX_TOKENS_EXTENSION   = 1100\n"
        "TEMPERATURE            = 0.3",
        language=None,
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.page_link("app.py", label="← Back to Agri Doctor", icon="🌿")
st.caption("Credentials are encrypted at rest (Fernet AES-128-CBC + HMAC-SHA256).")
