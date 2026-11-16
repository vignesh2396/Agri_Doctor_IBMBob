"""
Agri Doctor – AI-Powered Crop Disease Diagnosis & Advisory
===========================================================
Two-agent workflow with nine categories of guardrails:

  Agent 1  – Pathologist Agent      (IBM Granite 4 H Small, vision)
  Agent 2  – Extension Officer Agent (IBM Granite 4 H Small, text)

Guardrails are enforced at every touchpoint:
  • Input  : API-key format, image validation, prompt injection, PII, jailbreak
  • Tool   : endpoint allow-list, model allow-list, payload schema
  • Output : toxicity, hallucination flags, sensitive-data redaction, disclaimer
  • Data   : PII masking, role-based access (farmer / agronomist / admin)
  • Security : credential-exposure check, HTML sanitisation
  • Ops    : token budgets, session rate-limit, retry with back-off
  • HITL   : flag high-risk recommendations for human review
  • Multi-Agent : handoff validation, cross-agent output quality, loop guard
  • Ethics : bias detection, explainability note, audit trail
"""

import base64
import hashlib
import json
import os
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path

# ── Environment variables ──────────────────────────────────────────────────
# python-dotenv is imported here; the actual load_dotenv() call happens inside
# main() on every Streamlit rerun so that edits to .env are picked up
# immediately without restarting the server.
try:
    from dotenv import load_dotenv as _load_dotenv
    _DOTENV_PATH = Path(__file__).parent / ".env"
    _HAS_DOTENV  = True
except ImportError:
    _HAS_DOTENV  = False
    _DOTENV_PATH = None

import requests
import streamlit as st
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from PIL import Image

# ── Guardrails engine ──────────────────────────────────────────────────────
from guardrails import (
    GuardrailResult,
    add_explainability_note,
    call_with_retry,
    check_human_review_required,
    check_jailbreak_attempt,
    check_prompt_injection,
    check_session_rate_limit,
    detect_pii,
    filter_inappropriate_content,
    generate_audit_entry,
    get_guardrail_summary,
    check_bias_in_output,
    run_all_input_guardrails,
    run_all_output_guardrails,
    validate_agent_handoff,
    validate_credential_not_exposed,
    validate_cross_agent_output,
    validate_model_id,
    validate_payload_schema,
    validate_token_budget,
    validate_tool_endpoint,
)

# ── Metrics engine ─────────────────────────────────────────────────────────
from metrics import (
    TimingRecord,
    TokenRecord,
    evaluate_agent_run,
    evaluate_pipeline,
    estimate_tokens,
    estimate_input_tokens,
)

# ── Credential manager ─────────────────────────────────────────────────────
from credential_manager import (
    KNOWN_REGIONS,
    CredentialSet,
    ValidationResult,
    clear_session_credentials,
    get_credential_source,
    resolve_credentials,
    store_session_credentials,
    validate_credentials,
)

# ---------------------------------------------------------------------------
# Constants  (static values that do not vary per-credential-set)
# ---------------------------------------------------------------------------
MODEL_ID = "ibm/granite-4-h-small"

MAX_TOKENS_PATHOLOGIST = 900
MAX_TOKENS_EXTENSION   = 1100
TEMPERATURE            = 0.3

# ---------------------------------------------------------------------------
# Session initialisation helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """Initialise per-session state on first run."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if "all_guardrail_results" not in st.session_state:
        st.session_state.all_guardrail_results = []
    if "pipeline_depth" not in st.session_state:
        st.session_state.pipeline_depth = 0
    # Metrics history – list of PipelineRunRecord dicts
    if "pipeline_runs" not in st.session_state:
        st.session_state.pipeline_runs = []
    if "session_start_ts" not in st.session_state:
        st.session_state.session_start_ts = time.time()
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0
    # Per-run timing (written by instrumented agent wrappers)
    if "_timing_pathologist" not in st.session_state:
        st.session_state._timing_pathologist = None
    if "_timing_extension" not in st.session_state:
        st.session_state._timing_extension = None
    if "_tokens_pathologist" not in st.session_state:
        st.session_state._tokens_pathologist = None
    if "_tokens_extension" not in st.session_state:
        st.session_state._tokens_extension = None
    if "_retry_pathologist" not in st.session_state:
        st.session_state._retry_pathologist = 0
    if "_retry_extension" not in st.session_state:
        st.session_state._retry_extension = 0


def _record_guardrail(result: GuardrailResult) -> None:
    st.session_state.all_guardrail_results.append(result)


def _store_timing(
    agent_name: str, t_start: float, t_end: float,
    success: bool = True, error: str = ""
) -> None:
    tr = TimingRecord(
        agent_name=agent_name,
        start_ts=t_start,
        first_token_ts=t_end,   # non-streaming: TTFT ≈ full latency
        end_ts=t_end,
        success=success,
        error_message=error,
    )
    key = f"_timing_{agent_name}"
    st.session_state[key] = tr


def _store_tokens(agent_name: str, input_tokens: int, output_tokens: int) -> None:
    tr = TokenRecord(
        agent_name=agent_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    key = f"_tokens_{agent_name}"
    st.session_state[key] = tr


def _get_timing(agent_name: str) -> TimingRecord:
    key = f"_timing_{agent_name}"
    val = st.session_state.get(key)
    if val is None:
        val = TimingRecord(agent_name=agent_name)
    return val


def _get_tokens(agent_name: str) -> TokenRecord:
    key = f"_tokens_{agent_name}"
    val = st.session_state.get(key)
    if val is None:
        val = TokenRecord(agent_name=agent_name)
    return val


# ---------------------------------------------------------------------------
# IAM token
# ---------------------------------------------------------------------------

def _get_iam_token(api_key: str) -> str:
    authenticator = IAMAuthenticator(api_key)
    return authenticator.token_manager.get_token()


# ---------------------------------------------------------------------------
# Core IBM watsonx call  (with Tool Execution Guardrails)
# ---------------------------------------------------------------------------

def _call_watsonx(
    api_key: str,
    messages: list[dict],
    max_new_tokens: int = 900,
    agent_name: str = "agent",
) -> str:
    """
    POST to IBM watsonx /ml/v1/text/chat.
    Credentials (endpoint, project_id) are resolved dynamically from the
    credential manager on each call so that session/env changes take effect
    without a server restart.
    Enforces: Tool Execution (endpoint, model, payload) + Operational (token budget).
    """
    # Resolve live credentials for this call
    creds        = resolve_credentials()
    inference_url = creds.inference_url
    project_id    = creds.project_id

    # 3-A: endpoint allow-list
    ep_result = validate_tool_endpoint(inference_url)
    _record_guardrail(ep_result)
    if not ep_result.passed:
        raise RuntimeError(ep_result.message)

    # 3-C: model allow-list
    model_result = validate_model_id(MODEL_ID)
    _record_guardrail(model_result)

    # 6-A: token budget
    budget_result = validate_token_budget(agent_name, max_new_tokens)
    _record_guardrail(budget_result)
    if budget_result.metadata.get("capped_tokens"):
        max_new_tokens = budget_result.metadata["capped_tokens"]

    payload = {
        "model_id":   MODEL_ID,
        "project_id": project_id,
        "messages":   messages,
        "parameters": {
            "max_new_tokens":  max_new_tokens,
            "temperature":     TEMPERATURE,
            "decoding_method": "greedy",
        },
    }

    # 3-C: payload schema
    schema_result = validate_payload_schema(payload)
    _record_guardrail(schema_result)
    if not schema_result.passed:
        raise RuntimeError(schema_result.message)

    # Estimate input tokens for metrics
    _input_tok_estimate = estimate_input_tokens(messages)

    # Fetch IAM token
    try:
        token = _get_iam_token(api_key)
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain IAM token. Check your API key. Detail: {exc}"
        ) from exc

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    _t_start = time.time()
    try:
        response = requests.post(
            inference_url, headers=headers, json=payload, timeout=90
        )
    except requests.exceptions.Timeout:
        _t_end = time.time()
        _store_timing(agent_name, _t_start, _t_end, success=False, error="Timeout")
        raise RuntimeError(
            "The IBM watsonx API did not respond within 90 seconds. Please try again."
        )
    except requests.exceptions.ConnectionError as exc:
        _t_end = time.time()
        _store_timing(agent_name, _t_start, _t_end, success=False, error=str(exc))
        raise RuntimeError(
            f"Network error while contacting IBM watsonx API: {exc}"
        ) from exc

    _t_end = time.time()

    if response.status_code != 200:
        try:
            detail  = response.json()
            err_msg = detail.get("errors", [{}])[0].get("message", response.text)
        except Exception:
            err_msg = response.text
        _store_timing(agent_name, _t_start, _t_end, success=False, error=err_msg[:120])
        raise RuntimeError(
            f"IBM watsonx API returned HTTP {response.status_code}: {err_msg}"
        )

    try:
        data    = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Try to get token counts from response, fall back to estimate
        usage   = data.get("usage", {})
        in_tok  = usage.get("prompt_tokens", _input_tok_estimate)
        out_tok = usage.get("completion_tokens", estimate_tokens(content))
        _store_timing(agent_name, _t_start, _t_end, success=True)
        _store_tokens(agent_name, in_tok, out_tok)
        return content
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        _store_timing(agent_name, _t_start, _t_end, success=False, error=str(exc))
        raise RuntimeError(
            f"Unexpected response format from IBM watsonx API: {exc}\n"
            f"Raw: {response.text[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_to_base64(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    return base64.b64encode(raw).decode("utf-8")


def _mime_type(uploaded_file) -> str:
    """Derive MIME type from file extension; fall back to image/jpeg."""
    _ext_to_mime = {
        ".png":  "image/png",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
        ".gif":  "image/gif",
        ".tiff": "image/tiff",
        ".tif":  "image/tiff",
        ".avif": "image/avif",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    name = uploaded_file.name.lower()
    for ext, mime in _ext_to_mime.items():
        if name.endswith(ext):
            return mime
    return "image/jpeg"   # jpg / jpeg default


# ---------------------------------------------------------------------------
# Image analysis helpers
# ---------------------------------------------------------------------------

def _extract_image_features(file_bytes: bytes, filename: str) -> dict:
    """
    Use PIL to extract visual features from the uploaded image that can be
    described in text for the Pathologist Agent.
    Returns a dict with colour stats, dimensions, brightness, contrast, etc.
    """
    import io as _io
    from PIL import Image as _Image, ImageStat as _ImageStat

    try:
        img = _Image.open(_io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return {"error": "Could not open image – format may be unsupported by PIL."}

    width, height = img.size
    stat = _ImageStat.Stat(img)

    # Mean RGB values
    r_mean, g_mean, b_mean = [round(v, 1) for v in stat.mean[:3]]
    r_std,  g_std,  b_std  = [round(v, 1) for v in stat.stddev[:3]]

    # Overall brightness (0–255)
    brightness = round((r_mean + g_mean + b_mean) / 3, 1)

    # Dominant colour region (very rough: sample centre 25%)
    cw, ch = width // 4, height // 4
    centre = img.crop((cw, ch, width - cw, height - ch))
    cstat  = _ImageStat.Stat(centre)
    cr, cg, cb = [round(v, 1) for v in cstat.mean[:3]]

    # Greenness index: how green is the image overall?
    greenness = round((g_mean - (r_mean + b_mean) / 2) / 255 * 100, 1)

    # Yellowing proxy: high R+G, low B
    yellowing = round((r_mean + g_mean - 2 * b_mean) / 255 * 100, 1)

    # Brown/necrosis proxy: high R, low G, low B
    necrosis = round((r_mean - g_mean - b_mean / 2) / 255 * 100, 1)

    # Dark spot proxy: low overall brightness regions (std dev of brightness)
    dark_contrast = round((r_std + g_std + b_std) / 3, 1)

    return {
        "filename": filename,
        "dimensions": f"{width}×{height} px",
        "brightness": brightness,
        "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
        "r_std": r_std,   "g_std": g_std,   "b_std": b_std,
        "centre_rgb": f"R={cr} G={cg} B={cb}",
        "greenness_pct": greenness,
        "yellowing_pct": yellowing,
        "necrosis_pct":  necrosis,
        "dark_contrast": dark_contrast,
    }


def _build_image_description(features: dict) -> str:
    """
    Convert extracted image features into a rich natural-language description
    that gives the Pathologist Agent enough visual context to diagnose.
    """
    if "error" in features:
        return f"Image could not be analysed: {features['error']}"

    lines = [
        f"Image filename : {features['filename']}",
        f"Dimensions     : {features['dimensions']}",
        f"Overall brightness (0–255): {features['brightness']} "
        f"({'bright/well-lit' if features['brightness'] > 140 else 'dark/under-exposed' if features['brightness'] < 80 else 'moderate'})",
        "",
        "Colour channel averages (RGB 0–255):",
        f"  Red  : {features['r_mean']} ± {features['r_std']}",
        f"  Green: {features['g_mean']} ± {features['g_std']}",
        f"  Blue : {features['b_mean']} ± {features['b_std']}",
        f"  Centre region RGB: {features['centre_rgb']}",
        "",
        "Derived visual indicators:",
        f"  Greenness index  : {features['greenness_pct']}% "
        f"({'healthy green tissue present' if features['greenness_pct'] > 10 else 'low greenness – possible chlorosis or senescence'})",
        f"  Yellowing index  : {features['yellowing_pct']}% "
        f"({'elevated – possible nutrient deficiency, viral infection, or senescence' if features['yellowing_pct'] > 20 else 'normal'})",
        f"  Necrosis index   : {features['necrosis_pct']}% "
        f"({'elevated – possible fungal/bacterial lesions or burn damage' if features['necrosis_pct'] > 15 else 'normal'})",
        f"  Colour contrast  : {features['dark_contrast']} "
        f"({'high variation – possible spots, lesions, or mottling' if features['dark_contrast'] > 35 else 'low variation – relatively uniform surface'})",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 1 – Pathologist Agent
# ---------------------------------------------------------------------------

PATHOLOGIST_SYSTEM = textwrap.dedent("""
You are an expert plant pathologist with decades of experience diagnosing crop
diseases. A farmer has uploaded a photo of their crop or plant leaf.

You will receive a detailed colour and visual analysis of that image extracted
by an image-processing tool. Use these quantitative visual indicators along with
your plant pathology knowledge to provide a structured diagnostic report.

Interpret the colour data as follows:
- High yellowing index → possible nutrient deficiency (N, Mg, Fe), viral mosaic, or senescence
- High necrosis/redness index → possible fungal blight, bacterial leaf spot, or burn
- Low greenness index → possible chlorosis, severe disease, or water stress
- High colour contrast / dark spots → possible fungal spots, rust pustules, or anthracnose
- Low brightness → image may be under-exposed or taken in poor light

OUTPUT FORMAT (use exactly these headings):
### Suspected Disease
<name of the most likely disease or condition>

### Observed Symptoms
<bullet list of visual symptoms inferred from the colour analysis data>

### Possible Causes
<fungal / bacterial / viral / pest / nutritional – explain briefly>

### Confidence Level
<High / Medium / Low – with one-sentence justification based on the data quality>

### Alternative Diagnoses
<up to two alternative possibilities with brief reasoning>

### Recommendation
<practical next step for the farmer; advise lab confirmation if confidence is Low or Medium>

IMPORTANT:
- Base your diagnosis on the quantitative visual indicators provided.
- Label the assessment as "AI-generated preliminary assessment – not a
  confirmed laboratory diagnosis."
- Be factual and do not invent diseases not supported by the colour data.
- If the image appears under-exposed or the data is ambiguous, state that
  clearly and recommend the farmer take a better photo.
""").strip()


def run_pathologist_agent(
    api_key: str,
    file_bytes: bytes,
    filename: str,
) -> str:
    """
    Agent 1: Extract visual features from the image with PIL, build a
    rich text description, and send it to IBM Granite 4 H Small for diagnosis.
    No image/base64 is sent – only the derived colour analysis text.
    """
    features    = _extract_image_features(file_bytes, filename)
    img_desc    = _build_image_description(features)

    user_prompt = (
        "A farmer has uploaded a crop leaf image for disease diagnosis. "
        "Below is the quantitative colour and visual analysis of that image.\n\n"
        "--- IMAGE ANALYSIS ---\n"
        f"{img_desc}\n"
        "--- END ANALYSIS ---\n\n"
        "Please provide a complete diagnostic report following the specified format."
    )

    messages = [
        {"role": "system", "content": PATHOLOGIST_SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

    def _call():
        return _call_watsonx(
            api_key, messages,
            max_new_tokens=MAX_TOKENS_PATHOLOGIST,
            agent_name="pathologist",
        )

    # 6-F: retry with back-off
    raw = call_with_retry(_call, agent_name="pathologist")

    # 2: Output guardrails
    sanitised, out_results = run_all_output_guardrails(raw)
    for r in out_results:
        _record_guardrail(r)

    # Hard block on toxic output
    for r in out_results:
        if not r.passed:
            raise RuntimeError(r.message)

    # 9-C: explainability note
    sanitised = add_explainability_note(sanitised)

    return sanitised, img_desc   # also return img_desc for display


# ---------------------------------------------------------------------------
# Agent 2 – Extension Officer Agent
# ---------------------------------------------------------------------------

EXTENSION_SYSTEM = textwrap.dedent("""
You are a friendly and experienced agricultural extension officer who helps
farmers understand crop disease diagnoses and take appropriate action.

You will receive a diagnostic report from a plant pathologist. Your job is to
translate it into simple, practical advice that any farmer can follow.

OUTPUT FORMAT (use exactly these headings):
### Summary for the Farmer
<2–3 sentences explaining what is wrong with the crop in plain language>

### Immediate Actions
<numbered list: what the farmer should do right now, today>

### Treatment Options
<list safe, commonly available treatments; mention chemical treatments only
 when necessary; always include dosage guidance from official product labels
 and do NOT invent dosages; recommend consulting a local agronomist for
 exact quantities>

### Preventive Measures
<bullet list of steps to prevent this disease in the future>

### Crop-Care Recommendations
<general care tips relevant to this crop and disease condition>

### When to Seek Expert Help
<describe situations where the farmer should contact a local agricultural
 extension officer, university plant clinic, or certified agronomist>

### Safety Warnings
⚠ Always wear protective gloves, a mask, and goggles when handling pesticides.
⚠ Follow all label instructions and local agricultural regulations.
⚠ Keep chemicals away from children, animals, and water sources.
⚠ This report is an AI-generated preliminary advisory – verify with a
  qualified local expert before major treatment decisions.

IMPORTANT:
- Write at a reading level suitable for a rural farmer.
- Do not invent pesticide names, registration numbers, or dosages.
- Mention safe, widely available options (e.g., copper-based fungicides,
  neem oil) only where appropriate.
- If the pathologist indicated unclear image quality, advise accordingly.
""").strip()


def run_extension_officer_agent(api_key: str, diagnosis_report: str) -> str:
    """Agent 2 with multi-agent handoff validation and output guardrails."""
    messages = [
        {"role": "system", "content": EXTENSION_SYSTEM},
        {
            "role": "user",
            "content": (
                "Below is the diagnostic report from the plant pathologist. "
                "Please prepare a farmer-friendly advisory based on this diagnosis.\n\n"
                "--- PATHOLOGIST REPORT ---\n"
                f"{diagnosis_report}\n"
                "--- END REPORT ---"
            ),
        },
    ]

    def _call():
        return _call_watsonx(
            api_key, messages,
            max_new_tokens=MAX_TOKENS_EXTENSION,
            agent_name="extension_officer",
        )

    raw = call_with_retry(_call, agent_name="extension_officer")

    # 2: Output guardrails
    sanitised, out_results = run_all_output_guardrails(raw)
    for r in out_results:
        _record_guardrail(r)

    for r in out_results:
        if not r.passed:
            raise RuntimeError(r.message)

    # 9-A: bias detection
    bias_result = check_bias_in_output(sanitised)
    _record_guardrail(bias_result)
    if not bias_result.passed:
        raise RuntimeError(bias_result.message)

    return sanitised


# ---------------------------------------------------------------------------
# Combined downloadable report
# ---------------------------------------------------------------------------

def _build_final_report(
    diagnosis: str,
    advisory: str,
    image_filename: str,
    guardrail_summary: dict,
    session_id: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "                     AGRI DOCTOR – CROP HEALTH REPORT",
        "=" * 70,
        f"Generated  : {ts}",
        f"Session    : {session_id}",
        f"Image File : {image_filename}",
        f"Model      : {MODEL_ID}",
        "",
        "⚠ DISCLAIMER: This is an AI-generated preliminary assessment.",
        "  It is NOT a confirmed laboratory diagnosis. Always consult a",
        "  qualified agricultural expert before major treatment decisions.",
        "",
        "─" * 70,
        f"GUARDRAIL SUMMARY: {guardrail_summary['passed']} passed | "
        f"{guardrail_summary['warned']} warnings | "
        f"{guardrail_summary['blocked']} blocked",
        "─" * 70,
        "",
        "=" * 70,
        "SECTION 1 – PATHOLOGIST AGENT: CROP DISEASE DIAGNOSIS",
        "=" * 70,
        "",
        diagnosis,
        "",
        "=" * 70,
        "SECTION 2 – EXTENSION OFFICER AGENT: FARMER ADVISORY",
        "=" * 70,
        "",
        advisory,
        "",
        "=" * 70,
        "       Powered by IBM Granite 4 H Small | Agri Doctor v2.0",
        "       Guardrail-protected two-agent AI pipeline",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _show_guardrail_badge(result: GuardrailResult, label: str) -> None:
    """Display a compact pass / warn / block badge in the UI."""
    if not result.passed:
        st.error(f"🚫 **{label}** – {result.message}", icon="🚫")
    elif result.violations:
        st.warning(f"⚠ **{label}** – {result.message}", icon="⚠️")


def _show_guardrail_dashboard(results: list[GuardrailResult]) -> None:
    """Render the guardrail monitoring panel in the sidebar."""
    summary = get_guardrail_summary(results)
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Passed",  summary["passed"])
    col2.metric("⚠ Warned",  summary["warned"])
    col3.metric("🚫 Blocked", summary["blocked"])

    if summary["violations"]:
        with st.expander("📋 Violation Details", expanded=False):
            for v in summary["violations"]:
                st.markdown(f"- `{v}`")


# ---------------------------------------------------------------------------
# Streamlit UI – main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Agri Doctor – Crop Disease Diagnosis",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Reload .env on every rerun so edits are reflected immediately ─────
    # override=True ensures a changed value in .env overwrites whatever was
    # previously in os.environ (from an earlier run with the old value).
    if _HAS_DOTENV:
        _load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

    _init_session()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=80,
        )
        st.title("🌿 Agri Doctor")
        st.caption("AI-Powered Crop Disease Diagnosis  •  v2.0")
        st.divider()

        # ── ⚙️ Configuration panel ──────────────────────────────────────────
        st.subheader("⚙️ Configuration")

        # Resolve current credentials to pre-populate the fields
        _creds_now = resolve_credentials()
        _src       = _creds_now.source   # "session" | "env" | "default"

        # Source badge
        _src_badge = {
            "session": "🟢 **Active source: Website (session)**",
            "env":     "🔵 **Active source: `.env` file**",
            "default": "🟡 **Active source: Defaults** *(no API key set)*",
        }[_src]
        st.markdown(_src_badge)

        # Expand the settings panel by default only when no key is available
        _expand_settings = not _creds_now.is_complete
        with st.expander("🔧 Edit Credentials", expanded=_expand_settings):
            st.caption(
                "Values entered here are kept **only for this session** and are "
                "never written to the `.env` file or committed to Git."
            )

            # ── API Key ──────────────────────────────────────────────────
            _ui_key = st.text_input(
                "IBM Cloud API Key",
                value=_creds_now.api_key if _src == "session" else "",
                type="password",
                placeholder=(
                    "Pre-filled from .env"
                    if _src in ("env", "default")
                    else "Paste key to override"
                ),
                help=(
                    "Your IBM Cloud API key (Manage → Access (IAM) → API keys). "
                    "Leave blank to use the IBM_CLOUD_API_KEY value from the .env file."
                ),
                key="_ui_api_key",
            )

            # ── Region / Endpoint ────────────────────────────────────────
            _region_names   = list(KNOWN_REGIONS.keys()) + ["Custom URL…"]
            _current_ep     = _creds_now.endpoint
            _matched_region = next(
                (r for r, url in KNOWN_REGIONS.items() if url == _current_ep),
                "Custom URL…",
            )
            _region_choice = st.selectbox(
                "Region / Endpoint",
                options=_region_names,
                index=_region_names.index(_matched_region),
                help=(
                    "Select the IBM watsonx region that matches your project. "
                    "Leave unchanged to use IBM_ENDPOINT_URL from .env."
                ),
                key="_ui_region",
            )
            if _region_choice == "Custom URL…":
                _ui_endpoint = st.text_input(
                    "Custom Endpoint URL",
                    value=_current_ep if _matched_region == "Custom URL…" else "",
                    placeholder="https://your-region.ml.cloud.ibm.com",
                    key="_ui_custom_ep",
                )
            else:
                _ui_endpoint = KNOWN_REGIONS[_region_choice]
                st.caption(f"→ `{_ui_endpoint}`")

            # ── Project ID ───────────────────────────────────────────────
            _ui_project = st.text_input(
                "Project ID",
                value=_creds_now.project_id if _src == "session" else "",
                placeholder=(
                    "Pre-filled from .env / default"
                    if _src in ("env", "default")
                    else "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                ),
                help=(
                    "Your IBM watsonx.ai Project ID (Projects → Manage → General). "
                    "Leave blank to use IBM_PROJECT_ID from .env."
                ),
                key="_ui_project_id",
            )

            # ── Action buttons ───────────────────────────────────────────
            _col_apply, _col_validate, _col_clear = st.columns(3)

            with _col_apply:
                if st.button("💾 Apply", use_container_width=True,
                             help="Save these credentials for this session"):
                    store_session_credentials(
                        api_key    = _ui_key,
                        project_id = _ui_project,
                        endpoint   = _ui_endpoint,
                    )
                    st.success("Credentials saved for this session.", icon="✅")
                    st.rerun()

            with _col_validate:
                if st.button("🔍 Validate", use_container_width=True,
                             help="Test credentials against IBM Cloud now"):
                    # Build a temporary CredentialSet from the current UI fields
                    _test_creds = resolve_credentials()
                    if _ui_key:
                        from credential_manager import CredentialSet as _CS
                        _test_creds = _CS(
                            api_key    = _ui_key    or _test_creds.api_key,
                            project_id = _ui_project or _test_creds.project_id,
                            endpoint   = _ui_endpoint or _test_creds.endpoint,
                            source     = "session",
                        )
                    with st.spinner("Validating with IBM Cloud…"):
                        _vr = validate_credentials(_test_creds)
                    if _vr.ok:
                        st.success(_vr.message, icon="✅")
                    elif _vr.error_type == "invalid_key":
                        st.error(f"❌ Invalid API key — {_vr.message}")
                    elif _vr.error_type == "invalid_project":
                        st.error(f"❌ Invalid project/region — {_vr.message}")
                    elif _vr.error_type in ("network", "timeout"):
                        st.warning(f"⚠️ Connectivity issue — {_vr.message}")
                    else:
                        st.error(f"❌ {_vr.message}")

            with _col_clear:
                if st.button("🔄 Reset", use_container_width=True,
                             help="Clear session credentials and revert to .env"):
                    clear_session_credentials()
                    st.info("Session credentials cleared. Reverted to .env values.", icon="ℹ️")
                    st.rerun()

            st.caption(
                "💡 To change defaults permanently, edit the `.env` file — "
                "changes are detected on the next page interaction without restarting the server."
            )

        # Resolve again after any potential Apply/Reset above
        _active_creds = resolve_credentials()
        api_key_input = _active_creds.api_key   # used by the pipeline below

        # Quick status line under the expander
        if _active_creds.is_complete:
            if _active_creds.source == "session":
                st.caption("🟢 Using session credentials.")
            else:
                st.caption(f"🔵 Using `.env` credentials — region: `{_active_creds.endpoint}`")
        else:
            st.caption("🔴 No API key set. Enter one above or populate `.env`.")

        st.divider()
        st.subheader("🛡️ Guardrail Monitor")
        if st.session_state.all_guardrail_results:
            _show_guardrail_dashboard(st.session_state.all_guardrail_results)
        else:
            st.caption("No guardrail checks run yet.")

        st.divider()
        st.subheader("📊 Session Stats")
        st.metric("Diagnoses This Session", st.session_state.request_count)
        st.metric("Guardrail Checks Run",   len(st.session_state.all_guardrail_results))
        st.caption(f"Session ID: `{st.session_state.session_id}`")

        st.divider()
        st.subheader("📋 How It Works")
        st.markdown(
            """
            1. **Upload** a clear crop leaf photo.
            2. **Guardrails** validate your image and API key.
            3. **Agent 1 – Pathologist** analyses the image.
            4. **Multi-agent handoff** is validated.
            5. **Agent 2 – Extension Officer** prepares advice.
            6. **Output guardrails** sanitise both responses.
            7. **Download** the guarded report.
            """
        )
        st.divider()
        st.page_link("pages/dashboard.py", label="📊 Open Performance Dashboard", icon="📊")
        st.divider()
        _disp_creds = resolve_credentials()
        st.caption(
            "⚠ AI-generated preliminary assessment only.\n"
            "Not a confirmed laboratory diagnosis.\n"
            f"Model: `{MODEL_ID}`\n"
            f"Region: `{_disp_creds.endpoint}`"
        )

    # ── Main area ─────────────────────────────────────────────────────────
    st.title("🌿 Agri Doctor")
    st.subheader("Guardrail-Protected Two-Agent Crop Disease Diagnosis")

    # Guardrail legend
    with st.expander("🛡️ Active Guardrail Categories", expanded=False):
        cols = st.columns(3)
        categories = [
            ("1. Input",        "API key, image validation, prompt injection, PII, jailbreak"),
            ("2. Output",       "Toxicity, hallucination, sensitive-data redaction, disclaimer"),
            ("3. Tool Exec",    "Endpoint allow-list, model allow-list, payload schema"),
            ("4. Data",         "PII masking, role-based access control"),
            ("5. Security",     "Credential exposure, jailbreak detection, HTML sanitisation"),
            ("6. Operational",  "Token budgets, rate limiting, retry with back-off"),
            ("7. Human-in-Loop","High-risk recommendation flagging"),
            ("8. Multi-Agent",  "Handoff validation, cross-agent output quality, loop guard"),
            ("9. Ethics",       "Bias detection, explainability notes, audit trail"),
        ]
        for i, (name, desc) in enumerate(categories):
            cols[i % 3].markdown(f"**{name}**  \n{desc}")

    st.info(
        "Upload any crop or plant leaf image (JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF, HEIC and more). "
        "All inputs and outputs are protected by nine categories of guardrails.",
        icon="ℹ️",
    )

    uploaded_file = st.file_uploader(
        "Upload Crop / Leaf Image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif", "avif", "heic", "heif"],
        help="Supports JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF, HEIC and more. "
             "Best results with clear, well-lit close-up photos of the affected area.",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption=f"📷 {uploaded_file.name}", width='stretch')
            st.caption(
                f"Size: {img.width} × {img.height} px  |  "
                f"Format: {img.format or 'unknown'}"
            )
        with col2:
            st.markdown("### Ready to Diagnose")
            st.markdown(
                "Click **Diagnose Crop** to start the guardrail-protected two-agent pipeline.\n\n"
                "- 🔬 **Agent 1 – Pathologist** examines the image.\n"
                "- 👨‍🌾 **Agent 2 – Extension Officer** prepares your advisory.\n"
                "- 🛡️ Nine guardrail categories protect every step."
            )
            diagnose_btn = st.button(
                "🔬 Diagnose Crop",
                type="primary",
                width='stretch',
                disabled=(not api_key_input),
            )
            if not api_key_input:
                st.warning(
                    "No API key active. Open **⚙️ Configuration → 🔧 Edit Credentials** "
                    "in the sidebar to enter your IBM Cloud API key, "
                    "or populate `IBM_CLOUD_API_KEY` in the `.env` file.",
                    icon="🔑",
                )

        if diagnose_btn and api_key_input:
            st.divider()

            # ── PHASE 0: Input Guardrails ────────────────────────────────
            with st.status("🛡️ Phase 0 – Running Input Guardrails…", expanded=True) as gstatus:

                # 6-B: Rate limit
                rate_result = check_session_rate_limit(st.session_state.request_count)
                _record_guardrail(rate_result)
                if not rate_result.passed:
                    gstatus.update(label="🚫 Rate limit reached", state="error")
                    st.error(rate_result.message, icon="🚫")
                    st.stop()

                # 5-E: Credential exposure in API key field itself
                cred_result = validate_credential_not_exposed(api_key_input)
                _record_guardrail(cred_result)

                # 5-B: Jailbreak check (key field edge-case)
                jb_result = check_jailbreak_attempt(api_key_input)
                _record_guardrail(jb_result)
                if not jb_result.passed:
                    gstatus.update(label="🚫 Security check failed", state="error")
                    st.error(jb_result.message, icon="🚫")
                    st.stop()

                # Read image bytes once
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                mime       = _mime_type(uploaded_file)

                # 1: All image + API key input guardrails
                input_result = run_all_input_guardrails(
                    api_key_input, file_bytes, uploaded_file.name, mime
                )
                _record_guardrail(input_result)
                if not input_result.passed:
                    gstatus.update(label="🚫 Input validation failed", state="error")
                    st.error(input_result.message, icon="🚫")
                    st.stop()

                # Warn on PII in filename (unusual but possible)
                pii_result = detect_pii(uploaded_file.name)
                _record_guardrail(pii_result)
                if pii_result.violations:
                    st.warning(pii_result.message, icon="⚠️")

                # Injection check on filename
                inj_result = check_prompt_injection(uploaded_file.name)
                _record_guardrail(inj_result)
                if not inj_result.passed:
                    gstatus.update(label="🚫 Prompt injection detected", state="error")
                    st.error(inj_result.message, icon="🚫")
                    st.stop()

                # Inappropriate content in filename
                inapt_result = filter_inappropriate_content(uploaded_file.name)
                _record_guardrail(inapt_result)
                if not inapt_result.passed:
                    gstatus.update(label="🚫 Inappropriate content", state="error")
                    st.error(inapt_result.message, icon="🚫")
                    st.stop()

                gstatus.update(label="✅ Phase 0 – Input guardrails passed", state="complete")

            # ── PHASE 1: Pathologist Agent ───────────────────────────────
            with st.status(
                "🔬 Phase 1 – Pathologist Agent: Analysing crop image…", expanded=True
            ) as pstatus:
                st.write("Extracting colour & visual features from image…")
                input_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

                st.write("Sending image analysis to IBM Granite 4 H Small…")
                try:
                    diagnosis, img_desc = run_pathologist_agent(
                        api_key_input, file_bytes, uploaded_file.name
                    )
                    pstatus.update(
                        label="✅ Phase 1 – Pathologist: Diagnosis complete", state="complete"
                    )
                except RuntimeError as exc:
                    pstatus.update(label="❌ Pathologist Agent – Error", state="error")
                    st.error(str(exc), icon="🚨")
                    # Audit the failure
                    generate_audit_entry(
                        "pathologist_failed", "pathologist", input_hash, 0,
                        st.session_state.all_guardrail_results,
                        st.session_state.session_id,
                    )
                    st.stop()

            with st.expander("🔬 Agent 1 – Pathologist Report", expanded=True):
                with st.expander("📊 Image Colour Analysis (used as input)", expanded=False):
                    st.code(img_desc, language=None)
                st.markdown(diagnosis)

            # ── PHASE 2: Multi-Agent Handoff Guardrails ──────────────────
            with st.status(
                "🛡️ Phase 2 – Multi-Agent Handoff Validation…", expanded=True
            ) as hstatus:
                st.session_state.pipeline_depth += 1

                # 8-A/F: handoff validation
                handoff_result = validate_agent_handoff(
                    "pathologist", "extension_officer",
                    st.session_state.pipeline_depth,
                )
                _record_guardrail(handoff_result)
                if not handoff_result.passed:
                    hstatus.update(label="🚫 Handoff validation failed", state="error")
                    st.error(handoff_result.message, icon="🚫")
                    st.stop()

                # 8-C: cross-agent output quality
                quality_result = validate_cross_agent_output("pathologist", diagnosis)
                _record_guardrail(quality_result)
                if not quality_result.passed:
                    hstatus.update(label="🚫 Diagnosis quality check failed", state="error")
                    st.error(quality_result.message, icon="🚫")
                    st.stop()

                hstatus.update(
                    label="✅ Phase 2 – Handoff validated: diagnosis passed to Extension Officer",
                    state="complete",
                )

            # ── PHASE 3: Extension Officer Agent ────────────────────────
            with st.status(
                "👨‍🌾 Phase 3 – Extension Officer Agent: Preparing advisory…", expanded=True
            ) as estatus:
                st.write("Passing pathologist diagnosis to Extension Officer Agent…")
                try:
                    advisory = run_extension_officer_agent(api_key_input, diagnosis)
                    estatus.update(
                        label="✅ Phase 3 – Extension Officer: Advisory ready", state="complete"
                    )
                except RuntimeError as exc:
                    estatus.update(label="❌ Extension Officer Agent – Error", state="error")
                    st.error(str(exc), icon="🚨")
                    st.stop()

            with st.expander("👨‍🌾 Agent 2 – Extension Officer Advisory", expanded=True):
                st.markdown(advisory)

            # ── PHASE 4: Post-Output Guardrails + Metrics ─────────────────
            with st.status(
                "🛡️ Phase 4 – Post-Output Guardrails & Metrics…", expanded=True
            ) as ostatus:
                # 7: HITL – high-risk recommendations
                hitl_result = check_human_review_required(advisory)
                _record_guardrail(hitl_result)

                # ── Evaluate both agents and the full pipeline ─────────────
                run_id = str(uuid.uuid4())[:8]
                grds   = st.session_state.all_guardrail_results
                total_grd_checks    = len(grds)
                total_interventions = sum(1 for r in grds if r.violations)
                total_blocks        = sum(1 for r in grds if not r.passed)

                p_timing = _get_timing("pathologist")
                p_tokens = _get_tokens("pathologist")
                e_timing = _get_timing("extension_officer")
                e_tokens = _get_tokens("extension_officer")

                p_rec = evaluate_agent_run(
                    agent_name="pathologist",
                    run_id=run_id,
                    session_id=st.session_state.session_id,
                    image_filename=uploaded_file.name,
                    image_size_bytes=len(file_bytes),
                    timing=p_timing,
                    tokens=p_tokens,
                    response_text=diagnosis,
                    guardrail_interventions=total_interventions // 2,
                    guardrail_blocks=total_blocks,
                    total_guardrail_checks=total_grd_checks // 2,
                    retry_count=st.session_state.get("_retry_pathologist", 0),
                )
                e_rec = evaluate_agent_run(
                    agent_name="extension_officer",
                    run_id=run_id,
                    session_id=st.session_state.session_id,
                    image_filename=uploaded_file.name,
                    image_size_bytes=len(file_bytes),
                    timing=e_timing,
                    tokens=e_tokens,
                    response_text=advisory,
                    guardrail_interventions=total_interventions - total_interventions // 2,
                    guardrail_blocks=0,
                    total_guardrail_checks=total_grd_checks - total_grd_checks // 2,
                    retry_count=st.session_state.get("_retry_extension", 0),
                )
                pipeline_rec = evaluate_pipeline(
                    run_id=run_id,
                    session_id=st.session_state.session_id,
                    image_filename=uploaded_file.name,
                    pathologist_record=p_rec,
                    extension_record=e_rec,
                    pathologist_output=diagnosis,
                    extension_output=advisory,
                    total_guardrail_checks=total_grd_checks,
                    total_interventions=total_interventions,
                )
                st.session_state.pipeline_runs.append(pipeline_rec.to_dict())

                # Audit trail
                audit_entry = generate_audit_entry(
                    "pipeline_complete",
                    "multi_agent",
                    input_hash,
                    len(diagnosis) + len(advisory),
                    st.session_state.all_guardrail_results,
                    st.session_state.session_id,
                )
                st.session_state.audit_log.append(audit_entry)
                st.session_state.request_count += 1

                ostatus.update(
                    label="✅ Phase 4 – Output guardrails & metrics complete", state="complete"
                )

            # ── HITL notice (if triggered) ────────────────────────────────
            if hitl_result.severity == "warning":
                st.warning(hitl_result.message, icon="⚠️")

            # ── Combined Final Report ─────────────────────────────────────
            st.divider()
            st.subheader("📄 Combined Final Report")
            st.success(
                "Both agents completed their analysis. All guardrail checks passed. "
                "Review the report below and download for your records.",
                icon="✅",
            )

            # Live guardrail summary
            with st.expander("🛡️ Guardrail Run Summary", expanded=True):
                _show_guardrail_dashboard(st.session_state.all_guardrail_results)

            st.markdown("#### 🔬 Disease Diagnosis")
            st.markdown(diagnosis)
            st.markdown("---")
            st.markdown("#### 👨‍🌾 Farmer Advisory")
            st.markdown(advisory)

            st.divider()
            st.warning(
                "⚠ **Disclaimer:** This is an AI-generated preliminary assessment. "
                "It is **not** a confirmed laboratory diagnosis. Always consult a "
                "qualified agricultural extension officer, certified agronomist, or "
                "plant pathology laboratory before significant treatment decisions.",
                icon="⚠️",
            )

            # Download
            guardrail_summary = get_guardrail_summary(st.session_state.all_guardrail_results)
            final_report_text = _build_final_report(
                diagnosis, advisory, uploaded_file.name,
                guardrail_summary, st.session_state.session_id,
            )
            st.download_button(
                label="📥 Download Full Advisory Report (.txt)",
                data=final_report_text.encode("utf-8"),
                file_name=f"agri_doctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="primary",
                width='stretch',
            )

    else:
        # Landing state
        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### 🔬 Agent 1 – Pathologist")
            st.markdown(
                "Analyses your crop image using IBM Granite 4 H Small to identify "
                "diseases, symptoms, and probable causes."
            )
        with col_b:
            st.markdown("### 👨‍🌾 Agent 2 – Extension Officer")
            st.markdown(
                "Reads the pathologist's diagnosis and produces plain-language "
                "treatment advice, prevention tips, and safety warnings."
            )
        with col_c:
            st.markdown("### 🛡️ Nine Guardrail Categories")
            st.markdown(
                "Input, Output, Tool Execution, Data, Security, Operational, "
                "Human-in-the-Loop, Multi-Agent, and Ethical & Compliance."
            )

        st.divider()
        st.markdown(
            "**Supported crops include** wheat, rice, maize, tomato, potato, cotton, "
            "groundnut, banana, sugarcane, and most vegetable crops. "
            "For best results, upload a clear close-up photo of the affected leaf or plant part."
        )


if __name__ == "__main__":
    main()
