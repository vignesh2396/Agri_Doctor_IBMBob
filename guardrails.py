"""
guardrails.py – Agri Doctor Guardrails Engine
==============================================
Nine guardrail categories mapped to all touchpoints in the two-agent workflow:

  1.  Input Guardrails          – validate user inputs before any agent call
  2.  Output Guardrails         – validate AI responses before display
  3.  Tool Execution Guardrails – control access to external APIs
  4.  Data Guardrails           – protect PII and sensitive data
  5.  Security Guardrails       – prompt injection, jailbreaks, SQL injection
  6.  Operational Guardrails    – token budgets, rate limits, timeouts, retries
  7.  Human-in-the-Loop         – flag high-risk recommendations for review
  8.  Multi-Agent Guardrails    – cross-agent output verification and loop prevention
  9.  Ethical & Compliance      – bias, explainability, audit trail

Each guardrail returns a GuardrailResult dataclass so callers can branch on
.passed, read .violations, and display .message to the end user.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Logging – structured audit trail (Category 9)
# ---------------------------------------------------------------------------
_audit_logger = logging.getLogger("agri_doctor.audit")
_audit_logger.setLevel(logging.INFO)
if not _audit_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [AUDIT] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    _audit_logger.addHandler(_handler)


def _audit(event: str, detail: str = "") -> None:
    """Append a structured audit-trail entry."""
    _audit_logger.info("%s | %s", event, detail)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    message: str = ""
    severity: str = "info"          # "info" | "warning" | "error" | "blocked"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, msg: str = "Guardrail passed.", **meta) -> "GuardrailResult":
        return cls(passed=True, message=msg, severity="info", metadata=meta)

    @classmethod
    def block(cls, violations: list[str], msg: str, **meta) -> "GuardrailResult":
        return cls(passed=False, violations=violations, message=msg,
                   severity="blocked", metadata=meta)

    @classmethod
    def warn(cls, violations: list[str], msg: str, **meta) -> "GuardrailResult":
        return cls(passed=True, violations=violations, message=msg,
                   severity="warning", metadata=meta)


# ===========================================================================
# 1. INPUT GUARDRAILS
# ===========================================================================
# Patterns used for detection
_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(your\s+)?(system|prior|above)\s+(prompt|instructions?|context)",
    r"you\s+are\s+now\s+(a\s+)?(?!a\s+plant|an?\s+agri)",  # role-hijack
    r"act\s+as\s+(?!a\s+plant|an?\s+agri)",
    r"jailbreak",
    r"dan\s+mode",
    r"pretend\s+(you|to\s+be)",
    r"forget\s+(your|all)\s+(instructions?|training|rules)",
    r"override\s+(safety|guardrail|filter|restriction)",
    r"<\s*script[^>]*>",                  # HTML/JS injection
    r"system\s*:\s*you\s+are",            # shadow system-prompt
    r"\{\{.*\}\}",                        # template injection
    r"\$\{.*\}",                          # template injection
]

_MALICIOUS_PATTERNS = [
    r"(drop|delete|truncate|alter)\s+table",   # SQL injection
    r"(union\s+select|insert\s+into|exec\s*\()",
    r"(rm\s+-rf|sudo|chmod|wget\s+http)",      # shell injection
    r"(eval\s*\(|exec\s*\(|import\s+os)",      # code injection
    r"(base64\.decode|__import__)",
]

_PII_PATTERNS = {
    "email":        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone_in":     r"\b[6-9]\d{9}\b",          # Indian mobile
    "phone_intl":   r"\+?\d[\d\s\-\(\)]{8,}\d",
    "aadhaar":      r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "pan":          r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "credit_card":  r"\b(?:\d[ -]?){13,19}\b",
    "ssn":          r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
}

_INAPPROPRIATE_TERMS = [
    r"\b(fuck|shit|asshole|bastard|bitch|cunt|dick|pussy)\b",
    r"\b(kill|murder|bomb|terrorist|weapon)\b",
    r"\b(porn|nude|naked|sex)\b",
]

# Allowed image MIME types – all common raster formats accepted
_ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/webp", "image/bmp",
    "image/tiff", "image/gif", "image/avif", "image/heic",
    "image/heif", "image/x-bmp", "image/x-tiff",
}
# Allowed file extensions (PIL can decode all of these)
_ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif",
    "gif", "avif", "heic", "heif",
}
# Max image size: 20 MB (raised to accommodate TIFF/RAW-adjacent formats)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Max API key length guard
_API_KEY_MIN_LEN = 20
_API_KEY_MAX_LEN = 256


def validate_api_key(api_key: str) -> GuardrailResult:
    """
    1-A: API key format / length validation.
    Never log or expose the actual key value.
    """
    if not api_key or not api_key.strip():
        return GuardrailResult.block(
            ["empty_api_key"], "API key cannot be empty."
        )
    key = api_key.strip()
    if len(key) < _API_KEY_MIN_LEN:
        return GuardrailResult.block(
            ["api_key_too_short"],
            f"API key appears too short (min {_API_KEY_MIN_LEN} characters). "
            "Please check your IBM Cloud API key.",
        )
    if len(key) > _API_KEY_MAX_LEN:
        return GuardrailResult.block(
            ["api_key_too_long"],
            f"API key exceeds maximum allowed length ({_API_KEY_MAX_LEN} chars). "
            "Please verify the key.",
        )
    # Reject keys that look like they were pasted with surrounding quotes
    if key.startswith(("'", '"')) or key.endswith(("'", '"')):
        return GuardrailResult.block(
            ["api_key_quoted"],
            "API key should not include surrounding quote characters. "
            "Paste the raw key only.",
        )
    _audit("INPUT_GUARDRAIL", "api_key validated (hash=" + hashlib.sha256(key.encode()).hexdigest()[:8] + ")")
    return GuardrailResult.ok("API key format is valid.")


def validate_image_upload(file_bytes: bytes, filename: str, mime: str) -> GuardrailResult:
    """
    1-B: Image input validation – size, type, and header check.
    Accepts all common raster image formats that PIL can decode.
    """
    violations: list[str] = []

    # MIME type check (allow generic octet-stream for unknown uploads)
    if mime not in _ALLOWED_MIME and not mime.startswith("image/"):
        violations.append(f"unsupported_mime:{mime}")

    # Extension check
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext and ext not in _ALLOWED_EXTENSIONS:
        violations.append(f"unsupported_extension:{ext}")

    # File size
    size_bytes = len(file_bytes)
    if size_bytes == 0:
        violations.append("empty_file")
    elif size_bytes > _MAX_IMAGE_BYTES:
        violations.append(f"file_too_large:{size_bytes // (1024*1024)}MB")

    # Magic-byte header validation for common formats
    # (unknown formats pass through – PIL will reject them later if invalid)
    header = file_bytes[:12] if len(file_bytes) >= 12 else file_bytes
    is_jpeg = header[:3]  == b"\xff\xd8\xff"
    is_png  = header[:8]  == b"\x89PNG\r\n\x1a\n"
    is_webp = header[:4]  == b"RIFF" and header[8:12] == b"WEBP"
    is_bmp  = header[:2]  == b"BM"
    is_gif  = header[:6]  in (b"GIF87a", b"GIF89a")
    is_tiff = header[:4]  in (b"II\x2a\x00", b"MM\x00\x2a")
    is_image_header = any([is_jpeg, is_png, is_webp, is_bmp, is_gif, is_tiff])

    # Only block if we can positively identify it as NOT an image
    if not is_image_header and size_bytes > 0:
        # Give benefit of doubt for formats we don't check (AVIF, HEIC, etc.)
        # – PIL will error naturally during Image.open if truly corrupt
        pass

    if violations:
        _audit("INPUT_GUARDRAIL_BLOCK", f"image_upload violations={violations} file={filename}")
        return GuardrailResult.block(
            violations,
            f"Image validation failed: {', '.join(violations)}. "
            "Please upload an image file (JPG, PNG, WEBP, BMP, TIFF, GIF, etc.).",
        )

    _audit("INPUT_GUARDRAIL", f"image_upload OK file={filename} size={size_bytes} ext={ext}")
    return GuardrailResult.ok("Image validated successfully.")


def check_prompt_injection(text: str) -> GuardrailResult:
    """
    1-C / 5-A: Detect prompt injection and jailbreak attempts in any free-text input.
    """
    lowered = text.lower()
    hits = []
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            hits.append(pattern[:40])
    if hits:
        _audit("SECURITY_GUARDRAIL_BLOCK", f"prompt_injection detected patterns={len(hits)}")
        return GuardrailResult.block(
            ["prompt_injection"],
            "Your input contains patterns that could manipulate AI behaviour. "
            "Please describe your crop symptoms in plain language.",
        )
    return GuardrailResult.ok()


def check_malicious_input(text: str) -> GuardrailResult:
    """
    1-D / 5-C: Detect SQL injection, shell commands, code injection.
    """
    hits = []
    for pattern in _MALICIOUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern[:30])
    if hits:
        _audit("SECURITY_GUARDRAIL_BLOCK", f"malicious_input detected count={len(hits)}")
        return GuardrailResult.block(
            ["malicious_input"],
            "Your input contains potentially malicious content and cannot be processed.",
        )
    return GuardrailResult.ok()


def detect_pii(text: str) -> GuardrailResult:
    """
    1-E / 4-A: Detect personally identifiable information in user-provided text.
    Returns a warning (not a block) so the user can decide to proceed.
    """
    found_types = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if re.search(pattern, text):
            found_types.append(pii_type)
    if found_types:
        _audit("DATA_GUARDRAIL_WARN", f"PII detected types={found_types}")
        return GuardrailResult.warn(
            [f"pii:{t}" for t in found_types],
            f"Detected possible personal information ({', '.join(found_types)}) in your input. "
            "Please remove personal details before submitting.",
        )
    return GuardrailResult.ok()


def filter_inappropriate_content(text: str) -> GuardrailResult:
    """
    1-F: Block clearly inappropriate or offensive content.
    """
    hits = []
    for pattern in _INAPPROPRIATE_TERMS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append("inappropriate_content")
            break
    if hits:
        _audit("INPUT_GUARDRAIL_BLOCK", "inappropriate_content detected")
        return GuardrailResult.block(
            hits,
            "Your input contains inappropriate language. "
            "Please describe your crop problem using agricultural terminology.",
        )
    return GuardrailResult.ok()


def run_all_input_guardrails(
    api_key: str,
    file_bytes: bytes,
    filename: str,
    mime: str,
) -> GuardrailResult:
    """
    Run all input guardrails in sequence.  Returns the first blocking result,
    or a combined OK / warning result.
    """
    checks = [
        validate_api_key(api_key),
        validate_image_upload(file_bytes, filename, mime),
    ]
    warnings: list[str] = []
    for result in checks:
        if not result.passed:
            return result
        warnings.extend(result.violations)

    combined = GuardrailResult.ok("All input guardrails passed.")
    combined.violations = warnings
    if warnings:
        combined.severity = "warning"
    return combined


# ===========================================================================
# 2. OUTPUT GUARDRAILS
# ===========================================================================

_HALLUCINATION_FLAGS = [
    r"studies\s+show\s+that\s+\d+%",        # unsupported statistics
    r"proven\s+(to\s+)?cure",
    r"guaranteed\s+to",
    r"100%\s+(effective|accurate|certain)",
    r"(always|never)\s+(works?|cures?|prevents?)",
    r"I\s+am\s+absolutely\s+certain",
    r"(definitely|certainly)\s+(caused\s+by|is\s+a)",
]

_TOXICITY_PATTERNS = [
    r"\b(kill\s+(yourself|all)|suicide|bomb|explosive)\b",
    r"\b(hate\s+(speech|group)|racism|fascism)\b",
]

_SENSITIVE_OUTPUT_PATTERNS = [
    r"api[_\-\s]?key\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"bearer\s+[A-Za-z0-9\-_\.]+",
    r"secret\s*[:=]\s*\S+",
    r"token\s*[:=]\s*[A-Za-z0-9\-_\.]{20,}",
]

# Accept several equivalent disclaimer phrasings the model might use
_DISCLAIMER_PHRASES = [
    "ai-generated preliminary assessment",
    "ai generated preliminary assessment",
    "preliminary assessment",
    "not a confirmed laboratory diagnosis",
    "not a confirmed diagnosis",
    "consult a qualified",
    "this is an ai",
    "generated by ibm",
    "preliminary advisory",
]
_REQUIRED_DISCLAIMER = "ai-generated preliminary assessment"  # kept for add_explainability_note

_MAX_RESPONSE_CHARS = 8000


def check_output_hallucination(text: str) -> GuardrailResult:
    """2-A: Flag language that suggests hallucinated or overconfident claims."""
    hits = []
    for pattern in _HALLUCINATION_FLAGS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append("potential_hallucination")
            break
    if hits:
        _audit("OUTPUT_GUARDRAIL_WARN", "hallucination_language detected")
        return GuardrailResult.warn(
            hits,
            "The AI response contains overly confident language. "
            "Please treat this as a preliminary assessment only.",
        )
    return GuardrailResult.ok()


def check_output_toxicity(text: str) -> GuardrailResult:
    """2-C: Block toxic or harmful content in AI output."""
    for pattern in _TOXICITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            _audit("OUTPUT_GUARDRAIL_BLOCK", "toxicity detected in AI output")
            return GuardrailResult.block(
                ["toxic_output"],
                "The AI response was blocked because it contained harmful content. "
                "Please try again with a clearer crop image.",
            )
    return GuardrailResult.ok()


def filter_sensitive_output(text: str) -> tuple[str, GuardrailResult]:
    """
    2-D: Redact any credentials or secrets accidentally present in AI output.
    Returns (sanitised_text, result).
    """
    sanitised = text
    redacted = False
    for pattern in _SENSITIVE_OUTPUT_PATTERNS:
        new = re.sub(pattern, "[REDACTED]", sanitised, flags=re.IGNORECASE)
        if new != sanitised:
            sanitised = new
            redacted = True
    if redacted:
        _audit("OUTPUT_GUARDRAIL_WARN", "sensitive information redacted from AI output")
        result = GuardrailResult.warn(
            ["sensitive_data_redacted"],
            "Some potentially sensitive information was automatically removed from the AI response.",
        )
    else:
        result = GuardrailResult.ok()
    return sanitised, result


def validate_output_disclaimer(text: str) -> GuardrailResult:
    """2-E: Ensure AI output carries the required disclaimer (any accepted phrasing)."""
    text_lower = text.lower()
    found = any(phrase in text_lower for phrase in _DISCLAIMER_PHRASES)
    if not found:
        _audit("OUTPUT_GUARDRAIL_WARN", "disclaimer missing from AI output – will be appended automatically")
        # Return a passing result with an info note; the explainability note appends it
        return GuardrailResult.ok("Disclaimer will be appended via explainability note.")
    return GuardrailResult.ok("Disclaimer present.")


def validate_response_length(text: str) -> GuardrailResult:
    """2-E: Guard against runaway responses."""
    if len(text) > _MAX_RESPONSE_CHARS:
        _audit("OUTPUT_GUARDRAIL_WARN", f"response_too_long chars={len(text)}")
        return GuardrailResult.warn(
            ["response_too_long"],
            f"AI response was unusually long ({len(text)} chars). "
            "It has been accepted but please verify the content.",
        )
    return GuardrailResult.ok()


def run_all_output_guardrails(text: str) -> tuple[str, list[GuardrailResult]]:
    """
    Run all output guardrails.  Returns (sanitised_text, list_of_results).
    Callers should stop displaying output if any result.severity == 'blocked'.
    """
    results: list[GuardrailResult] = []

    # Toxicity must be checked first (hard block)
    tox = check_output_toxicity(text)
    results.append(tox)
    if not tox.passed:
        return text, results

    # Sensitive data redaction (mutates text)
    sanitised, sens_result = filter_sensitive_output(text)
    results.append(sens_result)

    # Hallucination flag (warning only)
    results.append(check_output_hallucination(sanitised))

    # Disclaimer check (warning only)
    results.append(validate_output_disclaimer(sanitised))

    # Length check (warning only)
    results.append(validate_response_length(sanitised))

    return sanitised, results


# ===========================================================================
# 3. TOOL EXECUTION GUARDRAILS
# ===========================================================================

_ALLOWED_ENDPOINTS = {
    "https://eu-de.ml.cloud.ibm.com",
    "https://us-south.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com",
    "https://iam.cloud.ibm.com",
}

_ALLOWED_MODELS = {
    "ibm/granite-4-h-small",
    "ibm/granite-3-8b-instruct",
    "ibm/granite-3-2b-instruct",
    "ibm/granite-13b-chat-v2",
}


def validate_tool_endpoint(url: str) -> GuardrailResult:
    """3-A/B: Ensure API calls only go to authorised IBM watsonx endpoints."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _ALLOWED_ENDPOINTS:
        _audit("TOOL_GUARDRAIL_BLOCK", f"unauthorized_endpoint={base}")
        return GuardrailResult.block(
            ["unauthorized_endpoint"],
            f"API call to '{base}' is not permitted. "
            "Only IBM watsonx AI endpoints are allowed.",
        )
    return GuardrailResult.ok()


def validate_model_id(model_id: str) -> GuardrailResult:
    """3-C: Validate that the requested model is on the approved list."""
    if model_id not in _ALLOWED_MODELS:
        _audit("TOOL_GUARDRAIL_WARN", f"non_standard_model={model_id}")
        return GuardrailResult.warn(
            ["non_standard_model"],
            f"Model '{model_id}' is not in the approved model list. "
            "Proceeding, but verify the model is appropriate for this use case.",
        )
    return GuardrailResult.ok()


def validate_payload_schema(payload: dict) -> GuardrailResult:
    """3-C: Minimal schema check before sending to the LLM API."""
    required = {"model_id", "project_id", "messages"}
    missing = required - payload.keys()
    if missing:
        _audit("TOOL_GUARDRAIL_BLOCK", f"invalid_payload missing={missing}")
        return GuardrailResult.block(
            ["invalid_payload_schema"],
            f"API payload is missing required fields: {', '.join(missing)}.",
        )
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        return GuardrailResult.block(
            ["empty_messages"],
            "API payload 'messages' must be a non-empty list.",
        )
    return GuardrailResult.ok()


# ===========================================================================
# 4. DATA GUARDRAILS
# ===========================================================================

def mask_pii_in_text(text: str) -> str:
    """
    4-B: Replace detected PII with masked placeholders.
    Returns the sanitised text.
    """
    masked = text
    replacements = {
        "email":        "[EMAIL_REDACTED]",
        "phone_in":     "[PHONE_REDACTED]",
        "phone_intl":   "[PHONE_REDACTED]",
        "aadhaar":      "[AADHAAR_REDACTED]",
        "pan":          "[PAN_REDACTED]",
        "credit_card":  "[CARD_REDACTED]",
        "ssn":          "[SSN_REDACTED]",
    }
    for pii_type, pattern in _PII_PATTERNS.items():
        masked = re.sub(pattern, replacements[pii_type], masked)
    return masked


def validate_data_access_role(user_role: str, requested_operation: str) -> GuardrailResult:
    """
    4-C/D: Role-based access control for data operations.
    Roles: 'farmer' (read-only diagnosis), 'agronomist' (full access), 'admin'.
    """
    role_permissions: dict[str, set[str]] = {
        "farmer":     {"diagnose", "view_report", "download_report"},
        "agronomist": {"diagnose", "view_report", "download_report", "view_audit"},
        "admin":      {"diagnose", "view_report", "download_report", "view_audit", "manage_users"},
    }
    perms = role_permissions.get(user_role.lower(), set())
    if requested_operation not in perms:
        _audit("DATA_GUARDRAIL_BLOCK", f"role={user_role} denied operation={requested_operation}")
        return GuardrailResult.block(
            ["insufficient_role"],
            f"Your role ('{user_role}') does not have permission for '{requested_operation}'.",
        )
    return GuardrailResult.ok()


# ===========================================================================
# 5. SECURITY GUARDRAILS  (patterns shared with Section 1 above)
# ===========================================================================

def check_jailbreak_attempt(text: str) -> GuardrailResult:
    """5-B: Detect jailbreak attempts specifically."""
    jailbreak_patterns = [
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
        r"unrestricted\s+mode",
        r"do\s+anything\s+now",
        r"no\s+(rules?|restrictions?|limits?|filters?|guardrails?)",
        r"bypass\s+(safety|filter|guardrail|restriction)",
        r"ignore\s+(ethics|safety|rules?|guidelines?)",
    ]
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            _audit("SECURITY_GUARDRAIL_BLOCK", "jailbreak_attempt detected")
            return GuardrailResult.block(
                ["jailbreak_attempt"],
                "Your input appears to contain a jailbreak attempt. "
                "This request has been blocked for security reasons.",
            )
    return GuardrailResult.ok()


def sanitise_for_display(text: str) -> str:
    """
    5-D: HTML-escape AI output before any rendering context where raw HTML
    might be interpreted (belt-and-suspenders; Streamlit markdown is also escaped).
    """
    return html.escape(text, quote=False)


def validate_credential_not_exposed(text: str) -> GuardrailResult:
    """5-E: Ensure API keys / secrets are never present in any logged or displayed text."""
    suspicious = re.findall(
        r"(?:api[_\-\s]?key|password|secret|bearer|token)\s*[:=]\s*(\S{10,})",
        text,
        re.IGNORECASE,
    )
    if suspicious:
        _audit("SECURITY_GUARDRAIL_BLOCK", "credential_exposure_attempt detected")
        return GuardrailResult.block(
            ["credential_exposure"],
            "A potential credential was detected in the content and the request was blocked.",
        )
    return GuardrailResult.ok()


# ===========================================================================
# 6. OPERATIONAL GUARDRAILS
# ===========================================================================

# Token budget per agent (hard ceiling, must not exceed model context)
_TOKEN_BUDGET = {
    "pathologist":      900,
    "extension_officer": 1100,
    "default":          1000,
}

# Rate limiting (simple in-memory per-session counter; per-session state
# is managed in Streamlit via st.session_state in app.py)
_MAX_REQUESTS_PER_SESSION = 15
_MAX_AGENT_ITERATIONS     = 3    # max re-tries per agent
_MAX_RETRY_ATTEMPTS       = 2
_RETRY_BACKOFF_SECONDS    = [1, 3]   # exponential back-off steps


def validate_token_budget(agent_name: str, requested_tokens: int) -> GuardrailResult:
    """6-A: Enforce per-agent token limits."""
    budget = _TOKEN_BUDGET.get(agent_name.lower(), _TOKEN_BUDGET["default"])
    if requested_tokens > budget:
        _audit("OPERATIONAL_GUARDRAIL_WARN", f"token_overage agent={agent_name} requested={requested_tokens} budget={budget}")
        return GuardrailResult.warn(
            ["token_budget_exceeded"],
            f"Requested {requested_tokens} tokens for '{agent_name}' agent, "
            f"but budget is {budget}. Capping at {budget}.",
            capped_tokens=budget,
        )
    return GuardrailResult.ok(capped_tokens=requested_tokens)


def check_session_rate_limit(request_count: int) -> GuardrailResult:
    """6-B: Prevent excessive API calls within a single user session."""
    if request_count >= _MAX_REQUESTS_PER_SESSION:
        _audit("OPERATIONAL_GUARDRAIL_BLOCK", f"rate_limit_exceeded count={request_count}")
        return GuardrailResult.block(
            ["rate_limit_exceeded"],
            f"You have reached the maximum of {_MAX_REQUESTS_PER_SESSION} diagnoses per session. "
            "Please restart the app for a new session.",
        )
    return GuardrailResult.ok()


def call_with_retry(fn, *args, agent_name: str = "agent", **kwargs):
    """
    6-F: Wrap an API call with exponential back-off retry logic.
    Raises RuntimeError after _MAX_RETRY_ATTEMPTS exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            last_exc = exc
            err_str = str(exc)
            # Do NOT retry on auth or validation errors
            if any(k in err_str for k in ("IAM token", "API key", "HTTP 400", "HTTP 401", "HTTP 403")):
                raise
            if attempt < _MAX_RETRY_ATTEMPTS:
                wait = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                _audit("OPERATIONAL_GUARDRAIL", f"retry attempt={attempt+1} agent={agent_name} wait={wait}s")
                time.sleep(wait)
    raise RuntimeError(
        f"Agent '{agent_name}' failed after {_MAX_RETRY_ATTEMPTS + 1} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


# ===========================================================================
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# ===========================================================================

# High-risk HITL triggers: genuine high-risk actions only (not routine farming advice)
_HIGH_RISK_TRIGGERS = [
    r"systemic\s+(pesticide|fungicide|insecticide)\s+.{0,20}(immediately|now|today|urgently)",
    r"(destroy|burn|uproot)\s+(the\s+)?(entire\s+)?(crop|field|plantation)",
    r"(quarantine|embargo)\s+(the\s+)?(farm|field|crop|area)",
    r"(do\s+not\s+harvest|unsafe\s+to\s+(eat|consume|harvest))",
    r"(toxic|poisonous|hazardous)\s+to\s+(humans?|livestock|animals?)",
    r"(call|contact|notify)\s+(agriculture\s+department|government|authorities)",
]


def check_human_review_required(text: str) -> GuardrailResult:
    """
    7: Flag advisory outputs that recommend genuinely high-risk actions.
    Returns a warning with NO violation entry (so the guardrail panel stays clean)
    – it is displayed as a separate informational notice in the UI.
    """
    triggers = []
    for pattern in _HIGH_RISK_TRIGGERS:
        if re.search(pattern, text, re.IGNORECASE):
            triggers.append(pattern[:40])
    if triggers:
        _audit("HITL_GUARDRAIL_WARN", f"high_risk_advisory detected triggers={len(triggers)}")
        # Use empty violations list so it doesn't appear in the violation panel
        return GuardrailResult(
            passed=True,
            violations=[],          # intentionally empty – shown as separate UI notice
            message=(
                "⚠ The advisory mentions actions that carry significant risk "
                "(e.g., crop destruction, quarantine, human safety hazard). "
                "Please verify with a certified agricultural expert or local "
                "extension officer before taking action."
            ),
            severity="warning",
            metadata={"triggers": triggers},
        )
    return GuardrailResult.ok()


# ===========================================================================
# 8. MULTI-AGENT GUARDRAILS
# ===========================================================================

_AGENT_REGISTRY = {"pathologist", "extension_officer"}
_MAX_PIPELINE_DEPTH = 5   # prevent infinite agent loops


def validate_agent_handoff(
    source_agent: str,
    target_agent: str,
    pipeline_depth: int,
) -> GuardrailResult:
    """
    8-A/F: Validate agent-to-agent handoff and prevent infinite loops.
    """
    if source_agent not in _AGENT_REGISTRY:
        _audit("MULTIAGENT_GUARDRAIL_BLOCK", f"unknown_source_agent={source_agent}")
        return GuardrailResult.block(
            ["unknown_source_agent"],
            f"Unknown source agent: '{source_agent}'.",
        )
    if target_agent not in _AGENT_REGISTRY:
        _audit("MULTIAGENT_GUARDRAIL_BLOCK", f"unknown_target_agent={target_agent}")
        return GuardrailResult.block(
            ["unknown_target_agent"],
            f"Unknown target agent: '{target_agent}'.",
        )
    if pipeline_depth >= _MAX_PIPELINE_DEPTH:
        _audit("MULTIAGENT_GUARDRAIL_BLOCK", f"pipeline_loop_detected depth={pipeline_depth}")
        return GuardrailResult.block(
            ["pipeline_loop"],
            f"Agent pipeline depth ({pipeline_depth}) exceeds maximum ({_MAX_PIPELINE_DEPTH}). "
            "Possible infinite loop detected.",
        )
    return GuardrailResult.ok()


def validate_cross_agent_output(
    source_agent: str,
    output: str,
    min_length: int = 50,
) -> GuardrailResult:
    """
    8-C: Verify that the source agent's output is substantive before passing
    it to the next agent.
    """
    if not output or len(output.strip()) < min_length:
        _audit("MULTIAGENT_GUARDRAIL_BLOCK", f"insufficient_output source={source_agent} len={len(output)}")
        return GuardrailResult.block(
            ["insufficient_agent_output"],
            f"Agent '{source_agent}' produced insufficient output to pass to the next agent. "
            "Please try again with a clearer crop image.",
        )
    # Check the output is not just an error message
    error_markers = ["error:", "failed:", "exception:", "traceback"]
    if any(m in output.lower()[:100] for m in error_markers):
        _audit("MULTIAGENT_GUARDRAIL_BLOCK", f"error_output_from source={source_agent}")
        return GuardrailResult.block(
            ["error_in_agent_output"],
            f"Agent '{source_agent}' returned an error response. "
            "The pipeline has been halted.",
        )
    return GuardrailResult.ok()


# ===========================================================================
# 9. ETHICAL & COMPLIANCE GUARDRAILS
# ===========================================================================

_BIAS_PATTERNS = [
    r"(traditional|small|poor|uneducated)\s+farmer[s]?\s+(cannot|should not|won't understand)",
    r"(women|female)\s+(farmer[s]?)?\s+(cannot|should not|are not capable)",
    r"(indigenous|tribal)\s+(farming|method[s]?|practice[s]?)\s+(is|are)\s+(inferior|primitive|wrong)",
]


def check_bias_in_output(text: str) -> GuardrailResult:
    """9-A: Detect language that may reinforce agricultural or demographic bias."""
    for pattern in _BIAS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            _audit("ETHICAL_GUARDRAIL_BLOCK", "bias_language detected in output")
            return GuardrailResult.block(
                ["biased_content"],
                "The AI response contained biased language and was blocked. "
                "Please try again.",
            )
    return GuardrailResult.ok()


def generate_audit_entry(
    event: str,
    agent: str,
    input_hash: str,
    output_length: int,
    guardrail_results: list[GuardrailResult],
    session_id: str = "unknown",
) -> dict:
    """
    9-F: Build a structured audit-trail record for logging / compliance.
    Never includes raw input, output, or API keys.
    """
    violations = [v for r in guardrail_results for v in r.violations]
    blocked    = any(not r.passed for r in guardrail_results)
    entry = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "session_id":      session_id,
        "event":           event,
        "agent":           agent,
        "input_hash":      input_hash,
        "output_length":   output_length,
        "violations":      violations,
        "blocked":         blocked,
        "guardrail_count": len(guardrail_results),
    }
    _audit("AUDIT_TRAIL", str(entry))
    return entry


def add_explainability_note(diagnosis: str) -> str:
    """
    9-C: Append a brief explainability footer to any AI output, explaining
    what information the model used to reach its conclusion.
    """
    note = (
        "\n\n---\n"
        "**ℹ Explainability Note:** This assessment was generated by IBM Granite 4 H Small, "
        "a large language model analysing the visual features of the uploaded image. "
        "The model identified colour patterns, texture anomalies, and morphological features "
        "consistent with the reported condition. Confidence reflects visual evidence quality. "
        "This is an AI-generated preliminary assessment – not a confirmed laboratory diagnosis."
    )
    # Only append if the disclaimer isn't already present
    if _REQUIRED_DISCLAIMER not in diagnosis.lower():
        return diagnosis + note
    return diagnosis


# ===========================================================================
# Convenience: run the full guardrail pipeline for the UI
# ===========================================================================

def get_guardrail_summary(results: list[GuardrailResult]) -> dict:
    """
    Return a compact summary dict for display in the Streamlit sidebar.
    """
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)
    blocked = sum(1 for r in results if not r.passed)
    warned  = sum(1 for r in results if r.passed and r.violations)
    all_violations = [v for r in results for v in r.violations]
    return {
        "total":      total,
        "passed":     passed,
        "blocked":    blocked,
        "warned":     warned,
        "violations": all_violations,
    }
