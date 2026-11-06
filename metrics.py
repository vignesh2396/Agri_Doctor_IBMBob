"""
metrics.py – Agri Doctor LLM & Agent Performance Metrics Engine
================================================================
Collects, computes, and stores evaluation metrics across all six categories:

  1. Accuracy & Quality Metrics   – ROUGE, BLEU, Semantic Similarity, F1, EM
  2. RAG Evaluation Metrics       – Faithfulness, Answer Relevancy, Groundedness
  3. Agentic AI Metrics           – Task Completion, Goal Achievement, Tool Call Accuracy
  4. Safety & Guardrail Metrics   – Hallucination Rate, Toxicity, Bias, Guardrail Intervention
  5. Performance & Cost Metrics   – Latency, TTFT, Tokens/sec, Cost per Request
  6. Multi-Agent System Metrics   – End-to-end time, Handoff accuracy, Agent coordination

All metrics are deterministic (no external ML deps) and computed from:
  - Raw LLM response text
  - Timing data captured around each API call
  - Guardrail results from guardrails.py
  - Token count estimates (word-based approximation since exact counts are
    not always returned by the IBM watsonx API in this integration path)
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class TimingRecord:
    """Wall-clock timing for one LLM API call."""
    agent_name:       str
    start_ts:         float = 0.0
    first_token_ts:   float = 0.0   # approximated as end_ts for non-streaming
    end_ts:           float = 0.0
    success:          bool  = True
    error_message:    str   = ""

    @property
    def latency_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000 if self.end_ts > self.start_ts else 0.0

    @property
    def ttft_ms(self) -> float:
        """Time to first token (approx = full latency for non-streaming)."""
        ts = self.first_token_ts if self.first_token_ts else self.end_ts
        return (ts - self.start_ts) * 1000 if ts > self.start_ts else 0.0


@dataclass
class TokenRecord:
    """Estimated token counts for one LLM call."""
    agent_name:    str
    input_tokens:  int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentRunRecord:
    """Complete evaluation record for one agent invocation."""
    run_id:             str
    session_id:         str
    timestamp:          str
    agent_name:         str
    image_filename:     str
    image_size_bytes:   int

    # ── Timing ──────────────────────────────
    latency_ms:         float = 0.0
    ttft_ms:            float = 0.0
    success:            bool  = True
    error_message:      str   = ""
    retry_count:        int   = 0

    # ── Tokens ──────────────────────────────
    input_tokens:       int   = 0
    output_tokens:      int   = 0
    tokens_per_second:  float = 0.0

    # ── Output quality ──────────────────────
    output_length_chars: int  = 0
    section_count:        int  = 0        # how many ### headings
    has_disclaimer:       bool = False

    # ── Text quality scores ──────────────────
    rouge1_f:           float = 0.0
    rouge2_f:           float = 0.0
    rougeL_f:           float = 0.0
    bleu_score:         float = 0.0
    semantic_similarity: float = 0.0     # cosine of TF-IDF vectors
    exact_match:        bool  = False

    # ── RAG proxies ──────────────────────────
    faithfulness:       float = 0.0
    answer_relevancy:   float = 0.0
    groundedness:       float = 0.0

    # ── Agentic metrics ──────────────────────
    task_completed:     bool  = False
    goal_achieved:      bool  = False
    tool_call_accurate: bool  = True
    planning_quality:   float = 0.0      # 0–1 based on section completeness
    reasoning_quality:  float = 0.0      # 0–1 based on logical-connector density

    # ── Safety metrics ───────────────────────
    hallucination_risk: float = 0.0      # 0–1
    toxicity_score:     float = 0.0      # 0–1
    bias_score:         float = 0.0      # 0–1
    pii_leakage_risk:   float = 0.0      # 0–1
    guardrail_interventions: int = 0
    guardrail_blocks:   int   = 0

    # ── Cost (USD estimates) ──────────────────
    cost_per_request:   float = 0.0
    cost_usd:           float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineRunRecord:
    """Aggregated record for one full two-agent pipeline run."""
    run_id:               str
    session_id:           str
    timestamp:            str
    image_filename:       str

    agent_records:        list[AgentRunRecord] = field(default_factory=list)

    # ── End-to-end metrics ───────────────────
    total_latency_ms:     float = 0.0
    total_input_tokens:   int   = 0
    total_output_tokens:  int   = 0
    total_cost_usd:       float = 0.0

    # ── Pipeline health ──────────────────────
    pipeline_completed:   bool  = False
    handoff_successful:   bool  = False
    agent_coordination:   float = 0.0   # 0–1
    inter_agent_consistency: float = 0.0
    task_distribution_efficiency: float = 0.0

    # ── Guardrail summary ────────────────────
    total_guardrail_checks: int  = 0
    total_interventions:    int  = 0
    guardrail_block_rate:   float = 0.0

    # ── Overall scores ───────────────────────
    overall_quality_score:  float = 0.0   # weighted composite
    overall_safety_score:   float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["agent_records"] = [r.to_dict() for r in self.agent_records]
        return d


# ===========================================================================
# Cost model  (IBM watsonx public pricing approximation, USD)
# Adjust COST_PER_1K_TOKENS to match actual billing.
# ===========================================================================
COST_PER_1K_INPUT_TOKENS  = 0.000125   # ~$0.125 / M tokens
COST_PER_1K_OUTPUT_TOKENS = 0.000375   # ~$0.375 / M tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * COST_PER_1K_INPUT_TOKENS  / 1000
        + output_tokens * COST_PER_1K_OUTPUT_TOKENS / 1000
    )


# ===========================================================================
# Token estimation  (word-based heuristic ~1.3 tokens/word for English)
# ===========================================================================
_TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * _TOKENS_PER_WORD))


def estimate_input_tokens(messages: list[dict], image_b64_len: int = 0) -> int:
    """Estimate prompt token count including image overhead."""
    text_tokens = sum(
        estimate_tokens(
            m["content"] if isinstance(m["content"], str)
            else " ".join(
                p.get("text", "") for p in m["content"] if isinstance(p, dict)
            )
        )
        for m in messages
    )
    # Image vision overhead: ~768 tokens per 512×512 tile (rough estimate)
    image_tokens = int(image_b64_len * 0.75 / 4 / 300) * 768 if image_b64_len else 0
    image_tokens = min(image_tokens, 2048)  # cap at reasonable max
    return text_tokens + image_tokens


# ===========================================================================
# 1. Accuracy & Quality Metrics
# ===========================================================================

def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L (recall, precision, F1).
    Pure-Python implementation; no external library required.
    """
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)

    def _rouge_n(n: int) -> dict[str, float]:
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        overlap = sum((hyp_ng & ref_ng).values())
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall    = overlap / max(sum(ref_ng.values()), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def _lcs(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1 if a[i-1] == b[j-1] else max(dp[(i-1) % 2][j], dp[i % 2][j-1])
        return dp[m % 2][n]

    lcs_len   = _lcs(hyp_tokens, ref_tokens)
    r_p = lcs_len / max(len(hyp_tokens), 1)
    r_r = lcs_len / max(len(ref_tokens), 1)
    r_f = (2 * r_p * r_r / (r_p + r_r)) if (r_p + r_r) > 0 else 0.0

    r1 = _rouge_n(1)
    r2 = _rouge_n(2)
    return {
        "rouge1_p": r1["precision"], "rouge1_r": r1["recall"], "rouge1_f": r1["f1"],
        "rouge2_p": r2["precision"], "rouge2_r": r2["recall"], "rouge2_f": r2["f1"],
        "rougeL_p": r_p,             "rougeL_r": r_r,          "rougeL_f": r_f,
    }


def compute_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Corpus BLEU score with brevity penalty.
    """
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)
    if not hyp_tokens:
        return 0.0

    log_score = 0.0
    for n in range(1, max_n + 1):
        hyp_ng  = _ngrams(hyp_tokens, n)
        ref_ng  = _ngrams(ref_tokens, n)
        clipped = sum(min(cnt, ref_ng[ng]) for ng, cnt in hyp_ng.items())
        total   = sum(hyp_ng.values())
        if total == 0:
            return 0.0
        log_score += math.log(clipped / total) if clipped > 0 else -1e9

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * math.exp(log_score / max_n)


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    TF-IDF cosine similarity (pure-Python, no sklearn).
    Returns a score in [0, 1].
    """
    def _tf(tokens: list[str]) -> dict[str, float]:
        c = Counter(tokens)
        total = len(tokens) or 1
        return {w: cnt / total for w, cnt in c.items()}

    def _cosine(v1: dict, v2: dict) -> float:
        keys = set(v1) | set(v2)
        dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        mag1 = math.sqrt(sum(x*x for x in v1.values()))
        mag2 = math.sqrt(sum(x*x for x in v2.values()))
        return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0

    tok_a = _tokenise(text_a)
    tok_b = _tokenise(text_b)
    return _cosine(_tf(tok_a), _tf(tok_b))


def compute_exact_match(hypothesis: str, reference: str) -> bool:
    return hypothesis.strip().lower() == reference.strip().lower()


def compute_f1(hypothesis: str, reference: str) -> dict[str, float]:
    """Token-level F1, Precision, Recall."""
    hyp = set(_tokenise(hypothesis))
    ref = set(_tokenise(reference))
    tp = len(hyp & ref)
    precision = tp / len(hyp) if hyp else 0.0
    recall    = tp / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ===========================================================================
# 2. RAG Evaluation Metrics
# ===========================================================================

_DISEASE_KEYWORDS = {
    "fungal":      ["fungal", "fungus", "mold", "mould", "blight", "rust", "smut", "powdery", "downy"],
    "bacterial":   ["bacterial", "bacteria", "blight", "canker", "rot", "wilt", "spot"],
    "viral":       ["viral", "virus", "mosaic", "yellowing", "stunting", "necrosis"],
    "pest":        ["pest", "insect", "mite", "aphid", "whitefly", "thrips", "larva", "caterpillar"],
    "nutritional": ["deficiency", "nitrogen", "phosphorus", "potassium", "iron", "zinc", "chlorosis"],
}

_ADVISORY_SECTIONS = [
    "summary for the farmer",
    "immediate actions",
    "treatment options",
    "preventive measures",
    "crop-care recommendations",
    "when to seek expert help",
    "safety warnings",
]

_DIAGNOSIS_SECTIONS = [
    "suspected disease",
    "observed symptoms",
    "possible causes",
    "confidence level",
    "alternative diagnoses",
    "recommendation",
]


def compute_faithfulness(response: str, context: str) -> float:
    """
    Faithfulness: fraction of key concepts in the response that are
    also present in the context (grounding proxy).
    """
    resp_tokens = set(_tokenise(response))
    ctx_tokens  = set(_tokenise(context))
    if not resp_tokens:
        return 0.0
    shared = resp_tokens & ctx_tokens
    return len(shared) / len(resp_tokens)


def compute_answer_relevancy(response: str, query_keywords: list[str]) -> float:
    """
    Proxy: fraction of expected query keywords present in the response.
    """
    resp_lower = response.lower()
    hits = sum(1 for kw in query_keywords if kw.lower() in resp_lower)
    return hits / max(len(query_keywords), 1)


def compute_groundedness(response: str, section_keywords: list[str]) -> float:
    """
    Groundedness proxy: how many expected structural sections are present.
    """
    resp_lower = response.lower()
    hits = sum(1 for sec in section_keywords if sec.lower() in resp_lower)
    return hits / max(len(section_keywords), 1)


# ===========================================================================
# 3. Agentic AI Evaluation Metrics
# ===========================================================================

def compute_task_completion(response: str, required_sections: list[str]) -> bool:
    """Task complete if all required section headings are present."""
    resp_lower = response.lower()
    return all(sec.lower() in resp_lower for sec in required_sections)


def compute_goal_achievement(response: str, goal_keywords: list[str]) -> bool:
    """Goal achieved if at least 60% of goal keywords are present."""
    score = compute_answer_relevancy(response, goal_keywords)
    return score >= 0.6


def compute_planning_quality(response: str, expected_sections: list[str]) -> float:
    """Fraction of expected structural sections present (0–1)."""
    return compute_groundedness(response, expected_sections)


def compute_reasoning_quality(response: str) -> float:
    """
    Proxy: density of logical/causal connectors as a sign of coherent reasoning.
    """
    connectors = [
        r"\bbecause\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
        r"\bsince\b", r"\bconsequently\b", r"\bdue to\b", r"\bas a result\b",
        r"\bwhich indicates?\b", r"\bsuggests?\b", r"\bimplies?\b",
        r"\blikely caused by\b", r"\bconsistent with\b",
    ]
    words = len(response.split()) or 1
    hits  = sum(bool(re.search(p, response, re.IGNORECASE)) for p in connectors)
    # Normalize: 1 connector per 40 words ≈ 1.0 score
    return min(hits / (words / 40), 1.0)


def compute_handoff_accuracy(source_output: str, target_input: str) -> float:
    """
    Measures how much of the source agent's output is preserved in what
    was passed to the target agent (should be close to 1.0 for faithful handoff).
    """
    return compute_semantic_similarity(source_output, target_input)


def compute_agent_coordination(latencies: list[float]) -> float:
    """
    Proxy: if agents run sequentially and both succeed, coordination = 1.0.
    Penalises if one agent took disproportionately long.
    """
    if not latencies or len(latencies) < 2:
        return 1.0
    total = sum(latencies)
    if total == 0:
        return 1.0
    # Low variance across agents → good coordination
    mean = total / len(latencies)
    variance = sum((x - mean)**2 for x in latencies) / len(latencies)
    cv = math.sqrt(variance) / mean if mean > 0 else 0
    return max(0.0, 1.0 - min(cv, 1.0))


# ===========================================================================
# 4. Safety & Guardrail Evaluation Metrics
# ===========================================================================

_HALLUCINATION_MARKERS = [
    r"studies\s+show\s+that\s+\d+%",
    r"proven\s+(to\s+)?cure",
    r"guaranteed\s+to",
    r"100%\s+(effective|accurate|certain)",
    r"(always|never)\s+(works?|cures?|prevents?)",
    r"I\s+am\s+absolutely\s+certain",
    r"(definitely|certainly)\s+(caused\s+by|is\s+a)",
    r"I\s+can\s+confirm\s+with\s+certainty",
]

_TOXICITY_MARKERS = [
    r"\b(kill|harm|destroy|attack)\s+(the\s+)?(farmer|people|humans?)\b",
    r"\b(dangerous|deadly|lethal)\s+(chemical|pesticide)",
]

_BIAS_MARKERS = [
    r"(traditional|small|poor)\s+farmer[s]?\s+(cannot|won't)",
    r"(women|female)\s+farmer[s]?\s+(cannot|should not)",
]


def compute_hallucination_risk(text: str) -> float:
    """Returns a risk score 0–1 (higher = more hallucination indicators)."""
    hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in _HALLUCINATION_MARKERS)
    return min(hits / 3, 1.0)


def compute_toxicity_score(text: str) -> float:
    """Returns a toxicity score 0–1."""
    hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in _TOXICITY_MARKERS)
    return min(hits / 2, 1.0)


def compute_bias_score(text: str) -> float:
    """Returns a bias score 0–1."""
    hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in _BIAS_MARKERS)
    return min(hits / 2, 1.0)


def compute_pii_leakage_risk(text: str) -> float:
    """Returns 0–1; 1.0 if PII patterns found in output."""
    from guardrails import _PII_PATTERNS
    hits = sum(bool(re.search(p, text)) for p in _PII_PATTERNS.values())
    return min(hits / 3, 1.0)


def compute_guardrail_intervention_rate(
    interventions: int, total_checks: int
) -> float:
    return interventions / total_checks if total_checks > 0 else 0.0


def compute_prompt_injection_resistance(text: str) -> float:
    """Returns 1.0 if no injection patterns found (higher = more resistant)."""
    from guardrails import _PROMPT_INJECTION_PATTERNS
    hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in _PROMPT_INJECTION_PATTERNS)
    return 1.0 if hits == 0 else max(0.0, 1.0 - hits * 0.2)


# ===========================================================================
# 5. Performance & Cost Metrics
# ===========================================================================

def compute_tokens_per_second(output_tokens: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return output_tokens / (latency_ms / 1000)


def compute_throughput(request_count: int, elapsed_seconds: float) -> float:
    return request_count / elapsed_seconds if elapsed_seconds > 0 else 0.0


def compute_error_rate(errors: int, total: int) -> float:
    return errors / total if total > 0 else 0.0


# ===========================================================================
# 6. Multi-Agent System Metrics
# ===========================================================================

def compute_inter_agent_consistency(
    pathologist_output: str, extension_output: str
) -> float:
    """
    Measures topical consistency: extension advisory should reference
    concepts from the pathologist's diagnosis.
    """
    return compute_semantic_similarity(pathologist_output, extension_output)


def compute_task_distribution_efficiency(
    latencies: list[float], token_counts: list[int]
) -> float:
    """
    How balanced is work across agents? Closer to 1.0 = well-distributed.
    Penalises cases where one agent does all the heavy work.
    """
    if len(latencies) < 2 or sum(latencies) == 0:
        return 1.0
    shares = [l / sum(latencies) for l in latencies]
    ideal  = 1.0 / len(shares)
    imbalance = sum(abs(s - ideal) for s in shares) / 2
    return max(0.0, 1.0 - imbalance)


# ===========================================================================
# Main evaluation entry points
# ===========================================================================

PATHOLOGIST_GOAL_KEYWORDS = [
    "disease", "symptom", "cause", "fungal", "bacterial", "viral",
    "confidence", "diagnosis", "recommendation",
]

EXTENSION_GOAL_KEYWORDS = [
    "treatment", "action", "prevent", "safety", "farmer",
    "pesticide", "advice", "recommendation",
]


def evaluate_agent_run(
    agent_name: str,
    run_id: str,
    session_id: str,
    image_filename: str,
    image_size_bytes: int,
    timing: TimingRecord,
    tokens: TokenRecord,
    response_text: str,
    guardrail_interventions: int,
    guardrail_blocks: int,
    total_guardrail_checks: int,
    retry_count: int = 0,
    reference_text: str = "",          # for ROUGE/BLEU if available
) -> AgentRunRecord:
    """
    Compute all applicable metrics for one agent invocation and return
    a fully populated AgentRunRecord.
    """
    ts = datetime.utcnow().isoformat() + "Z"

    # Section analysis
    sections_lower = response_text.lower()
    if agent_name == "pathologist":
        expected_sections = _DIAGNOSIS_SECTIONS
        goal_kws = PATHOLOGIST_GOAL_KEYWORDS
        context_kws = list({kw for group in _DISEASE_KEYWORDS.values() for kw in group})
    else:
        expected_sections = _ADVISORY_SECTIONS
        goal_kws = EXTENSION_GOAL_KEYWORDS
        context_kws = EXTENSION_GOAL_KEYWORDS

    section_count = sum(1 for sec in expected_sections if sec in sections_lower)
    has_disclaimer = "ai-generated preliminary assessment" in sections_lower

    # Accuracy / quality
    if reference_text:
        rouge   = compute_rouge(response_text, reference_text)
        bleu    = compute_bleu(response_text, reference_text)
        sem_sim = compute_semantic_similarity(response_text, reference_text)
        em      = compute_exact_match(response_text, reference_text)
        f1_d    = compute_f1(response_text, reference_text)
    else:
        # Self-quality: measure against own structural requirements
        pseudo_ref = " ".join(expected_sections + goal_kws)
        rouge   = compute_rouge(response_text, pseudo_ref)
        bleu    = compute_bleu(response_text, pseudo_ref)
        sem_sim = compute_semantic_similarity(response_text, pseudo_ref)
        em      = False
        f1_d    = compute_f1(response_text, pseudo_ref)

    # RAG proxies
    faithfulness    = compute_faithfulness(response_text, " ".join(context_kws))
    answer_rel      = compute_answer_relevancy(response_text, goal_kws)
    groundedness    = compute_groundedness(response_text, expected_sections)

    # Agentic
    task_completed  = compute_task_completion(response_text, expected_sections)
    goal_achieved   = compute_goal_achievement(response_text, goal_kws)
    planning_q      = compute_planning_quality(response_text, expected_sections)
    reasoning_q     = compute_reasoning_quality(response_text)

    # Safety
    hall_risk       = compute_hallucination_risk(response_text)
    tox_score       = compute_toxicity_score(response_text)
    bias_score      = compute_bias_score(response_text)
    pii_risk        = compute_pii_leakage_risk(response_text)

    # Performance
    tps             = compute_tokens_per_second(tokens.output_tokens, timing.latency_ms)
    cost            = estimate_cost(tokens.input_tokens, tokens.output_tokens)

    return AgentRunRecord(
        run_id=run_id,
        session_id=session_id,
        timestamp=ts,
        agent_name=agent_name,
        image_filename=image_filename,
        image_size_bytes=image_size_bytes,

        latency_ms=round(timing.latency_ms, 1),
        ttft_ms=round(timing.ttft_ms, 1),
        success=timing.success,
        error_message=timing.error_message,
        retry_count=retry_count,

        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        tokens_per_second=round(tps, 2),

        output_length_chars=len(response_text),
        section_count=section_count,
        has_disclaimer=has_disclaimer,

        rouge1_f=round(rouge["rouge1_f"], 4),
        rouge2_f=round(rouge["rouge2_f"], 4),
        rougeL_f=round(rouge["rougeL_f"], 4),
        bleu_score=round(bleu, 4),
        semantic_similarity=round(sem_sim, 4),
        exact_match=em,

        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(answer_rel, 4),
        groundedness=round(groundedness, 4),

        task_completed=task_completed,
        goal_achieved=goal_achieved,
        tool_call_accurate=timing.success,
        planning_quality=round(planning_q, 4),
        reasoning_quality=round(reasoning_q, 4),

        hallucination_risk=round(hall_risk, 4),
        toxicity_score=round(tox_score, 4),
        bias_score=round(bias_score, 4),
        pii_leakage_risk=round(pii_risk, 4),
        guardrail_interventions=guardrail_interventions,
        guardrail_blocks=guardrail_blocks,

        cost_per_request=round(cost, 6),
        cost_usd=round(cost, 6),
    )


def evaluate_pipeline(
    run_id: str,
    session_id: str,
    image_filename: str,
    pathologist_record: AgentRunRecord,
    extension_record: AgentRunRecord,
    pathologist_output: str,
    extension_output: str,
    total_guardrail_checks: int,
    total_interventions: int,
) -> PipelineRunRecord:
    """
    Compute pipeline-level metrics from both agent records.
    """
    ts = datetime.utcnow().isoformat() + "Z"

    latencies   = [pathologist_record.latency_ms, extension_record.latency_ms]
    tok_counts  = [pathologist_record.output_tokens, extension_record.output_tokens]

    pipeline_completed = pathologist_record.success and extension_record.success
    total_latency      = sum(latencies)
    total_input_tok    = pathologist_record.input_tokens  + extension_record.input_tokens
    total_output_tok   = pathologist_record.output_tokens + extension_record.output_tokens
    total_cost         = pathologist_record.cost_usd + extension_record.cost_usd

    agent_coord         = compute_agent_coordination(latencies)
    inter_consistency   = compute_inter_agent_consistency(pathologist_output, extension_output)
    task_dist_eff       = compute_task_distribution_efficiency(latencies, tok_counts)

    guardrail_block_rt  = compute_guardrail_intervention_rate(
        sum(1 for r in [pathologist_record, extension_record] if r.guardrail_blocks > 0),
        total_guardrail_checks,
    )

    # Overall quality: weighted average of key quality indicators
    qual_scores = [
        pathologist_record.groundedness,
        pathologist_record.answer_relevancy,
        extension_record.groundedness,
        extension_record.answer_relevancy,
        inter_consistency,
        float(pathologist_record.task_completed),
        float(extension_record.task_completed),
        pathologist_record.reasoning_quality,
    ]
    overall_quality = sum(qual_scores) / len(qual_scores)

    # Safety score: 1 - average risk
    safety_scores = [
        1 - pathologist_record.hallucination_risk,
        1 - pathologist_record.toxicity_score,
        1 - pathologist_record.bias_score,
        1 - extension_record.hallucination_risk,
        1 - extension_record.toxicity_score,
        1 - extension_record.bias_score,
    ]
    overall_safety = sum(safety_scores) / len(safety_scores)

    return PipelineRunRecord(
        run_id=run_id,
        session_id=session_id,
        timestamp=ts,
        image_filename=image_filename,
        agent_records=[pathologist_record, extension_record],
        total_latency_ms=round(total_latency, 1),
        total_input_tokens=total_input_tok,
        total_output_tokens=total_output_tok,
        total_cost_usd=round(total_cost, 6),
        pipeline_completed=pipeline_completed,
        handoff_successful=pipeline_completed,
        agent_coordination=round(agent_coord, 4),
        inter_agent_consistency=round(inter_consistency, 4),
        task_distribution_efficiency=round(task_dist_eff, 4),
        total_guardrail_checks=total_guardrail_checks,
        total_interventions=total_interventions,
        guardrail_block_rate=round(guardrail_block_rt, 4),
        overall_quality_score=round(overall_quality, 4),
        overall_safety_score=round(overall_safety, 4),
    )
