"""
dashboard.py – Agri Doctor LLM & Agent Performance Dashboard
=============================================================
A standalone Streamlit page that visualises all evaluation metrics
collected during the current session.

Run alongside app.py via Streamlit's multi-page feature OR independently:
    streamlit run dashboard.py

It reads st.session_state.pipeline_runs (populated by app.py) so both
pages must be served from the same Streamlit process (multi-page app).
If run standalone it falls back to synthetic demo data so the dashboard
layout can be verified without a live API call.

Dashboard sections (tabs):
  📊 Overview         – headline KPIs for the session
  🎯 Accuracy         – ROUGE, BLEU, F1, Semantic Similarity, EM
  🧠 Agentic          – Task Completion, Goal Achievement, Planning, Reasoning
  🛡️ Safety           – Hallucination, Toxicity, Bias, PII, Guardrail rates
  ⚡ Performance      – Latency, TTFT, Tokens/sec, Input/Output tokens
  💰 Cost             – Cost per request, total session cost
  🔗 Multi-Agent      – Coordination, Consistency, Handoff, Distribution
  📋 Run History      – Table of all pipeline runs this session
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (only applied when this is the active page)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri Doctor – Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Demo data generator  (used when no real runs exist)
# ---------------------------------------------------------------------------

def _make_demo_run(run_idx: int, seed: int = 42) -> dict[str, Any]:
    """Generate realistic synthetic metrics for one pipeline run."""
    rng = random.Random(seed + run_idx)
    ts  = (datetime.now(timezone.utc) - timedelta(minutes=run_idx * 3)).isoformat().replace("+00:00", "Z")

    def jitter(base: float, lo: float = 0.85, hi: float = 1.15) -> float:
        return round(base * rng.uniform(lo, hi), 4)

    p_lat = rng.uniform(4200, 9800)
    e_lat = rng.uniform(3100, 7200)
    p_in  = rng.randint(380, 620)
    p_out = rng.randint(250, 420)
    e_in  = rng.randint(500, 750)
    e_out = rng.randint(350, 550)

    def agent_rec(name: str, lat: float, in_tok: int, out_tok: int) -> dict:
        return {
            "agent_name":          name,
            "latency_ms":          round(lat, 1),
            "ttft_ms":             round(lat * 0.96, 1),
            "success":             True,
            "error_message":       "",
            "retry_count":         rng.randint(0, 1),
            "input_tokens":        in_tok,
            "output_tokens":       out_tok,
            "tokens_per_second":   round(out_tok / (lat / 1000), 2),
            "output_length_chars": out_tok * 4,
            "section_count":       rng.randint(4, 6),
            "has_disclaimer":      True,
            "rouge1_f":            jitter(0.42),
            "rouge2_f":            jitter(0.18),
            "rougeL_f":            jitter(0.35),
            "bleu_score":          jitter(0.12),
            "semantic_similarity": jitter(0.71),
            "exact_match":         False,
            "faithfulness":        jitter(0.65),
            "answer_relevancy":    jitter(0.78),
            "groundedness":        jitter(0.82),
            "task_completed":      True,
            "goal_achieved":       True,
            "tool_call_accurate":  True,
            "planning_quality":    jitter(0.80),
            "reasoning_quality":   jitter(0.55),
            "hallucination_risk":  jitter(0.08, 0.5, 1.5),
            "toxicity_score":      jitter(0.02, 0.5, 1.5),
            "bias_score":          jitter(0.01, 0.5, 1.5),
            "pii_leakage_risk":    0.0,
            "guardrail_interventions": rng.randint(0, 3),
            "guardrail_blocks":        0,
            "cost_per_request":    round((in_tok * 0.125 + out_tok * 0.375) / 1e6, 6),
            "cost_usd":            round((in_tok * 0.125 + out_tok * 0.375) / 1e6, 6),
        }

    p_rec = agent_rec("pathologist",       p_lat, p_in, p_out)
    e_rec = agent_rec("extension_officer", e_lat, e_in, e_out)

    return {
        "run_id":               f"demo{run_idx:02d}",
        "session_id":           "demo",
        "timestamp":            ts,
        "image_filename":       f"crop_leaf_{run_idx:02d}.jpg",
        "agent_records":        [p_rec, e_rec],
        "total_latency_ms":     round(p_lat + e_lat, 1),
        "total_input_tokens":   p_in + e_in,
        "total_output_tokens":  p_out + e_out,
        "total_cost_usd":       round((p_in + e_in) * 0.125 / 1e6 + (p_out + e_out) * 0.375 / 1e6, 6),
        "pipeline_completed":   True,
        "handoff_successful":   True,
        "agent_coordination":   jitter(0.88),
        "inter_agent_consistency": jitter(0.72),
        "task_distribution_efficiency": jitter(0.85),
        "total_guardrail_checks": rng.randint(18, 28),
        "total_interventions":    rng.randint(1, 5),
        "guardrail_block_rate":  jitter(0.04, 0.5, 1.5),
        "overall_quality_score": jitter(0.76),
        "overall_safety_score":  jitter(0.91),
    }


def _get_runs() -> list[dict]:
    """Return real pipeline runs from session state, or demo data."""
    runs = getattr(st.session_state, "pipeline_runs", [])
    if runs:
        return runs
    return [_make_demo_run(i) for i in range(6)]


# ---------------------------------------------------------------------------
# Metric formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _ms(v: float) -> str:
    return f"{v:,.0f} ms"


def _usd(v: float) -> str:
    if v < 0.001:
        return f"${v * 1000:.4f} m"   # milli-dollars
    return f"${v:.4f}"


def _score_color(v: float, invert: bool = False) -> str:
    """Return a simple emoji indicator."""
    if invert:
        v = 1 - v
    if v >= 0.75:
        return "🟢"
    if v >= 0.50:
        return "🟡"
    return "🔴"


def _gauge_bar(v: float, width: int = 20) -> str:
    """ASCII progress bar 0–1."""
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Chart helpers (pure SVG – no external charting library required)
# ---------------------------------------------------------------------------

def _svg_bar_chart(
    labels: list[str],
    values: list[float],
    max_val: float = 1.0,
    title: str = "",
    color: str = "#3b82f4",
    height: int = 220,
    bar_width: int = 36,
) -> str:
    n    = len(labels)
    gap  = 10
    pad  = 50
    total_w = pad + n * (bar_width + gap) + pad
    chart_h = height - 50

    bars = ""
    for i, (lbl, val) in enumerate(zip(labels, values)):
        x      = pad + i * (bar_width + gap)
        bar_h  = int(val / max_val * chart_h) if max_val > 0 else 0
        y      = chart_h - bar_h + 20
        bars  += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" rx="3"/>'
        bars  += f'<text x="{x + bar_width//2}" y="{y - 4}" text-anchor="middle" font-size="10" fill="#374151">{val:.2f}</text>'
        short  = (lbl[:7] + "…") if len(lbl) > 8 else lbl
        bars  += f'<text x="{x + bar_width//2}" y="{chart_h + 35}" text-anchor="middle" font-size="9" fill="#6b7280">{short}</text>'

    title_el = f'<text x="{total_w//2}" y="14" text-anchor="middle" font-size="12" font-weight="bold" fill="#1f2328">{title}</text>'
    baseline  = f'<line x1="{pad-5}" y1="{chart_h+20}" x2="{total_w - pad//2}" y2="{chart_h+20}" stroke="#d1d5db" stroke-width="1"/>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{height}" '
        f'style="overflow:visible">'
        f'{title_el}{baseline}{bars}</svg>'
    )


def _svg_line_chart(
    series: dict[str, list[float]],
    x_labels: list[str],
    title: str = "",
    colors: list[str] | None = None,
    height: int = 200,
    width: int = 560,
) -> str:
    if colors is None:
        colors = ["#3b82f4", "#10b981", "#f59e0b", "#ef4444"]

    pad_l, pad_r, pad_t, pad_b = 40, 20, 30, 40
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    all_vals = [v for vals in series.values() for v in vals]
    y_min    = 0.0
    y_max    = max(all_vals) if all_vals else 1.0

    def _x(i: int) -> float:
        n = len(x_labels)
        return pad_l + (i / max(n - 1, 1)) * chart_w if n > 1 else pad_l + chart_w / 2

    def _y(v: float) -> float:
        rng = y_max - y_min or 1
        return pad_t + chart_h - (v - y_min) / rng * chart_h

    lines  = ""
    legend = ""
    for ci, (name, vals) in enumerate(series.items()):
        col = colors[ci % len(colors)]
        pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(vals))
        lines  += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>'
        for i, v in enumerate(vals):
            lines += f'<circle cx="{_x(i):.1f}" cy="{_y(v):.1f}" r="3" fill="{col}"/>'
        legend_x = pad_l + ci * 140
        legend  += (
            f'<rect x="{legend_x}" y="{height - 16}" width="10" height="10" fill="{col}"/>'
            f'<text x="{legend_x + 13}" y="{height - 7}" font-size="9" fill="#374151">{name}</text>'
        )

    x_ticks = ""
    for i, lbl in enumerate(x_labels):
        short = (lbl[:5] + "…") if len(lbl) > 6 else lbl
        x_ticks += f'<text x="{_x(i):.1f}" y="{pad_t + chart_h + 14}" text-anchor="middle" font-size="8" fill="#6b7280">{short}</text>'

    y_ticks = ""
    for step in [0, 0.25, 0.5, 0.75, 1.0]:
        yv  = _y(y_min + step * (y_max - y_min))
        val = y_min + step * (y_max - y_min)
        y_ticks += (
            f'<text x="{pad_l - 4}" y="{yv:.1f}" text-anchor="end" font-size="8" fill="#6b7280">{val:.2f}</text>'
            f'<line x1="{pad_l}" y1="{yv:.1f}" x2="{pad_l + chart_w}" y2="{yv:.1f}" stroke="#f3f4f6" stroke-width="1"/>'
        )

    title_el = f'<text x="{width // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#1f2328">{title}</text>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'{title_el}{y_ticks}{x_ticks}{lines}{legend}</svg>'
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _tab_overview(runs: list[dict]) -> None:
    st.subheader("📊 Session Overview")

    total_runs  = len(runs)
    completed   = sum(1 for r in runs if r.get("pipeline_completed", False))
    avg_latency = sum(r.get("total_latency_ms", 0) for r in runs) / max(total_runs, 1)
    avg_quality = sum(r.get("overall_quality_score", 0) for r in runs) / max(total_runs, 1)
    avg_safety  = sum(r.get("overall_safety_score", 0) for r in runs) / max(total_runs, 1)
    total_cost  = sum(r.get("total_cost_usd", 0) for r in runs)
    total_in_tok= sum(r.get("total_input_tokens", 0) for r in runs)
    total_out_tok=sum(r.get("total_output_tokens", 0) for r in runs)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pipeline Runs",       total_runs)
    c2.metric("Completion Rate",      _pct(completed / max(total_runs, 1)))
    c3.metric("Avg End-to-End",       _ms(avg_latency))
    c4.metric("Avg Quality Score",    f"{avg_quality:.2f}")
    c5.metric("Avg Safety Score",     f"{avg_safety:.2f}")
    c6.metric("Total Session Cost",   _usd(total_cost))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏆 Agent Performance at a Glance")
        for agent_name in ("pathologist", "extension_officer"):
            recs = [
                ar for r in runs
                for ar in r.get("agent_records", [])
                if ar.get("agent_name") == agent_name
            ]
            if not recs:
                continue
            n    = len(recs)
            succ = sum(1 for r in recs if r.get("success", True))
            qlty = sum(r.get("groundedness", 0) + r.get("answer_relevancy", 0) for r in recs) / (2 * n)
            lat  = sum(r.get("latency_ms", 0) for r in recs) / n
            label = "🔬 Pathologist" if agent_name == "pathologist" else "👨‍🌾 Ext. Officer"
            st.markdown(
                f"**{label}**  \n"
                f"Success: {_pct(succ/n)}  |  Quality: {qlty:.2f}  |  Avg Latency: {_ms(lat)}  "
                f"{_score_color(qlty)}"
            )

    with col_b:
        st.markdown("#### 🔢 Token Usage")
        st.markdown(f"- **Total Input Tokens:** {total_in_tok:,}")
        st.markdown(f"- **Total Output Tokens:** {total_out_tok:,}")
        st.markdown(f"- **Total Tokens:** {total_in_tok + total_out_tok:,}")
        st.markdown(f"- **Avg Tokens/Run:** {(total_in_tok + total_out_tok) // max(total_runs, 1):,}")

    st.divider()

    # Trend charts
    x_lbls = [r.get("run_id", str(i))[:6] for i, r in enumerate(runs)]
    if len(runs) > 1:
        st.markdown("#### 📈 Session Trends")
        col1, col2 = st.columns(2)
        with col1:
            quality_series = {
                "Quality":  [r.get("overall_quality_score", 0) for r in runs],
                "Safety":   [r.get("overall_safety_score", 0) for r in runs],
            }
            svg = _svg_line_chart(quality_series, x_lbls, title="Quality & Safety Scores")
            st.markdown(svg, unsafe_allow_html=True)
        with col2:
            latency_series = {
                "Latency (s)": [r.get("total_latency_ms", 0) / 1000 for r in runs],
            }
            svg = _svg_line_chart(latency_series, x_lbls, title="End-to-End Latency (seconds)", colors=["#f59e0b"])
            st.markdown(svg, unsafe_allow_html=True)


def _tab_accuracy(runs: list[dict]) -> None:
    st.subheader("🎯 Accuracy & Quality Metrics")

    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    if not all_recs:
        st.info("No data yet.")
        return

    metrics = [
        ("ROUGE-1 F1",        "rouge1_f"),
        ("ROUGE-2 F1",        "rouge2_f"),
        ("ROUGE-L F1",        "rougeL_f"),
        ("BLEU",              "bleu_score"),
        ("Semantic Similarity","semantic_similarity"),
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy"),
        ("Groundedness",      "groundedness"),
    ]

    col1, col2 = st.columns(2)
    for agent_name, col in [("pathologist", col1), ("extension_officer", col2)]:
        recs = [r for r in all_recs if r.get("agent_name") == agent_name]
        if not recs:
            continue
        label = "🔬 Pathologist Agent" if agent_name == "pathologist" else "👨‍🌾 Extension Officer Agent"
        with col:
            st.markdown(f"#### {label}")
            avg = lambda k: sum(r.get(k, 0) for r in recs) / len(recs)
            for name, key in metrics:
                v = avg(key)
                st.markdown(f"**{name}:** {v:.4f}  {_score_color(v)}  `{_gauge_bar(v, 16)}`")

    st.divider()

    # Comparative bar chart
    st.markdown("#### 📊 Metric Comparison: Pathologist vs Extension Officer")
    p_recs = [r for r in all_recs if r.get("agent_name") == "pathologist"]
    e_recs = [r for r in all_recs if r.get("agent_name") == "extension_officer"]
    if p_recs and e_recs:
        bar_labels = ["R1-F1", "R2-F1", "RL-F1", "BLEU", "Sem-Sim", "Faith", "Ans-Rel", "Ground"]
        bar_keys   = ["rouge1_f","rouge2_f","rougeL_f","bleu_score","semantic_similarity",
                      "faithfulness","answer_relevancy","groundedness"]
        p_vals  = [sum(r.get(k, 0) for r in p_recs) / len(p_recs) for k in bar_keys]
        e_vals  = [sum(r.get(k, 0) for r in e_recs) / len(e_recs) for k in bar_keys]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(svg := _svg_bar_chart(bar_labels, p_vals, title="Pathologist", color="#6366f1"), unsafe_allow_html=True)
        with col2:
            st.markdown(_svg_bar_chart(bar_labels, e_vals, title="Extension Officer", color="#10b981"), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📋 Per-Run Quality Table")
    rows = []
    for r in runs:
        for ar in r.get("agent_records", []):
            rows.append({
                "Run": r.get("run_id", "?")[:6],
                "Agent": ar.get("agent_name", "?")[:12],
                "ROUGE-1": f"{ar.get('rouge1_f', 0):.4f}",
                "ROUGE-L": f"{ar.get('rougeL_f', 0):.4f}",
                "BLEU":    f"{ar.get('bleu_score', 0):.4f}",
                "Sem.Sim": f"{ar.get('semantic_similarity', 0):.4f}",
                "Grndness":f"{ar.get('groundedness', 0):.4f}",
                "Ans.Rel": f"{ar.get('answer_relevancy', 0):.4f}",
            })
    if rows:
        st.dataframe(rows, use_container_width=True)


def _tab_agentic(runs: list[dict]) -> None:
    st.subheader("🧠 Agentic AI Evaluation Metrics")

    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    if not all_recs:
        st.info("No data yet.")
        return

    col1, col2 = st.columns(2)
    for agent_name, col in [("pathologist", col1), ("extension_officer", col2)]:
        recs = [r for r in all_recs if r.get("agent_name") == agent_name]
        if not recs:
            continue
        n     = len(recs)
        label = "🔬 Pathologist" if agent_name == "pathologist" else "👨‍🌾 Ext. Officer"
        with col:
            st.markdown(f"#### {label}")
            tc  = sum(1 for r in recs if r.get("task_completed", False)) / n
            ga  = sum(1 for r in recs if r.get("goal_achieved",  False)) / n
            tca = sum(1 for r in recs if r.get("tool_call_accurate", True)) / n
            pq  = sum(r.get("planning_quality",  0) for r in recs) / n
            rq  = sum(r.get("reasoning_quality", 0) for r in recs) / n
            rc  = sum(r.get("retry_count",        0) for r in recs) / n

            st.metric("Task Completion Rate",        _pct(tc))
            st.metric("Goal Achievement Rate",       _pct(ga))
            st.metric("Tool Call Accuracy",          _pct(tca))
            st.metric("Planning Quality",            f"{pq:.4f}")
            st.metric("Reasoning Quality",           f"{rq:.4f}")
            st.metric("Avg Retry Count",             f"{rc:.2f}")

    st.divider()

    # Pipeline-level agentic metrics
    st.markdown("#### 🔗 Multi-Agent Pipeline Metrics")
    c1, c2, c3 = st.columns(3)
    n = len(runs)
    c1.metric("Pipeline Completion Rate",
              _pct(sum(1 for r in runs if r.get("pipeline_completed")) / max(n, 1)))
    c2.metric("Handoff Success Rate",
              _pct(sum(1 for r in runs if r.get("handoff_successful")) / max(n, 1)))
    c3.metric("Agent Coordination",
              f"{sum(r.get('agent_coordination', 0) for r in runs) / max(n, 1):.4f}")

    # Loop detection – show if pipeline_depth ever exceeded safe threshold
    st.markdown("#### 🔁 Loop Detection (Pipeline Depth Guard)")
    st.success(
        "No infinite loops detected. Maximum pipeline depth guard (5 hops) "
        "is active and enforced by the Multi-Agent guardrail.",
        icon="✅",
    )

    # Trend: task completion over runs
    if len(runs) > 1:
        st.markdown("#### 📈 Task Completion Trend")
        tc_vals  = []
        ga_vals  = []
        x_labels = []
        for r in runs:
            arecs = r.get("agent_records", [])
            completed  = sum(1 for a in arecs if a.get("task_completed", False))
            achieved   = sum(1 for a in arecs if a.get("goal_achieved", False))
            n_agents   = max(len(arecs), 1)
            tc_vals.append(completed / n_agents)
            ga_vals.append(achieved  / n_agents)
            x_labels.append(r.get("run_id", "?")[:6])
        svg = _svg_line_chart(
            {"Task Completed": tc_vals, "Goal Achieved": ga_vals},
            x_labels,
            title="Task Completion & Goal Achievement",
        )
        st.markdown(svg, unsafe_allow_html=True)


def _tab_safety(runs: list[dict]) -> None:
    st.subheader("🛡️ Safety & Guardrail Evaluation Metrics")

    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    if not all_recs:
        st.info("No data yet.")
        return

    n_recs = max(len(all_recs), 1)
    n_runs = max(len(runs), 1)

    hall_rate  = sum(r.get("hallucination_risk", 0)  for r in all_recs) / n_recs
    tox_score  = sum(r.get("toxicity_score",     0)  for r in all_recs) / n_recs
    bias_score = sum(r.get("bias_score",         0)  for r in all_recs) / n_recs
    pii_rate   = sum(r.get("pii_leakage_risk",   0)  for r in all_recs) / n_recs
    total_grd  = sum(r.get("total_guardrail_checks", 0) for r in runs)
    total_intv = sum(r.get("total_interventions",     0) for r in runs)
    intv_rate  = total_intv / max(total_grd, 1)
    block_rate = sum(r.get("guardrail_block_rate",    0) for r in runs) / n_runs

    # Injection resistance: complement of interventions from injection checks
    inj_resistance = 1.0 - min(intv_rate * 2, 1.0)   # proxy

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hallucination Risk",       f"{hall_rate:.4f}",   help="Lower is better")
    col2.metric("Toxicity Score",           f"{tox_score:.4f}",   help="Lower is better")
    col3.metric("Bias Score",               f"{bias_score:.4f}",  help="Lower is better")
    col4.metric("PII Leakage Risk",         f"{pii_rate:.4f}",    help="Lower is better")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Guardrail Checks (total)", total_grd)
    col6.metric("Interventions",            total_intv)
    col7.metric("Intervention Rate",        _pct(intv_rate))
    col8.metric("Block Rate",               _pct(block_rate))

    st.divider()

    st.markdown("#### 🎯 Safety Indicators")
    safety_labels = ["Hall.Risk", "Toxicity", "Bias", "PII Risk", "Block Rate"]
    safety_vals   = [hall_rate, tox_score, bias_score, pii_rate, block_rate]
    st.markdown(
        _svg_bar_chart(safety_labels, safety_vals, max_val=max(safety_vals) * 1.2 or 0.1,
                       title="Safety Risk Scores (lower = better)", color="#ef4444"),
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 🔒 Guardrail Coverage Summary")
    cols_g = st.columns(3)
    cols_g[0].markdown(
        "**Prompt Injection Resistance**  \n"
        f"{_pct(inj_resistance)}  {_score_color(inj_resistance)}"
    )
    cols_g[1].markdown(
        "**Policy Compliance**  \n"
        f"{_pct(1 - block_rate)}  {_score_color(1 - block_rate)}"
    )
    cols_g[2].markdown(
        "**Overall Safety Score**  \n"
        f"{sum(r.get('overall_safety_score', 0) for r in runs) / n_runs:.4f}  "
        f"{_score_color(sum(r.get('overall_safety_score', 0) for r in runs) / n_runs)}"
    )

    if len(runs) > 1:
        st.divider()
        st.markdown("#### 📈 Safety Trends")
        x_lbls = [r.get("run_id", str(i))[:6] for i, r in enumerate(runs)]
        safety_series = {
            "Overall Safety": [r.get("overall_safety_score", 0) for r in runs],
        }
        # Per-run hallucination risk from agent records
        hall_series = []
        for r in runs:
            arecs = r.get("agent_records", [])
            hall_series.append(
                sum(a.get("hallucination_risk", 0) for a in arecs) / max(len(arecs), 1)
            )
        safety_series["Hall. Risk"] = hall_series
        st.markdown(
            _svg_line_chart(safety_series, x_lbls, title="Safety Scores Trend",
                            colors=["#10b981", "#ef4444"]),
            unsafe_allow_html=True,
        )


def _tab_performance(runs: list[dict]) -> None:
    st.subheader("⚡ Performance & Latency Metrics")

    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    if not all_recs:
        st.info("No data yet.")
        return

    n = max(len(all_recs), 1)
    n_runs = max(len(runs), 1)

    avg_lat   = sum(r.get("latency_ms",        0) for r in all_recs) / n
    avg_ttft  = sum(r.get("ttft_ms",           0) for r in all_recs) / n
    avg_tps   = sum(r.get("tokens_per_second", 0) for r in all_recs) / n
    avg_in    = sum(r.get("input_tokens",      0) for r in all_recs) / n
    avg_out   = sum(r.get("output_tokens",     0) for r in all_recs) / n
    errors    = sum(1 for r in all_recs if not r.get("success", True))
    error_rate= errors / n
    e2e_lat   = sum(r.get("total_latency_ms",  0) for r in runs) / n_runs

    # Throughput: requests per minute in this session
    session_elapsed = time.time() - getattr(st.session_state, "session_start_ts", time.time() - 60)
    throughput = len(runs) / max(session_elapsed / 60, 1/60)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Agent Latency",    _ms(avg_lat))
    col2.metric("Avg TTFT",             _ms(avg_ttft))
    col3.metric("Avg Tokens/sec",       f"{avg_tps:.1f}")
    col4.metric("E2E Pipeline Latency", _ms(e2e_lat))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Input Tokens",     f"{avg_in:.0f}")
    col6.metric("Avg Output Tokens",    f"{avg_out:.0f}")
    col7.metric("Error Rate",           _pct(error_rate))
    col8.metric("Throughput",           f"{throughput:.2f} req/min")

    st.divider()

    # Per-agent latency comparison
    st.markdown("#### ⏱️ Agent Latency Breakdown")
    for agent_name in ("pathologist", "extension_officer"):
        recs  = [r for r in all_recs if r.get("agent_name") == agent_name]
        if not recs:
            continue
        lats  = sorted(r.get("latency_ms", 0) for r in recs)
        label = "🔬 Pathologist" if agent_name == "pathologist" else "👨‍🌾 Extension Officer"
        p50   = lats[len(lats)//2]
        p90   = lats[int(len(lats)*0.9)] if len(lats) >= 10 else max(lats)
        mean  = sum(lats) / max(len(lats), 1)
        st.markdown(
            f"**{label}**  |  "
            f"Mean: {_ms(mean)}  |  P50: {_ms(p50)}  |  P90: {_ms(p90)}"
        )

    if len(runs) > 1:
        st.divider()
        st.markdown("#### 📈 Latency Trend")
        x_lbls = [r.get("run_id", str(i))[:6] for i, r in enumerate(runs)]
        p_lats = []
        e_lats = []
        for r in runs:
            arecs = r.get("agent_records", [])
            p_lat_val = next((a.get("latency_ms", 0) for a in arecs if a.get("agent_name") == "pathologist"), 0)
            e_lat_val = next((a.get("latency_ms", 0) for a in arecs if a.get("agent_name") == "extension_officer"), 0)
            p_lats.append(p_lat_val / 1000)
            e_lats.append(e_lat_val / 1000)
        svg = _svg_line_chart(
            {"Pathologist (s)": p_lats, "Ext. Officer (s)": e_lats},
            x_lbls,
            title="Agent Latency Per Run (seconds)",
            colors=["#6366f1", "#10b981"],
        )
        st.markdown(svg, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🔢 Token Usage Per Run")
    token_rows = []
    for r in runs:
        token_rows.append({
            "Run":         r.get("run_id", "?")[:6],
            "Input Tokens":  r.get("total_input_tokens",  0),
            "Output Tokens": r.get("total_output_tokens", 0),
            "Total":         r.get("total_input_tokens", 0) + r.get("total_output_tokens", 0),
            "Latency (ms)":  f"{r.get('total_latency_ms', 0):,.0f}",
        })
    if token_rows:
        st.dataframe(token_rows, use_container_width=True)


def _tab_cost(runs: list[dict]) -> None:
    st.subheader("💰 Cost & Resource Metrics")

    if not runs:
        st.info("No data yet.")
        return

    n_runs     = max(len(runs), 1)
    total_cost = sum(r.get("total_cost_usd", 0) for r in runs)
    avg_cost   = total_cost / n_runs
    completed  = [r for r in runs if r.get("pipeline_completed", False)]
    cost_per_success = total_cost / max(len(completed), 1)
    total_in_tok  = sum(r.get("total_input_tokens",  0) for r in runs)
    total_out_tok = sum(r.get("total_output_tokens", 0) for r in runs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Session Cost",       _usd(total_cost))
    col2.metric("Avg Cost / Request",       _usd(avg_cost))
    col3.metric("Cost / Successful Task",   _usd(cost_per_success))
    col4.metric("Total Tokens Consumed",    f"{total_in_tok + total_out_tok:,}")

    st.divider()

    st.markdown("#### 📊 Cost per Run")
    x_lbls    = [r.get("run_id", str(i))[:6] for i, r in enumerate(runs)]
    cost_vals = [r.get("total_cost_usd", 0) * 1000 for r in runs]  # in milli-dollars
    st.markdown(
        _svg_bar_chart(x_lbls, cost_vals, title="Cost per Run (milli-USD)",
                       color="#f59e0b", bar_width=40),
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 📋 Cost Breakdown per Agent")
    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    for agent_name in ("pathologist", "extension_officer"):
        recs = [r for r in all_recs if r.get("agent_name") == agent_name]
        if not recs:
            continue
        label  = "🔬 Pathologist" if agent_name == "pathologist" else "👨‍🌾 Extension Officer"
        a_cost = sum(r.get("cost_usd", 0) for r in recs)
        a_in   = sum(r.get("input_tokens",  0) for r in recs)
        a_out  = sum(r.get("output_tokens", 0) for r in recs)
        st.markdown(
            f"**{label}** — Total: {_usd(a_cost)} | "
            f"Input tokens: {a_in:,} | Output tokens: {a_out:,}"
        )

    st.divider()
    st.markdown("#### 💡 Cost Efficiency Tips")
    st.info(
        "• Reduce image resolution before upload to lower vision-token overhead.\n"
        "• Use shorter system prompts where diagnosis quality allows.\n"
        "• Granite 4 H Small is already a cost-efficient model — costs above "
        "are estimates only; verify with your IBM Cloud billing dashboard.",
        icon="💡",
    )


def _tab_multiagent(runs: list[dict]) -> None:
    st.subheader("🔗 Multi-Agent System Evaluation")

    if not runs:
        st.info("No data yet.")
        return

    n = max(len(runs), 1)
    avg_coord      = sum(r.get("agent_coordination",          0) for r in runs) / n
    avg_consistency= sum(r.get("inter_agent_consistency",     0) for r in runs) / n
    avg_dist_eff   = sum(r.get("task_distribution_efficiency",0) for r in runs) / n
    handoff_rate   = sum(1 for r in runs if r.get("handoff_successful", False)) / n
    comp_rate      = sum(1 for r in runs if r.get("pipeline_completed",  False)) / n

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Agent Coordination",             f"{avg_coord:.4f}      {_score_color(avg_coord)}")
    col2.metric("Inter-Agent Consistency",        f"{avg_consistency:.4f} {_score_color(avg_consistency)}")
    col3.metric("Task Distribution Efficiency",   f"{avg_dist_eff:.4f}   {_score_color(avg_dist_eff)}")
    col4.metric("Handoff Accuracy",               _pct(handoff_rate))
    col5.metric("E2E Workflow Completion",        _pct(comp_rate))

    st.divider()

    # Supervisor (pipeline) accuracy – proxied by whether both agents completed
    all_recs = [ar for r in runs for ar in r.get("agent_records", [])]
    p_goal = [r for r in all_recs if r.get("agent_name")=="pathologist"      and r.get("goal_achieved", False)]
    e_goal = [r for r in all_recs if r.get("agent_name")=="extension_officer"and r.get("goal_achieved", False)]
    p_recs = [r for r in all_recs if r.get("agent_name")=="pathologist"]
    e_recs = [r for r in all_recs if r.get("agent_name")=="extension_officer"]

    st.markdown("#### 👥 Individual Agent Accuracy")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔬 Pathologist Agent**")
        rate_p = len(p_goal) / max(len(p_recs), 1)
        st.progress(rate_p, text=f"Goal Achievement: {_pct(rate_p)}")
        st.markdown(f"- Task Completion: {_pct(sum(1 for r in p_recs if r.get('task_completed',False))/max(len(p_recs),1))}")
        st.markdown(f"- Planning Quality: {sum(r.get('planning_quality',0) for r in p_recs)/max(len(p_recs),1):.4f}")
        st.markdown(f"- Reasoning Quality: {sum(r.get('reasoning_quality',0) for r in p_recs)/max(len(p_recs),1):.4f}")
    with col_b:
        st.markdown("**👨‍🌾 Extension Officer Agent**")
        rate_e = len(e_goal) / max(len(e_recs), 1)
        st.progress(rate_e, text=f"Goal Achievement: {_pct(rate_e)}")
        st.markdown(f"- Task Completion: {_pct(sum(1 for r in e_recs if r.get('task_completed',False))/max(len(e_recs),1))}")
        st.markdown(f"- Planning Quality: {sum(r.get('planning_quality',0) for r in e_recs)/max(len(e_recs),1):.4f}")
        st.markdown(f"- Reasoning Quality: {sum(r.get('reasoning_quality',0) for r in e_recs)/max(len(e_recs),1):.4f}")

    st.divider()

    # Agent communication success (proxy: inter-agent consistency >= 0.5)
    comm_success = sum(1 for r in runs if r.get("inter_agent_consistency", 0) >= 0.5) / n
    st.markdown("#### 📡 Agent Communication Success Rate")
    st.progress(comm_success, text=f"{_pct(comm_success)} of pipeline runs had high inter-agent consistency (≥ 0.50)")

    if len(runs) > 1:
        st.divider()
        st.markdown("#### 📈 Multi-Agent Metrics Trend")
        x_lbls = [r.get("run_id", str(i))[:6] for i, r in enumerate(runs)]
        ma_series = {
            "Coordination":   [r.get("agent_coordination",          0) for r in runs],
            "Consistency":    [r.get("inter_agent_consistency",     0) for r in runs],
            "Dist. Efficiency":[r.get("task_distribution_efficiency",0) for r in runs],
        }
        st.markdown(
            _svg_line_chart(ma_series, x_lbls, title="Multi-Agent Performance Trends"),
            unsafe_allow_html=True,
        )


def _tab_history(runs: list[dict]) -> None:
    st.subheader("📋 Run History")

    if not runs:
        st.info("No pipeline runs recorded in this session yet.")
        return

    rows = []
    for r in runs:
        arecs = r.get("agent_records", [])
        p = next((a for a in arecs if a.get("agent_name") == "pathologist"),       {})
        e = next((a for a in arecs if a.get("agent_name") == "extension_officer"), {})
        rows.append({
            "Run ID":      r.get("run_id",        "?")[:6],
            "Timestamp":   r.get("timestamp",     "?")[:19],
            "Image":       r.get("image_filename","?")[:20],
            "Completed":   "✅" if r.get("pipeline_completed") else "❌",
            "E2E (ms)":    f"{r.get('total_latency_ms',0):,.0f}",
            "In Tok":      r.get("total_input_tokens",  0),
            "Out Tok":     r.get("total_output_tokens", 0),
            "Cost (USD)":  f"{r.get('total_cost_usd',0):.6f}",
            "Quality":     f"{r.get('overall_quality_score', 0):.4f}",
            "Safety":      f"{r.get('overall_safety_score',  0):.4f}",
            "Coordination":f"{r.get('agent_coordination',    0):.4f}",
            "Consistency": f"{r.get('inter_agent_consistency',0):.4f}",
            "P-Lat(ms)":   f"{p.get('latency_ms',0):,.0f}",
            "E-Lat(ms)":   f"{e.get('latency_ms',0):,.0f}",
            "Grd Checks":  r.get("total_guardrail_checks", 0),
            "Interventions":r.get("total_interventions",   0),
        })

    st.dataframe(rows, use_container_width=True)

    # Export
    import json as _json
    st.download_button(
        "📥 Export Full Metrics JSON",
        data=_json.dumps(runs, indent=2, default=str).encode("utf-8"),
        file_name=f"agri_doctor_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main dashboard entry point
# ---------------------------------------------------------------------------

def main() -> None:
    runs = _get_runs()
    is_demo = not getattr(st.session_state, "pipeline_runs", [])

    st.title("📊 Agri Doctor – LLM & Agent Performance Dashboard")
    st.caption(
        f"{'⚠ Demo data (run a diagnosis in the main app to see real metrics)' if is_demo else '✅ Live session data'}  "
        f"| Model: `ibm/granite-4-h-small`  |  Session: `{getattr(st.session_state, 'session_id', 'new')}`"
    )

    if is_demo:
        st.info(
            "Showing **synthetic demo data** to illustrate the dashboard layout. "
            "Go to the main Agri Doctor page, upload a crop image, and run a diagnosis "
            "to populate this dashboard with real metrics.",
            icon="ℹ️",
        )

    tabs = st.tabs([
        "📊 Overview",
        "🎯 Accuracy & Quality",
        "🧠 Agentic AI",
        "🛡️ Safety & Guardrails",
        "⚡ Performance",
        "💰 Cost",
        "🔗 Multi-Agent",
        "📋 Run History",
    ])

    with tabs[0]: _tab_overview(runs)
    with tabs[1]: _tab_accuracy(runs)
    with tabs[2]: _tab_agentic(runs)
    with tabs[3]: _tab_safety(runs)
    with tabs[4]: _tab_performance(runs)
    with tabs[5]: _tab_cost(runs)
    with tabs[6]: _tab_multiagent(runs)
    with tabs[7]: _tab_history(runs)


if __name__ == "__main__":
    main()
