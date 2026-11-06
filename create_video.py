"""
Agri Doctor — Animated Slideshow Video Generator
=================================================
Creates a narrated MP4 slideshow video using moviepy + PIL.

Features:
  • 10 content slides matching the PowerPoint presentation
  • Dark green theme (#0D1B0A background, #7FD47F accent)
  • Animated text: fade-in, slide-in from left
  • Slide transitions: cross-fade dissolve
  • Progress bar at bottom of each slide
  • Auto-generated narration text overlays
  • Final output: agri_doctor_slideshow.mp4

Usage:
    python create_video.py

Requirements (auto-installed by this script if missing):
    pip install moviepy pillow numpy
"""

import sys
import subprocess

# ── Auto-install dependencies ────────────────────────────────────────────────
def _ensure(pkg, import_name=None):
    import importlib
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Installing {pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("moviepy")
_ensure("Pillow", "PIL")
_ensure("numpy")

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# moviepy >=2.x API
try:
    from moviepy import VideoClip, ImageClip, concatenate_videoclips, CompositeVideoClip
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import VideoClip, ImageClip, concatenate_videoclips, CompositeVideoClip
    MOVIEPY_V2 = False

# ── Constants ────────────────────────────────────────────────────────────────
W, H         = 1280, 720
FPS          = 24
SLIDE_DUR    = 7.0      # seconds per slide
TRANS_DUR    = 0.8      # cross-fade duration
OUTPUT_FILE  = "agri_doctor_slideshow.mp4"

BG_COLOR     = (13, 27, 10)        # dark green background
ACCENT_COLOR = (127, 212, 127)     # green accent
WHITE        = (255, 255, 255)
LIGHT_GRAY   = (200, 200, 200)
DARK_GRAY    = (80, 80, 80)
GOLD         = (255, 215, 0)

# ── Font loader ──────────────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try to load a system font; fall back to PIL default."""
    candidates = []
    if bold:
        candidates = [
            "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Text wrapping helper ──────────────────────────────────────────────────────
def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


# ── Draw a single slide frame ─────────────────────────────────────────────────
def make_slide_frame(
    title: str,
    body: str,
    slide_num: int,
    total_slides: int,
    t: float,              # current time within slide (0 → SLIDE_DUR)
    badge: str = "",
) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # ── Background subtle grid ────────────────────────────────────────────
    for x in range(0, W, 80):
        draw.line([(x, 0), (x, H)], fill=(20, 40, 20), width=1)
    for y in range(0, H, 80):
        draw.line([(0, y), (W, y)], fill=(20, 40, 20), width=1)

    # ── Progress bar ──────────────────────────────────────────────────────
    bar_w = int(W * (slide_num / total_slides))
    draw.rectangle([(0, H - 6), (W, H)], fill=(30, 60, 30))
    draw.rectangle([(0, H - 6), (bar_w, H)], fill=ACCENT_COLOR)

    # ── Slide number ──────────────────────────────────────────────────────
    num_font = _font(16)
    draw.text((W - 80, H - 30), f"{slide_num}/{total_slides}", font=num_font, fill=DARK_GRAY)

    # ── Top accent line ───────────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, 4)], fill=ACCENT_COLOR)

    # ── Animate: fade in based on t ──────────────────────────────────────
    alpha = min(1.0, t / 0.6)   # fade in over 0.6 s

    # ── Title ─────────────────────────────────────────────────────────────
    title_font = _font(42, bold=True)
    title_x = int(60 + (1 - alpha) * (-200))  # slide in from left
    title_alpha_color = tuple(int(c * alpha) for c in ACCENT_COLOR)
    draw.text((title_x, 30), title, font=title_font, fill=title_alpha_color)
    draw.line([(60, 90), (W - 60, 90)], fill=(50, 100, 50), width=2)

    # ── Body text ─────────────────────────────────────────────────────────
    body_font = _font(20)
    body_x = int(60 + (1 - alpha) * (-100))
    body_alpha_color = tuple(int(c * alpha) for c in LIGHT_GRAY)

    lines = []
    for line in body.split("\n"):
        lines.append(line)

    y_pos = 105
    line_h = 26
    max_lines = (H - 150) // line_h
    for i, line in enumerate(lines[:max_lines]):
        if not line:
            y_pos += line_h // 2
            continue
        # Highlight lines that start with special markers
        color = body_alpha_color
        if line.startswith("✅") or line.startswith("PASS"):
            color = tuple(int(c * alpha) for c in (100, 220, 100))
        elif line.startswith("⚠") or line.startswith("WARN"):
            color = tuple(int(c * alpha) for c in GOLD)
        elif line.startswith("•") or line.startswith("-") or line.startswith("→"):
            color = tuple(int(c * alpha) for c in (180, 220, 180))
        elif line.isupper() and len(line) > 4:
            color = tuple(int(c * alpha) for c in ACCENT_COLOR)

        # Truncate long lines
        max_chars = 95
        display_line = line if len(line) <= max_chars else line[:max_chars - 1] + "…"
        draw.text((body_x, y_pos), display_line, font=body_font, fill=color)
        y_pos += line_h

    # ── Badge ─────────────────────────────────────────────────────────────
    if badge:
        badge_font = _font(16)
        badge_color = tuple(int(c * alpha) for c in DARK_GRAY)
        draw.text((60, H - 40), badge, font=badge_font, fill=badge_color)

    return np.array(img)


# ── Slide definitions ─────────────────────────────────────────────────────────
SLIDES = [
    {
        "title": "🌿 Agri Doctor",
        "body": (
            "AI-Powered Crop Disease Diagnosis & Advisory\n"
            "\n"
            "Powered by IBM Granite 4 H Small via IBM watsonx AI\n"
            "\n"
            "Two-agent Streamlit application:\n"
            "• Agent 1 — Pathologist (image analysis + diagnosis)\n"
            "• Agent 2 — Extension Officer (farmer advisory)\n"
            "\n"
            "Protected by 9 categories of AI safety guardrails\n"
            "\n"
            "IBM SkillsBuild 2026  •  Gartner Program  •  Capstone Project"
        ),
        "badge": "IBM SkillsBuild 2026 | Gartner Program | Capstone Project",
    },
    {
        "title": "Project Overview",
        "body": (
            "WHAT IS AGRI DOCTOR?\n"
            "\n"
            "A guardrail-protected two-agent AI system that:\n"
            "• Identifies crop diseases from uploaded photos\n"
            "• Provides instant, practical treatment advisory\n"
            "• Runs entirely in a browser — no app install needed\n"
            "• Uses IBM Granite 4 H Small for AI inference\n"
            "\n"
            "THE PROBLEM BEING SOLVED\n"
            "\n"
            "→ 20–40% of crop yields lost globally to disease each year\n"
            "→ Smallholder farmers lack access to plant pathologists\n"
            "→ Diagnosis requires lab visits that take days/weeks\n"
            "\n"
            "THE SOLUTION\n"
            "\n"
            "→ Upload a leaf photo  →  Get AI diagnosis in seconds\n"
            "→ Download combined diagnosis + advisory report"
        ),
        "badge": "",
    },
    {
        "title": "Two-Agent Architecture",
        "body": (
            "AGENT 1 — PATHOLOGIST AGENT\n"
            "\n"
            "• Receives uploaded crop image\n"
            "• Extracts visual features with PIL (greenness, yellowing, necrosis)\n"
            "• Sends colour analysis text to IBM Granite 4 H Small\n"
            "• Returns: Disease name, Symptoms, Confidence, Alternatives\n"
            "• Max tokens: 900  |  Temperature: 0.3\n"
            "\n"
            "→ MULTI-AGENT HANDOFF VALIDATION →\n"
            "\n"
            "AGENT 2 — EXTENSION OFFICER AGENT\n"
            "\n"
            "• Receives pathologist diagnosis as input\n"
            "• Translates to plain farmer-friendly language\n"
            "• Returns: Immediate Actions, Treatment Options, Prevention\n"
            "• Includes Safety Warnings + When to Seek Expert Help\n"
            "• Max tokens: 1100  |  Temperature: 0.3"
        ),
        "badge": "Both agents run on IBM Granite 4 H Small via IBM watsonx AI REST API",
    },
    {
        "title": "9 Categories of Guardrails",
        "body": (
            "1. INPUT      API key validation, image format/size, prompt injection, PII, jailbreak\n"
            "2. OUTPUT     Toxicity filter, hallucination flags, PII redaction, disclaimer check\n"
            "3. TOOL EXEC  Endpoint allow-list, model allow-list, payload schema validation\n"
            "4. DATA       PII masking in outputs, role-based access (farmer/agronomist/admin)\n"
            "5. SECURITY   Credential exposure check, HTML sanitisation, jailbreak detection\n"
            "6. OPS        Per-agent token budgets, session rate limit (15/session), retry+backoff\n"
            "7. HITL       High-risk advisory flagging (crop destruction, human safety hazards)\n"
            "8. MULTI-AGENT Handoff validation, cross-agent quality check, loop guard (depth≤5)\n"
            "9. ETHICS     Bias detection, explainability notes, full audit trail"
        ),
        "badge": "60+ individual guardrail checks protect every touchpoint in the pipeline",
    },
    {
        "title": "Image Analysis Pipeline (PIL)",
        "body": (
            "INPUT: Farmer uploads JPG / PNG / WEBP / BMP / TIFF crop leaf image\n"
            "\n"
            "STEP 1 — PIL Feature Extraction (no raw bytes sent to API)\n"
            "• Image dimensions\n"
            "• Overall brightness (0–255): bright / moderate / dark classification\n"
            "• RGB channel means ± standard deviation\n"
            "• Centre region RGB sampling\n"
            "\n"
            "STEP 2 — Derived Visual Indicators\n"
            "• Greenness index (%)  — chlorophyll / healthy tissue proxy\n"
            "• Yellowing index (%)  — nutrient deficiency / viral infection indicator\n"
            "• Necrosis index (%)   — fungal / bacterial lesion indicator\n"
            "• Dark contrast value  — spot / lesion variation detector\n"
            "\n"
            "STEP 3 — Text description sent to IBM Granite 4 H Small\n"
            "STEP 4 — Structured diagnosis returned with 6 mandatory sections"
        ),
        "badge": "Privacy-preserving: only derived text metrics sent, never raw image bytes",
    },
    {
        "title": "LLM Test Results — 97.5% Pass Rate",
        "body": (
            "79 tests across 11 evaluation dimensions\n"
            "Passed: 77  |  Failed: 2  |  Average Score: 0.909/1.000\n"
            "\n"
            "✅ PASS   1. Correctness & Factual Accuracy       6/6   score 1.000\n"
            "✅ PASS   2. Format / Prompt Adherence             6/6   score 0.884\n"
            "⚠ WARN   3. NLP Metrics (ROUGE/BLEU/Semantic)    6/8   score 0.339\n"
            "✅ PASS   4. Input Guardrails                    14/14  score 1.000\n"
            "✅ PASS   5. Output Guardrails                   11/11  score 0.924\n"
            "✅ PASS   6. Tool Execution Guardrails             6/6   score 1.000\n"
            "✅ PASS   7. Operational Guardrails                4/4   score 1.000\n"
            "✅ PASS   8. Multi-Agent Guardrails                7/7   score 1.000\n"
            "✅ PASS   9. Ethics Guardrails                     6/6   score 1.000\n"
            "✅ PASS  10. Performance & Token Efficiency        5/5   score 0.927\n"
            "✅ PASS  11. Robustness & Edge Cases               6/6   score 1.000"
        ),
        "badge": "NLP metric warnings are expected: lightweight TF-IDF vs full sentence-transformers",
    },
    {
        "title": "Project File Structure",
        "body": (
            "agri-doctor/\n"
            "├── app.py               Main Streamlit app (two agents + UI + metrics)\n"
            "├── guardrails.py        9-category guardrail engine (60+ checks, audit trail)\n"
            "├── metrics.py           LLM evaluation engine (ROUGE, BLEU, RAG, agent metrics)\n"
            "├── test_agent.py        79-test LLM test suite — 97.5% pass rate\n"
            "├── pages/\n"
            "│   └── dashboard.py     Performance analytics (7 tabs, SVG charts)\n"
            "├── requirements.txt     Python dependencies\n"
            "├── README.md            Full documentation with badges\n"
            "├── .gitignore           GitHub gitignore\n"
            "├── Agri_Doctor_Presentation.pptx\n"
            "└── create_video.py      This video generator\n"
            "\n"
            "TECH STACK\n"
            "Python 3.11+  •  Streamlit 1.35+  •  IBM Granite 4 H Small\n"
            "IBM watsonx AI  •  Pillow (PIL)  •  ibm-cloud-sdk-core  •  requests"
        ),
        "badge": "",
    },
    {
        "title": "Performance Dashboard — 7 Analytics Tabs",
        "body": (
            "📊  OVERVIEW        Pipeline run history, quality scores, safety scores, KPIs\n"
            "\n"
            "🎯  ACCURACY        ROUGE-1/2/L, BLEU, semantic similarity, faithfulness\n"
            "                   groundedness, answer relevancy, exact match\n"
            "\n"
            "🤖  AGENTIC         Task completion, goal achievement, planning quality\n"
            "                   reasoning quality, tool call accuracy\n"
            "\n"
            "🛡️   SAFETY          Hallucination risk, toxicity score, bias score\n"
            "                   PII leakage risk, guardrail intervention rate\n"
            "\n"
            "⚡  PERFORMANCE     Latency distribution, tokens/sec, error rate, throughput\n"
            "\n"
            "💰  COST            Cost per request, total cost, token breakdown\n"
            "\n"
            "🔗  MULTI-AGENT    Agent coordination, handoff accuracy, inter-agent consistency"
        ),
        "badge": "All metrics computed in real-time from pipeline execution data",
    },
    {
        "title": "How to Run the Application",
        "body": (
            "QUICK START — 3 STEPS\n"
            "\n"
            "Step 1 — Install dependencies\n"
            "  pip install -r requirements.txt\n"
            "\n"
            "Step 2 — Run the Streamlit app\n"
            "  streamlit run app.py\n"
            "\n"
            "Step 3 — Use the app\n"
            "  → Open http://localhost:8501 in your browser\n"
            "  → Paste your IBM Cloud API key in the sidebar\n"
            "  → Upload a crop leaf photo (JPG / PNG / WEBP)\n"
            "  → Click 'Diagnose Crop'\n"
            "  → View Pathologist report + Extension Officer advisory\n"
            "  → Download the combined .txt report\n"
            "\n"
            "OPTIONAL — Run the 79-test suite\n"
            "  python test_agent.py"
        ),
        "badge": "GitHub: https://github.com/vignesh2396/Agri_Doctor_IBMBob",
    },
    {
        "title": "🌿 Thank You",
        "body": (
            "Agri Doctor — AI-Powered Crop Disease Diagnosis & Advisory\n"
            "\n"
            "IBM Granite 4 H Small  •  IBM watsonx AI  •  Streamlit\n"
            "\n"
            "• 2 AI agents working in a guardrail-protected pipeline\n"
            "• 9 categories of safety guardrails (60+ individual checks)\n"
            "• 79-test LLM evaluation suite with 97.5% pass rate\n"
            "• 7-tab performance analytics dashboard\n"
            "• Privacy-preserving PIL image feature extraction\n"
            "• Download-ready combined diagnosis + advisory report\n"
            "\n"
            "GitHub Repository:\n"
            "https://github.com/vignesh2396/Agri_Doctor_IBMBob\n"
            "\n"
            "IBM SkillsBuild 2026  •  Gartner Program  •  Capstone Project"
        ),
        "badge": "",
    },
]


# ── Build one slide clip ──────────────────────────────────────────────────────
def build_slide_clip(slide_data: dict, slide_idx: int, total: int) -> VideoClip:
    title       = slide_data["title"]
    body        = slide_data["body"]
    badge       = slide_data.get("badge", "")
    slide_num   = slide_idx + 1

    def make_frame(t):
        return make_slide_frame(title, body, slide_num, total, t, badge)

    clip = VideoClip(make_frame, duration=SLIDE_DUR)
    clip = clip.set_fps(FPS)
    return clip


# ── Cross-fade transition ────────────────────────────────────────────────────
def crossfade(clip_a, clip_b, duration=TRANS_DUR):
    """
    Simple cross-fade: clip_a fades out, clip_b fades in, overlapping for `duration`.
    Uses alpha blending on numpy frames.
    """
    STEPS = int(FPS * duration)
    transition_frames = []
    for i in range(STEPS):
        alpha = i / STEPS
        t_a = clip_a.duration - duration + i / FPS
        t_b = i / FPS
        frame_a = clip_a.get_frame(min(t_a, clip_a.duration - 0.01))
        frame_b = clip_b.get_frame(t_b)
        blended = ((1 - alpha) * frame_a + alpha * frame_b).astype(np.uint8)
        transition_frames.append(blended)

    def trans_make_frame(t):
        idx = min(int(t * FPS), len(transition_frames) - 1)
        return transition_frames[idx]

    trans_clip = VideoClip(trans_make_frame, duration=duration)
    trans_clip = trans_clip.set_fps(FPS)
    return trans_clip


# ── Main render ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AGRI DOCTOR — SLIDESHOW VIDEO GENERATOR")
    print("=" * 60)
    print(f"  Resolution : {W}×{H}")
    print(f"  FPS        : {FPS}")
    print(f"  Slides     : {len(SLIDES)}")
    print(f"  Duration/slide: {SLIDE_DUR}s")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 60)

    total = len(SLIDES)
    slide_clips = []

    for i, slide_data in enumerate(SLIDES):
        print(f"  Rendering slide {i+1}/{total}: {slide_data['title'][:40]}…")
        clip = build_slide_clip(slide_data, i, total)
        slide_clips.append(clip)

    # Build final video with cross-fades
    print("\n  Compositing clips with cross-fade transitions…")
    segments = []
    for i, clip in enumerate(slide_clips):
        # Keep clip minus transition overlap at the end (except last)
        if i < len(slide_clips) - 1:
            trimmed = clip.subclip(0, SLIDE_DUR - TRANS_DUR / 2)
        else:
            trimmed = clip
        segments.append(trimmed)

        if i < len(slide_clips) - 1:
            next_clip = slide_clips[i + 1]
            trans = crossfade(clip, next_clip, TRANS_DUR)
            segments.append(trans)

    final = concatenate_videoclips(segments, method="compose")

    total_dur = final.duration
    print(f"  Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
    print(f"\n  Writing {OUTPUT_FILE}…")

    final.write_videofile(
        OUTPUT_FILE,
        fps=FPS,
        codec="libx264",
        audio=False,
        logger=None,
    )

    size = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    print(f"\n  ✅ Done! File: {OUTPUT_FILE} ({size:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
