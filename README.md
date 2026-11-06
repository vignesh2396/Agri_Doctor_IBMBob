# 🌿 Agri Doctor — AI-Powered Crop Disease Diagnosis & Advisory

[![IBM watsonx AI](https://img.shields.io/badge/IBM%20watsonx-AI-052FAD?logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![IBM Granite](https://img.shields.io/badge/Model-IBM%20Granite%204%20H%20Small-7FD47F)](https://www.ibm.com/granite)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-77%2F79%20Passing%20%2897.5%25%29-brightgreen)](test_agent.py)
[![Guardrails](https://img.shields.io/badge/Guardrails-9%20Categories-blue)](guardrails.py)

> **IBM SkillsBuild 2026 • Gartner Program • Capstone Project**

A **guardrail-protected, two-agent AI application** that helps farmers identify crop diseases from photos and receive practical, safe treatment advice — powered by **IBM Granite 4 H Small** via the **IBM watsonx AI** API.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Two-Agent Architecture](#two-agent-architecture)
- [9 Categories of Guardrails](#9-categories-of-guardrails)
- [Image Analysis Pipeline](#image-analysis-pipeline)
- [LLM Test Results](#llm-test-results)
- [Performance Dashboard](#performance-dashboard)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Test Suite](#running-the-test-suite)
- [Generating the Video](#generating-the-video)
- [Safety & Disclaimer](#safety--disclaimer)
- [License](#license)

---

## Overview

Agri Doctor addresses the global challenge of crop disease diagnosis:

- **20–40% of crop yields** are lost globally to disease each year
- **Smallholder farmers** lack access to plant pathologists
- Lab diagnosis takes **days or weeks**; farmers need answers today

Agri Doctor provides **instant AI-powered preliminary assessment** 24/7 — directly in a browser, with no app install required.

---

## Two-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FARMER uploads crop leaf image (JPG / PNG / WEBP / BMP / TIFF)         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                    Input Guardrails
                    (API key, image, PII, injection, jailbreak)
                           │
           ┌───────────────▼──────────────────┐
           │   AGENT 1 — Pathologist Agent     │
           │                                   │
           │  • PIL extracts image features    │
           │  • Greenness / yellowing / necrosis│
           │  • Text description → IBM Granite  │
           │  • Returns structured diagnosis    │
           │    (6 mandatory sections)          │
           └───────────────┬──────────────────┘
                           │
                 Multi-Agent Handoff Guardrails
                 (quality check, loop guard, pipeline depth)
                           │
           ┌───────────────▼──────────────────┐
           │  AGENT 2 — Extension Officer       │
           │                                   │
           │  • Receives diagnosis report       │
           │  • Translates to farmer language   │
           │  • Returns 7-section advisory      │
           │    (treatment, prevention, safety) │
           └───────────────┬──────────────────┘
                           │
               Output Guardrails + Ethics + HITL
                           │
           ┌───────────────▼──────────────────┐
           │   Combined Report + Download      │
           └──────────────────────────────────┘
```

| Agent | Role | LLM Task | Max Tokens |
|---|---|---|---|
| **Pathologist Agent** | Examines the crop image | Image feature analysis → structured diagnostic report | 900 |
| **Extension Officer Agent** | Reads the diagnosis | Text generation → farmer-friendly advisory | 1100 |

**Model**: `ibm/granite-4-h-small` | **Temperature**: `0.3` | **Decoding**: greedy

---

## 9 Categories of Guardrails

Agri Doctor enforces **60+ individual guardrail checks** at every touchpoint:

| # | Category | Checks |
|---|---|---|
| **1** | **Input Validation** | API key format, image file type/size (≤20 MB), prompt injection detection, PII detection, jailbreak blocking |
| **2** | **Output Quality** | Toxicity filtering, hallucination language flags, PII redaction, disclaimer enforcement, response length |
| **3** | **Tool Execution** | Endpoint allow-list, model ID allow-list, payload schema validation |
| **4** | **Data Privacy** | PII masking in outputs, role-based access control (farmer / agronomist / admin) |
| **5** | **Security** | Credential exposure check, HTML sanitisation, jailbreak detection |
| **6** | **Operational** | Per-agent token budgets (900 / 1100), session rate limit (15/session), retry with exponential back-off |
| **7** | **Human-in-the-Loop** | High-risk advisory flagging: crop destruction, quarantine, toxic to humans/livestock |
| **8** | **Multi-Agent** | Handoff validation (source/target registry), cross-agent output quality, pipeline loop guard (depth ≤ 5) |
| **9** | **Ethics** | Bias detection, explainability notes appended to output, full audit trail (structured JSON) |

---

## Image Analysis Pipeline

The Pathologist Agent uses **PIL** to extract quantitative visual features — **no raw image bytes are sent to the API** (privacy-preserving):

```
Upload Image
     │
     ▼
PIL Feature Extraction
  • Dimensions (W × H px)
  • Overall brightness (0–255)
  • RGB channel means ± std deviation
  • Centre region RGB
  • Greenness index (%)   — chlorophyll proxy
  • Yellowing index (%)   — nutrient deficiency / viral indicator
  • Necrosis index (%)    — fungal / bacterial lesion indicator
  • Dark contrast value   — spot / lesion variation detector
     │
     ▼
Natural-language description built from metrics
     │
     ▼
Sent as text to IBM Granite 4 H Small via watsonx REST API
     │
     ▼
Structured diagnosis (6 mandatory sections):
  ### Suspected Disease
  ### Observed Symptoms
  ### Possible Causes
  ### Confidence Level
  ### Alternative Diagnoses
  ### Recommendation
```

---

## LLM Test Results

**79 tests across 11 evaluation dimensions — 97.5% pass rate**

Run: `python test_agent.py`

| Section | Pass | Total | Score | Status |
|---|---|---|---|---|
| 1. Correctness & Factual Accuracy | 6 | 6 | 1.000 | ✅ PASS |
| 2. Format / Prompt Adherence | 6 | 6 | 0.884 | ✅ PASS |
| 3. NLP Metrics (ROUGE/BLEU/Semantic) | 6 | 8 | 0.339 | ⚠ WARN |
| 4. Input Guardrails | 14 | 14 | 1.000 | ✅ PASS |
| 5. Output Guardrails | 11 | 11 | 0.924 | ✅ PASS |
| 6. Tool Execution Guardrails | 6 | 6 | 1.000 | ✅ PASS |
| 7. Operational Guardrails | 4 | 4 | 1.000 | ✅ PASS |
| 8. Multi-Agent Guardrails | 7 | 7 | 1.000 | ✅ PASS |
| 9. Ethics Guardrails | 6 | 6 | 1.000 | ✅ PASS |
| 10. Performance & Token Efficiency | 5 | 5 | 0.927 | ✅ PASS |
| 11. Robustness & Edge Cases | 6 | 6 | 1.000 | ✅ PASS |
| **TOTAL** | **77** | **79** | **0.909** | **⚠ MOSTLY PASS** |

> The 2 ⚠ WARN results in Section 3 are expected: the built-in TF-IDF-based similarity metrics produce lower cosine scores compared to full neural sentence-transformer models. All safety-critical guardrail tests pass.

---

## Performance Dashboard

The app includes a **7-tab performance analytics dashboard** at `pages/dashboard.py`:

| Tab | Metrics |
|---|---|
| 📊 Overview | Pipeline run history, quality/safety scores, KPIs |
| 🎯 Accuracy | ROUGE-1/2/L, BLEU, semantic similarity, faithfulness, groundedness |
| 🤖 Agentic | Task completion, goal achievement, planning/reasoning quality |
| 🛡️ Safety | Hallucination risk, toxicity, bias, PII leakage, guardrail rate |
| ⚡ Performance | Latency distribution, tokens/sec, error rate, throughput |
| 💰 Cost | Cost per request, total cost, input/output token breakdown |
| 🔗 Multi-Agent | Agent coordination, handoff accuracy, inter-agent consistency |

---

## Project Structure

```
agri-doctor/
├── app.py                      Main Streamlit application (all agents + UI)
├── guardrails.py               9-category guardrail engine (60+ checks)
├── metrics.py                  LLM evaluation engine (ROUGE, BLEU, RAG metrics)
├── test_agent.py               79-test comprehensive LLM test suite
├── create_video.py             Animated MP4 slideshow video generator
├── pages/
│   └── dashboard.py            Performance analytics dashboard (7 tabs)
├── Agri_Doctor_Presentation.pptx  10-slide project presentation
├── requirements.txt            Python dependencies
├── README.md                   This file
├── .gitignore                  Git ignore rules
└── test_results.json           Latest test suite output (auto-generated)
```

---

## Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/vignesh2396/Agri_Doctor_IBMBob.git
cd Agri_Doctor_IBMBob
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the app

```bash
streamlit run app.py
```

### 4 — Use the app

1. Open `http://localhost:8501` in your browser
2. Enter your **IBM Cloud API key** in the sidebar *(never stored, used only for IAM token)*
3. Upload a **JPG / PNG / WEBP / BMP / TIFF** photo of an affected crop leaf
4. Click **🔬 Diagnose Crop**
5. Review the **Pathologist report** and **Extension Officer advisory**
6. Download the combined `.txt` report
7. Click **📊 Open Performance Dashboard** to view analytics

---

## Configuration

| Variable | Location | Default |
|---|---|---|
| `IBM_PUBLIC_ENDPOINT` | `app.py` line 72 | `https://eu-de.ml.cloud.ibm.com` |
| `IBM_PROJECT_ID` | `app.py` line 73 | Project ID (update to your own) |
| `MODEL_ID` | `app.py` line 75 | `ibm/granite-4-h-small` |
| `MAX_TOKENS_PATHOLOGIST` | `app.py` line 79 | `900` |
| `MAX_TOKENS_EXTENSION` | `app.py` line 80 | `1100` |
| `TEMPERATURE` | `app.py` line 81 | `0.3` |
| IBM Cloud API Key | Sidebar (runtime) | Never hardcoded |

> **To use your own IBM Cloud project**: update `IBM_PROJECT_ID` in `app.py` and ensure your API key has access to that project.

---

## Running the Test Suite

```bash
python test_agent.py
```

The test suite requires no IBM Cloud API key — all tests run offline against synthetic responses and the local guardrail/metrics engines.

Results are saved to `test_results.json`.

---

## Generating the Video

```bash
python create_video.py
```

This auto-installs `moviepy`, `Pillow`, and `numpy` if not already installed, then renders `agri_doctor_slideshow.mp4` (~70 seconds, 1280×720, H.264).

---

## Safety & Disclaimer

> ⚠ **This application provides AI-generated preliminary assessments only.**
>
> Diagnoses are **not** confirmed laboratory results.
>
> Always consult a qualified agricultural extension officer, certified agronomist, or plant pathology laboratory before major treatment decisions.
>
> Follow all pesticide label instructions and local agricultural regulations.
>
> Keep all chemicals away from children, animals, and water sources.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using IBM Granite 4 H Small via IBM watsonx AI**

*IBM SkillsBuild 2026 • Gartner Program • Capstone Project*

</div>
