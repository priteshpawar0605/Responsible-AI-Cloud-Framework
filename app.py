"""
app.py -- Streamlit dashboard for the Responsible AI Framework.

Start with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Responsible AI Platform",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG        = "#F5F4EF"
SURFACE   = "#FFFFFF"
TEXT_MAIN = "#252923"
TEXT_SEC  = "#77796F"
ACCENT    = "#536B4F"
WARNING   = "#A66B32"
BORDER    = "#E0DFD7"
DANGER    = "#B04020"

CUSTOM_CSS = f"""
<style>
/* ---- Google Fonts import ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Reset & base ---- */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT_MAIN};
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}}

/* ---- Hide Default Streamlit Sidebar & Chrome ---- */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu,
footer,
header {{
    display: none !important;
    visibility: hidden !important;
}}

/* ---- Centered layout container ---- */
.main .block-container {{
    max-width: 820px !important;
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}}

/* ---- Typography ---- */
h1 {{ font-size: 24px; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 4px; letter-spacing: -0.4px; }}
h2 {{ font-size: 18px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 0; margin-bottom: 4px; }}
p {{ line-height: 1.6; }}

/* ---- Surface Cards / Panels ---- */
.panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.panel-title {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 6px;
}}

/* ---- Text Area ---- */
.stTextArea textarea {{
    background-color: {SURFACE} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    color: {TEXT_MAIN} !important;
    padding: 14px 16px !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}}
.stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(83, 107, 79, 0.15) !important;
}}

/* ---- Action Buttons ---- */
.stButton button[kind="primary"] {{
    background-color: {ACCENT} !important;
    color: {SURFACE} !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 4px rgba(83,107,79,0.2) !important;
    transition: all 0.15s ease !important;
}}
.stButton button[kind="primary"]:hover {{
    background-color: #42553F !important;
    box-shadow: 0 3px 8px rgba(83,107,79,0.3) !important;
    transform: translateY(-1px) !important;
}}

/* ---- Badges & Banners ---- */
.warn-banner {{
    background: #FDF4E7;
    border: 1px solid {WARNING};
    border-radius: 8px;
    padding: 12px 18px;
    color: {WARNING};
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 16px;
}}
.info-banner {{
    background: #EFF4EE;
    border: 1px solid {ACCENT};
    border-radius: 8px;
    padding: 12px 18px;
    color: {ACCENT};
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 16px;
}}

/* ---- Separator ---- */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 24px 0 !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "dev-insecure-key-change-me")
HEADERS  = {
    "X-API-Key":     API_KEY,
    "Content-Type":  "application/json",
}

# ---------------------------------------------------------------------------
# Backend status check
# ---------------------------------------------------------------------------
def backend_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def show_backend_status():
    if not backend_alive():
        st.markdown(
            '<div class="warn-banner">Backend API is not reachable at '
            f'{API_BASE}. Run: <code>uvicorn api:app --reload --port 8000</code>'
            "</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Main Application: AI Text Decision & Fairness Evaluator
# ===========================================================================

def main():
    show_backend_status()

    # Clean Header
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 28px;">
            <div style="font-size: 24px; font-weight: 700; color: {TEXT_MAIN}; letter-spacing: -0.4px;">
                Responsible AI Platform
            </div>
            <div style="font-size: 13px; color: {TEXT_SEC}; margin-top: 4px;">
                Trustworthy Decision-Making & AI Decision Auditor
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Box Section
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <div style="font-size: 15px; font-weight: 600; color: {TEXT_MAIN}; margin-bottom: 4px;">
                AI-Generated Decision Text
            </div>
            <div style="font-size: 13px; color: {TEXT_SEC};">
                Paste any AI decision, explanation, or automated reasoning output to audit its outcome, fairness, and review routing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Input Text",
        height=130,
        placeholder="Paste AI-generated decision text here...",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1.5, 4])
    with col_btn:
        analyze_clicked = st.button("Analyze Decision", type="primary", use_container_width=True)

    # Processing & Analysis
    if analyze_clicked or (user_text and user_text.strip()):
        if not user_text.strip():
            st.warning("Please paste or type AI text to evaluate.")
            return

        from text_analyzer import analyze_ai_text
        result = analyze_ai_text(user_text)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 3 Result Cards
        c1, c2, c3 = st.columns(3)

        with c1:
            dec_color = ACCENT if "APPROV" in result.detected_decision else (DANGER if "REJECT" in result.detected_decision else WARNING)
            dec_icon = "&#10003;" if "APPROV" in result.detected_decision else ("&#10007;" if "REJECT" in result.detected_decision else "&#8212;")
            st.markdown(
                f"""
                <div class="panel" style="min-height: 125px; border-top: 3px solid {dec_color};">
                    <div class="panel-title">1. AI Decision</div>
                    <div style="font-size: 22px; font-weight: 700; color: {dec_color}; margin: 8px 0 4px 0;">
                        {dec_icon} {result.detected_decision}
                    </div>
                    <div style="font-size: 12px; color: {TEXT_SEC};">
                        Confidence: <b>{result.decision_confidence}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            is_fair = not ("WARNING" in result.fairness_status or "BIASED" in result.fairness_status)
            fair_color = ACCENT if is_fair else WARNING
            fair_icon = "&#9679;" if is_fair else "&#9888;"
            fair_title = "Fair / Unbiased" if is_fair else "Bias Alert"
            fair_desc = "No sensitive attributes cited" if is_fair else "Contains demographic references"
            st.markdown(
                f"""
                <div class="panel" style="min-height: 125px; border-top: 3px solid {fair_color};">
                    <div class="panel-title">2. Fairness Status</div>
                    <div style="font-size: 22px; font-weight: 700; color: {fair_color}; margin: 8px 0 4px 0;">
                        {fair_icon} {fair_title}
                    </div>
                    <div style="font-size: 12px; color: {TEXT_SEC};">
                        {fair_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            rev_color = WARNING if result.requires_human_review else ACCENT
            rev_icon = "&#9998;" if result.requires_human_review else "&#10003;"
            rev_title = "Required" if result.requires_human_review else "Not Required"
            rev_desc = f"Priority: {result.review_urgency}" if result.requires_human_review else "Safe to proceed automatically"
            st.markdown(
                f"""
                <div class="panel" style="min-height: 125px; border-top: 3px solid {rev_color};">
                    <div class="panel-title">3. Human Review</div>
                    <div style="font-size: 22px; font-weight: 700; color: {rev_color}; margin: 8px 0 4px 0;">
                        {rev_icon} {rev_title}
                    </div>
                    <div style="font-size: 12px; color: {TEXT_SEC};">
                        {rev_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Rationale Panel
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">Evaluation Summary & Rationale</div>
                <div style="font-size: 13px; line-height: 1.7; color: {TEXT_MAIN};">
                    {result.summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sensitive flags alert
        if result.fairness_flags:
            st.markdown(
                f"""
                <div class="warn-banner">
                    <b>Sensitive Demographic References Detected:</b> {', '.join(result.fairness_flags)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Future Scope Card
        st.markdown(
            f"""
            <div class="panel" style="border-left: 3px solid {ACCENT};">
                <div class="panel-title">Future Scope & Cloud Pipeline</div>
                <div style="font-size: 12px; line-height: 1.7; color: {TEXT_SEC};">
                    In the next phase, this evaluator will be deployed as an automated <b>LLM-as-a-Judge</b> proxy service in the cloud.
                    All customer-facing explanations and automated AI recommendations will be intercepted in real-time to prevent bias and ensure regulatory compliance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
