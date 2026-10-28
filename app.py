"""
app.py -- Streamlit dashboard for the Responsible AI Framework.

Start with:
    streamlit run app.py

The dashboard calls the FastAPI backend via HTTP for all data operations.
If the backend is not running, a warning is shown and some pages degrade
gracefully using locally-loaded data.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Responsible AI Platform",
    layout="wide",
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

/* ---- Hide Default Streamlit Sidebar & Collapse Control ---- */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] {{
    display: none !important;
}}

/* ---- Center and constrain max-width ---- */
.main .block-container {{
    max-width: 960px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}}

/* ---- Tab Navigation ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 28px;
}}
.stTabs [data-baseweb="tab"] {{
    height: 38px;
    padding: 0 28px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: {TEXT_SEC};
    background-color: transparent;
    border: none;
    transition: all 0.2s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_MAIN};
    background-color: rgba(0,0,0,0.025);
}}
.stTabs [aria-selected="true"] {{
    color: {SURFACE} !important;
    font-weight: 600 !important;
    background-color: {ACCENT} !important;
    border-bottom: none !important;
    box-shadow: 0 1px 3px rgba(83,107,79,0.25);
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ---- Main headers ---- */
h1 {{ font-size: 22px; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 2px; letter-spacing: -0.3px; }}
h2 {{ font-size: 18px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 12px; margin-bottom: 4px; }}
h3 {{ font-size: 15px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 8px; margin-bottom: 4px; }}

/* ---- Surface Cards ---- */
.panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    transition: box-shadow 0.15s ease;
}}
.panel:hover {{
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}
.panel-title {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 8px;
}}

/* ---- Text Area ---- */
.stTextArea textarea {{
    background-color: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    color: {TEXT_MAIN} !important;
    padding: 14px !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.03) !important;
}}
.stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(83, 107, 79, 0.12) !important;
}}
.stTextArea label {{
    font-size: 12px !important;
    font-weight: 600 !important;
    color: {TEXT_SEC} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* ---- Primary Button ---- */
.stButton button[kind="primary"] {{
    background-color: {ACCENT} !important;
    color: {SURFACE} !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 1px 3px rgba(83,107,79,0.2) !important;
    transition: all 0.15s ease !important;
}}
.stButton button[kind="primary"]:hover {{
    background-color: #42553F !important;
    box-shadow: 0 2px 6px rgba(83,107,79,0.3) !important;
    transform: translateY(-1px) !important;
}}

/* ---- Secondary / Download Button ---- */
.stButton button, .stDownloadButton button {{
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border: 1px solid {BORDER} !important;
    background-color: {SURFACE} !important;
    color: {TEXT_MAIN} !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}}
.stDownloadButton button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}

/* ---- Badges ---- */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.badge-low    {{ background: #EBF0EA; color: {ACCENT}; }}
.badge-medium {{ background: #FBF1E5; color: {WARNING}; }}
.badge-high   {{ background: #FCE9E0; color: {DANGER}; }}
.badge-ok     {{ background: #EBF0EA; color: {ACCENT}; }}
.badge-fail   {{ background: #FCE9E0; color: {DANGER}; }}

/* ---- Tables ---- */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
    text-align: left;
    font-size: 10px;
    font-weight: 700;
    color: {TEXT_SEC};
    padding: 10px 14px;
    border-bottom: 2px solid {BORDER};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(224,223,215,0.5);
    color: {TEXT_MAIN};
    vertical-align: top;
}}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background-color: rgba(83,107,79,0.02); }}

/* ---- Banners ---- */
.warn-banner {{
    background: #FBF1E5;
    border: 1px solid {WARNING};
    border-radius: 8px;
    padding: 12px 16px;
    color: {WARNING};
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 14px;
}}
.info-banner {{
    background: #EBF0EA;
    border: 1px solid {ACCENT};
    border-radius: 8px;
    padding: 12px 16px;
    color: {ACCENT};
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 14px;
}}

/* ---- Caveat text ---- */
.caveat {{
    font-size: 11px;
    color: {TEXT_SEC};
    border-left: 3px solid {BORDER};
    padding-left: 12px;
    margin-top: 12px;
    line-height: 1.7;
}}

/* ---- Metric widgets (Streamlit native) ---- */
[data-testid="stMetricValue"] {{
    font-size: 26px !important;
    font-weight: 700 !important;
    color: {TEXT_MAIN} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px !important;
    font-weight: 600 !important;
    color: {TEXT_SEC} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* ---- Separator ---- */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 24px 0 !important;
}}

/* ---- Hide Streamlit chrome ---- */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}
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

MODEL_PATH = Path(__file__).parent / "models" / "loan_model.pkl"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=HEADERS,
                         params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_post(path: str, payload: dict) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", headers=HEADERS,
                          json=payload, timeout=15)
        if r.ok:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def backend_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local model loader (fallback when API is down)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_local_bundle():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# Utility renderers
# ---------------------------------------------------------------------------

def risk_badge(level: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(
        level.upper(), "badge-medium"
    )
    return f'<span class="badge {cls}">{level}</span>'


def check_badge(passed: bool) -> str:
    cls  = "badge-ok"   if passed else "badge-fail"
    text = "PASS"       if passed else "FAIL"
    return f'<span class="badge {cls}">{text}</span>'


def fmt_prob(p: float) -> str:
    return f"{p:.1%}"


def section(title: str):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend status banner
# ---------------------------------------------------------------------------

def show_backend_status():
    alive = backend_alive()
    if not alive:
        st.markdown(
            '<div class="warn-banner">Backend API is not reachable at '
            f'{API_BASE}.  Start it with: <code>uvicorn api:app --reload --port 8000</code>'
            "</div>",
            unsafe_allow_html=True,
        )
    return alive


# ===========================================================================
# Page: AI Text Decision Auditor
# ===========================================================================

def page_text_audit():
    st.markdown(
        f"""<div style="margin-bottom:24px;">
            <h2 style="margin:0 0 4px 0;">AI Decision & Trust Evaluator</h2>
            <p style="color:{TEXT_SEC};font-size:13px;margin:0;">
                Paste any AI-generated decision text below. The system will classify the outcome,
                check for fairness concerns, and determine if human review is needed.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Input Text",
        height=140,
        placeholder="Paste AI-generated decision text here...",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("Analyze Decision", type="primary", use_container_width=True)

    if analyze_clicked or (user_text and user_text.strip()):
        if not user_text.strip():
            st.warning("Please paste or type AI text to evaluate.")
            return

        from text_analyzer import analyze_ai_text
        result = analyze_ai_text(user_text)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # --- 3 Result Cards ---
        c1, c2, c3 = st.columns(3)

        with c1:
            dec_color = ACCENT if "APPROV" in result.detected_decision else (DANGER if "REJECT" in result.detected_decision else WARNING)
            dec_icon = "&#10003;" if "APPROV" in result.detected_decision else ("&#10007;" if "REJECT" in result.detected_decision else "&#8212;")
            st.markdown(
                f"""<div class="panel" style="min-height:130px;border-top:3px solid {dec_color};">
                    <div class="panel-title">AI Decision</div>
                    <div style="font-size:24px;font-weight:700;color:{dec_color};margin:10px 0 6px 0;">{dec_icon} {result.detected_decision}</div>
                    <div style="font-size:12px;color:{TEXT_SEC};">Confidence: <b>{result.decision_confidence}</b></div>
                </div>""",
                unsafe_allow_html=True,
            )

        with c2:
            is_fair = not ("WARNING" in result.fairness_status or "BIASED" in result.fairness_status)
            fair_color = ACCENT if is_fair else WARNING
            fair_icon = "&#9679;" if is_fair else "&#9888;"
            fair_title = "No Bias Detected" if is_fair else "Bias Alert"
            fair_desc = "No prohibited demographic attributes cited" if is_fair else "Contains sensitive demographic references"
            st.markdown(
                f"""<div class="panel" style="min-height:130px;border-top:3px solid {fair_color};">
                    <div class="panel-title">Fairness</div>
                    <div style="font-size:24px;font-weight:700;color:{fair_color};margin:10px 0 6px 0;">{fair_icon} {fair_title}</div>
                    <div style="font-size:12px;color:{TEXT_SEC};">{fair_desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with c3:
            rev_color = WARNING if result.requires_human_review else ACCENT
            rev_icon = "&#9998;" if result.requires_human_review else "&#10003;"
            rev_title = f"Required" if result.requires_human_review else "Not Required"
            rev_desc = f"Urgency: {result.review_urgency}" if result.requires_human_review else "Safe to proceed automatically"
            st.markdown(
                f"""<div class="panel" style="min-height:130px;border-top:3px solid {rev_color};">
                    <div class="panel-title">Human Review</div>
                    <div style="font-size:24px;font-weight:700;color:{rev_color};margin:10px 0 6px 0;">{rev_icon} {rev_title}</div>
                    <div style="font-size:12px;color:{TEXT_SEC};">{rev_desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # --- Rationale Panel ---
        st.markdown(
            f"""<div class="panel">
                <div class="panel-title">Evaluation Summary</div>
                <div style="font-size:13px;line-height:1.7;color:{TEXT_MAIN};">{result.summary}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        if result.fairness_flags:
            st.markdown(
                f"""<div class="warn-banner">
                    <b>Flagged References:</b> {', '.join(result.fairness_flags)}
                </div>""",
                unsafe_allow_html=True,
            )

        # --- Future Scope ---
        st.markdown(
            f"""<div class="panel" style="border-left:3px solid {ACCENT};">
                <div class="panel-title">Future Scope</div>
                <div style="font-size:12px;line-height:1.7;color:{TEXT_SEC};">
                    In the next phase, this text evaluator will be integrated as an automated
                    <b>LLM-as-a-Judge</b> proxy service in the cloud pipeline.
                    All customer-facing AI recommendations will be intercepted in real-time
                    to prevent bias and ensure regulatory compliance.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Page: Model Evaluation
# ===========================================================================

def page_evaluation(alive: bool):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    import matplotlib.pyplot as plt

    st.markdown(
        f"""<div style="margin-bottom:24px;">
            <h2 style="margin:0 0 4px 0;">Model Evaluation</h2>
            <p style="color:{TEXT_SEC};font-size:13px;margin:0;">
                Baseline vs. Responsible AI Framework comparison on the held-out test set.
                All numbers are measured, not illustrative.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    eval_path = Path(__file__).parent / "data" / "evaluation_results.csv"
    detail_path = Path(__file__).parent / "data" / "evaluation_detail.json"

    if not eval_path.exists():
        st.warning(
            "Evaluation results not found.  "
            "Run `python evaluate.py` to generate them."
        )
        return

    df = pd.read_csv(eval_path)
    metrics = api_get("/metrics") if alive else None

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown(
            f"""<div class="panel">
                <div class="panel-title">Comparison Table</div>
            </div>""",
            unsafe_allow_html=True,
        )
        rows = ""
        for _, row in df.iterrows():
            m  = row.get("metric", "")
            b  = row.get("baseline")
            fw = row.get("framework")
            b_str = f"{b}" if pd.notna(b) else "-"
            fw_str = f"{fw}" if pd.notna(fw) else "-"
            rows += (
                f"<tr><td>{m}</td>"
                f"<td style='font-weight:500'>{b_str}</td>"
                f"<td style='font-weight:500'>{fw_str}</td></tr>"
            )
        st.markdown(
            "<table><thead><tr><th>Metric</th><th>Baseline</th><th>Framework</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )
        if eval_path.exists():
            csv = df.to_csv(index=False)
            st.download_button("Export CSV", csv,
                               file_name="evaluation_results.csv", mime="text/csv")

    with col_right:
        st.markdown(
            f"""<div class="panel">
                <div class="panel-title">Performance Chart</div>
            </div>""",
            unsafe_allow_html=True,
        )
        row_acc  = df[df["metric"] == "accuracy"]
        row_f1   = df[df["metric"] == "f1_score"]
        if not row_acc.empty and not row_f1.empty:
            cats  = ["Accuracy", "F1-score"]
            base  = [float(row_acc["baseline"].values[0]),
                     float(row_f1["baseline"].values[0])]
            frame = [float(row_acc["framework"].values[0]),
                     float(row_f1["framework"].values[0])]
            x = range(len(cats))
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            bar1 = ax.bar([i - 0.17 for i in x], base,  0.28, label="Baseline",  color=BORDER, edgecolor="none")
            bar2 = ax.bar([i + 0.17 for i in x], frame, 0.28, label="Framework", color=ACCENT, edgecolor="none")
            # Value labels on bars
            for bar in bar1:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=TEXT_SEC)
            for bar in bar2:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=ACCENT)
            ax.set_xticks(list(x))
            ax.set_xticklabels(cats, fontsize=11, fontweight=500)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("Score", fontsize=10, color=TEXT_SEC)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.tick_params(colors=TEXT_SEC, labelsize=9)
            ax.legend(fontsize=9, framealpha=0, loc="upper right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Confusion matrix
    if metrics and "confusion_matrix" in metrics:
        st.markdown("---")
        cm = metrics["confusion_matrix"]
        col_cm, _ = st.columns([1, 2])
        with col_cm:
            st.markdown(
                f"""<div class="panel">
                    <div class="panel-title">Confusion Matrix (Test Set)</div>
                </div>""",
                unsafe_allow_html=True,
            )
            fig, ax = plt.subplots(figsize=(3.5, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(cm, cmap="Greens", aspect="auto")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: Reject", "Pred: Approve"], fontsize=9)
            ax.set_yticklabels(["Actual: Reject", "Actual: Approve"], fontsize=9)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center",
                            fontsize=14, fontweight=600, color=TEXT_MAIN)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.markdown(
        '<div class="caveat">'
        "Adding Responsible AI checks does not automatically improve model accuracy. "
        "The framework adds explainability, fairness evaluation, and human-review "
        "routing at the cost of additional processing time.  Results are from "
        "synthetic data and do not reflect real-world lending outcomes.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Navigation
# ===========================================================================

def main():
    alive = show_backend_status()

    # Clean header
    st.markdown(
        f"""<div style="text-align:center; padding:8px 0 20px 0; margin-bottom:8px;">
            <div style="font-size:20px; font-weight:700; color:{TEXT_MAIN}; letter-spacing:-0.3px;">
                Responsible AI Platform
            </div>
            <div style="font-size:12px; color:{TEXT_SEC}; margin-top:2px;">
                Trustworthy Decision-Making in Cloud Computing
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    tab_audit, tab_eval = st.tabs([
        "  AI Text Audit  ",
        "  Model Evaluation  ",
    ])

    with tab_audit:
        page_text_audit()
    with tab_eval:
        page_evaluation(alive)


if __name__ == "__main__":
    main()
