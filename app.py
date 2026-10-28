"""
app.py — Streamlit dashboard for the Responsible AI Framework.

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
    page_title="Responsible AI Framework",
    layout="wide",
    initial_sidebar_state="expanded",
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

CUSTOM_CSS = f"""
<style>
/* ---- Reset & base ---- */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT_MAIN};
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 14px;
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13px;
    color: {TEXT_MAIN};
}}

/* ---- Main header ---- */
h1 {{ font-size: 20px; font-weight: 600; color: {TEXT_MAIN}; margin-bottom: 2px; }}
h2 {{ font-size: 16px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 18px; }}
h3 {{ font-size: 14px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 12px; }}

/* ---- Panels ---- */
.panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 14px;
}}
.panel-title {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 8px;
}}

/* ---- Stat row ---- */
.stat-row {{ display: flex; gap: 12px; margin-bottom: 14px; }}
.stat-cell {{
    flex: 1;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 12px 16px;
}}
.stat-num {{
    font-size: 24px;
    font-weight: 700;
    color: {TEXT_MAIN};
    line-height: 1;
}}
.stat-label {{
    font-size: 11px;
    color: {TEXT_SEC};
    margin-top: 4px;
}}

/* ---- Badges ---- */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.badge-low    {{ background: #EBF0EA; color: {ACCENT}; }}
.badge-medium {{ background: #FBF1E5; color: {WARNING}; }}
.badge-high   {{ background: #FCE9E0; color: #B04020; }}
.badge-ok     {{ background: #EBF0EA; color: {ACCENT}; }}
.badge-fail   {{ background: #FCE9E0; color: #B04020; }}

/* ---- Tables ---- */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_SEC};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    background: {BG};
}}
td {{
    padding: 7px 10px;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_MAIN};
    vertical-align: top;
}}
tr:last-child td {{ border-bottom: none; }}

/* ---- Warning banner ---- */
.warn-banner {{
    background: #FBF1E5;
    border: 1px solid {WARNING};
    border-radius: 4px;
    padding: 10px 14px;
    color: {WARNING};
    font-size: 13px;
    margin-bottom: 12px;
}}

/* ---- Info banner ---- */
.info-banner {{
    background: #EBF0EA;
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 10px 14px;
    color: {ACCENT};
    font-size: 13px;
    margin-bottom: 12px;
}}

/* ---- Caveat text ---- */
.caveat {{
    font-size: 11px;
    color: {TEXT_SEC};
    border-left: 3px solid {BORDER};
    padding-left: 10px;
    margin-top: 10px;
    line-height: 1.6;
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
            '<div class="warn-banner">⚠ Backend API is not reachable at '
            f'{API_BASE}.  Start it with: <code>uvicorn api:app --reload --port 8000</code>'
            "</div>",
            unsafe_allow_html=True,
        )
    return alive


# ===========================================================================
# Page: Dashboard
# ===========================================================================

def page_dashboard(alive: bool):
    st.markdown("# Responsible AI Framework")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;margin-bottom:20px;">'
        "Academic prototype — synthetic loan-application decisions with "
        "explainability, fairness, privacy, security, and human-review layers."
        "</p>",
        unsafe_allow_html=True,
    )

    # Stats row
    stats = api_get("/stats") if alive else {}
    stats = stats or {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Decisions",     stats.get("total_decisions", "—"))
    with col2:
        st.metric("High-Risk Decisions", stats.get("high_risk", "—"))
    with col3:
        st.metric("Pending Review",      stats.get("pending_review", "—"))
    with col4:
        st.metric("Reviewed",            stats.get("reviewed", "—"))

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    # Model performance
    with col_left:
        section("MODEL PERFORMANCE")
        metrics = api_get("/metrics") if alive else None
        if metrics:
            for k in ("accuracy", "precision", "recall", "f1_score"):
                v = metrics.get(k)
                if v is not None:
                    st.markdown(
                        f"<table><tr><td style='color:{TEXT_SEC};width:140px'>{k.replace('_',' ').title()}</td>"
                        f"<td><b>{v:.4f}</b></td></tr></table>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Run train_model.py then start the API to see metrics.")

    # Fairness summary
    with col_right:
        section("FAIRNESS SUMMARY")
        fairness = api_get("/fairness") if alive else None
        if fairness:
            dpd = fairness.get("demographic_parity_difference")
            fw  = fairness.get("fairness_warning", False)
            if fw:
                st.markdown(
                    f'<div class="warn-banner">Fairness warning: DPD = {dpd:.4f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="info-banner">Fairness OK: DPD = {dpd:.4f}</div>',
                    unsafe_allow_html=True,
                )
            gm = fairness.get("group_metrics", [])
            if gm:
                rows = "".join(
                    f"<tr><td>{g['group']}</td><td>{g['size']}</td>"
                    f"<td>{g['selection_rate']:.2%}</td></tr>"
                    for g in gm
                )
                st.markdown(
                    f"<table><thead><tr><th>Group</th><th>N</th>"
                    f"<th>Selection Rate</th></tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Fairness data not available.")

    # Recent decisions
    st.markdown("---")
    section("RECENT ACTIVITY")
    decisions = api_get("/decisions", {"limit": 8}) if alive else []
    decisions = decisions or []
    if decisions:
        rows = ""
        for d in decisions:
            ts = d.get("created_at", "")[:19].replace("T", " ")
            rl = d.get("risk_level", "—")
            ro = d.get("routing_outcome", "—")
            pred = "✓ Approved" if d.get("original_prediction") == 1 else "✗ Rejected"
            rows += (
                f"<tr><td style='font-family:monospace;font-size:11px'>{d['decision_id'][:8]}…</td>"
                f"<td>{ts}</td><td>{pred}</td>"
                f"<td>{risk_badge(rl)}</td><td>{ro}</td></tr>"
            )
        st.markdown(
            f"<table><thead><tr><th>ID</th><th>Time (UTC)</th><th>Prediction</th>"
            f"<th>Risk</th><th>Routing</th></tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No decisions recorded yet.  Use 'Make a Prediction' to create one.")


# ===========================================================================
# Page: Make a Prediction
# ===========================================================================

def page_predict(alive: bool):
    st.markdown("# Make a Prediction")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;">Enter synthetic applicant data. '
        "The system runs the full Responsible AI assessment pipeline.</p>",
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            income    = st.number_input("Annual Income ($)", 0.0, 1_000_000.0, 55000.0, step=1000.0)
            cs        = st.number_input("Credit Score",      300,  850,         650)
            debt      = st.number_input("Existing Debt ($)", 0.0, 500_000.0,   10000.0, step=500.0)
        with col2:
            emp_dur   = st.number_input("Employment Duration (years)", 0.0, 50.0, 5.0, step=0.5)
            loan_amt  = st.number_input("Loan Amount ($)",  1000.0, 1_000_000.0, 25000.0, step=500.0)

        submitted = st.form_submit_button("Run Assessment", use_container_width=True)

    if not submitted:
        return

    if not alive:
        st.error("Backend is not running.  Start uvicorn api:app --reload --port 8000")
        return

    payload = {
        "income":              income,
        "credit_score":        float(cs),
        "existing_debt":       debt,
        "employment_duration": emp_dur,
        "loan_amount":         loan_amt,
    }

    with st.spinner("Running Responsible AI assessment …"):
        result = api_post("/predict", payload)

    if result is None or "error" in result:
        st.error(f"Prediction failed: {result}")
        return

    # --- Results ---
    st.markdown("---")
    col_pred, col_risk, col_route = st.columns(3)

    with col_pred:
        section("PREDICTION")
        label = result.get("prediction_label", "—")
        prob  = result.get("prediction_prob", 0.0)
        color = ACCENT if result.get("prediction") == 1 else "#B04020"
        st.markdown(
            f"<div style='font-size:22px;font-weight:700;color:{color}'>{label}</div>"
            f"<div style='font-size:13px;color:{TEXT_SEC};margin-top:4px'>"
            f"Model probability: {fmt_prob(prob)}<br>"
            f"<span style='font-size:11px'>Probability is not a calibrated confidence interval.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_risk:
        section("RISK ASSESSMENT")
        rl    = result.get("risk_level", "—")
        score = result.get("risk_score", 0)
        st.markdown(
            f"{risk_badge(rl)}"
            f"<div style='margin-top:8px;font-size:13px'>Score: <b>{score}</b></div>",
            unsafe_allow_html=True,
        )
        flags = result.get("risk_flags", [])
        if flags:
            for f in flags:
                st.markdown(f"<li style='font-size:12px;color:{WARNING}'>{f}</li>",
                            unsafe_allow_html=True)

    with col_route:
        section("ROUTING")
        routing = result.get("routing", "—")
        st.markdown(
            f"<div style='font-size:15px;font-weight:600'>{routing}</div>"
            f"<div style='font-size:12px;color:{TEXT_SEC};margin-top:6px'>"
            f"{result.get('routing_reason','')}</div>",
            unsafe_allow_html=True,
        )

    # RA checks
    st.markdown("---")
    section("RESPONSIBLE AI CHECKS")
    checks = {
        "Privacy":     result.get("privacy_passed",     False),
        "Security":    result.get("security_passed",    False),
        "Reliability": result.get("reliability_passed", False),
    }
    row_html = "".join(
        f"<tr><td>{k}</td><td>{check_badge(v)}</td></tr>"
        for k, v in checks.items()
    )
    fw = result.get("fairness_warning", False)
    fw_badge = (
        '<span class="badge badge-medium">WARNING</span>'
        if fw
        else '<span class="badge badge-ok">OK</span>'
    )
    row_html += f"<tr><td>Fairness (batch)</td><td>{fw_badge}</td></tr>"
    st.markdown(
        f"<table style='max-width:340px'><thead><tr><th>Check</th><th>Status</th>"
        f"</tr></thead><tbody>{row_html}</tbody></table>",
        unsafe_allow_html=True,
    )

    # Explanation
    expl = result.get("explanation_summary", "")
    if expl:
        st.markdown("---")
        section("SHAP EXPLANATION")
        st.markdown(f"<div style='font-size:13px'>{expl}</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="caveat">SHAP values show which features most influenced '
            "this model output.  They do not prove the decision is correct or fair.</div>",
            unsafe_allow_html=True,
        )

    # Decision ID
    st.markdown(
        f"<div style='font-size:11px;color:{TEXT_SEC};margin-top:14px'>"
        f"Decision ID: <code>{result.get('decision_id','—')}</code></div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page: Fairness Analysis
# ===========================================================================

def page_fairness(alive: bool):
    st.markdown("# Fairness Analysis")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;">'
        "Group-level fairness metrics computed on the held-out test set with "
        "ground-truth labels.  Fairness is not inferred from individual predictions.</p>",
        unsafe_allow_html=True,
    )

    fairness = api_get("/fairness") if alive else None
    if not fairness:
        st.warning("Fairness data not available.  Start the API and train the model first.")
        return

    # Threshold control
    threshold = st.slider(
        "Demographic-parity-difference warning threshold",
        min_value=0.01, max_value=0.30, value=float(fairness.get("threshold_used", 0.10)),
        step=0.01,
        help="Adjust to see how different thresholds affect the warning status.",
    )

    dpd = fairness.get("demographic_parity_difference", 0.0)
    eod = fairness.get("equalized_odds_difference")
    fw  = abs(dpd) > threshold

    col_left, col_right = st.columns([1, 1])

    with col_left:
        section("GLOBAL METRICS")
        table_rows = [
            ("Overall selection rate", f"{fairness.get('overall_selection_rate',0):.2%}"),
            ("Demographic parity diff", f"{dpd:.4f}"),
            ("Equalized odds diff",
             f"{eod:.4f}" if eod is not None else "N/A"),
            ("Sample size", str(fairness.get("n_samples", "—"))),
            ("Warning threshold", f"{threshold:.2f}"),
        ]
        rows = "".join(
            f"<tr><td style='color:{TEXT_SEC}'>{k}</td><td><b>{v}</b></td></tr>"
            for k, v in table_rows
        )
        st.markdown(
            f"<table><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

        if fw:
            st.markdown(
                f'<div class="warn-banner" style="margin-top:12px">⚠ {fairness.get("warning_message","Fairness warning.")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-banner" style="margin-top:12px">✓ Demographic-parity difference '
                f"({dpd:.4f}) is within the threshold ({threshold:.2f}).</div>",
                unsafe_allow_html=True,
            )

    with col_right:
        section("GROUP-LEVEL SELECTION RATES")
        gm = fairness.get("group_metrics", [])
        if gm:
            df_gm = pd.DataFrame(gm)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            colors = [ACCENT if abs(g["selection_rate"] - fairness.get("overall_selection_rate", 0)) <= threshold else WARNING
                      for g in gm]
            ax.bar(df_gm["group"], df_gm["selection_rate"], color=colors, width=0.5)
            ax.axhline(fairness.get("overall_selection_rate", 0), color=TEXT_SEC,
                       linestyle="--", linewidth=0.8, label="Overall")
            ax.set_ylabel("Selection Rate", fontsize=10, color=TEXT_MAIN)
            ax.set_ylim(0, 1)
            ax.tick_params(colors=TEXT_MAIN, labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.legend(fontsize=9, framealpha=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Group table
    st.markdown("---")
    section("GROUP DETAIL TABLE")
    if gm:
        table_rows = []
        for g in gm:
            acc_val = g.get("accuracy")
            acc_str = f"{acc_val:.2%}" if acc_val is not None else "N/A"
            table_rows.append(
                f"<tr><td>{g['group']}</td><td>{g['size']}</td>"
                f"<td>{g['selection_rate']:.2%}</td>"
                f"<td>{acc_str}</td></tr>"
            )
        rows_str = "".join(table_rows)
        st.markdown(
            "<table><thead><tr><th>Group</th><th>N</th>"
            "<th>Selection Rate</th><th>Accuracy</th></tr></thead>"
            f"<tbody>{rows_str}</tbody></table>",
            unsafe_allow_html=True,
        )

    # Metric definitions
    st.markdown("---")
    section("METRIC DEFINITIONS")
    for name, defn in fairness.get("metric_definitions", {}).items():
        st.markdown(
            f"<div style='font-size:13px'><b>{name}</b>: {defn}</div><br>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="caveat">{fairness.get("caveat","")}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page: Human Review
# ===========================================================================

def page_review(alive: bool):
    st.markdown("# Human Review")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;">Decisions with high risk scores '
        "are routed here.  Reviewer actions are recorded separately from the "
        "original model prediction.</p>",
        unsafe_allow_html=True,
    )

    pending = api_get("/review/pending") if alive else []
    pending = pending or []

    if not pending:
        st.markdown(
            '<div class="info-banner">No decisions are currently awaiting review.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"**{len(pending)} decision(s) pending review**")

    for dec in pending:
        did    = dec.get("decision_id", "")
        ts     = dec.get("created_at", "")[:19].replace("T", " ")
        rl     = dec.get("risk_level", "—")
        score  = dec.get("risk_score", 0)
        pred   = "Approved" if dec.get("original_prediction") == 1 else "Rejected"
        prob   = dec.get("prediction_prob")
        flags  = json.loads(dec.get("risk_reasons", "[]") or "[]")
        expl   = dec.get("explanation_summary", "No explanation available.")

        with st.expander(f"Decision {did[:12]}… — {ts} UTC — {rl} risk"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(
                    f"<table>"
                    f"<tr><td style='color:{TEXT_SEC}'>Prediction</td><td><b>{pred}</b></td></tr>"
                    f"<tr><td style='color:{TEXT_SEC}'>Prob.</td><td><b>{fmt_prob(prob) if prob else '—'}</b></td></tr>"
                    f"<tr><td style='color:{TEXT_SEC}'>Risk Level</td><td>{risk_badge(rl)}</td></tr>"
                    f"<tr><td style='color:{TEXT_SEC}'>Risk Score</td><td><b>{score}</b></td></tr>"
                    f"</table>",
                    unsafe_allow_html=True,
                )
            with col2:
                if flags:
                    st.markdown("**Active risk flags:**")
                    for f in flags:
                        st.markdown(f"<li style='color:{WARNING};font-size:12px'>{f}</li>",
                                    unsafe_allow_html=True)

            st.markdown(f"**Explanation:** {expl}")

            with st.form(f"review_form_{did}"):
                reviewer_id = st.text_input("Reviewer ID", value="reviewer-1", key=f"rid_{did}")
                action      = st.selectbox(
                    "Action", ["APPROVE", "REJECT", "REQUEST_REVIEW"], key=f"act_{did}"
                )
                comments    = st.text_area("Comments (optional)", key=f"cmt_{did}")
                submit_rev  = st.form_submit_button("Submit Review")

            if submit_rev:
                r = api_post(f"/review/{did}", {
                    "reviewer_id": reviewer_id,
                    "action":      action,
                    "comments":    comments,
                })
                if r and "error" not in r:
                    st.success(f"Review recorded: {action}")
                    api_get.clear()
                else:
                    st.error(f"Failed to record review: {r}")


# ===========================================================================
# Page: Audit Logs
# ===========================================================================

def page_audit(alive: bool):
    st.markdown("# Audit Log")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;">Complete record of all decisions, '
        "assessments, and reviewer actions.</p>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        f_risk = st.selectbox("Risk level", ["All", "LOW", "MEDIUM", "HIGH"])
    with col_f2:
        f_routing = st.selectbox(
            "Routing", ["All", "AUTO_APPROVED", "AUTO_REJECTED", "FLAGGED", "HUMAN_REVIEW"]
        )
    with col_f3:
        f_review = st.selectbox("Review status", ["All", "PENDING", "REVIEWED", "NA"])
    with col_f4:
        f_limit = st.number_input("Rows", min_value=10, max_value=500, value=50)

    params = {"limit": f_limit}
    if f_risk    != "All": params["risk_level"]    = f_risk
    if f_routing != "All": params["routing"]       = f_routing
    if f_review  != "All": params["review_status"] = f_review

    decisions = api_get("/decisions", params) if alive else []
    decisions = decisions or []

    if not decisions:
        st.caption("No decisions match the selected filters.")
        return

    rows = ""
    for d in decisions:
        ts    = d.get("created_at", "")[:19].replace("T", " ")
        rl    = d.get("risk_level", "—")
        pred  = "Approved" if d.get("original_prediction") == 1 else "Rejected"
        final = d.get("final_decision")
        final_str = ("Approved" if final == 1 else "Rejected") if final is not None else "—"
        rv    = d.get("reviewer_action", "—") or "—"
        rs    = d.get("review_status", "—")
        rows += (
            f"<tr>"
            f"<td style='font-family:monospace;font-size:11px'>{d['decision_id'][:12]}…</td>"
            f"<td>{ts}</td>"
            f"<td>{pred}</td>"
            f"<td>{risk_badge(rl)}</td>"
            f"<td>{d.get('routing_outcome','—')}</td>"
            f"<td>{rs}</td>"
            f"<td>{rv}</td>"
            f"<td>{final_str}</td>"
            f"</tr>"
        )

    st.markdown(
        "<div style='overflow-x:auto'>"
        "<table><thead><tr>"
        "<th>Decision ID</th><th>Time (UTC)</th><th>Prediction</th>"
        "<th>Risk</th><th>Routing</th><th>Review Status</th>"
        "<th>Reviewer Action</th><th>Final Decision</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # Export
    if st.button("Export to CSV"):
        df = pd.DataFrame(decisions)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV", csv,
            file_name="audit_log.csv", mime="text/csv",
        )


# ===========================================================================
# Page: Model Evaluation
# ===========================================================================

def page_evaluation(alive: bool):
    import matplotlib.pyplot as plt

    st.markdown("# Model Evaluation")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;">'
        "Baseline vs. Responsible AI Framework comparison on the held-out test set. "
        "All numbers are measured, not illustrative.</p>",
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

    col_left, col_right = st.columns([1, 1])

    with col_left:
        section("COMPARISON TABLE")
        rows = ""
        for _, row in df.iterrows():
            m  = row.get("metric", "")
            b  = row.get("baseline")
            fw = row.get("framework")
            rows += (
                f"<tr><td>{m}</td>"
                f"<td>{b if pd.notna(b) else '—'}</td>"
                f"<td>{fw if pd.notna(fw) else '—'}</td></tr>"
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
        section("ACCURACY COMPARISON")
        row_acc  = df[df["metric"] == "accuracy"]
        row_f1   = df[df["metric"] == "f1_score"]
        if not row_acc.empty and not row_f1.empty:
            cats  = ["Accuracy", "F1-score"]
            base  = [float(row_acc["baseline"].values[0]),
                     float(row_f1["baseline"].values[0])]
            frame = [float(row_acc["framework"].values[0]),
                     float(row_f1["framework"].values[0])]
            x = range(len(cats))
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.bar([i - 0.18 for i in x], base,  0.3, label="Baseline",   color=BORDER)
            ax.bar([i + 0.18 for i in x], frame, 0.3, label="Framework",  color=ACCENT)
            ax.set_xticks(list(x))
            ax.set_xticklabels(cats, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.legend(fontsize=9, framealpha=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Confusion matrix from metrics endpoint
    if metrics and "confusion_matrix" in metrics:
        st.markdown("---")
        section("CONFUSION MATRIX (test set)")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Rej", "Pred: App"], fontsize=9)
        ax.set_yticklabels(["Act: Rej", "Act: App"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        fontsize=12, color=TEXT_MAIN)
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
# Page: AI Text Decision Auditor
# ===========================================================================

def page_text_audit():
    st.markdown("# AI Text Decision Auditor")
    st.markdown(
        '<p style="color:#77796F;font-size:13px;margin-bottom:16px;">'
        "Paste any AI-generated response, decision text, or explanation. "
        "The system evaluates what decision was made, whether it is fair and bias-free, "
        "and whether a human reviewer must intervene."
        "</p>",
        unsafe_allow_html=True,
    )

    # Preset quick-load examples
    st.markdown("<span style='font-size:12px;color:#77796F;font-weight:600;'>TRY A SAMPLE:</span>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    if "audit_input" not in st.session_state:
        st.session_state.audit_input = ""

    with c1:
        if st.button("Sample 1: Fair Approval", use_container_width=True):
            st.session_state.audit_input = (
                "After reviewing the applicant's profile, the loan application is approved. "
                "The applicant maintains an annual income of $82,000, a credit score of 740, "
                "and an existing debt-to-income ratio below 20%. Repayment capacity is strong."
            )
    with c2:
        if st.button("Sample 2: Potentially Biased Rejection", use_container_width=True):
            st.session_state.audit_input = (
                "The loan application has been rejected. Although the applicant's credit score is 680, "
                "the applicant is a single female living in a high-risk inner-city neighborhood, "
                "and our demographic risk profile indicates potential loan volatility for this group."
            )
    with c3:
        if st.button("Sample 3: Uncertain / Borderline", use_container_width=True):
            st.session_state.audit_input = (
                "The model generated a borderline outcome. The applicant might be eligible for a loan, "
                "but conflicting financial records make this determination unclear. We cannot verify "
                "if income covers the debt load. Decision is currently inconclusive."
            )

    user_text = st.text_area(
        "Paste AI-Generated Text Here",
        value=st.session_state.audit_input,
        height=130,
        placeholder="e.g. 'The application is rejected due to credit score 580 and high existing debt...'"
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("Analyze Decision", use_container_width=True)

    if analyze_clicked or user_text:
        if not user_text.strip():
            st.warning("Please paste or type AI-generated text to analyze.")
            return

        from text_analyzer import analyze_ai_text
        result = analyze_ai_text(user_text)

        st.markdown("---")
        
        # Top 3 KPI blocks
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            section("AI DECISION DETECTED")
            dec_color = ACCENT if "APPROV" in result.detected_decision else ("#B04020" if "REJECT" in result.detected_decision else WARNING)
            st.markdown(
                f"<div style='font-size:20px;font-weight:700;color:{dec_color}'>{result.detected_decision}</div>"
                f"<div style='font-size:12px;color:{TEXT_SEC};margin-top:4px'>Confidence: <b>{result.decision_confidence}</b></div>",
                unsafe_allow_html=True
            )
        
        with kpi2:
            section("FAIRNESS EVALUATION")
            if "WARNING" in result.fairness_status or "BIASED" in result.fairness_status:
                st.markdown(
                    f"<div style='font-size:18px;font-weight:700;color:{WARNING}'>ALERT / BIASED</div>"
                    f"<div style='font-size:12px;color:{WARNING};margin-top:4px'>Demographic references found</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='font-size:18px;font-weight:700;color:{ACCENT}'>FAIR</div>"
                    f"<div style='font-size:12px;color:{TEXT_SEC};margin-top:4px'>No demographic bias detected</div>",
                    unsafe_allow_html=True
                )
                
        with kpi3:
            section("HUMAN REVIEW ROUTING")
            if result.requires_human_review:
                urgency_color = "#B04020" if result.review_urgency == "IMMEDIATE" else WARNING
                st.markdown(
                    f"<div style='font-size:18px;font-weight:700;color:{urgency_color}'>REQUIRED ({result.review_urgency})</div>"
                    f"<div style='font-size:12px;color:{urgency_color};margin-top:4px'>Escalate to human reviewer</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='font-size:18px;font-weight:700;color:{ACCENT}'>NOT REQUIRED</div>"
                    f"<div style='font-size:12px;color:{TEXT_SEC};margin-top:4px'>Safe for automated decision</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        
        # Detailed findings
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            section("WHY HUMAN REVIEW IS " + ("REQUIRED" if result.requires_human_review else "NOT NEEDED"))
            if result.reasons_for_review:
                for r in result.reasons_for_review:
                    st.markdown(f"<li style='font-size:13px;color:{WARNING};margin-bottom:6px;'>{r}</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li style='font-size:13px;color:{ACCENT};margin-bottom:6px;'>All criteria strictly rely on objective financial metrics.</li>", unsafe_allow_html=True)
                st.markdown(f"<li style='font-size:13px;color:{ACCENT};margin-bottom:6px;'>No uncertainty or ambiguous language detected.</li>", unsafe_allow_html=True)

        with col_detail2:
            section("SUMMARY & RATIONALE")
            st.markdown(f"<div style='font-size:13px;line-height:1.6;color:{TEXT_MAIN};'>{result.summary}</div>", unsafe_allow_html=True)
            if result.fairness_flags:
                st.markdown(f"<div style='font-size:12px;color:{WARNING};margin-top:8px;'><b>Flags:</b> {', '.join(result.fairness_flags)}</div>", unsafe_allow_html=True)

        # Future scope box
        st.markdown("---")
        section("FUTURE SCOPE & CLOUD INTEGRATION")
        st.markdown(
            f'<div class="panel" style="font-size:12px;color:{TEXT_SEC};line-height:1.6;">'
            "<b>Phase 2 Architecture Roadmap:</b><br>"
            "This text analysis module will be integrated into the cloud pipeline as an automated <i>LLM-as-a-Judge</i> proxy. "
            "Every AI-generated response from loan models or customer chatbots will be audited in real-time. "
            "If sensitive demographic terms or ungrounded rationales are detected, the response is intercepted and rerouted "
            "to human underwriters before reaching the applicant."
            "</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Navigation
# ===========================================================================

def main():
    alive = show_backend_status()

    with st.sidebar:
        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:{TEXT_MAIN};"
            "padding:8px 0 4px 0;border-bottom:1px solid #E0DFD7;"
            "margin-bottom:14px'>Responsible AI Framework</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Make a Prediction",
                "AI Text Audit",
                "Fairness Analysis",
                "Human Review",
                "Audit Log",
                "Model Evaluation",
            ],
            label_visibility="collapsed",
        )
        st.markdown(
            f"<div style='font-size:11px;color:{TEXT_SEC};margin-top:24px;'>"
            "Academic prototype · Synthetic data only<br>"
            "Not for production use.</div>",
            unsafe_allow_html=True,
        )

    if page == "Dashboard":
        page_dashboard(alive)
    elif page == "Make a Prediction":
        page_predict(alive)
    elif page == "AI Text Audit":
        page_text_audit()
    elif page == "Fairness Analysis":
        page_fairness(alive)
    elif page == "Human Review":
        page_review(alive)
    elif page == "Audit Log":
        page_audit(alive)
    elif page == "Model Evaluation":
        page_evaluation(alive)


if __name__ == "__main__":
    main()
