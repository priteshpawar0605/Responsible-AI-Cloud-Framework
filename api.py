"""
api.py — FastAPI REST backend for the Responsible AI Framework.

Start with:
    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /health                  Public — liveness check
GET  /model-info              Public — model version and feature list
POST /predict                 Authenticated — run full RA prediction pipeline
GET  /decisions               Authenticated — list audit log entries
GET  /decisions/{id}          Authenticated — single decision
GET  /review/pending          Authenticated — pending human-review queue
POST /review/{id}             Authenticated — submit reviewer action
GET  /metrics                 Public — model performance metrics
GET  /fairness                Public — batch fairness metrics

Authentication
--------------
Protected endpoints require the X-API-Key header.  The key is read from the
API_KEY environment variable (set in .env).  This is a prototype mechanism —
not production OAuth/JWT.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
import audit
from config import (
    API_KEY,
    API_KEY_HEADER,
    DEMOGRAPHIC_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_VERSION,
)
from explainability import explain_prediction
from fairness import evaluate_fairness, load_fairness_results_from_test_set
from responsible_ai import run_privacy_check, run_reliability_check, run_security_check
from risk_engine import RoutingOutcome, compute_risk

# ---------------------------------------------------------------------------
# Initialise DB on startup
# ---------------------------------------------------------------------------
audit.init_db()

# ---------------------------------------------------------------------------
# Load model bundle (singleton)
# ---------------------------------------------------------------------------
_model_bundle: Optional[dict] = None

def get_model_bundle() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found.  Run python train_model.py first.",
            )
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Responsible AI Framework API",
    version="1.0.0",
    description=(
        "REST API for a prototype Responsible AI decision-making system "
        "applied to synthetic loan-application data.  "
        "This is an academic prototype, not a production system."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

def require_api_key(key: Optional[str] = Security(api_key_scheme)):
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return key


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    income:               float = Field(..., ge=0,    le=1_000_000)
    credit_score:         float = Field(..., ge=300,  le=850)
    existing_debt:        float = Field(..., ge=0,    le=500_000)
    employment_duration:  float = Field(..., ge=0,    le=50)
    loan_amount:          float = Field(..., ge=1000, le=1_000_000)
    applicant_id:         Optional[str] = None

    @field_validator("credit_score")
    @classmethod
    def credit_score_int_range(cls, v):
        if not (300 <= v <= 850):
            raise ValueError("credit_score must be between 300 and 850")
        return v


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=64)
    action:      str = Field(..., pattern="^(APPROVE|REJECT|REQUEST_REVIEW)$")
    comments:    Optional[str] = Field(default="", max_length=2000)


class PredictResponse(BaseModel):
    decision_id:         str
    prediction:          int
    prediction_label:    str
    prediction_prob:     float
    risk_level:          str
    risk_score:          int
    risk_flags:          List[str]
    routing:             str
    routing_reason:      str
    explanation_summary: str
    privacy_passed:      bool
    security_passed:     bool
    reliability_passed:  bool
    fairness_warning:    bool
    caveat:              str = (
        "This is a prototype.  Prediction probability is not a calibrated "
        "confidence interval.  Results on synthetic data do not reflect "
        "real-world outcomes."
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/model-info", tags=["System"])
def model_info():
    bundle = get_model_bundle()
    return {
        "model_version":   bundle.get("model_version", MODEL_VERSION),
        "model_type":      type(bundle["model"]).__name__,
        "feature_columns": bundle["feature_columns"],
        "metrics":         bundle.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(
    req: PredictRequest,
    raw_request: Request,
    _key: str = Depends(require_api_key),
):
    bundle = get_model_bundle()
    clf    = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["feature_columns"]

    import pandas as pd
    input_dict = req.model_dump(exclude={"applicant_id"})

    # Scale and predict
    row_df   = pd.DataFrame([input_dict])[feats]
    X_scaled = scaler.transform(row_df)
    pred     = int(clf.predict(X_scaled)[0])
    prob     = float(clf.predict_proba(X_scaled)[0, 1])

    # RA checks
    priv = run_privacy_check(input_dict)
    sec  = run_security_check(input_dict, raw_size_bytes=int(raw_request.headers.get("content-length", 0)))
    rel  = run_reliability_check(input_dict, prediction_probability=prob)

    # Explanation
    expl = explain_prediction(clf, scaler, row_df, feats)

    # Fairness (from pre-computed test set)
    fairness = load_fairness_results_from_test_set(bundle)
    fw = fairness.fairness_warning if fairness.success else False

    # Risk
    risk = compute_risk(
        prediction=pred,
        prediction_probability=prob,
        explanation=expl,
        privacy_check=priv,
        security_check=sec,
        reliability_check=rel,
        fairness_warning=fw,
    )

    # Auto-resolve only if routing is not HUMAN_REVIEW
    auto_resolve = risk.routing != RoutingOutcome.HUMAN_REVIEW

    # Audit log
    decision_id = audit.log_decision(
        input_snapshot=input_dict,
        original_prediction=pred,
        prediction_prob=prob,
        risk_level=risk.risk_level.value,
        risk_score=risk.risk_score,
        risk_flags=risk.risk_flags,
        explanation_summary=expl.human_readable_summary,
        privacy_passed=priv.passed,
        security_passed=sec.passed,
        reliability_passed=rel.passed,
        fairness_warning=fw,
        routing_outcome=risk.routing.value,
        routing_reason=risk.routing_reason,
        auto_resolve=auto_resolve,
    )

    return PredictResponse(
        decision_id=decision_id,
        prediction=pred,
        prediction_label="Approved" if pred == 1 else "Rejected",
        prediction_prob=round(prob, 4),
        risk_level=risk.risk_level.value,
        risk_score=risk.risk_score,
        risk_flags=[f["flag"] for f in risk.risk_flags],
        routing=risk.routing.value,
        routing_reason=risk.routing_reason,
        explanation_summary=expl.human_readable_summary,
        privacy_passed=priv.passed,
        security_passed=sec.passed,
        reliability_passed=rel.passed,
        fairness_warning=fw,
    )


@app.get("/decisions", tags=["Audit"])
def list_decisions(
    limit: int = 50,
    offset: int = 0,
    risk_level: Optional[str] = None,
    routing: Optional[str] = None,
    review_status: Optional[str] = None,
    _key: str = Depends(require_api_key),
):
    return audit.get_all_decisions(
        limit=limit, offset=offset,
        risk_level=risk_level, routing=routing, review_status=review_status,
    )


@app.get("/decisions/{decision_id}", tags=["Audit"])
def get_decision(decision_id: str, _key: str = Depends(require_api_key)):
    dec = audit.get_decision(decision_id)
    if dec is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    return dec


@app.get("/review/pending", tags=["Review"])
def pending_reviews(_key: str = Depends(require_api_key)):
    return audit.get_pending_reviews()


@app.post("/review/{decision_id}", tags=["Review"])
def submit_review(
    decision_id: str,
    req: ReviewRequest,
    _key: str = Depends(require_api_key),
):
    ok = audit.record_review(
        decision_id=decision_id,
        reviewer_id=req.reviewer_id,
        action=req.action,
        comments=req.comments or "",
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Decision not found.")
    return {"status": "ok", "decision_id": decision_id, "action": req.action}


@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    metrics_path = Path(MODEL_PATH).parent / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found.  Run train_model.py first.")
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/fairness", tags=["Evaluation"])
def get_fairness():
    bundle = get_model_bundle()
    result = load_fairness_results_from_test_set(bundle)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    return {
        "overall_selection_rate":        result.overall_selection_rate,
        "demographic_parity_difference": result.demographic_parity_difference,
        "equalized_odds_difference":     result.equalized_odds_difference,
        "fairness_warning":              result.fairness_warning,
        "warning_message":               result.warning_message,
        "threshold_used":                result.threshold_used,
        "n_samples":                     result.n_samples,
        "group_metrics": [
            {"group": g.group, "size": g.size,
             "selection_rate": g.selection_rate, "accuracy": g.accuracy}
            for g in result.group_metrics
        ],
        "metric_definitions": result.metric_definitions,
        "caveat": result.caveat,
    }


@app.get("/stats", tags=["System"])
def get_stats():
    return audit.get_summary_stats()


class TextAuditRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


@app.post("/analyze-text", tags=["NLP Audit"])
def analyze_text_endpoint(req: TextAuditRequest):
    from text_analyzer import analyze_ai_text
    res = analyze_ai_text(req.text)
    return {
        "detected_decision": res.detected_decision,
        "decision_confidence": res.decision_confidence,
        "decision_snippet": res.decision_snippet,
        "requires_human_review": res.requires_human_review,
        "review_urgency": res.review_urgency,
        "fairness_status": res.fairness_status,
        "fairness_flags": res.fairness_flags,
        "reasons_for_review": res.reasons_for_review,
        "positive_indicators": res.positive_indicators,
        "summary": res.summary,
        "future_scope": res.future_scope,
    }
