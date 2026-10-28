"""tests/test_api.py — Tests for FastAPI endpoints."""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

# Use a temp DB so tests don't pollute the real audit log
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()

import config
config.DATABASE_PATH = _TMP_DB.name

import audit
audit.DATABASE_PATH = _TMP_DB.name

import api as api_module
api_module.audit.DATABASE_PATH = _TMP_DB.name
api_module.audit.init_db()

client = TestClient(api_module.app)

API_KEY = config.API_KEY
HEADERS = {"X-API-Key": API_KEY}

VALID_PAYLOAD = {
    "income":              55000.0,
    "credit_score":        650.0,
    "existing_debt":       10000.0,
    "employment_duration": 5.0,
    "loan_amount":         25000.0,
}


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_public():
    r = client.get("/model-info")
    # Either 200 (model loaded) or 503 (model not trained yet)
    assert r.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def test_predict_requires_auth():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 403


def test_decisions_requires_auth():
    r = client.get("/decisions")
    assert r.status_code == 403


def test_review_pending_requires_auth():
    r = client.get("/review/pending")
    assert r.status_code == 403


def test_invalid_api_key_rejected():
    r = client.post("/predict", json=VALID_PAYLOAD,
                    headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_predict_rejects_credit_score_out_of_range():
    bad = {**VALID_PAYLOAD, "credit_score": 200}  # below 300
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_negative_income():
    bad = {**VALID_PAYLOAD, "income": -1}
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amount"}
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Predict (only if model is trained)
# ---------------------------------------------------------------------------

def test_predict_valid_input():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    if r.status_code == 503:
        pytest.skip("Model not trained — skipping predict test")
    assert r.status_code == 200
    body = r.json()
    assert "decision_id"      in body
    assert "prediction"       in body
    assert "risk_level"       in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["prediction_prob"] <= 1.0


def test_predict_response_has_explanation():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    if r.status_code == 503:
        pytest.skip("Model not trained")
    body = r.json()
    assert "explanation_summary" in body


def test_predict_creates_audit_record():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    if r.status_code == 503:
        pytest.skip("Model not trained")
    did = r.json()["decision_id"]
    r2  = client.get(f"/decisions/{did}", headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["decision_id"] == did


# ---------------------------------------------------------------------------
# Review endpoints
# ---------------------------------------------------------------------------

def test_submit_review_invalid_action():
    r = client.post("/review/fake-id",
                    json={"reviewer_id": "r1", "action": "DANCE"},
                    headers=HEADERS)
    assert r.status_code == 422


def test_submit_review_nonexistent_decision():
    r = client.post(
        "/review/00000000-0000-0000-0000-000000000000",
        json={"reviewer_id": "r1", "action": "APPROVE", "comments": "ok"},
        headers=HEADERS,
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

def test_e2e_predict_risk_audit_review():
    """
    End-to-end: predict → check audit record → submit review → verify final.
    """
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    if r.status_code == 503:
        pytest.skip("Model not trained")

    body = r.json()
    did  = body["decision_id"]

    # Audit record exists
    audit_r = client.get(f"/decisions/{did}", headers=HEADERS)
    assert audit_r.status_code == 200
    rec = audit_r.json()
    assert rec["original_prediction"] in [0, 1]
    assert rec["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    # If routed to human review, submit a review
    if rec["review_status"] == "PENDING":
        rev_r = client.post(
            f"/review/{did}",
            json={"reviewer_id": "test-reviewer", "action": "APPROVE",
                  "comments": "E2E test approval"},
            headers=HEADERS,
        )
        assert rev_r.status_code == 200

        updated = client.get(f"/decisions/{did}", headers=HEADERS).json()
        assert updated["review_status"]   == "REVIEWED"
        assert updated["reviewer_action"] == "APPROVE"
        # Original prediction must be preserved
        assert updated["original_prediction"] == rec["original_prediction"]
