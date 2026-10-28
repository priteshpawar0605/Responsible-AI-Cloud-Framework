"""tests/test_audit.py — Tests for audit logging and review workflow."""

import os
import tempfile
import pytest

# Redirect DB to a temp file for testing
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ["DATABASE_PATH_OVERRIDE"] = _TMP_DB.name


# ---------------------------------------------------------------------------
# Patch DATABASE_PATH before importing audit
# ---------------------------------------------------------------------------
import config
config.DATABASE_PATH = _TMP_DB.name

import audit
audit.DATABASE_PATH = _TMP_DB.name
audit.init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(prediction=1, risk_level="LOW", routing="AUTO_APPROVED", auto=True):
    return audit.log_decision(
        input_snapshot={"income": 55000, "credit_score": 650,
                        "existing_debt": 10000, "employment_duration": 5,
                        "loan_amount": 25000},
        original_prediction=prediction,
        prediction_prob=0.8,
        risk_level=risk_level,
        risk_score=0,
        risk_flags=[],
        explanation_summary="Test explanation",
        privacy_passed=True,
        security_passed=True,
        reliability_passed=True,
        fairness_warning=False,
        routing_outcome=routing,
        routing_reason="Test routing reason",
        auto_resolve=auto,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_log_decision_returns_id():
    did = _log()
    assert isinstance(did, str)
    assert len(did) > 0


def test_get_decision_returns_record():
    did = _log()
    rec = audit.get_decision(did)
    assert rec is not None
    assert rec["decision_id"] == did


def test_original_prediction_preserved_after_review():
    did = _log(prediction=1, risk_level="HIGH", routing="HUMAN_REVIEW", auto=False)
    ok  = audit.record_review(did, "reviewer-1", "REJECT", "Override")
    assert ok

    rec = audit.get_decision(did)
    assert rec["original_prediction"] == 1   # original preserved
    assert rec["final_decision"]       == 0   # reviewer rejected → flip
    assert rec["reviewer_action"]      == "REJECT"


def test_reviewer_approve_keeps_original():
    did = _log(prediction=1, risk_level="HIGH", routing="HUMAN_REVIEW", auto=False)
    audit.record_review(did, "reviewer-2", "APPROVE", "Looks good")
    rec = audit.get_decision(did)
    assert rec["final_decision"] == rec["original_prediction"]


def test_review_status_updated():
    did = _log(risk_level="HIGH", routing="HUMAN_REVIEW", auto=False)
    rec = audit.get_decision(did)
    assert rec["review_status"] == "PENDING"

    audit.record_review(did, "r1", "APPROVE")
    rec = audit.get_decision(did)
    assert rec["review_status"] == "REVIEWED"


def test_get_pending_reviews():
    did = _log(risk_level="HIGH", routing="HUMAN_REVIEW", auto=False)
    pending = audit.get_pending_reviews()
    ids = [p["decision_id"] for p in pending]
    assert did in ids


def test_auto_resolve_sets_na_status():
    did = _log(auto=True)
    rec = audit.get_decision(did)
    assert rec["review_status"] == "NA"
    assert rec["final_decision"] is not None


def test_summary_stats_counts():
    stats = audit.get_summary_stats()
    assert "total_decisions" in stats
    assert "pending_review"  in stats
    assert stats["total_decisions"] >= 0


def test_get_all_decisions_filter_by_risk():
    _log(risk_level="HIGH", routing="HUMAN_REVIEW", auto=False)
    results = audit.get_all_decisions(risk_level="HIGH")
    assert all(r["risk_level"] == "HIGH" for r in results)


def test_nonexistent_decision_returns_none():
    rec = audit.get_decision("00000000-0000-0000-0000-000000000000")
    assert rec is None


def test_review_nonexistent_decision_returns_false():
    ok = audit.record_review("nonexistent-id", "r1", "APPROVE")
    assert ok is False
