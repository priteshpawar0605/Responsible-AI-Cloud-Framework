"""tests/test_risk_engine.py — Tests for risk scoring and decision routing."""

import pytest
from unittest.mock import MagicMock

from explainability import ExplanationResult
from responsible_ai import PrivacyCheckResult, ReliabilityCheckResult, SecurityCheckResult
from risk_engine import RiskLevel, RoutingOutcome, compute_risk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checks(
    privacy_ok=True,
    security_ok=True,
    reliability_ok=True,
    low_confidence=False,
    missing=False,
    out_of_range=False,
):
    priv = PrivacyCheckResult(passed=privacy_ok, concerns=[] if privacy_ok else ["concern"])
    sec  = SecurityCheckResult(passed=security_ok, failures=[] if security_ok else ["fail"])
    rel  = ReliabilityCheckResult(
        passed=reliability_ok,
        low_confidence_detected=low_confidence,
        missing_values_detected=missing,
        out_of_range_detected=out_of_range,
        prediction_probability=0.5 if low_confidence else 0.9,
        issues=[] if reliability_ok else ["issue"],
    )
    return priv, sec, rel


def _ok_explanation():
    return ExplanationResult(success=True, human_readable_summary="OK")


def _failed_explanation():
    return ExplanationResult(success=False, error_message="SHAP failed")


# ---------------------------------------------------------------------------
# Tests: low-risk path
# ---------------------------------------------------------------------------

def test_low_risk_auto_approved():
    priv, sec, rel = _make_checks()
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    assert result.risk_level == RiskLevel.LOW
    assert result.routing    == RoutingOutcome.AUTO_APPROVED


def test_low_risk_auto_rejected():
    priv, sec, rel = _make_checks()
    result = compute_risk(
        prediction=0, prediction_probability=0.1,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    assert result.routing == RoutingOutcome.AUTO_REJECTED


# ---------------------------------------------------------------------------
# Tests: individual risk flags
# ---------------------------------------------------------------------------

def test_high_uncertainty_raises_score():
    priv, sec, rel = _make_checks(low_confidence=True, reliability_ok=False)
    result = compute_risk(
        prediction=1, prediction_probability=0.51,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    flag_names = [f["flag"] for f in result.risk_flags]
    assert "high_uncertainty" in flag_names


def test_fairness_warning_raises_score():
    priv, sec, rel = _make_checks()
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=True,
    )
    flag_names = [f["flag"] for f in result.risk_flags]
    assert "fairness_warning" in flag_names


def test_privacy_failure_raises_score():
    priv, sec, rel = _make_checks(privacy_ok=False)
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    flag_names = [f["flag"] for f in result.risk_flags]
    assert "privacy_concern" in flag_names


def test_security_failure_routes_to_review():
    priv, sec, rel = _make_checks(security_ok=False)
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    assert result.routing == RoutingOutcome.HUMAN_REVIEW


def test_invalid_input_raises_score():
    priv, sec, rel = _make_checks(reliability_ok=False, out_of_range=True)
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    flag_names = [f["flag"] for f in result.risk_flags]
    assert "invalid_input" in flag_names


def test_missing_assessment_raises_score():
    priv, sec, rel = _make_checks()
    result = compute_risk(
        prediction=1, prediction_probability=0.9,
        explanation=_failed_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    flag_names = [f["flag"] for f in result.risk_flags]
    assert "missing_assessment" in flag_names


# ---------------------------------------------------------------------------
# Tests: combined high-risk
# ---------------------------------------------------------------------------

def test_multiple_flags_produce_high_risk():
    priv, sec, rel = _make_checks(
        privacy_ok=False, security_ok=False,
        low_confidence=True, reliability_ok=False,
    )
    result = compute_risk(
        prediction=1, prediction_probability=0.51,
        explanation=_failed_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=True,
    )
    assert result.risk_level == RiskLevel.HIGH
    assert result.routing    == RoutingOutcome.HUMAN_REVIEW


def test_risk_score_is_non_negative():
    priv, sec, rel = _make_checks()
    result = compute_risk(
        prediction=0, prediction_probability=0.1,
        explanation=_ok_explanation(),
        privacy_check=priv, security_check=sec, reliability_check=rel,
        fairness_warning=False,
    )
    assert result.risk_score >= 0
