"""tests/test_text_analyzer.py — Tests for NLP AI text decision auditor."""

import pytest
from text_analyzer import analyze_ai_text


def test_fair_approval_detected():
    text = (
        "The loan application is approved. The applicant has an annual income "
        "of $75,000, credit score of 720, and low existing debt."
    )
    res = analyze_ai_text(text)
    assert res.detected_decision == "APPROVED"
    assert res.requires_human_review is False
    assert len(res.fairness_flags) == 0
    assert "PASSED" in res.fairness_status


def test_biased_rejection_detected():
    text = (
        "The application is rejected. The applicant is an elderly single female "
        "living in an inner-city neighborhood."
    )
    res = analyze_ai_text(text)
    assert res.detected_decision == "REJECTED"
    assert res.requires_human_review is True
    assert res.review_urgency == "IMMEDIATE"
    assert len(res.fairness_flags) > 0
    assert "WARNING" in res.fairness_status


def test_ambiguous_outcome_routes_to_review():
    text = "The application might be accepted or denied. Data is conflicting."
    res = analyze_ai_text(text)
    assert res.requires_human_review is True
    assert "AMBIGUOUS" in res.detected_decision or "UNCERTAIN" in res.detected_decision


def test_empty_input_handled_safely():
    res = analyze_ai_text("")
    assert res.requires_human_review is True
