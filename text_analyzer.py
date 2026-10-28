"""
text_analyzer.py — NLP/Text audit analyzer for AI-generated decisions.

Analyzes unstructured AI-generated text (e.g., LLM responses, automated rejection
letters, explanation notes) to evaluate:
1. What decision the AI made (Approved, Rejected, Uncertain).
2. Whether fairness/bias red flags or protected demographic attributes exist.
3. Whether the decision should be routed for Human Review.
4. Future scope recommendations for cloud deployment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextAuditResult:
    detected_decision: str                # APPROVED, REJECTED, UNCERTAIN, AMBIGUOUS
    decision_confidence: str              # HIGH, MEDIUM, LOW
    decision_snippet: str                 # Sentence where decision was detected
    requires_human_review: bool           # True / False
    review_urgency: str                   # IMMEDIATE, RECOMMENDED, NOT REQUIRED
    fairness_status: str                  # FAIR / WARNING / BIASED
    fairness_flags: List[str] = field(default_factory=list)
    reasons_for_review: List[str] = field(default_factory=list)
    positive_indicators: List[str] = field(default_factory=list)
    summary: str = ""
    future_scope: str = (
        "In future releases, this text auditor can be deployed as an LLM-as-a-Judge "
        "microservice or sidecar proxy in cloud environments (AWS/GCP/Azure) to automatically "
        "screen all model explanations and customer correspondence before delivery."
    )


# Patterns for decision extraction
_APPROVE_PATTERNS = [
    r"\b(approv(ed|al|e)?|accepted?|granted?|eligible|sanctioned?|qualified)\b"
]
_REJECT_PATTERNS = [
    r"\b(reject(ed|ion|s)?|deni(ed|al|es)?|declin(ed|es)?|ineligible|disqualified|not approved)\b"
]

# Sensitive/protected demographic attributes mentioned in rationale
_DEMOGRAPHIC_PATTERNS = {
    "Gender / Sex": r"\b(gender|male|female|woman|man|sex|maternity|paternity|marital status|married|single)\b",
    "Race / Ethnicity / Caste": r"\b(race|ethnicity|black|white|hispanic|latino|asian|caste|minority|native|tribe)\b",
    "Age": r"\b(age|elderly|old applicant|too young|retiree|pensioner|senior citizen|years old)\b",
    "Religion / Beliefs": r"\b(religion|religious|faith|church|mosque|temple|creed)\b",
    "Disability / Health": r"\b(disability|handicap|chronic illness|medical condition|pregnant)\b",
    "Geographic / Proxy Bias": r"\b(neighborhood|zip code|postal code|slum|inner city|foreign national|immigrant|citizenship)\b",
}

# Uncertainty patterns
_UNCERTAINTY_PATTERNS = [
    r"\b(uncertain|not sure|unclear|inconclusive|maybe|might be|possibly|conflicting data|insufficient data|borderline)\b",
    r"\b(cannot determine|unable to verify|ambiguous)\b",
]

# Valid financial rationale patterns (positive sign of objective decision-making)
_OBJECTIVE_PATTERNS = [
    r"\b(credit score|income|debt|dti|repayment|employment|balance|collateral|salary|financial)\b"
]


def analyze_ai_text(text: str) -> TextAuditResult:
    """
    Analyze AI-generated decision text for decision intent, fairness, and human review routing.
    """
    if not text or not text.strip():
        return TextAuditResult(
            detected_decision="NO TEXT PROVIDED",
            decision_confidence="LOW",
            decision_snippet="N/A",
            requires_human_review=True,
            review_urgency="IMMEDIATE",
            fairness_status="WARNING",
            reasons_for_review=["Empty input received; cannot perform AI audit."],
            summary="Please enter AI-generated text to analyze.",
        )

    clean_text = text.strip()
    sentences = re.split(r"[.\n;!?]+", clean_text)
    lower_text = clean_text.lower()

    # 1. Decision detection
    has_approval = any(re.search(p, lower_text) for p in _APPROVE_PATTERNS)
    has_rejection = any(re.search(p, lower_text) for p in _REJECT_PATTERNS)

    detected_decision = "UNCERTAIN"
    decision_snippet = ""

    if has_approval and not has_rejection:
        detected_decision = "APPROVED"
    elif has_rejection and not has_approval:
        detected_decision = "REJECTED"
    elif has_approval and has_rejection:
        detected_decision = "AMBIGUOUS (Both Approve and Reject terms detected)"
    else:
        detected_decision = "UNDETERMINED (No clear decision verb found)"

    # Find the most relevant decision sentence
    for s in sentences:
        s_lower = s.lower()
        if any(re.search(p, s_lower) for p in _APPROVE_PATTERNS + _REJECT_PATTERNS):
            decision_snippet = s.strip()
            break

    # 2. Check for fairness & protected attributes
    fairness_flags: List[str] = []
    for category, pattern in _DEMOGRAPHIC_PATTERNS.items():
        matches = re.findall(pattern, lower_text)
        if matches:
            unique_terms = sorted(set(matches))
            fairness_flags.append(
                f"{category} referenced ({', '.join(unique_terms)})"
            )

    # 3. Check for uncertainty
    uncertainty_detected = False
    for pat in _UNCERTAINTY_PATTERNS:
        found = re.findall(pat, lower_text)
        if found:
            uncertainty_detected = True
            break

    # 4. Check for objective financial rationale
    has_objective_basis = any(re.search(p, lower_text) for p in _OBJECTIVE_PATTERNS)
    positive_indicators: List[str] = []
    if has_objective_basis:
        positive_indicators.append("Contains objective financial metrics (e.g. income, credit score, debt).")
    if not fairness_flags:
        positive_indicators.append("No prohibited demographic or protected attributes were cited.")

    # 5. Determine Human Review Routing & Urgency
    reasons_for_review: List[str] = []

    if fairness_flags:
        reasons_for_review.append(
            f"Fairness alert: AI text cites sensitive demographic attributes: {'; '.join(fairness_flags)}."
        )

    if detected_decision in ["AMBIGUOUS (Both Approve and Reject terms detected)", "UNDETERMINED (No clear decision verb found)"]:
        reasons_for_review.append(
            f"Decision ambiguity: AI output does not state an unequivocal decision outcome ({detected_decision})."
        )

    if uncertainty_detected:
        reasons_for_review.append(
            "High model uncertainty: The text indicates doubt, missing data, or inconclusive criteria."
        )

    if not has_objective_basis:
        reasons_for_review.append(
            "Lack of grounded financial justification: No recognized credit or financial factors were cited."
        )

    requires_human_review = len(reasons_for_review) > 0

    if fairness_flags:
        fairness_status = "WARNING (Demographic References Detected)"
        review_urgency = "IMMEDIATE"
    elif uncertainty_detected or detected_decision.startswith("AMBIGUOUS"):
        fairness_status = "UNCERTAIN"
        review_urgency = "RECOMMENDED"
    else:
        fairness_status = "PASSED (No demographic bias detected)"
        review_urgency = "NOT REQUIRED" if not requires_human_review else "RECOMMENDED"

    decision_confidence = "HIGH" if (has_objective_basis and not uncertainty_detected and not fairness_flags) else "MEDIUM"
    if detected_decision.startswith("UNDETERMINED"):
        decision_confidence = "LOW"

    # Build concise human readable summary
    if requires_human_review:
        summary = (
            f"The AI appears to have {detected_decision.lower()}, but this decision CANNOT be safely auto-processed. "
            f"Reason: {reasons_for_review[0]} Human review is {review_urgency.lower()}."
        )
    else:
        summary = (
            f"The AI has decided: {detected_decision}. The rationale references objective financial criteria "
            "without invoking protected demographic attributes. This decision is fair according to policy rules "
            "and can proceed without mandatory human escalation."
        )

    return TextAuditResult(
        detected_decision=detected_decision,
        decision_confidence=decision_confidence,
        decision_snippet=decision_snippet or clean_text[:120] + "...",
        requires_human_review=requires_human_review,
        review_urgency=review_urgency,
        fairness_status=fairness_status,
        fairness_flags=fairness_flags,
        reasons_for_review=reasons_for_review,
        positive_indicators=positive_indicators,
        summary=summary,
    )
