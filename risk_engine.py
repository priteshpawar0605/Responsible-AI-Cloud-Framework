"""
risk_engine.py — Transparent, configurable risk scoring and decision routing.

How it works
------------
Each assessment check contributes a documented point value to a cumulative
risk score.  When the total exceeds the RISK_SCORE_HIGH threshold the decision
is routed for human review.  When it exceeds RISK_SCORE_MEDIUM it is flagged
but may proceed automatically in the demonstration workflow.

Routing rules are defined in one place (ROUTING_RULES) so that students and
reviewers can inspect them without reading business logic spread across modules.

Principle: a missing check is treated as a risk factor, not as safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import RISK_SCORE_HIGH, RISK_SCORE_MEDIUM, RISK_WEIGHTS
from explainability import ExplanationResult
from responsible_ai import PrivacyCheckResult, ReliabilityCheckResult, SecurityCheckResult


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class RoutingOutcome(str, Enum):
    AUTO_APPROVED  = "AUTO_APPROVED"   # Low risk, model prediction accepted
    AUTO_REJECTED  = "AUTO_REJECTED"   # Low risk, model prediction (rejection) accepted
    FLAGGED        = "FLAGGED"         # Medium risk, flagged for attention
    HUMAN_REVIEW   = "HUMAN_REVIEW"    # High risk, routed to human reviewer


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessmentResult:
    """
    Complete risk assessment for one prediction request.

    Fields
    ------
    risk_score    : cumulative score from all active risk factors
    risk_level    : LOW / MEDIUM / HIGH (derived from score thresholds)
    risk_flags    : list of (flag_name, score_contribution, description)
    routing       : how the decision should be handled
    routing_reason: plain-English explanation of the routing decision
    """
    risk_score: int
    risk_level: RiskLevel
    risk_flags: List[Dict] = field(default_factory=list)
    routing: RoutingOutcome = RoutingOutcome.AUTO_APPROVED
    routing_reason: str = ""
    threshold_high: int = RISK_SCORE_HIGH
    threshold_medium: int = RISK_SCORE_MEDIUM

    def to_dict(self) -> dict:
        return {
            "risk_score":     self.risk_score,
            "risk_level":     self.risk_level.value,
            "risk_flags":     self.risk_flags,
            "routing":        self.routing.value,
            "routing_reason": self.routing_reason,
            "threshold_high":   self.threshold_high,
            "threshold_medium": self.threshold_medium,
        }


# ---------------------------------------------------------------------------
# Routing rules (single source of truth)
# ---------------------------------------------------------------------------
#
# Each entry describes a risk factor:
#   key          : matches a key in the assessment inputs
#   weight       : points added when this flag is active
#   description  : plain-English reason shown in the UI and audit log
#
ROUTING_RULES = {
    "high_uncertainty": {
        "weight":      RISK_WEIGHTS["high_uncertainty"],
        "description": (
            "The model's prediction probability is near the decision boundary "
            "(below the configured uncertainty threshold).  The model is not "
            "confident; a human reviewer should confirm."
        ),
    },
    "fairness_warning": {
        "weight":      RISK_WEIGHTS["fairness_warning"],
        "description": (
            "The batch fairness evaluation found a demographic-parity difference "
            "above the configured threshold.  This does not prove individual bias "
            "but warrants review."
        ),
    },
    "privacy_concern": {
        "weight":      RISK_WEIGHTS["privacy_concern"],
        "description": (
            "The privacy check identified one or more unexpected or potentially "
            "sensitive fields in the request."
        ),
    },
    "security_failure": {
        "weight":      RISK_WEIGHTS["security_failure"],
        "description": (
            "The security check detected disallowed characters, oversized payload, "
            "or other input validation failures."
        ),
    },
    "invalid_input": {
        "weight":      RISK_WEIGHTS["invalid_input"],
        "description": (
            "One or more feature values are missing, non-numeric, or outside the "
            "expected domain range."
        ),
    },
    "missing_assessment": {
        "weight":      RISK_WEIGHTS["missing_assessment"],
        "description": (
            "One or more Responsible AI checks could not be completed.  An "
            "incomplete assessment is treated as a risk factor, not as safe."
        ),
    },
}


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------

def compute_risk(
    prediction: int,
    prediction_probability: float,
    explanation: ExplanationResult,
    privacy_check: PrivacyCheckResult,
    security_check: SecurityCheckResult,
    reliability_check: ReliabilityCheckResult,
    fairness_warning: bool = False,
) -> RiskAssessmentResult:
    """
    Compute the risk score and route the decision.

    Parameters
    ----------
    prediction            : 0 or 1 (model output)
    prediction_probability: probability of the positive class
    explanation           : result of SHAP explanation
    privacy_check         : result of privacy assessment
    security_check        : result of security validation
    reliability_check     : result of reliability check
    fairness_warning      : whether the batch fairness evaluation flagged a warning

    Returns
    -------
    RiskAssessmentResult
    """
    score = 0
    flags: List[Dict] = []

    def add_flag(key: str, condition: bool, detail: str = ""):
        nonlocal score
        if condition:
            rule = ROUTING_RULES[key]
            score += rule["weight"]
            flags.append({
                "flag":        key,
                "weight":      rule["weight"],
                "description": rule["description"],
                "detail":      detail,
            })

    # --- Evaluate each risk factor ---

    # 1. Model uncertainty
    confidence = max(prediction_probability, 1 - prediction_probability)
    add_flag(
        "high_uncertainty",
        reliability_check.low_confidence_detected,
        f"Confidence={confidence:.4f}, threshold={reliability_check.uncertainty_threshold}",
    )

    # 2. Fairness warning (from batch evaluation)
    add_flag(
        "fairness_warning",
        fairness_warning,
        "Demographic-parity difference exceeds configured threshold on test set.",
    )

    # 3. Privacy concern
    add_flag(
        "privacy_concern",
        not privacy_check.passed,
        "; ".join(privacy_check.concerns),
    )

    # 4. Security failure
    add_flag(
        "security_failure",
        not security_check.passed,
        "; ".join(security_check.failures),
    )

    # 5. Invalid/out-of-range input
    add_flag(
        "invalid_input",
        (reliability_check.missing_values_detected or
         reliability_check.out_of_range_detected),
        "; ".join(reliability_check.issues),
    )

    # 6. Missing assessment (failed explainability or other check)
    missing_assessment = not explanation.success
    add_flag(
        "missing_assessment",
        missing_assessment,
        explanation.error_message[:200] if not explanation.success else "",
    )

    # --- Derive risk level ---
    if score >= RISK_SCORE_HIGH:
        level = RiskLevel.HIGH
    elif score >= RISK_SCORE_MEDIUM:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    # --- Routing ---
    if level == RiskLevel.HIGH:
        routing = RoutingOutcome.HUMAN_REVIEW
        reason  = (
            f"Risk score {score} ≥ high threshold {RISK_SCORE_HIGH}.  "
            "Decision routed to human reviewer.  Active flags: "
            + ", ".join(f["flag"] for f in flags) + "."
        )
    elif level == RiskLevel.MEDIUM:
        routing = RoutingOutcome.FLAGGED
        reason  = (
            f"Risk score {score} ≥ medium threshold {RISK_SCORE_MEDIUM}.  "
            "Decision is flagged for attention but may proceed in the demonstration "
            "workflow.  Active flags: "
            + ", ".join(f["flag"] for f in flags) + "."
        )
    else:
        routing = (
            RoutingOutcome.AUTO_APPROVED if prediction == 1
            else RoutingOutcome.AUTO_REJECTED
        )
        reason = (
            f"Risk score {score} is below both thresholds.  "
            "No risk factors detected; decision proceeds automatically."
        )

    return RiskAssessmentResult(
        risk_score=score,
        risk_level=level,
        risk_flags=flags,
        routing=routing,
        routing_reason=reason,
    )
