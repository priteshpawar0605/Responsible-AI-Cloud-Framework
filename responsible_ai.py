"""
responsible_ai.py — Privacy, security, and reliability assessment checks.

Each check returns a structured result that is recorded in the audit log.

Limitations
-----------
- These are basic prototype checks.  They do not constitute a production
  security audit or legal privacy compliance assessment.
- Security check results are recorded as informational flags, not guarantees.
- Privacy checks verify that unnecessary sensitive fields are absent; they do
  not perform PII detection on free-text fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import FEATURE_COLUMNS, FEATURE_RANGES, UNCERTAINTY_THRESHOLD


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class PrivacyCheckResult:
    passed: bool
    concerns: List[str] = field(default_factory=list)
    checked_fields: List[str] = field(default_factory=list)
    caveat: str = (
        "Privacy checks verify that unnecessary fields are absent and that "
        "inputs match expected formats.  They do not constitute GDPR or legal "
        "compliance validation."
    )


@dataclass
class SecurityCheckResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    caveat: str = (
        "Security checks apply basic input validation and API authentication. "
        "This prototype is not hardened for production deployment."
    )


@dataclass
class ReliabilityCheckResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    missing_values_detected: bool = False
    out_of_range_detected: bool = False
    low_confidence_detected: bool = False
    prediction_probability: Optional[float] = None
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD
    caveat: str = (
        "Prediction probability reflects the model's internal score, not a "
        "calibrated confidence interval.  Low probability does not automatically "
        "mean the prediction is wrong."
    )


# ---------------------------------------------------------------------------
# Sensitive field patterns to reject
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERNS = [
    r"\bssn\b", r"\bsocial.?security\b", r"\bpassport\b",
    r"\bdate.?of.?birth\b", r"\bdob\b", r"\bfull.?name\b",
    r"\bemail\b", r"\bphone\b", r"\baddress\b",
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Privacy check
# ---------------------------------------------------------------------------

def run_privacy_check(input_data: Dict[str, Any]) -> PrivacyCheckResult:
    """
    Verify that the input does not contain unnecessary sensitive fields.

    Allowed fields are the model feature columns plus 'applicant_id' (an
    opaque identifier).  Any additional field that matches a sensitive-data
    pattern triggers a concern.
    """
    allowed_fields = set(FEATURE_COLUMNS) | {"applicant_id", "demographic_group"}
    concerns: List[str] = []
    checked: List[str] = list(input_data.keys())

    # Check for unexpected sensitive-sounding fields
    for key in input_data:
        if key not in allowed_fields and _SENSITIVE_RE.search(key):
            concerns.append(
                f"Field '{key}' appears to contain sensitive personal data "
                "and was not expected in the request."
            )

    # Check for unexpected non-numeric values in feature fields
    for feat in FEATURE_COLUMNS:
        val = input_data.get(feat)
        if val is not None and not isinstance(val, (int, float)):
            concerns.append(
                f"Feature '{feat}' has non-numeric value '{val}'; "
                "this may indicate a data-type issue."
            )

    return PrivacyCheckResult(
        passed=len(concerns) == 0,
        concerns=concerns,
        checked_fields=checked,
    )


# ---------------------------------------------------------------------------
# Security check
# ---------------------------------------------------------------------------

def run_security_check(
    input_data: Dict[str, Any],
    raw_size_bytes: int = 0,
    max_size_bytes: int = 65_536,
) -> SecurityCheckResult:
    """
    Apply basic input security checks.

    Checks
    ------
    - Request payload does not exceed the configured size limit.
    - Feature values are numeric (strings could indicate injection).
    - Values do not contain SQL/shell metacharacters.
    - No obviously malformed JSON keys.
    """
    failures: List[str] = []
    warnings: List[str] = []

    # Size guard
    if raw_size_bytes > max_size_bytes:
        failures.append(
            f"Request payload ({raw_size_bytes} bytes) exceeds the "
            f"maximum allowed size ({max_size_bytes} bytes)."
        )

    # Type and injection check on feature values
    dangerous = re.compile(r"[;'\"\\/\x00-\x1f]")
    for key, val in input_data.items():
        if isinstance(val, str):
            if dangerous.search(val):
                failures.append(
                    f"Field '{key}' contains potentially dangerous characters."
                )
            else:
                warnings.append(
                    f"Field '{key}' is a string; expected numeric for model features."
                )

    # Key name safety
    for key in input_data:
        if dangerous.search(key):
            failures.append(f"Field name '{key}' contains disallowed characters.")

    passed = len(failures) == 0
    return SecurityCheckResult(passed=passed, failures=failures, warnings=warnings)


# ---------------------------------------------------------------------------
# Reliability check
# ---------------------------------------------------------------------------

def run_reliability_check(
    input_data: Dict[str, Any],
    prediction_probability: Optional[float] = None,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> ReliabilityCheckResult:
    """
    Check input quality and model confidence.

    Checks
    ------
    - No missing (None/NaN) feature values.
    - Feature values within configured domain ranges.
    - Prediction probability above the uncertainty threshold.
    """
    import math

    issues: List[str] = []
    missing = False
    out_of_range = False
    low_conf = False

    # Missing values
    for feat in FEATURE_COLUMNS:
        val = input_data.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            issues.append(f"Feature '{feat}' is missing or NaN.")
            missing = True

    # Range validation
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = input_data.get(feat)
        if val is not None:
            try:
                v = float(val)
                if not (lo <= v <= hi):
                    issues.append(
                        f"Feature '{feat}' value {v} is outside the expected "
                        f"range [{lo}, {hi}]."
                    )
                    out_of_range = True
            except (TypeError, ValueError):
                issues.append(f"Feature '{feat}' is not numeric.")
                out_of_range = True

    # Confidence check
    if prediction_probability is not None:
        # The "confidence" here is max(p, 1-p) — how sure the model is
        confidence = max(prediction_probability, 1 - prediction_probability)
        if confidence < threshold:
            issues.append(
                f"Model probability ({prediction_probability:.4f}) is near the "
                f"decision boundary (confidence {confidence:.4f} < "
                f"threshold {threshold}).  The prediction is uncertain."
            )
            low_conf = True

    return ReliabilityCheckResult(
        passed=len(issues) == 0,
        issues=issues,
        missing_values_detected=missing,
        out_of_range_detected=out_of_range,
        low_confidence_detected=low_conf,
        prediction_probability=prediction_probability,
        uncertainty_threshold=threshold,
    )
