"""
explainability.py — SHAP-based individual prediction explanations.

Provides human-readable, per-prediction feature-contribution explanations.

Important caveats communicated to users
----------------------------------------
- SHAP values show which features pushed the model's prediction up or down
  relative to the expected value.  They do NOT prove that the decision is
  correct or fair.
- SHAP TreeExplainer is used for tree-based models.  If the model type
  changes, the explainer selection must be updated.
- Explanation errors are handled gracefully and recorded in the audit log.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class ExplanationResult:
    """Structured result of a SHAP explanation attempt."""
    success: bool
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    top_positive_features: List[str] = field(default_factory=list)
    top_negative_features: List[str] = field(default_factory=list)
    base_value: Optional[float] = None
    prediction_value: Optional[float] = None
    human_readable_summary: str = ""
    error_message: str = ""
    caveat: str = (
        "SHAP values indicate which features most influenced this model output. "
        "They do not prove the decision is correct, fair, or causally valid."
    )


def explain_prediction(
    model,
    scaler,
    input_data: pd.DataFrame,
    feature_columns: List[str],
    max_features: int = 5,
) -> ExplanationResult:
    """
    Generate a SHAP explanation for a single prediction.

    Parameters
    ----------
    model : fitted sklearn estimator
    scaler : fitted sklearn scaler
    input_data : DataFrame with one row and feature_columns columns
    feature_columns : list of feature names
    max_features : number of top contributors to include in the summary

    Returns
    -------
    ExplanationResult
    """
    if not SHAP_AVAILABLE:
        return ExplanationResult(
            success=False,
            error_message="SHAP is not installed.  Run: pip install shap",
        )

    try:
        # Scale the input (must match training pipeline)
        X_scaled = scaler.transform(input_data[feature_columns])

        # Select explainer based on model type
        if hasattr(model, "estimators_"):
            # Tree-based ensemble (Random Forest, Gradient Boosting, etc.)
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_scaled)

            # For binary classification, shap_values is a list of two arrays
            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                shap_vals = shap_values_raw[1][0]  # positive class, first sample
                base_val  = explainer.expected_value[1]
            else:
                shap_vals = shap_values_raw[0]
                base_val  = (
                    explainer.expected_value[0]
                    if hasattr(explainer.expected_value, "__len__")
                    else explainer.expected_value
                )
        else:
            # Fallback: KernelExplainer (slower, model-agnostic)
            background = shap.maskers.Independent(X_scaled, max_samples=50)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], background
            )
            shap_vals = explainer.shap_values(X_scaled, nsamples=100)[0]
            base_val  = explainer.expected_value

        # Map feature names to SHAP values
        contributions = {
            col: round(float(shap_vals[i]), 6)
            for i, col in enumerate(feature_columns)
        }

        # Sort by absolute contribution
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        positive = [k for k, v in sorted_feats if v > 0][:max_features]
        negative = [k for k, v in sorted_feats if v < 0][:max_features]

        # Human-readable summary
        pred_value = float(base_val) + float(np.sum(shap_vals))
        summary_parts = []
        for feat, val in sorted_feats[:max_features]:
            direction = "increased" if val > 0 else "decreased"
            summary_parts.append(
                f"{feat} ({direction} approval probability by "
                f"{abs(val):.4f})"
            )
        summary = (
            f"Top {min(max_features, len(sorted_feats))} features influencing "
            f"this prediction: {'; '.join(summary_parts)}."
        )

        return ExplanationResult(
            success=True,
            feature_contributions=contributions,
            top_positive_features=positive,
            top_negative_features=negative,
            base_value=round(float(base_val), 6),
            prediction_value=round(pred_value, 6),
            human_readable_summary=summary,
        )

    except Exception as exc:
        return ExplanationResult(
            success=False,
            error_message=f"Explanation failed: {exc}\n{traceback.format_exc()}",
        )
