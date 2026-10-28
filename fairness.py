"""
fairness.py — Fairlearn-based group fairness evaluation.

Important design decisions
--------------------------
- Fairness is evaluated at the dataset/batch level using held-out test data
  and ground-truth labels.  It is NOT computed from a single prediction.
- The demographic_group column is used only here, never as a model feature.
- Metric results above the configured threshold generate a warning, but a
  warning is not evidence of discrimination — it is a flag for further
  investigation.
- Synthetic-data results do not establish or disprove real-world fairness.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from config import DEMOGRAPHIC_COLUMN, FAIRNESS_THRESHOLD, FEATURE_COLUMNS


@dataclass
class GroupMetrics:
    group: str
    size: int
    selection_rate: float
    accuracy: Optional[float] = None


@dataclass
class FairnessResult:
    """Structured result of a batch fairness evaluation."""
    success: bool
    overall_selection_rate: float = 0.0
    group_metrics: List[GroupMetrics] = field(default_factory=list)
    demographic_parity_difference: Optional[float] = None
    equalized_odds_difference: Optional[float] = None
    fairness_warning: bool = False
    warning_message: str = ""
    threshold_used: float = FAIRNESS_THRESHOLD
    n_samples: int = 0
    error_message: str = ""
    caveat: str = (
        "Fairness metrics compare group-level statistical outcomes. "
        "A metric above the threshold is a signal for further investigation, "
        "not evidence of discrimination.  Synthetic-data results do not "
        "establish or disprove real-world fairness."
    )
    metric_definitions: dict = field(default_factory=lambda: {
        "selection_rate": (
            "Fraction of applicants in each group who received a positive "
            "prediction (approved)."
        ),
        "demographic_parity_difference": (
            "Maximum difference in selection rate between any two groups. "
            "A value of 0 means all groups have the same selection rate."
        ),
        "equalized_odds_difference": (
            "Maximum difference in true-positive rate or false-positive rate "
            "between groups (requires ground-truth labels)."
        ),
    })


def evaluate_fairness(
    predictions: np.ndarray,
    demographic_groups: pd.Series,
    y_true: Optional[np.ndarray] = None,
    threshold: float = FAIRNESS_THRESHOLD,
) -> FairnessResult:
    """
    Evaluate group-level fairness metrics on a batch of predictions.

    Parameters
    ----------
    predictions : array of predicted labels (0/1)
    demographic_groups : Series of group identifiers, aligned with predictions
    y_true : optional array of ground-truth labels (enables equalized odds)
    threshold : demographic-parity-difference value that triggers a warning

    Returns
    -------
    FairnessResult
    """
    if not FAIRLEARN_AVAILABLE:
        return FairnessResult(
            success=False,
            error_message="Fairlearn is not installed.  Run: pip install fairlearn",
        )

    try:
        predictions = np.array(predictions)
        groups = np.array(demographic_groups)
        n = len(predictions)

        # Overall selection rate
        overall_sr = float(predictions.mean())

        # Per-group metrics
        group_metrics_list: List[GroupMetrics] = []
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            gp = predictions[mask]
            sr = float(gp.mean()) if len(gp) > 0 else 0.0
            acc = None
            if y_true is not None:
                yt = np.array(y_true)[mask]
                acc = float((gp == yt).mean()) if len(yt) > 0 else None
            group_metrics_list.append(
                GroupMetrics(group=str(g), size=int(mask.sum()),
                             selection_rate=round(sr, 4), accuracy=acc)
            )

        # Demographic parity difference
        dpd = float(demographic_parity_difference(
            y_true=predictions,          # we treat predictions as the "outcome"
            y_pred=predictions,
            sensitive_features=groups,
        ))

        # Equalized odds difference (requires ground-truth labels)
        eod = None
        if y_true is not None:
            try:
                eod = float(equalized_odds_difference(
                    y_true=np.array(y_true),
                    y_pred=predictions,
                    sensitive_features=groups,
                ))
            except Exception:
                eod = None

        # Warning
        warning = abs(dpd) > threshold
        warning_msg = ""
        if warning:
            worst_high = max(group_metrics_list, key=lambda g: g.selection_rate)
            worst_low  = min(group_metrics_list, key=lambda g: g.selection_rate)
            warning_msg = (
                f"Demographic parity difference ({dpd:.4f}) exceeds the "
                f"configured threshold ({threshold}).  "
                f"Group '{worst_high.group}' has the highest selection rate "
                f"({worst_high.selection_rate:.2%}) and "
                f"'{worst_low.group}' the lowest "
                f"({worst_low.selection_rate:.2%}).  "
                "This warrants further investigation but does not by itself "
                "indicate discrimination."
            )

        return FairnessResult(
            success=True,
            overall_selection_rate=round(overall_sr, 4),
            group_metrics=group_metrics_list,
            demographic_parity_difference=round(dpd, 6),
            equalized_odds_difference=round(eod, 6) if eod is not None else None,
            fairness_warning=warning,
            warning_message=warning_msg,
            threshold_used=threshold,
            n_samples=n,
        )

    except Exception as exc:
        return FairnessResult(
            success=False,
            error_message=f"Fairness evaluation failed: {exc}\n{traceback.format_exc()}",
        )


def load_fairness_results_from_test_set(model_bundle: dict) -> FairnessResult:
    """
    Convenience function: run fairness evaluation on the saved test set.

    This is what the dashboard and API use so that fairness is always computed
    from held-out batch data with ground-truth labels, not from one prediction.
    """
    from pathlib import Path
    test_path = Path(__file__).parent / "data" / "test_set.csv"
    if not test_path.exists():
        return FairnessResult(
            success=False,
            error_message="Test set not found.  Run train_model.py first.",
        )

    df = pd.read_csv(test_path)
    clf   = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["feature_columns"]

    X_scaled = scaler.transform(df[features])
    preds = clf.predict(X_scaled)
    y_true = df["approved"].values
    groups = df[DEMOGRAPHIC_COLUMN]

    return evaluate_fairness(preds, groups, y_true=y_true)
