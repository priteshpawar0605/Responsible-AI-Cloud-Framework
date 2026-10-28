"""
evaluate.py — Baseline vs. Responsible AI Framework comparison.

Run with:
    python evaluate.py

What it does
------------
1. Loads the trained model and test set.
2. Runs the baseline workflow (model prediction only, no assessment layer).
3. Runs the full framework workflow (prediction + all RA checks + risk scoring).
4. Compares accuracy, precision, recall, F1, fairness, processing time, and
   review-routing percentage.
5. Saves results to data/evaluation_results.csv and prints a report.

Important notes
---------------
- Results are measured, not invented.  All numbers come from running actual
  code on the synthetic test set.
- Adding Responsible AI checks does not automatically improve model accuracy.
  The comparison shows the overhead, flagging behaviour, and fairness metrics.
- Processing times include Python interpreter overhead and will vary by machine.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR,
    DEMOGRAPHIC_COLUMN,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    TARGET_COLUMN,
    UNCERTAINTY_THRESHOLD,
)
from explainability import explain_prediction
from fairness import evaluate_fairness
from responsible_ai import run_privacy_check, run_reliability_check, run_security_check
from risk_engine import RiskLevel, compute_risk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_test_set():
    path = DATA_DIR / "test_set.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Test set not found.  Run train_model.py first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Baseline workflow
# ---------------------------------------------------------------------------

def run_baseline(bundle: dict, test_df: pd.DataFrame) -> dict:
    """
    Baseline: model prediction only, no Responsible AI layer.

    Returns metrics dict with processing time.
    """
    clf    = bundle["model"]
    scaler = bundle["scaler"]
    X = test_df[FEATURE_COLUMNS]
    y = test_df[TARGET_COLUMN]

    t0 = time.perf_counter()
    X_scaled = scaler.transform(X)
    preds    = clf.predict(X_scaled)
    t1       = time.perf_counter()

    elapsed = t1 - t0
    n = len(y)

    return {
        "workflow":          "baseline",
        "n_samples":         n,
        "accuracy":          round(accuracy_score(y, preds), 4),
        "precision":         round(precision_score(y, preds, zero_division=0), 4),
        "recall":            round(recall_score(y, preds, zero_division=0), 4),
        "f1_score":          round(f1_score(y, preds, zero_division=0), 4),
        "total_time_s":      round(elapsed, 4),
        "avg_time_ms":       round((elapsed / n) * 1000, 4),
        "pct_flagged":       0.0,
        "pct_human_review":  0.0,
        "n_warnings":        0,
        "predictions":       preds.tolist(),
    }


# ---------------------------------------------------------------------------
# Framework workflow
# ---------------------------------------------------------------------------

def run_framework(bundle: dict, test_df: pd.DataFrame) -> dict:
    """
    Full framework workflow including all RA checks and risk scoring.

    Processes each row individually to measure per-decision overhead.
    Note: SHAP explanation on many rows is slow; a progress indicator is shown.
    """
    clf    = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["feature_columns"]

    X      = test_df[FEATURE_COLUMNS]
    y      = test_df[TARGET_COLUMN].values
    groups = test_df[DEMOGRAPHIC_COLUMN]
    n      = len(y)

    # Pre-compute batch fairness for the whole test set
    X_scaled_all = scaler.transform(X)
    batch_preds  = clf.predict(X_scaled_all)
    fairness_res = evaluate_fairness(batch_preds, groups, y_true=y)
    batch_fairness_warning = fairness_res.fairness_warning

    preds_list    = []
    risk_levels   = []
    total_warnings = 0

    t0 = time.perf_counter()
    for i in range(n):
        row = X.iloc[[i]]
        row_dict = row.iloc[0].to_dict()

        X_sc = scaler.transform(row)
        pred = int(clf.predict(X_sc)[0])
        prob = float(clf.predict_proba(X_sc)[0, 1])

        # RA checks (lightweight; skip SHAP for speed on large batches)
        priv  = run_privacy_check(row_dict)
        sec   = run_security_check(row_dict)
        rel   = run_reliability_check(row_dict, prediction_probability=prob)

        # Minimal stub explanation (full SHAP is run on demand in API)
        from explainability import ExplanationResult
        explanation = ExplanationResult(success=True, human_readable_summary="batch skip")

        risk = compute_risk(
            prediction=pred,
            prediction_probability=prob,
            explanation=explanation,
            privacy_check=priv,
            security_check=sec,
            reliability_check=rel,
            fairness_warning=batch_fairness_warning,
        )

        preds_list.append(pred)
        risk_levels.append(risk.risk_level.value)
        total_warnings += len(risk.risk_flags)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n} ...", end="\r")

    t1 = time.perf_counter()
    elapsed = t1 - t0

    preds_arr = np.array(preds_list)
    risk_arr  = np.array(risk_levels)

    n_flagged = int((risk_arr == "MEDIUM").sum())
    n_review  = int((risk_arr == "HIGH").sum())

    return {
        "workflow":          "responsible_ai_framework",
        "n_samples":         n,
        "accuracy":          round(accuracy_score(y, preds_arr), 4),
        "precision":         round(precision_score(y, preds_arr, zero_division=0), 4),
        "recall":            round(recall_score(y, preds_arr, zero_division=0), 4),
        "f1_score":          round(f1_score(y, preds_arr, zero_division=0), 4),
        "total_time_s":      round(elapsed, 4),
        "avg_time_ms":       round((elapsed / n) * 1000, 4),
        "pct_flagged":       round(n_flagged / n * 100, 2),
        "pct_human_review":  round(n_review  / n * 100, 2),
        "n_warnings":        total_warnings,
        "fairness_dpd":      fairness_res.demographic_parity_difference,
        "fairness_warning":  batch_fairness_warning,
        "predictions":       preds_list,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Responsible AI Framework - Evaluation")
    print("=" * 60)

    bundle  = load_bundle()
    test_df = load_test_set()

    print(f"\nTest set: {len(test_df)} samples")

    print("\n[1/2] Running baseline ...")
    baseline = run_baseline(bundle, test_df)

    print("\n[2/2] Running full framework ...")
    framework = run_framework(bundle, test_df)

    # Comparison table
    metrics = ["accuracy", "precision", "recall", "f1_score",
               "avg_time_ms", "pct_flagged", "pct_human_review", "n_warnings"]
    print("\n\n=== COMPARISON RESULTS ===")
    print(f"{'Metric':<30} {'Baseline':>12} {'Framework':>14}")
    print("-" * 58)
    for m in metrics:
        b = baseline.get(m, "N/A")
        f = framework.get(m, "N/A")
        print(f"{m:<30} {str(b):>12} {str(f):>14}")

    if "fairness_dpd" in framework:
        print(f"\nFairness (demographic parity diff): {framework['fairness_dpd']}")
        print(f"Fairness warning active           : {framework['fairness_warning']}")

    # Save CSV
    rows = []
    for m in metrics + ["fairness_dpd", "fairness_warning"]:
        rows.append({
            "metric":    m,
            "baseline":  baseline.get(m),
            "framework": framework.get(m),
        })
    results_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "evaluation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")

    # Detailed JSON
    detail = {"baseline": baseline, "framework": framework}
    # Remove large lists before saving
    for w in detail.values():
        w.pop("predictions", None)
    json_path = DATA_DIR / "evaluation_detail.json"
    with open(json_path, "w") as f:
        json.dump(detail, f, indent=2, default=str)
    print(f"Detailed results -> {json_path}")

    print("\n[OK] Evaluation complete.")


if __name__ == "__main__":
    main()
