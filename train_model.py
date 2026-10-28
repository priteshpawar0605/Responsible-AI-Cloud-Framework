"""
train_model.py — Synthetic-data generation and model training.

Run this script before starting the API or dashboard:
    python train_model.py

What it does
------------
1. Generates a synthetic loan-application dataset (no real personal data).
2. Splits data into train / validation / test sets.
3. Trains a Random Forest classifier.
4. Evaluates the model on the held-out test set.
5. Saves the trained model and the test set to disk.

Important caveats
-----------------
- This is a synthetic prototype.  Results do not reflect real-world lending
  patterns or establish real-world fairness.
- The demographic_group column is used only for fairness evaluation and is
  never passed to the model as a feature.
- Prediction probability reflects the model's internal score, not a
  calibrated confidence interval.
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from config import (
    DATA_DIR,
    DEMOGRAPHIC_COLUMN,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_SEED,
    SYNTHETIC_N_SAMPLES,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = SYNTHETIC_N_SAMPLES,
                                random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic loan-application dataset.

    Feature correlations are designed to create a realistic-looking but
    entirely artificial dataset.  No real applicant data is used.

    Demographic group is assigned randomly and is intentionally independent
    of the outcome in the data-generation process.  Any remaining disparity
    in model outputs is a product of the learned decision boundary and
    demonstrates why post-hoc fairness evaluation is necessary.
    """
    rng = np.random.default_rng(random_seed)

    n = n_samples

    # Core financial features
    income = rng.normal(loc=55_000, scale=25_000, size=n).clip(10_000, 300_000)
    credit_score = rng.normal(loc=650, scale=80, size=n).clip(300, 850)
    existing_debt = rng.exponential(scale=15_000, size=n).clip(0, 200_000)
    employment_duration = rng.exponential(scale=5, size=n).clip(0, 40)
    loan_amount = rng.normal(loc=25_000, scale=15_000, size=n).clip(1_000, 200_000)

    # Demographic group (used only for fairness — not a predictor)
    demographic_group = rng.choice(["Group_A", "Group_B", "Group_C"], size=n)

    # Synthetic approval logic (not intended to reflect real lending policy)
    # A higher score → more likely to be approved
    score = (
        (income / 10_000)
        + ((credit_score - 300) / 50)
        - (existing_debt / 10_000)
        + (employment_duration * 0.5)
        - (loan_amount / 20_000)
        + rng.normal(0, 2, size=n)       # noise
    )
    # Sigmoid-style threshold
    prob_approve = 1 / (1 + np.exp(-0.15 * (score - 10)))
    approved = (rng.uniform(size=n) < prob_approve).astype(int)

    df = pd.DataFrame({
        "income":               np.round(income, 2),
        "credit_score":         np.round(credit_score).astype(int),
        "existing_debt":        np.round(existing_debt, 2),
        "employment_duration":  np.round(employment_duration, 1),
        "loan_amount":          np.round(loan_amount, 2),
        DEMOGRAPHIC_COLUMN:     demographic_group,
        TARGET_COLUMN:          approved,
    })

    return df


# ---------------------------------------------------------------------------
# 2. Train / validate / test split
# ---------------------------------------------------------------------------

def split_dataset(df: pd.DataFrame):
    """
    Split into train (70 %), validation (15 %), test (15 %).

    The test set is held out and used only for final evaluation.  The
    validation set can be used for threshold tuning without contaminating
    the test set.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    demo = df[DEMOGRAPHIC_COLUMN]

    # First split off 30 % for val+test
    X_train, X_temp, y_train, y_temp, demo_train, demo_temp = train_test_split(
        X, y, demo, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    # Split the 30 % equally into val and test
    X_val, X_test, y_val, y_test, demo_val, demo_test = train_test_split(
        X_temp, y_temp, demo_temp,
        test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        demo_train, demo_val, demo_test,
    )


# ---------------------------------------------------------------------------
# 3. Model training
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Random Forest classifier.

    The scaler is fitted on training data only to prevent leakage.
    Both objects are returned so they can be saved together.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_scaled, y_train)

    return clf, scaler


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(clf, scaler, X: pd.DataFrame, y: pd.Series,
                   split_name: str = "test") -> dict:
    """
    Evaluate the model on a held-out split.

    Returns a dict of metrics that is also printed to stdout.
    """
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    cm   = confusion_matrix(y, y_pred).tolist()

    metrics = {
        "split":     split_name,
        "n_samples": len(y),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "confusion_matrix": cm,
    }

    print(f"\n=== {split_name.upper()} EVALUATION ===")
    print(f"  Samples  : {metrics['n_samples']}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    print(f"  Confusion matrix (rows=actual, cols=pred):")
    for row in cm:
        print(f"    {row}")
    print(classification_report(y, y_pred, zero_division=0))

    return metrics


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

def save_artifacts(clf, scaler, feature_columns: list, metrics: dict):
    """Save the model bundle and metrics to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model":           clf,
        "scaler":          scaler,
        "feature_columns": feature_columns,
        "model_version":   MODEL_VERSION,
        "metrics":         metrics,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved -> {MODEL_PATH}")

    # Save metrics as JSON for the API
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Responsible AI Framework - Model Training")
    print("=" * 60)

    # 1. Generate data
    print(f"\nGenerating {SYNTHETIC_N_SAMPLES} synthetic samples ...")
    df = generate_synthetic_dataset()

    # 2. Save raw dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "loan_applications.csv"
    df.to_csv(raw_path, index=False)
    print(f"Dataset saved -> {raw_path}  (shape: {df.shape})")
    print(f"  Approval rate: {df[TARGET_COLUMN].mean():.2%}")
    print(f"  Demographic groups: {df[DEMOGRAPHIC_COLUMN].value_counts().to_dict()}")

    # 3. Split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     demo_train, demo_val, demo_test) = split_dataset(df)

    print(f"\nSplit sizes - train: {len(X_train)}, "
          f"val: {len(X_val)}, test: {len(X_test)}")

    # Save test set (with demographics) for fairness evaluation
    test_df = X_test.copy()
    test_df[DEMOGRAPHIC_COLUMN] = demo_test.values
    test_df[TARGET_COLUMN] = y_test.values
    test_path = DATA_DIR / "test_set.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Test set saved -> {test_path}")

    # 4. Train
    print("\nTraining Random Forest classifier ...")
    clf, scaler = train_model(X_train, y_train)

    # 5. Evaluate on validation then test
    val_metrics  = evaluate_model(clf, scaler, X_val,  y_val,  "validation")
    test_metrics = evaluate_model(clf, scaler, X_test, y_test, "test")

    # 6. Save
    save_artifacts(clf, scaler, FEATURE_COLUMNS, test_metrics)

    print("\n[OK] Training complete.  You can now start the API and dashboard.")


if __name__ == "__main__":
    main()
