"""tests/test_model.py — Tests for model loading and prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent.parent / "models" / "loan_model.pkl"


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip("Model not found.  Run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture()
def valid_input():
    return {
        "income":              55000.0,
        "credit_score":        650.0,
        "existing_debt":       10000.0,
        "employment_duration": 5.0,
        "loan_amount":         25000.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_bundle_has_required_keys(bundle):
    assert "model"           in bundle
    assert "scaler"          in bundle
    assert "feature_columns" in bundle
    assert "model_version"   in bundle


def test_model_predicts_valid_input(bundle, valid_input):
    clf    = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["feature_columns"]

    row = pd.DataFrame([valid_input])[feats]
    X_sc = scaler.transform(row)
    pred = clf.predict(X_sc)

    assert pred[0] in [0, 1], "Prediction must be 0 or 1"


def test_model_returns_probabilities(bundle, valid_input):
    clf    = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["feature_columns"]

    row  = pd.DataFrame([valid_input])[feats]
    X_sc = scaler.transform(row)
    prob = clf.predict_proba(X_sc)

    assert prob.shape[1] == 2, "predict_proba must return two columns"
    assert 0.0 <= prob[0, 1] <= 1.0, "Probability must be in [0, 1]"


def test_model_metrics_present(bundle):
    metrics = bundle.get("metrics", {})
    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert key in metrics, f"Metrics dict missing '{key}'"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} must be in [0, 1]"


def test_feature_columns_match(bundle):
    from config import FEATURE_COLUMNS
    assert bundle["feature_columns"] == FEATURE_COLUMNS
