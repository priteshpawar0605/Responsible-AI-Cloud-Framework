"""
config.py — Central configuration for the Responsible AI Framework.

All thresholds are documented and configurable. Hard-coded values are replaced
with named constants so that students can adjust them without hunting through
business logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = str(BASE_DIR / "audit.db")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = str(MODEL_DIR / "loan_model.pkl")
MODEL_VERSION = "1.0.0"

# Features used for prediction (demographic attributes are excluded from
# training features to avoid using protected characteristics in the model).
FEATURE_COLUMNS = [
    "income",
    "credit_score",
    "existing_debt",
    "employment_duration",
    "loan_amount",
]

# Demographic column used only for fairness evaluation — never as a predictor.
DEMOGRAPHIC_COLUMN = "demographic_group"

# Target label column
TARGET_COLUMN = "approved"

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------
# Uncertainty threshold: predictions whose probability is below this value
# (i.e., the model is not confident) trigger a high-uncertainty risk flag.
# Rationale: 0.65 means the model assigns less than 65 % probability to the
# predicted class, which is close to the decision boundary.
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.65"))

# Fairness threshold: demographic-parity-difference values above this trigger
# a fairness warning. 0.10 follows the "80 % rule" spirit used in employment-
# disparate-impact analysis (a difference > 10 pp is commonly flagged).
FAIRNESS_THRESHOLD = float(os.getenv("FAIRNESS_THRESHOLD", "0.10"))

# Risk scoring weights - each contributing factor adds its score to the total.
RISK_WEIGHTS = {
    "high_uncertainty":   40,   # Model probability near decision boundary
    "fairness_warning":   25,   # Fairness metric exceeds threshold
    "privacy_concern":    20,   # Privacy check identified an issue
    "security_failure":   40,   # Input failed security validation (routes to review)
    "invalid_input":      35,   # Input failed domain-range validation
    "missing_assessment": 20,   # One or more checks could not be run
}

# Total score thresholds for routing
RISK_SCORE_HIGH = int(os.getenv("RISK_SCORE_HIGH", "40"))   # Route to human review
RISK_SCORE_MEDIUM = int(os.getenv("RISK_SCORE_MEDIUM", "20"))  # Flag but allow

# ---------------------------------------------------------------------------
# API security
# ---------------------------------------------------------------------------
# The API key is read from the environment.  The .env.example file shows the
# required variable name.  Never hard-code a real key here.
API_KEY = os.getenv("API_KEY", "dev-insecure-key-change-me")
API_KEY_HEADER = "X-API-Key"

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
SYNTHETIC_N_SAMPLES = int(os.getenv("SYNTHETIC_N_SAMPLES", "2000"))
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
MAX_REQUEST_SIZE_BYTES = 65_536   # 64 KB — basic defence against large payloads

# ---------------------------------------------------------------------------
# Feature value ranges used for input validation
# ---------------------------------------------------------------------------
FEATURE_RANGES = {
    "income":               (0,    1_000_000),
    "credit_score":         (300,  850),
    "existing_debt":        (0,    500_000),
    "employment_duration":  (0,    50),       # years
    "loan_amount":          (1000, 1_000_000),
}
