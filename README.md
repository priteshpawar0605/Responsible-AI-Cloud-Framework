# Responsible AI Framework for Trustworthy Decision-Making in Cloud Computing

An academic prototype demonstrating responsible AI principles applied to synthetic loan-application decisions.  The system generates predictions, explains them with SHAP, evaluates fairness with Fairlearn, assesses privacy/security/reliability, routes high-risk decisions for human review, and maintains a complete audit log.

> **Important:** This is a student prototype built on synthetic data.  It is not production-ready and does not eliminate AI risks.  Results do not reflect real-world lending outcomes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)          │
│  Dashboard · Predict · Fairness · Review · Audit · Eval │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (REST)
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend (api.py)              │
│  /predict  /decisions  /review  /metrics  /fairness     │
└──┬──────────────┬──────────────┬────────────────────────┘
   │              │              │
   ▼              ▼              ▼
train_model.py  responsible_ai.py  audit.py (SQLite)
explainability.py  fairness.py     risk_engine.py
config.py          evaluate.py
```

### Workflow per prediction

1. Validate input (security + reliability checks)
2. Generate model prediction + probability
3. Run SHAP explanation
4. Retrieve batch fairness results from test set
5. Run privacy check
6. Compute risk score from documented rules
7. Route: LOW → auto, HIGH → human review queue
8. Write audit record to SQLite

---

## Module Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Central configuration — thresholds, paths, feature names |
| `train_model.py` | Synthetic data generation, model training, evaluation |
| `explainability.py` | SHAP-based individual prediction explanations |
| `fairness.py` | Fairlearn group-level fairness metrics |
| `responsible_ai.py` | Privacy, security, and reliability checks |
| `risk_engine.py` | Risk scoring and decision routing rules |
| `audit.py` | SQLite audit log and reviewer workflow |
| `evaluate.py` | Baseline vs. framework comparison |
| `api.py` | FastAPI REST backend |
| `app.py` | Streamlit interactive dashboard |
| `tests/` | Pytest test suite |

---

## Prerequisites

- Python 3.10 or later (tested on 3.14)
- Windows 10 / 11 (also runs on Linux/macOS)
- Internet access for initial package installation
- No GPU, no paid APIs required

---

## Windows Installation

### 1. Clone or copy the project

```powershell
cd C:\stc
# Project is already in responsible-ai-framework\
cd responsible-ai-framework
```

### 2. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env and set API_KEY to a strong secret before sharing the server
```

---

## Running the System

### Step 1 — Train the model

```powershell
python train_model.py
```

This creates:
- `data/loan_applications.csv` — full synthetic dataset
- `data/test_set.csv` — held-out test set
- `models/loan_model.pkl` — trained model bundle
- `models/metrics.json` — test-set performance metrics

### Step 2 — Start the API

```powershell
uvicorn api:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### Step 3 — Start the dashboard

Open a second terminal:

```powershell
streamlit run app.py
```

Dashboard: http://localhost:8501

### Step 4 — (Optional) Run evaluation

```powershell
python evaluate.py
```

Creates `data/evaluation_results.csv` and `data/evaluation_detail.json`.

---

## Running Tests

```powershell
pytest tests/ -v
```

Run a single test file:
```powershell
pytest tests/test_risk_engine.py -v
```

Note: `test_model.py` and `test_api.py` require a trained model (`python train_model.py` first).

---

## API Reference

### Authentication

Protected endpoints require the header:
```
X-API-Key: <your API_KEY from .env>
```

### Example: make a prediction

```powershell
$headers = @{ "X-API-Key" = "dev-insecure-key-change-me"; "Content-Type" = "application/json" }
$body = '{"income":55000,"credit_score":650,"existing_debt":10000,"employment_duration":5,"loan_amount":25000}'
Invoke-RestMethod -Method POST -Uri http://localhost:8000/predict -Headers $headers -Body $body
```

**Response:**
```json
{
  "decision_id": "3f2a...",
  "prediction": 1,
  "prediction_label": "Approved",
  "prediction_prob": 0.8234,
  "risk_level": "LOW",
  "risk_score": 0,
  "risk_flags": [],
  "routing": "AUTO_APPROVED",
  "routing_reason": "Risk score 0 is below both thresholds...",
  "explanation_summary": "Top 5 features influencing...",
  "privacy_passed": true,
  "security_passed": true,
  "reliability_passed": true,
  "fairness_warning": false
}
```

### Example: submit a reviewer action

```powershell
$body = '{"reviewer_id":"analyst-1","action":"APPROVE","comments":"Verified manually"}'
Invoke-RestMethod -Method POST -Uri http://localhost:8000/review/<decision_id> -Headers $headers -Body $body
```

### All endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | No | Liveness check |
| GET | /model-info | No | Model version and features |
| POST | /predict | Yes | Full RA prediction pipeline |
| GET | /decisions | Yes | List audit log entries |
| GET | /decisions/{id} | Yes | Single decision |
| GET | /review/pending | Yes | Pending review queue |
| POST | /review/{id} | Yes | Submit reviewer action |
| GET | /metrics | No | Model performance metrics |
| GET | /fairness | No | Batch fairness metrics |
| GET | /stats | No | Dashboard summary counts |

---

## Demo Scenarios

### Low-risk decision (auto-approved)

```json
{"income": 80000, "credit_score": 750, "existing_debt": 5000,
 "employment_duration": 10, "loan_amount": 20000}
```

Expected: LOW risk, AUTO_APPROVED routing.

### High-risk decision (routed to human review)

```json
{"income": 20000, "credit_score": 310, "existing_debt": 45000,
 "employment_duration": 0.5, "loan_amount": 50000}
```

Expected: HIGH risk (uncertainty + potential out-of-range), HUMAN_REVIEW routing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not found` | Run `python train_model.py` |
| `SHAP is not installed` | Run `pip install shap` |
| `Fairlearn is not installed` | Run `pip install fairlearn` |
| PowerShell script blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port 8000 in use | `uvicorn api:app --port 8001` and set `API_BASE_URL=http://localhost:8001` in `.env` |
| Streamlit blank page | Ensure the API is running and `.env` has the correct `API_BASE_URL` |

---

## Optional Cloud Deployment

The system can be deployed to AWS, Azure, or Google Cloud.  **All options incur costs — review pricing before deploying.**

### General approach

1. Containerise with Docker (`Dockerfile` not included in prototype).
2. Deploy the FastAPI backend to a managed container service (AWS App Runner, Azure Container Apps, Cloud Run).
3. Deploy the Streamlit app to the same service or to Streamlit Community Cloud (free tier, public apps only).
4. Use a managed SQLite replacement (AWS RDS SQLite is not managed; use PostgreSQL instead) or a cloud object-store-backed SQLite for small workloads.
5. Store secrets in AWS Secrets Manager / Azure Key Vault / Google Secret Manager.
6. Replace the prototype API-key authentication with OAuth 2.0 or IAM-based auth.

**Cost note:** A minimal deployment (Cloud Run + Cloud SQL) costs approximately \$5–\$20/month depending on traffic.  The prototype can run entirely locally at zero cost.

---

## Security Limitations

This prototype implements basic security measures for educational demonstration:

- API key authentication (not production OAuth/JWT)
- Input validation via Pydantic
- Parameterized SQL queries (no string concatenation)
- Basic request-size limits
- Secrets in environment variables (not hard-coded)

**It is NOT suitable for production use without:**
- Proper authentication and authorisation (OAuth 2.0, RBAC)
- TLS/HTTPS termination
- Rate limiting and DDoS protection
- Penetration testing
- GDPR/legal privacy compliance review
- Production-grade logging and monitoring

---

## Known Limitations and Future Improvements

- Fairness metrics are computed on synthetic data; results cannot be generalised.
- SHAP explanations show correlation, not causation.
- The model is not calibrated; probability ≠ confidence.
- SQLite is not suitable for concurrent production workloads.
- No real PII detection in privacy checks.
- Human-review UI is minimal; a production system would need role-based access.
- Drift detection and retraining pipelines are not implemented.
- The baseline/framework comparison uses the same model; a true comparison would include model selection with fairness constraints.

---

## License

Academic prototype — no warranty.  Do not use for real lending decisions.
