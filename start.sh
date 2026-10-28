#!/bin/bash
set -e

echo "Starting Responsible AI Platform..."

# Start FastAPI in background
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
