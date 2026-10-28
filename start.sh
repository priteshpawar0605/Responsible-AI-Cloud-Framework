#!/bin/bash
set -e

echo "Starting Responsible AI Platform..."

# Port for Streamlit (Render sets $PORT, default to 8501)
STREAMLIT_PORT=${PORT:-8501}

# Start FastAPI backend in background on port 8000
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in foreground
python -m streamlit run app.py \
    --server.port ${STREAMLIT_PORT} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
