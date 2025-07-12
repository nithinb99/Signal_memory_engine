#!/usr/bin/env bash

# ==============================================================================
# starter.sh — initialize environment and launch backend + frontend
# ==============================================================================

# 1) Load environment variables from .env
set -a
if [ -f .env ]; then
  . .env
else
  echo "Warning: .env file not found. Make sure PINECONE_API_KEY and OPENAI_API_KEY are set."
fi
set +a

# 2) Activate virtual environment (if exists)
if [ -f venv/bin/activate ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# 3) Install dependencies
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

# 4) Start FastAPI backend
echo "Starting FastAPI server at http://127.0.0.1:8000..."
uvicorn api.main:app --reload &

# 5) Start Streamlit frontend
# Replace `streamlit_app.py` with your actual Streamlit script name if different
echo "Starting Streamlit app at http://localhost:8501..."
streamlit run streamlit_app.py

# 6) Wait for jobs to finish (especially the Streamlit process)
wait